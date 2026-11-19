// Otsuka heuristic cluster optimizer for CU-1 prepared zero-field models.
// Ported from gpucluster.cu:63-257. Parallel component decisions are a heuristic.
// Union-find:86-97; colored moves:113-136; cluster/residual:160-252;
// final descent:150-158; per-anneal state:145-148,255-256.
// Standalone NVRTC source: CUDA vector types and intrinsics are built in.
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

struct Params {
    int sweeps, single_every, rounds, chunk;
    float beta_min, beta_knee, knee_frac, beta_max, tile_w, dimer_w;
};

struct Dev {
    int n, T, P, C;
    const int4* tiles; const char4* gauge; const int2* pairs;
    const int* weak_deg; const int* weak_ell; const int* all_deg; const int* all_ell;
    const int* color_off;
    const int* h; const int* tile_w; const int* link_w;  // tile_w, link_w null: the global weights
};

__device__ __forceinline__ uint64_t xs(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
__device__ __forceinline__ float urand(uint64_t& s) {
    return (float)(xs(s) >> 40) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}

#ifdef QUIP_OTSUKA_FAST
// Local build (QUIP_OTSUKA_FAST): 16-bit union-find parents (n <= 65536), so a
// block needs 7 instead of 9 shared bytes per node and three blocks fit an SM.
// Roots are still the smallest node of each component, so results match.
typedef unsigned short pid_t_;
__device__ __forceinline__ int find_root(pid_t_* p, int x) {
    while (true) { int y = ((volatile pid_t_*)p)[x]; if (y == x) return x; x = y; }
}
__device__ void unite(pid_t_* p, int a, int b) {
    while (true) {
        a = find_root(p, a); b = find_root(p, b);
        if (a == b) return;
        if (a < b) { int t = a; a = b; b = t; }
        unsigned short old = atomicCAS(&p[a], (unsigned short)a, (unsigned short)b);
        if (old == (unsigned short)a) return;
    }
}
#else
typedef int pid_t_;
__device__ __forceinline__ int find_root(int* p, int x) {
    while (true) { int y = atomicAdd(&p[x], 0); if (y == x) return x; x = y; }
}
__device__ void unite(int* p, int a, int b) {
    while (true) {
        a = find_root(p, a); b = find_root(p, b);
        if (a == b) return;
        if (a < b) { int t = a; a = b; b = t; }
        int old = atomicCAS(&p[a], a, b);
        if (old == a) return;
    }
}
#endif

// The k-th (0-based) node of a tile whose bit is set in `mask`, by selects
// rather than a dynamically indexed local array.
__device__ __forceinline__ int tile_pick(int4 nd, unsigned mask, unsigned k) {
    int at = __fns(mask, 0, (int)k + 1);
    return at == 0 ? nd.x : at == 1 ? nd.y : at == 2 ? nd.z : nd.w;
}

#ifdef QUIP_OTSUKA_PHASE_TIMING
// Profiling only: block 0, thread 0 clocks each barrier-to-barrier phase and prints the totals.
#define PT_DECL unsigned long long pt_acc[8] = {0,0,0,0,0,0,0,0}; long long pt_t = clock64();
#define PT(k) do { long long pt_n = clock64(); pt_acc[k] += pt_n - pt_t; pt_t = pt_n; } while (0)
#define PT_PRINT if (blockIdx.x == 0 && threadIdx.x == 0) printf("phase-cycles nsweeps %d unite %llu compress %llu accum %llu decide %llu flip %llu metro %llu\n", nsweeps, pt_acc[0], pt_acc[1], pt_acc[2], pt_acc[3], pt_acc[4], pt_acc[5]);
#else
#define PT_DECL
#define PT(k)
#define PT_PRINT
#endif

#ifdef QUIP_OTSUKA_SWEEP_V2
// Local build (QUIP_OTSUKA_SWEEP_V2): union-find with path halving and plain
// volatile loads. unite links the larger root under the smaller, so parents
// only decrease (p[x] <= x) and every root is its component's smallest node:
// any concurrent halving store writes an ancestor, and the roots, hence all
// draws and flips, match the stock kernel's.
__device__ __forceinline__ int find_v2(int* p, int x) {
    volatile int* vp = p;
    int par = vp[x];
    if (par != x) {
        int next, prev = x;
        while (par > (next = vp[par])) { vp[prev] = next; prev = par; par = next; }
    }
    return par;
}
// Compress-pass find. No unions run in this pass, so the root found is final:
// a second walk stores it on every node of the path. Every store is the final
// root, so racing walks can only shorten each other's paths.
__device__ __forceinline__ int find_compress(int* p, int x) {
    volatile int* vp = p;
    int r = vp[x];
    if (r == x) return x;
    int y;
    while ((y = vp[r]) != r) r = y;
    while ((y = vp[x]) != r) { vp[x] = r; x = y; }
    return r;
}
__device__ __forceinline__ void unite_v2(int* p, int a, int b) {
    a = find_v2(p, a); b = find_v2(p, b);
    while (a != b) {
        if (a < b) { int t = a; a = b; b = t; }
        int old = atomicCAS(&p[a], a, b);
        if (old == a) return;
        a = find_v2(p, old); b = find_v2(p, b);
    }
}
// Append up to two union edges (a << 16 | b; NO_EDGE for none) per lane to the
// shared list, one atomic per warp. All 32 lanes must call it. Past `cap` the
// lane unites at once instead: union order never changes the components.
#define NO_EDGE 0xffffffffu
__device__ __forceinline__ unsigned edge(int a, int b) { return ((unsigned)a << 16) | (unsigned)b; }
__device__ __forceinline__ void push_edges(unsigned e1, unsigned e2, unsigned* list, int* count,
                                           int cap, int* parent) {
    unsigned b1 = __ballot_sync(0xffffffffu, e1 != NO_EDGE), b2 = __ballot_sync(0xffffffffu, e2 != NO_EDGE);
    int tot = __popc(b1) + __popc(b2);
    if (tot == 0) return;
    int lane = threadIdx.x & 31, base = 0;
    if (lane == 0) base = atomicAdd(count, tot);
    base = __shfl_sync(0xffffffffu, base, 0);
    unsigned lt = (1u << lane) - 1u;
    if (e1 != NO_EDGE) {
        int k = base + __popc(b1 & lt);
        if (k < cap) list[k] = e1; else unite_v2(parent, (int)(e1 >> 16), (int)(e1 & 0xffffu));
    }
    if (e2 != NO_EDGE) {
        int k = base + __popc(b1) + __popc(b2 & lt);
        if (k < cap) list[k] = e2; else unite_v2(parent, (int)(e2 >> 16), (int)(e2 & 0xffffu));
    }
}
// Fields of two nodes at once (b < 0: none), loads interleaved for twice the
// memory-level parallelism. skip: exclude neighbors whose parent equals the
// node's own root (accumulate); null parent: plain field (Metropolis).
__device__ __forceinline__ void field2(const int* ell, int n, int ua, int da, int ub, int db,
                                       const signed char* s, const int* parent, int ra, int rb,
                                       int& fa, int& fb) {
    int dm = min(da, db), e = 0;
    for (; e < dm; ++e) {
        unsigned wa = (unsigned)__ldg(&ell[e * n + ua]);
        unsigned wb = (unsigned)__ldg(&ell[e * n + ub]);
        int va = wa >> 16, vb = wb >> 16;
        if (!parent || parent[va] != ra) fa += (int)(short)(wa & 0xffffu) * s[va];
        if (!parent || parent[vb] != rb) fb += (int)(short)(wb & 0xffffu) * s[vb];
    }
    for (; e < da; ++e) {
        unsigned wa = (unsigned)__ldg(&ell[e * n + ua]);
        int va = wa >> 16;
        if (!parent || parent[va] != ra) fa += (int)(short)(wa & 0xffffu) * s[va];
    }
    for (; e < db; ++e) {
        unsigned wb = (unsigned)__ldg(&ell[e * n + ub]);
        int vb = wb >> 16;
        if (!parent || parent[vb] != rb) fb += (int)(short)(wb & 0xffffu) * s[vb];
    }
}
#ifdef QUIP_OTSUKA_SWEEP_V3
// Local build (QUIP_OTSUKA_SWEEP_V3): the sweep's full ELL arrives repacked by the
// host (device.rs pack_ell4): int4 0 holds the chunk count K in .x, chunk k of
// node u is int4 1 + k * n + u (entries 4k..4k+3, zero past the degree). A
// node's K chunks load as K independent 16-byte reads issued before any use:
// one L2 round trip instead of one per four entries plus the degree load.
// Zero words add 0 * s[0], so the integer fields equal V2's.
#define V3_KMAX 6
__device__ __forceinline__ int term4(int4 w, const signed char* s, const int* parent, int r) {
    int f = 0;
    unsigned x = (unsigned)w.x, y = (unsigned)w.y, z = (unsigned)w.z, t = (unsigned)w.w;
    if (!parent || parent[x >> 16] != r) f += (int)(short)(x & 0xffffu) * s[x >> 16];
    if (!parent || parent[y >> 16] != r) f += (int)(short)(y & 0xffffu) * s[y >> 16];
    if (!parent || parent[z >> 16] != r) f += (int)(short)(z & 0xffffu) * s[z >> 16];
    if (!parent || parent[t >> 16] != r) f += (int)(short)(t & 0xffffu) * s[t >> 16];
    return f;
}
// Field of node u (0 if !valid); parent/r as in field2; deg: the row's
// degree array. The degree and all chunks load together (independent reads,
// one round trip); only chunks below ceil(deg / 4) are summed, so padding
// costs at most three zero words per row.
__device__ __forceinline__ int field4(const int4* __restrict__ e4, const int* __restrict__ deg,
                                      int n, int K, int u, bool valid,
                                      const signed char* s, const int* parent, int r) {
    int4 w[V3_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int d = valid ? __ldg(&deg[u]) : 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) w[k] = (valid && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
    int kd = (d + 3) >> 2, f = 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k)
        if (k < kd) f += term4(w[k], s, parent, r);
    for (int k = V3_KMAX; k < kd; ++k) f += term4(__ldg(&e4[1 + k * n + u]), s, parent, r);
    return f;
}
__device__ __forceinline__ void field4_2(const int4* __restrict__ e4, const int* __restrict__ deg,
                                         int n, int K, int ua, bool va, int ub, bool vb,
                                         const signed char* s, const int* parent, int ra, int rb,
                                         int& fa, int& fb) {
#ifndef QUIP_V3_NOPAIRLOAD
    // Both nodes' chunks in flight before either is summed (fits the
    // 64-register cap of __launch_bounds__(512, 2)).
    int4 wa[V3_KMAX], wb[V3_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int da = va ? __ldg(&deg[ua]) : 0, db = vb ? __ldg(&deg[ub]) : 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) {
        wa[k] = (va && k < K) ? __ldg(&e4[1 + k * n + ua]) : zero;
        wb[k] = (vb && k < K) ? __ldg(&e4[1 + k * n + ub]) : zero;
    }
    int ka = (da + 3) >> 2, kb = (db + 3) >> 2;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) {
        if (k < ka) fa += term4(wa[k], s, parent, ra);
        if (k < kb) fb += term4(wb[k], s, parent, rb);
    }
    for (int k = V3_KMAX; k < ka; ++k) fa += term4(__ldg(&e4[1 + k * n + ua]), s, parent, ra);
    for (int k = V3_KMAX; k < kb; ++k) fb += term4(__ldg(&e4[1 + k * n + ub]), s, parent, rb);
#else
    fa += field4(e4, deg, n, K, ua, va, s, parent, ra);
    fb += field4(e4, deg, n, K, ub, vb, s, parent, rb);
#endif
}
#define V3SIGN 1
// V3: from the compress pass to the flips, parent[i] holds root | sign bit of
// s[i] (owner-written, every read masks with V3M), so a weak edge costs one
// shared load for both the root test and the spin. J * s[v] is unchanged.
#ifdef QUIP_OTSUKA_SWEEP_V6
// V6: tagged parents also carry bit 30 (V3TAG), so for a tagged word p and a
// coupling sign c in bit 31, (p ^ c) >> 30 is the sign of J * s[v] (+1 or -1)
// in one shift. Roots are below 2^30; every reader masks with V3M.
#define V3M 0x3fffffff
#define V3TAG 0x40000000
#else
#define V3M 0x7fffffff
#define V3TAG 0
#endif
__device__ __forceinline__ int sbit(signed char x) { return x < 0 ? (int)0x80000000 : 0; }
__device__ __forceinline__ int wterm(unsigned w, const int* parent, int r) {
    int p = parent[w >> 16], j = (int)(short)(w & 0xffffu);
    return (p & V3M) != r ? (p < 0 ? -j : j) : 0;
}
// V3: the weak rows arrive int4-packed too. With the sign tag a word costs one
// shared load, so loading a row in one round trip now wins (the plain-s
// version of this measured slower).
__device__ __forceinline__ int field4s(const int4* __restrict__ e4, const int* __restrict__ deg,
                                       int n, int K, int u, bool valid, const int* parent, int r) {
    int4 w[V3_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int d = valid ? __ldg(&deg[u]) : 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) w[k] = (valid && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
    int kd = (d + 3) >> 2, f = 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k)
        if (k < kd)
            f += wterm((unsigned)w[k].x, parent, r) + wterm((unsigned)w[k].y, parent, r)
               + wterm((unsigned)w[k].z, parent, r) + wterm((unsigned)w[k].w, parent, r);
    for (int k = V3_KMAX; k < kd; ++k) {
        int4 x = __ldg(&e4[1 + k * n + u]);
        f += wterm((unsigned)x.x, parent, r) + wterm((unsigned)x.y, parent, r)
           + wterm((unsigned)x.z, parent, r) + wterm((unsigned)x.w, parent, r);
    }
    return f;
}
#endif
#ifdef QUIP_OTSUKA_SWEEP_V4
// Local build (QUIP_OTSUKA_SWEEP_V4, implies V3). No V4 change touches a
// thread's RNG stream or the order of its draws, so draws and results match V3:
// - Union pass: a flat per-lane loop. One step advances both finds of the
//   lane's edge by one hop (two independent shared loads) or tries the link;
//   a lane that finishes an edge starts its next one at once instead of
//   waiting for the slowest lane of its warp. The last V4_TAIL * blockDim
//   edges come from a block-wide queue, so warps also finish together. Unions
//   only link a larger root under a smaller one and halve paths to ancestors
//   (as in V2), so every component and its root (its smallest node) match V3
//   whatever order the edges run in.
// - Compress pass: four owned nodes' entries load before any is used. Its
//   output, root | sign bit for every node, is unique.
// - Tile draws read one packed int4 per tile (four 16-bit nodes, gauge sign
//   bits, tile weight) prefetched an iteration ahead and pick tile nodes from a
//   constant table instead of __fns; pair draws read one packed int2.
// - Shared accesses in the union, compress, Metropolis and accumulate loops use
//   32-bit shared-window addresses (ld/st/atom.shared). Through the generic
//   pointers nvcc rebuilt a 64-bit address per access (S2R SR_CgaCtaId,
//   IMAD.WIDE, IADD.64) and V3 spills inside the Metropolis chunk loop.
// Ablation (QUIP_OTSUKA_NVRTC_OPTS): -DV4_NO_UNION, -DV4_NO_COMPRESS and
// -DV4_NO_SHADDR put back the V3 code of that part; draws stay packed.
#ifndef V4_REFILL
#define V4_REFILL 8   // idle lanes that trigger a warp's queue refill
#endif
#ifndef V4_TAIL
#define V4_TAIL 2     // queue-served edges, in units of blockDim
#endif
__device__ __forceinline__ unsigned sh_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
// The same, pinned in a register for a loop: ptxas otherwise rebuilds it per
// access from S2R SR_CgaCtaId (a variable-latency read) on the critical path.
// All lanes must call it.
__device__ __forceinline__ unsigned sh_addr_pinned(const void* p) {
    return __shfl_sync(0xffffffffu, sh_addr(p), 0);
}
// Volatile shared word (the union and compress passes race on parents).
__device__ __forceinline__ int ldv32(unsigned a) {
    int v; asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v;
}
__device__ __forceinline__ void stv32(unsigned a, int v) {
    asm volatile("st.volatile.shared.b32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ int cas32(unsigned a, int cmp, int val) {
    int old; asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "r"(a), "r"(cmp), "r"(val) : "memory");
    return old;
}
// Phase-stable shared reads (no writer until the next barrier).
__device__ __forceinline__ int ld32(unsigned a) {
    int v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v;
}
__device__ __forceinline__ int lds8(unsigned a) {
    int v; asm volatile("ld.shared.s8 %0, [%1];" : "=r"(v) : "r"(a)); return v;
}
__device__ __forceinline__ void red_add(unsigned a, int v) {
    asm volatile("red.shared.add.s32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
// Position (0-3) of the k-th set bit of a 4-bit mask, two bits per k, one
// byte per mask: masks 0-7 in the low word, 8-15 in the high word.
#define V4_PICK_LO 0x2409080204010000ull
#define V4_PICK_HI 0xe439380e340d0c03ull
__device__ __forceinline__ int pick_pos(unsigned mask, unsigned k) {
    unsigned long long t = (mask & 8u) ? V4_PICK_HI : V4_PICK_LO;
    return (int)((t >> (((mask & 7u) << 3) + (k << 1))) & 3ull);
}
// Node at position 0-3 of a packed tile (x: nodes 0 | 1 << 16, y: 2 | 3 << 16).
__device__ __forceinline__ int tile_node(int4 rec, int at) {
    unsigned w = (unsigned)((at & 2) ? rec.y : rec.x);
    return (int)((w >> ((at & 1) << 4)) & 0xffffu);
}
__device__ __forceinline__ int tile_pick_v4(int4 rec, unsigned mask, unsigned k) {
    return tile_node(rec, pick_pos(mask, k));
}
// up mask of a packed tile: bit k set when s[node k] * gauge k > 0. rec.z
// holds the gauge-positive bits in 0-3 and the gauge-negative bits in 4-7.
__device__ __forceinline__ unsigned tile_up(int4 rec, unsigned s_sh) {
    unsigned x = (unsigned)rec.x, y = (unsigned)rec.y, g = (unsigned)rec.z;
    unsigned sp = (lds8(s_sh + (x & 0xffffu)) > 0 ? 1u : 0u) | (lds8(s_sh + (x >> 16)) > 0 ? 2u : 0u)
                | (lds8(s_sh + (y & 0xffffu)) > 0 ? 4u : 0u) | (lds8(s_sh + (y >> 16)) > 0 ? 8u : 0u);
    return ((sp & g) | (~sp & (g >> 4))) & 15u;
}
// One hop of a path-halving find: (prev, cur) with cur the loaded parent of
// prev; prev == cur marks a root. nx is cur's parent. p: parent array base.
__device__ __forceinline__ void hop(unsigned p, int& prev, int& cur, int nx) {
    if (nx < cur) stv32(p + 4u * (unsigned)prev, nx);
    prev = cur; cur = nx;
}
// All of a block's threads call this between barriers; m = list length.
__device__ __forceinline__ void union_pass_v4(int* parent, const unsigned* list, int m, int* qhead) {
    const unsigned p = sh_addr_pinned(parent), l = p + (unsigned)((const char*)list - (const char*)parent),
                   qh = p + (unsigned)((const char*)qhead - (const char*)parent);
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31, bd = blockDim.x;
    const unsigned lt = (1u << lane) - 1u;
    // Static share: edges threadIdx.x + j * bd for j < js; the queue serves
    // edges js * bd .. m - 1.
    const int js = max(0, m / bd - V4_TAIL), qbase = js * bd;
    int j = 0;
    unsigned ne = js > 0 ? (unsigned)ld32(l + 4u * threadIdx.x) : 0u;  // next static edge, loaded ahead
    int st = 0;                     // 0 needs an edge, 1 working, 2 done
    int pa = 0, ca = 0, pb = 0, cb = 0;
    for (;;) {
        if (st == 0 && j < js) {
            pa = (int)(ne >> 16); pb = (int)(ne & 0xffffu);
            ca = ldv32(p + 4u * pa); cb = ldv32(p + 4u * pb);
            st = 1;
            if (++j < js) ne = (unsigned)ld32(l + 4u * (threadIdx.x + j * bd));
        }
        unsigned idle = __ballot_sync(FULL, st == 0);
        unsigned busy = __ballot_sync(FULL, st == 1);
        if (idle && (!busy || __popc(idle) >= V4_REFILL)) {
            int got = 0;
            if (lane == 0) asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(got) : "r"(qh), "r"(__popc(idle)) : "memory");
            got = __shfl_sync(FULL, got, 0);
            if (st == 0) {
                int k = qbase + got + __popc(idle & lt);
                if (k < m) {
                    unsigned e = (unsigned)ld32(l + 4u * k);
                    pa = (int)(e >> 16); pb = (int)(e & 0xffffu);
                    ca = ldv32(p + 4u * pa); cb = ldv32(p + 4u * pb);
                    st = 1;
                } else {
                    st = 2;
                }
            }
            busy = __ballot_sync(FULL, st == 1);
        }
        if (!busy) break;
        if (st == 1) {
            bool da = pa == ca, db = pb == cb;
            if (da && db) {
                if (ca == cb) {
                    st = 0;
                } else {
                    int hi = max(ca, cb), lo = min(ca, cb);
                    int old = cas32(p + 4u * hi, hi, lo);
                    if (old == hi) {
                        st = 0;
                    } else {
                        // As unite_v2: find from old and from lo again.
                        pa = hi; ca = old; pb = lo; cb = ldv32(p + 4u * lo);
                    }
                }
            } else {
                int na = da ? ca : ldv32(p + 4u * ca);
                int nb = db ? cb : ldv32(p + 4u * cb);
                if (!da) hop(p, pa, ca, na);
                if (!db) hop(p, pb, cb, nb);
            }
        }
    }
}
// Compress pass of V3 (pointer jumping, sign tags), four owned nodes per batch.
__device__ __forceinline__ void compress_v4(int* parent, const signed char* s, int n) {
    const unsigned p = sh_addr_pinned(parent);
    const int bd = blockDim.x;
    for (bool again = true; again;) {
        again = false;
        for (int i0 = threadIdx.x; i0 < n; i0 += 4 * bd) {
            int v[4], pp[4], sg[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int i = i0 + q * bd;
                v[q] = i < n ? ldv32(p + 4u * i) : 0;
                sg[q] = i < n ? sbit(s[i]) : 0;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) pp[q] = i0 + q * bd < n ? ldv32(p + 4u * (v[q] & V3M)) & V3M : 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int i = i0 + q * bd;
                if (i < n) {
                    int pi = v[q] & V3M;
                    if (pp[q] != pi) again = true;
                    int nv = (pp[q] != pi ? pp[q] : pi) | sg[q] | V3TAG;
                    if (nv != v[q]) stv32(p + 4u * i, nv);
                }
            }
        }
    }
}
#ifdef QUIP_OTSUKA_UNION_SV
// Local build (QUIP_OTSUKA_UNION_SV, implies V4): a Shiloach-Vishkin-style
// union pass in flat, branch-light passes instead of union_pass_v4's per-lane
// find/CAS loops. Every write is red.min(p[x], y) with y < x in x's component,
// so parents only decrease (p[x] <= x) and stay in the component: the forest
// never has a cycle, every tree lies inside one component, and a tree's root
// is the smallest node of the tree. The pass ends when every listed edge has
// both ends under one root after a full jump (or after the CAS fix-up below),
// so trees are exactly the components and every root is its component's
// smallest node: the unique result union_pass_v4 and unite_v2 produce, hence
// the same compress output, draws and flips.
// - Hook (no edge overflowed into unite_v2 during the draws): red.min the
//   larger end of every edge to the smaller end. It can overwrite a link set
//   by another listed edge; the rounds below re-check every listed edge. With
//   an overflow the forest already holds CAS links that no listed edge would
//   re-check, so the hook is skipped and the rounds hook only roots.
// - Jump: every node to its root (no writer of roots in this phase).
// - Round (SV_ROUNDS of them): for each listed edge with different roots,
//   red.min the larger root to the smaller. Only roots of the round's start
//   are written, so the jumped non-root links stay. A round that hooks
//   nothing ends the pass (every node already points at its root).
// - Fix-up: after SV_ROUNDS hooking rounds, jump again and unite_v2 the edges
//   whose ends still have different roots (CAS links of roots, as V2).
#ifndef SV_ROUNDS
#define SV_ROUNDS 1
#endif
__device__ __forceinline__ void red_min(unsigned a, int v) {
    asm volatile("red.shared.min.s32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
// Every owned node to its root, four at a time, until this thread's nodes are
// done (other threads' jumps only shorten the walk; roots are fixed).
__device__ __forceinline__ void jump_sv(unsigned p, int n) {
    const int bd = blockDim.x;
    for (bool again = true; again;) {
        again = false;
        for (int i0 = threadIdx.x; i0 < n; i0 += 4 * bd) {
            int v[4], pp[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = i0 + q * bd < n ? ldv32(p + 4u * (i0 + q * bd)) : 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) pp[q] = i0 + q * bd < n ? ldv32(p + 4u * v[q]) : 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (i0 + q * bd < n && pp[q] != v[q]) {
                    again = true;
                    stv32(p + 4u * (i0 + q * bd), pp[q]);
                }
            }
        }
    }
}
// All of a block's threads call this between barriers; total = edges drawn
// (the list holds min(total, n) of them; the rest were united at once).
// Returns after a barrier.
__device__ void union_sv(int* parent, const unsigned* list, int total, int n) {
    const unsigned p = sh_addr_pinned(parent), l = p + (unsigned)((const char*)list - (const char*)parent);
    const int bd = blockDim.x, m = min(total, n);
    if (total <= n) {
        for (int k = threadIdx.x; k < m; k += bd) {
            const unsigned e = (unsigned)ld32(l + 4u * k);
            const int a = (int)(e >> 16), b = (int)(e & 0xffffu);
            if (a != b) red_min(p + 4u * (unsigned)max(a, b), min(a, b));
        }
        __syncthreads();
    }
    for (int round = 0; round < SV_ROUNDS; ++round) {
        jump_sv(p, n);
        __syncthreads();
        int hooked = 0;
        for (int k = threadIdx.x; k < m; k += bd) {
            const unsigned e = (unsigned)ld32(l + 4u * k);
            const int ra = ldv32(p + 4u * (e >> 16)), rb = ldv32(p + 4u * (e & 0xffffu));
            if (ra != rb) {
                red_min(p + 4u * (unsigned)max(ra, rb), min(ra, rb));
                hooked = 1;
            }
        }
        if (!__syncthreads_or(hooked)) return;
    }
    jump_sv(p, n);
    __syncthreads();
    for (int k = threadIdx.x; k < m; k += bd) {
        const unsigned e = (unsigned)ld32(l + 4u * k);
        const int a = (int)(e >> 16), b = (int)(e & 0xffffu);
        if (ldv32(p + 4u * a) != ldv32(p + 4u * b)) unite_v2(parent, a, b);
    }
    __syncthreads();
}
#endif
#ifndef V4_NO_SHADDR
#define V4_SHADDR 1
// term4 / field4_2 / wterm / field4s with 32-bit shared addresses; same
// integer sums (zero padding words add 0 * s[0]).
__device__ __forceinline__ int term4_sh(int4 w, unsigned s_sh) {
    unsigned x = (unsigned)w.x, y = (unsigned)w.y, z = (unsigned)w.z, t = (unsigned)w.w;
    return (int)(short)(x & 0xffffu) * lds8(s_sh + (x >> 16)) + (int)(short)(y & 0xffffu) * lds8(s_sh + (y >> 16))
         + (int)(short)(z & 0xffffu) * lds8(s_sh + (z >> 16)) + (int)(short)(t & 0xffffu) * lds8(s_sh + (t >> 16));
}
__device__ __forceinline__ void field4_2_sh(const int4* __restrict__ e4, const int* __restrict__ deg,
                                            int n, int K, int ua, bool va, int ub, bool vb,
                                            unsigned s_sh, int& fa, int& fb) {
    int4 wa[V3_KMAX], wb[V3_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int da = va ? __ldg(&deg[ua]) : 0, db = vb ? __ldg(&deg[ub]) : 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) {
        wa[k] = (va && k < K) ? __ldg(&e4[1 + k * n + ua]) : zero;
        wb[k] = (vb && k < K) ? __ldg(&e4[1 + k * n + ub]) : zero;
    }
    int ka = (da + 3) >> 2, kb = (db + 3) >> 2;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) {
        if (k < ka) fa += term4_sh(wa[k], s_sh);
        if (k < kb) fb += term4_sh(wb[k], s_sh);
    }
    for (int k = V3_KMAX; k < ka; ++k) fa += term4_sh(__ldg(&e4[1 + k * n + ua]), s_sh);
    for (int k = V3_KMAX; k < kb; ++k) fb += term4_sh(__ldg(&e4[1 + k * n + ub]), s_sh);
}
__device__ __forceinline__ int wterm_sh(unsigned w, unsigned p_sh, int r) {
    int p = ld32(p_sh + ((w >> 16) << 2)), j = (int)(short)(w & 0xffffu);
    return (p & V3M) != r ? (p < 0 ? -j : j) : 0;
}
__device__ __forceinline__ int field4s_sh(const int4* __restrict__ e4, const int* __restrict__ deg,
                                          int n, int K, int u, bool valid, unsigned p_sh, int r) {
    int4 w[V3_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int d = valid ? __ldg(&deg[u]) : 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k) w[k] = (valid && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
    int kd = (d + 3) >> 2, f = 0;
#pragma unroll
    for (int k = 0; k < V3_KMAX; ++k)
        if (k < kd)
            f += wterm_sh((unsigned)w[k].x, p_sh, r) + wterm_sh((unsigned)w[k].y, p_sh, r)
               + wterm_sh((unsigned)w[k].z, p_sh, r) + wterm_sh((unsigned)w[k].w, p_sh, r);
    for (int k = V3_KMAX; k < kd; ++k) {
        int4 x = __ldg(&e4[1 + k * n + u]);
        f += wterm_sh((unsigned)x.x, p_sh, r) + wterm_sh((unsigned)x.y, p_sh, r)
           + wterm_sh((unsigned)x.z, p_sh, r) + wterm_sh((unsigned)x.w, p_sh, r);
    }
    return f;
}
#ifdef QUIP_OTSUKA_SWEEP_V5
#define V5_ELL16 1
// Local build (QUIP_OTSUKA_SWEEP_V5, implies V4). 16-bit ELL entries for the
// Metropolis and accumulate gathers (device.rs pack_ell16): entry = neighbor
// << 2 | code, eight per int4, and the header int4 is {K16, 1, J0 | J1 << 16,
// J2 | J3 << 16}: a graph's ELL holds at most four distinct couplings (the
// dimer family: {+-1000, +-62} full, {+-62, +-750} weak at tile 250). A row
// of width 20 is three 16-byte loads instead of five. Padding entries point
// at the row's own node with code 0: in the accumulate they add 0 (the node's
// own root), and Metropolis subtracts their npad * J0 * s[u] (s[u] is only
// written by this thread, after its field), so the integer fields, hence every
// draw and flip, equal V4's. A header .y other
// than 1 (graph not eligible: >4 couplings or >16383 nodes) means the buffer
// is V4's int4 ELL, and the V4 gathers run (a block-uniform branch).
#define V5_KMAX 3
// Coupling of a 2-bit code: prmt picks the code's 16-bit half of {t01, t23}
// with sign replication (selector 0x9910 + code * 0x2222).
__device__ __forceinline__ int jcode(unsigned c, unsigned t01, unsigned t23) {
    int j; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(j) : "r"(t01), "r"(t23), "r"(c * 0x2222u + 0x9910u)); return j;
}
__device__ __forceinline__ int mterm16(unsigned e, unsigned s_sh, unsigned t01, unsigned t23) {
    return jcode(e & 3u, t01, t23) * lds8(s_sh + (e >> 2));
}
__device__ __forceinline__ int term16_sh(int4 w, unsigned s_sh, unsigned t01, unsigned t23) {
    unsigned x = (unsigned)w.x, y = (unsigned)w.y, z = (unsigned)w.z, t = (unsigned)w.w;
    return mterm16(x & 0xffffu, s_sh, t01, t23) + mterm16(x >> 16, s_sh, t01, t23)
         + mterm16(y & 0xffffu, s_sh, t01, t23) + mterm16(y >> 16, s_sh, t01, t23)
         + mterm16(z & 0xffffu, s_sh, t01, t23) + mterm16(z >> 16, s_sh, t01, t23)
         + mterm16(t & 0xffffu, s_sh, t01, t23) + mterm16(t >> 16, s_sh, t01, t23);
}
__device__ __forceinline__ void field16_2_sh(const int4* __restrict__ e4, const int* __restrict__ deg,
                                             int n, int K, unsigned t01, unsigned t23,
                                             int ua, bool va, int ub, bool vb,
                                             unsigned s_sh, int& fa, int& fb) {
    int4 wa[V5_KMAX], wb[V5_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int da = va ? __ldg(&deg[ua]) : 0, db = vb ? __ldg(&deg[ub]) : 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k) {
        wa[k] = (va && k < K) ? __ldg(&e4[1 + k * n + ua]) : zero;
        wb[k] = (vb && k < K) ? __ldg(&e4[1 + k * n + ub]) : zero;
    }
    int ka = (da + 7) >> 3, kb = (db + 7) >> 3;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k) {
        if (k < ka) fa += term16_sh(wa[k], s_sh, t01, t23);
        if (k < kb) fb += term16_sh(wb[k], s_sh, t01, t23);
    }
    for (int k = V5_KMAX; k < ka; ++k) fa += term16_sh(__ldg(&e4[1 + k * n + ua]), s_sh, t01, t23);
    for (int k = V5_KMAX; k < kb; ++k) fb += term16_sh(__ldg(&e4[1 + k * n + ub]), s_sh, t01, t23);
    // Padding entries (8 * k - deg per row) each added J0 * s[own node].
    fa -= ((ka << 3) - da) * mterm16((unsigned)ua << 2, s_sh, t01, t23);
    fb -= ((kb << 3) - db) * mterm16((unsigned)ub << 2, s_sh, t01, t23);
}
__device__ __forceinline__ int wterm16_sh(unsigned e, unsigned p_sh, int r, unsigned t01, unsigned t23) {
    int p = ld32(p_sh + (e & ~3u)), j = jcode(e & 3u, t01, t23);  // e & ~3 = neighbor * 4
    return (p & V3M) != r ? (p < 0 ? -j : j) : 0;
}
__device__ __forceinline__ int wquad16_sh(int4 w, unsigned p_sh, int r, unsigned t01, unsigned t23) {
    unsigned x = (unsigned)w.x, y = (unsigned)w.y, z = (unsigned)w.z, t = (unsigned)w.w;
    return wterm16_sh(x & 0xffffu, p_sh, r, t01, t23) + wterm16_sh(x >> 16, p_sh, r, t01, t23)
         + wterm16_sh(y & 0xffffu, p_sh, r, t01, t23) + wterm16_sh(y >> 16, p_sh, r, t01, t23)
         + wterm16_sh(z & 0xffffu, p_sh, r, t01, t23) + wterm16_sh(z >> 16, p_sh, r, t01, t23)
         + wterm16_sh(t & 0xffffu, p_sh, r, t01, t23) + wterm16_sh(t >> 16, p_sh, r, t01, t23);
}
__device__ __forceinline__ int field16s_sh(const int4* __restrict__ e4, const int* __restrict__ deg,
                                           int n, int K, unsigned t01, unsigned t23,
                                           int u, bool valid, unsigned p_sh, int r) {
    int4 w[V5_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    int d = valid ? __ldg(&deg[u]) : 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k) w[k] = (valid && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
    int kd = (d + 7) >> 3, f = 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k)
        if (k < kd) f += wquad16_sh(w[k], p_sh, r, t01, t23);
    for (int k = V5_KMAX; k < kd; ++k) f += wquad16_sh(__ldg(&e4[1 + k * n + u]), p_sh, r, t01, t23);
    return f;  // padding entries are the node itself: its own root, so 0
}
#endif
#endif
#endif
#if defined(V5_ELL16) && !defined(V5_NO_MPREF)
#define V5_MPREF 1
// V5 Metropolis on 16-bit rows (header .y == 1). Same nodes, order and draws
// as metropolis_v2: a thread's pairs (u, u + blockDim) in color order. The
// rows, degrees and h of the thread's NEXT pair (this color's next one, or the
// next color's first) are static, so they load as soon as this pair's fields
// are summed, into the same registers: the L2 round trip overlaps the draws,
// the flips and the color barrier instead of starting after it.
__device__ __forceinline__ void load16_2(const int4* __restrict__ e4, const int* __restrict__ deg,
                                         const int* __restrict__ h, int n, int K,
                                         int ua, bool va, int ub, bool vb,
                                         int4 (&wa)[V5_KMAX], int4 (&wb)[V5_KMAX],
                                         int& da, int& db, int& ha, int& hb) {
    const int4 zero = make_int4(0, 0, 0, 0);
    da = va ? __ldg(&deg[ua]) : 0; db = vb ? __ldg(&deg[ub]) : 0;
    ha = va ? __ldg(&h[ua]) : 0; hb = vb ? __ldg(&h[ub]) : 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k) {
        wa[k] = (va && k < K) ? __ldg(&e4[1 + k * n + ua]) : zero;
        wb[k] = (vb && k < K) ? __ldg(&e4[1 + k * n + ub]) : zero;
    }
}
__device__ __forceinline__ int sum16(const int4 (&w)[V5_KMAX], int d, const int4* __restrict__ e4, int n,
                                     int u, unsigned s_sh, unsigned t01, unsigned t23) {
    int kd = (d + 7) >> 3, f = 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k)
        if (k < kd) f += term16_sh(w[k], s_sh, t01, t23);
    for (int k = V5_KMAX; k < kd; ++k) f += term16_sh(__ldg(&e4[1 + k * n + u]), s_sh, t01, t23);
    return f - ((kd << 3) - d) * mterm16((unsigned)u << 2, s_sh, t01, t23);
}
#ifdef V5_ACC_PIPE
__device__ __forceinline__ void load16_1(const int4* __restrict__ e4, const int* __restrict__ deg,
                                         const int* __restrict__ h, int n, int K, int u, bool v,
                                         int4 (&w)[V5_KMAX], int& d, int& hh) {
    const int4 zero = make_int4(0, 0, 0, 0);
    d = v ? __ldg(&deg[u]) : 0; hh = v ? __ldg(&h[u]) : 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k) w[k] = (v && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
}
__device__ __forceinline__ int wsum16(const int4 (&w)[V5_KMAX], int d, const int4* __restrict__ e4, int n,
                                      int u, unsigned p_sh, int r, unsigned t01, unsigned t23) {
    int kd = (d + 7) >> 3, f = 0;
#pragma unroll
    for (int k = 0; k < V5_KMAX; ++k)
        if (k < kd) f += wquad16_sh(w[k], p_sh, r, t01, t23);
    for (int k = V5_KMAX; k < kd; ++k) f += wquad16_sh(__ldg(&e4[1 + k * n + u]), p_sh, r, t01, t23);
    return f;
}
#endif
__device__ void metropolis_v5(const Dev& g, signed char* s, float beta, uint64_t& rng,
                              int K, unsigned t01, unsigned t23, unsigned s_sh) {
    const int4* e4 = (const int4*)g.all_ell;
    const int bd = blockDim.x, n = g.n;
    int c = 0, end = __ldg(&g.color_off[1]);
    int nxt = g.C > 1 ? __ldg(&g.color_off[2]) : end;
    int u = __ldg(&g.color_off[0]) + threadIdx.x;
    int4 wa[V5_KMAX], wb[V5_KMAX];
    int da, db, ha, hb;
    load16_2(e4, g.all_deg, g.h, n, K, u, u < end, u + bd, u + bd < end, wa, wb, da, db, ha, hb);
    for (;;) {
        const int ub = u + bd;
        const bool va = u < end, vb = ub < end;
        // This pair's fields: neighbor spins as of the last barrier.
        int fa = ha + (va ? sum16(wa, da, e4, n, u, s_sh, t01, t23) : 0);
        int fb = hb + (vb ? sum16(wb, db, e4, n, ub, s_sh, t01, t23) : 0);
        // The next pair of this thread, then its rows in flight.
        int un = u + 2 * bd, endn = end;
        const bool done = un >= end;
        if (done) { un = end + threadIdx.x; endn = nxt; }
        if (!done || c + 1 < g.C)
            load16_2(e4, g.all_deg, g.h, n, K, un, un < endn, un + bd, un + bd < endn, wa, wb, da, db, ha, hb);
        if (va) {
            int delta = -2 * s[u] * fa;  // milli
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[u] = -s[u];
        }
        if (vb) {
            int delta = -2 * s[ub] * fb;
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[ub] = -s[ub];
        }
        if (done) {
            __syncthreads();
            if (++c == g.C) break;
            end = nxt;
            nxt = c + 1 < g.C ? __ldg(&g.color_off[c + 2]) : end;
        }
        u = un;
    }
}
#endif
#ifdef QUIP_OTSUKA_SWEEP_V6
#if !defined(V5_ELL16) || !defined(V4_SHADDR)
#error "QUIP_OTSUKA_SWEEP_V6 needs the V5 build (QUIP_OTSUKA_SWEEP_V5, V4 shared addressing)"
#endif
// Local build (QUIP_OTSUKA_SWEEP_V6, implies V5). The gathers' per-entry
// decode drops to an address, a shared load and an add. An ELL in V6 format
// (device.rs pack_ell6_*, header {K, 2, Js, Jw}) holds exactly two coupling
// magnitudes, Js (the strong class, every row's at most 8 strong entries in
// chunk 0) and Jw (chunks 1..). A row's chunk count is the int after the
// chunks (rows6). The sums per class are exact integers, so Js * strong +
// Jw * weak equals the per-entry sum of J * s[v] in any order: every field,
// hence every draw and flip, equals V3's.
// - Metropolis (all ELL): an entry is a byte offset into the spin window at s:
//   v for J > 0, V6_NEG_OFF(n) + v (the block's copy of -s) for J < 0, and
//   V6_NEG_OFF(n) + n (a zero byte) for padding. Every write of s also writes
//   the copy (spin load, cluster flips, Metropolis flips).
// - Accumulate (weak ELL): a 32-bit word holds two entries: 4 * v_lo in bits
//   0-15, v_hi in bits 16-29, the coupling sign of the low entry in bit 30 and
//   of the high one in bit 31. Padding is the row's own node (its own root, so
//   it adds 0). With V3TAG, (p ^ sign) >> 30 is the sign of J * s[v].
// A header .y other than 2 (graph not eligible) runs the V5/V4 gathers.
#define V6_FMT 2
#define V6_KMAX 3
// Offset of the -s copy from s: past s, the CU-1 flag word and the V4 queue head.
#define V6_NEG_OFF(n) ((((n) + 3) & ~3) + 8)
__device__ __forceinline__ const int* rows6(const int4* e4, int n, int K) {
    return (const int*)(e4 + 1 + (size_t)K * (size_t)n);
}
// Sum of the eight signed spins one int4 of Metropolis entries points at.
__device__ __forceinline__ int msum8(int4 w, unsigned s_sh) {
    unsigned x = (unsigned)w.x, y = (unsigned)w.y, z = (unsigned)w.z, t = (unsigned)w.w;
    return (lds8(s_sh + (x & 0xffffu)) + lds8(s_sh + (x >> 16)))
         + (lds8(s_sh + (y & 0xffffu)) + lds8(s_sh + (y >> 16)))
         + (lds8(s_sh + (z & 0xffffu)) + lds8(s_sh + (z >> 16)))
         + (lds8(s_sh + (t & 0xffffu)) + lds8(s_sh + (t >> 16)));
}
// One accumulate word: +-1 per entry whose root differs from the row's
// (r1 = r | V3TAG, r2 = r1 with the sign bit: the two tagged words of root r).
__device__ __forceinline__ int aword6(unsigned x, unsigned p_sh, int r1, int r2) {
    const int pl = ld32(p_sh + (x & 0xffffu));
    const int ph = ld32(p_sh + ((x >> 14) & 0xfffcu));
    const int tl = pl ^ (int)((x << 1) & 0x80000000u);
    const int th = ph ^ (int)(x & 0x80000000u);
    int f = 0;
    if (pl != r1 && pl != r2) f += tl >> 30;
    if (ph != r1 && ph != r2) f += th >> 30;
    return f;
}
__device__ __forceinline__ int asum8(int4 w, unsigned p_sh, int r1, int r2) {
    return (aword6((unsigned)w.x, p_sh, r1, r2) + aword6((unsigned)w.y, p_sh, r1, r2))
         + (aword6((unsigned)w.z, p_sh, r1, r2) + aword6((unsigned)w.w, p_sh, r1, r2));
}
// Accumulate field of node u with root r (0 if !valid).
__device__ __forceinline__ int field6s_sh(const int4* __restrict__ e4, const int* __restrict__ rows,
                                          int n, int K, int js, int jw,
                                          int u, bool valid, unsigned p_sh, int r) {
    int4 w[V6_KMAX];
    const int4 zero = make_int4(0, 0, 0, 0);
    const int kd = valid ? __ldg(&rows[u]) : 0;
#pragma unroll
    for (int k = 0; k < V6_KMAX; ++k) w[k] = (valid && k < K) ? __ldg(&e4[1 + k * n + u]) : zero;
    const int r1 = r | V3TAG, r2 = r1 | (int)0x80000000;
    int as = 0, aw = 0;
    if (kd > 0) as = asum8(w[0], p_sh, r1, r2);
#pragma unroll
    for (int k = 1; k < V6_KMAX; ++k)
        if (k < kd) aw += asum8(w[k], p_sh, r1, r2);
    for (int k = V6_KMAX; k < kd; ++k) aw += asum8(__ldg(&e4[1 + k * n + u]), p_sh, r1, r2);
    return js * as + jw * aw;
}
// Metropolis field sums of one row from its loaded chunks (as sum16).
__device__ __forceinline__ int sum6(const int4 (&w)[V6_KMAX], int kd, const int4* __restrict__ e4, int n,
                                    int u, unsigned s_sh, int js, int jw) {
    int as = 0, aw = 0;
    if (kd > 0) as = msum8(w[0], s_sh);
#pragma unroll
    for (int k = 1; k < V6_KMAX; ++k)
        if (k < kd) aw += msum8(w[k], s_sh);
    for (int k = V6_KMAX; k < kd; ++k) aw += msum8(__ldg(&e4[1 + k * n + u]), s_sh);
    return js * as + jw * aw;
}
__device__ __forceinline__ void load6_2(const int4* __restrict__ e4, const int* __restrict__ rows,
                                        const int* __restrict__ h, int n, int K,
                                        int ua, bool va, int ub, bool vb,
                                        int4 (&wa)[V6_KMAX], int4 (&wb)[V6_KMAX],
                                        int& ka, int& kb, int& ha, int& hb) {
    const int4 zero = make_int4(0, 0, 0, 0);
    ka = va ? __ldg(&rows[ua]) : 0; kb = vb ? __ldg(&rows[ub]) : 0;
    ha = va ? __ldg(&h[ua]) : 0; hb = vb ? __ldg(&h[ub]) : 0;
#pragma unroll
    for (int k = 0; k < V6_KMAX; ++k) {
        wa[k] = (va && k < K) ? __ldg(&e4[1 + k * n + ua]) : zero;
        wb[k] = (vb && k < K) ? __ldg(&e4[1 + k * n + ub]) : zero;
    }
}
// metropolis_v5 with V6 rows: same nodes, order and draws; the next pair's rows
// load while this pair flips (V5_MPREF). sn = s + V6_NEG_OFF(n), the -s copy.
__device__ void metropolis_v6(const Dev& g, signed char* s, signed char* sn, float beta, uint64_t& rng,
                              int K, int js, int jw, unsigned s_sh) {
    const int4* e4 = (const int4*)g.all_ell;
    const int bd = blockDim.x, n = g.n;
    const int* rows = rows6(e4, n, K);
    int c = 0, end = __ldg(&g.color_off[1]);
    int nxt = g.C > 1 ? __ldg(&g.color_off[2]) : end;
    int u = __ldg(&g.color_off[0]) + threadIdx.x;
    int4 wa[V6_KMAX], wb[V6_KMAX];
    int ka, kb, ha, hb;
    load6_2(e4, rows, g.h, n, K, u, u < end, u + bd, u + bd < end, wa, wb, ka, kb, ha, hb);
    for (;;) {
        const int ub = u + bd;
        const bool va = u < end, vb = ub < end;
        int fa = ha + (va ? sum6(wa, ka, e4, n, u, s_sh, js, jw) : 0);
        int fb = hb + (vb ? sum6(wb, kb, e4, n, ub, s_sh, js, jw) : 0);
        int un = u + 2 * bd, endn = end;
        const bool done = un >= end;
        if (done) { un = end + threadIdx.x; endn = nxt; }
        if (!done || c + 1 < g.C)
            load6_2(e4, rows, g.h, n, K, un, un < endn, un + bd, un + bd < endn, wa, wb, ka, kb, ha, hb);
        if (va) {
            int delta = -2 * s[u] * fa;  // milli
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) { sn[u] = s[u]; s[u] = -s[u]; }
        }
        if (vb) {
            int delta = -2 * s[ub] * fb;
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) { sn[ub] = s[ub]; s[ub] = -s[ub]; }
        }
        if (done) {
            __syncthreads();
            if (++c == g.C) break;
            end = nxt;
            nxt = c + 1 < g.C ? __ldg(&g.color_off[c + 2]) : end;
        }
        u = un;
    }
}
#endif
// Same updates and draws as metropolis() with LANES == 1. A thread's nodes
// u and u + blockDim share a color, so they are not adjacent: both fields are
// computed first, then the draws run in stock order (u, then u + blockDim).
#ifdef QUIP_OTSUKA_SWEEP_V3
// V3: int4 gathers, and the next color's end offset is fetched before the
// barrier so no warp waits on it after. Same nodes, order and draws as V2.
__device__ void metropolis_v2(const Dev& g, signed char* s, float beta, uint64_t& rng) {
    const int ka = __ldg((const int*)g.all_ell);
#ifdef V4_SHADDR
    const unsigned s_sh = sh_addr(s);
#endif
#ifdef V5_ELL16
    // Header {K, format, J0 | J1 << 16, J2 | J3 << 16}; format 1 = 16-bit entries.
    const int4 mh = __ldg((const int4*)g.all_ell);
    const bool m16 = mh.y == 1;
    const unsigned mt01 = (unsigned)mh.z, mt23 = (unsigned)mh.w;
#endif
    int beg = __ldg(&g.color_off[0]), end = __ldg(&g.color_off[1]);
    for (int c = 0; c < g.C; ++c) {
        int next = c + 1 < g.C ? __ldg(&g.color_off[c + 2]) : end;
        for (int u = beg + threadIdx.x; u < end; u += 2 * blockDim.x) {
            int ub = u + blockDim.x;
            bool hb = ub < end;
            int fa = g.h[u], fb = hb ? g.h[ub] : 0;
#ifdef V5_ELL16
            if (m16) field16_2_sh((const int4*)g.all_ell, g.all_deg, g.n, ka, mt01, mt23, u, true, ub, hb, s_sh, fa, fb);
            else field4_2_sh((const int4*)g.all_ell, g.all_deg, g.n, ka, u, true, ub, hb, s_sh, fa, fb);
#elif defined(V4_SHADDR)
            field4_2_sh((const int4*)g.all_ell, g.all_deg, g.n, ka, u, true, ub, hb, s_sh, fa, fb);
#else
            field4_2((const int4*)g.all_ell, g.all_deg, g.n, ka, u, true, ub, hb, s, nullptr, 0, 0, fa, fb);
#endif
            int delta = -2 * s[u] * fa;  // milli
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[u] = -s[u];
            if (hb) {
                delta = -2 * s[ub] * fb;
                if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[ub] = -s[ub];
            }
        }
        __syncthreads();
        beg = end; end = next;
    }
}
#else
__device__ void metropolis_v2(const Dev& g, signed char* s, float beta, uint64_t& rng) {
    for (int c = 0; c < g.C; ++c) {
        int end = g.color_off[c + 1];
        for (int u = g.color_off[c] + threadIdx.x; u < end; u += 2 * blockDim.x) {
            int ub = u + blockDim.x;
            bool hb = ub < end;
            int fa = g.h[u], fb = hb ? g.h[ub] : 0;
            field2(g.all_ell, g.n, u, g.all_deg[u], hb ? ub : u, hb ? g.all_deg[ub] : 0,
                   s, nullptr, 0, 0, fa, fb);
            int delta = -2 * s[u] * fa;  // milli
            if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[u] = -s[u];
            if (hb) {
                delta = -2 * s[ub] * fb;
                if (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f)) s[ub] = -s[ub];
            }
        }
        __syncthreads();
    }
}
#endif
#endif

__device__ float beta_at(const Params& q, int sweep) {
    float f = q.sweeps > 1 ? (float)sweep / (float)(q.sweeps - 1) : 1.0f;
    if (q.beta_knee > 0.0f && q.knee_frac > 0.0f && q.knee_frac < 1.0f) {
        if (f < q.knee_frac) return q.beta_min * powf(q.beta_knee / q.beta_min, f / q.knee_frac);
        return q.beta_knee * powf(q.beta_max / q.beta_knee, (f - q.knee_frac) / (1.0f - q.knee_frac));
    }
    return q.beta_min * powf(q.beta_max / q.beta_min, f);
}

// LANES threads share one spin (each sums every LANES-th neighbor, a shuffle reduces). One was fastest.
#ifndef LANES
#define LANES 1
#endif

__device__ void metropolis(const Dev& g, signed char* s, float beta, uint64_t& rng, bool greedy, int* changed) {
    int sub = threadIdx.x & (LANES - 1), group = threadIdx.x / LANES, groups = blockDim.x / LANES;
    for (int c = 0; c < g.C; ++c) {
        int end = g.color_off[c + 1];
        for (int base = g.color_off[c]; base < end; base += groups) {
            int u = base + group;
            int field = 0;
            if (u < end && sub == 0) field = g.h[u];
            if (u < end) {
                for (int e = sub, d = g.all_deg[u]; e < d; e += LANES) {
                    unsigned int w = (unsigned int)__ldg(&g.all_ell[e * g.n + u]);
                    field += (int)(short)(w & 0xffffu) * s[(unsigned)w >> 16];
                }
            }
            for (int o = 1; o < LANES; o <<= 1) field += __shfl_xor_sync(0xffffffffu, field, o);
            if (u < end && sub == 0) {
                int delta = -2 * s[u] * field;  // milli
                bool acc = greedy ? (delta < 0) : (delta <= 0 || urand(rng) < __expf(-beta * delta * 0.001f));
                if (acc) { s[u] = -s[u]; if (changed) atomicExch(changed, 1); }
            }
        }
        __syncthreads();
    }
}

// One block per anneal. The flat arguments preserve CU-1 upload arrays.
// Local build: FAST's 16-bit parents free shared memory for three blocks per
// SM; cap registers (65536 / (3 * 512)) so registers do not become the limit.
#ifdef QUIP_OTSUKA_FAST
#define QUIP_CLUSTER_BOUNDS __launch_bounds__(512, 3)
#elif defined(QUIP_OTSUKA_SWEEP_V3)
// Shared memory allows two blocks per SM; keep registers from capping it at one.
#define QUIP_CLUSTER_BOUNDS __launch_bounds__(512, 2)
#else
#define QUIP_CLUSTER_BOUNDS
#endif
extern "C" __global__ void QUIP_CLUSTER_BOUNDS quip_cluster_anneal(
    int n_nodes, int tiles_count, int pairs_count, int colors_count,
    const int* tiles, const signed char* gauges, const int* pairs,
    const int* weak_deg, const int* weak_ell, const int* all_deg,
    const int* all_ell, const int* color_off, const int* h,
    const int* tile_w, const int* link_w, Params q,
    int sweep0, int nsweeps, signed char* gspins, uint64_t* grng, int final_pass) {
    Dev g{n_nodes, tiles_count, pairs_count, colors_count,
          (const int4*)tiles, (const char4*)gauges, (const int2*)pairs,
          weak_deg, weak_ell, all_deg, all_ell, color_off, h, tile_w, link_w};
    extern __shared__ int smem[];
    int n = g.n;
#ifdef QUIP_OTSUKA_FAST
    int* acc = smem;
    pid_t_* parent = (pid_t_*)(acc + n);
    signed char* s = (signed char*)(parent + n);
#else
    int* parent = smem;
    int* acc = parent + n;
    signed char* s = (signed char*)(acc + n);
#endif
    // Use CU-1 dynamic padding for the flag. A static int is rounded to
    // 16 bytes by the compiler, exceeding the four-byte static reservation.
#ifdef QUIP_OTSUKA_FAST
    int* changed = (int*)((char*)smem + ((7 * n + 3) & ~3));
#else
    int* changed = (int*)(s + ((n + 3) & ~3));
#endif
    signed char* mine = gspins + (size_t)blockIdx.x * n;
    uint64_t rng = grng[(size_t)blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = mine[i];
#ifdef QUIP_OTSUKA_SWEEP_V6
    // V6: the -s copy and the zero byte the Metropolis padding points at.
    signed char* sn = s + V6_NEG_OFF(n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) sn[i] = -mine[i];
    if (threadIdx.x == 0) sn[n] = 0;
#endif
    __syncthreads();

    if (final_pass) {
        // Mode 2 lets the host check cancellation between greedy sweeps.
        int passes = final_pass == 2 ? 1 : 256;
        for (int pass = 0; pass < passes; ++pass) {
            if (threadIdx.x == 0) *changed = 0;
            __syncthreads();
            metropolis(g, s, 0.0f, rng, true, changed);
            __syncthreads();
            if (!*changed) break;
            __syncthreads();
        }
    } else {
#ifdef QUIP_OTSUKA_SWEEP_V2
        // Local build: the stock sweep below, restructured. Every thread takes
        // the same items in the same order, so its RNG stream and all results
        // match. Parent reset and accumulator zeroing ride on earlier phases
        // (two barriers fewer), finds halve paths, rounds == 1 skips the
        // group hash, and ELL rows load four words at a time.
        unsigned* elist = (unsigned*)acc;  // union edges between draws and unions
        int* ecount = changed;              // `changed` is only used by final passes
        for (int i = threadIdx.x; i < n; i += blockDim.x) { parent[i] = i; acc[i] = 0; }
        if (threadIdx.x == 0) *ecount = 0;
#ifdef V4_SHADDR
        const unsigned par_sh = sh_addr(parent), acc_sh = sh_addr(acc);
#endif
#ifdef QUIP_OTSUKA_SWEEP_V4
        // V4 union queue head: the int after `changed`, inside the CU-1 padding
        // (9n + 16 bytes leave at least 9 bytes past `changed`).
        int* qhead = changed + 1;
        if (threadIdx.x == 0) *qhead = 0;
        const unsigned s_sh = sh_addr(s);
#endif
        __syncthreads();
        const bool one_round = q.rounds == 1;
#ifdef QUIP_OTSUKA_SWEEP_V3
        const int kw = __ldg(g.weak_ell);  // int4 chunks per packed weak row (header)
#endif
#ifdef V5_MPREF
        const int4 mh = __ldg((const int4*)g.all_ell);
        const bool m16 = mh.y == 1;
#endif
#ifdef V5_ELL16
        const int4 wh = __ldg((const int4*)g.weak_ell);
        const bool w16 = wh.y == 1;
        const unsigned wt01 = (unsigned)wh.z, wt23 = (unsigned)wh.w;
#endif
#ifdef QUIP_OTSUKA_SWEEP_V6
        // V6 headers: {K, 2, Js, Jw}; anything else runs the V5/V4 gathers.
        const int4 mh6 = __ldg((const int4*)g.all_ell);
        const bool m6 = mh6.y == V6_FMT, w6 = wh.y == V6_FMT;
        const int* wrows6 = rows6((const int4*)g.weak_ell, n, kw);
#endif
        PT_DECL
        for (int sweep = sweep0; sweep < sweep0 + nsweeps; ++sweep) {
            float beta = beta_at(q, sweep);
            float bt = beta * q.tile_w * 0.001f;
            float z = __expf(-2.0f * bt);
            float w0 = z * z * z * z;
            float zero_ice = w0, single_ice = w0 + 4.0f * (z - w0) / 3.0f;
            float zero_defect = __expf(-6.0f * bt);
            float dimer_link = 1.0f - __expf(-2.0f * beta * q.dimer_w * 0.001f);

            // Draws: each thread takes its stock tiles and pairs in stock order
            // (warp-uniform trip counts, so lanes past the end just idle) and
            // lists the unions it picks. Then every thread takes an equal share
            // of the list: the union work no longer follows the 7-vs-5 item
            // split of the stock thread mapping.
#ifdef QUIP_OTSUKA_SWEEP_V4
            // V4: the same items, order and draws from packed records
            // (device.rs pack_tiles_v4 / pack_pairs_v4), the next record in flight.
            int2 pnext = threadIdx.x < g.P ? __ldg(&g.pairs[threadIdx.x]) : make_int2(0, 0);
            {
            int4 tcur = threadIdx.x < g.T ? __ldg(&g.tiles[threadIdx.x]) : make_int4(0, 0, 0, 0);
            for (int tb = 0; tb < g.T; tb += blockDim.x) {
                int t = tb + threadIdx.x, tn = t + blockDim.x;
                int4 rec = tcur;
                tcur = tn < g.T ? __ldg(&g.tiles[tn]) : make_int4(0, 0, 0, 0);
                unsigned e1 = NO_EDGE, e2 = NO_EDGE;
                if (t < g.T) {
                    unsigned up = tile_up(rec, s_sh);
                    unsigned dn = up ^ 15u;
                    int np = __popc(up), nn = 4 - np;
                    if (np != 0 && np != 4) {
                        float r = urand(rng);
                        float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                        if (g.tile_w && (float)rec.w != q.tile_w) {
                            float tbt = beta * (float)rec.w * 0.001f;
                            float tz = __expf(-2.0f * tbt);
                            tz0 = tz * tz * tz * tz;
                            tsi = tz0 + 4.0f * (tz - tz0) / 3.0f;
                            tzd = __expf(-6.0f * tbt);
                        }
                        if (np == 2) {
                            if (r >= tz0) {
                                if (r < tsi) {
                                    uint32_t b = (uint32_t)xs(rng);
                                    e1 = edge(tile_pick_v4(rec, up, b & 1), tile_pick_v4(rec, dn, (b >> 1) & 1));
                                } else {
                                    unsigned ch = (unsigned)(xs(rng) & 1);
                                    e1 = edge(tile_pick_v4(rec, up, 0), tile_pick_v4(rec, dn, ch));
                                    e2 = edge(tile_pick_v4(rec, up, 1), tile_pick_v4(rec, dn, 1 - ch));
                                }
                            }
                        } else if (r >= tzd) {
                            uint32_t b = (uint32_t)xs(rng);
                            e1 = edge(tile_pick_v4(rec, up, (b & 0xffff) % np), tile_pick_v4(rec, dn, (b >> 16) % nn));
                        }
                    }
                }
                push_edges(e1, e2, elist, ecount, n, parent);
            }
            }
            for (int pb = 0; pb < g.P; pb += blockDim.x) {
                int p = pb + threadIdx.x, pn = p + blockDim.x;
                int2 rec = pnext;
                pnext = pn < g.P ? __ldg(&g.pairs[pn]) : make_int2(0, 0);
                unsigned e1 = NO_EDGE;
                if (p < g.P) {
                    int px = (int)((unsigned)rec.x & 0xffffu), py = (int)((unsigned)rec.x >> 16);
                    int lw = g.link_w ? rec.y : (int)q.dimer_w;
                    float bond = (abs(lw) == (int)q.dimer_w) ? dimer_link : 1.0f - __expf(-2.0f * beta * fabsf((float)lw) * 0.001f);
                    bool satisfied = lw > 0 ? s[px] == s[py] : s[px] != s[py];
                    if (satisfied && urand(rng) < bond) e1 = edge(px, py);
                }
                push_edges(e1, NO_EDGE, elist, ecount, n, parent);
            }
            __syncthreads();
#if defined(QUIP_OTSUKA_UNION_SV)
            union_sv(parent, elist, *ecount, n);
#elif !defined(V4_NO_UNION)
            union_pass_v4(parent, elist, min(*ecount, n), qhead);
#else
            {
                int m = min(*ecount, n);
                for (int k = threadIdx.x; k < m; k += blockDim.x) {
                    unsigned e = elist[k];
                    unite_v2(parent, (int)(e >> 16), (int)(e & 0xffffu));
                }
            }
#endif
#else
            for (int tb = 0; tb < g.T; tb += blockDim.x) {
                int t = tb + threadIdx.x;
                unsigned e1 = NO_EDGE, e2 = NO_EDGE;
                if (t < g.T) {
                    int4 nd = g.tiles[t]; char4 ga = g.gauge[t];
                    unsigned up = (s[nd.x] * ga.x > 0 ? 1u : 0u) | (s[nd.y] * ga.y > 0 ? 2u : 0u)
                                | (s[nd.z] * ga.z > 0 ? 4u : 0u) | (s[nd.w] * ga.w > 0 ? 8u : 0u);
                    unsigned dn = up ^ 15u;
                    int np = __popc(up), nn = 4 - np;
                    if (np != 0 && np != 4) {
                        float r = urand(rng);
                        float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                        if (g.tile_w && (float)g.tile_w[t] != q.tile_w) {
                            float tbt = beta * (float)g.tile_w[t] * 0.001f;
                            float tz = __expf(-2.0f * tbt);
                            tz0 = tz * tz * tz * tz;
                            tsi = tz0 + 4.0f * (tz - tz0) / 3.0f;
                            tzd = __expf(-6.0f * tbt);
                        }
                        if (np == 2) {
                            if (r >= tz0) {
                                if (r < tsi) {
                                    uint32_t b = (uint32_t)xs(rng);
                                    e1 = edge(tile_pick(nd, up, b & 1), tile_pick(nd, dn, (b >> 1) & 1));
                                } else {
                                    unsigned ch = (unsigned)(xs(rng) & 1);
                                    e1 = edge(tile_pick(nd, up, 0), tile_pick(nd, dn, ch));
                                    e2 = edge(tile_pick(nd, up, 1), tile_pick(nd, dn, 1 - ch));
                                }
                            }
                        } else if (r >= tzd) {
                            uint32_t b = (uint32_t)xs(rng);
                            e1 = edge(tile_pick(nd, up, (b & 0xffff) % np), tile_pick(nd, dn, (b >> 16) % nn));
                        }
                    }
                }
                push_edges(e1, e2, elist, ecount, n, parent);
            }
            for (int pb = 0; pb < g.P; pb += blockDim.x) {
                int p = pb + threadIdx.x;
                unsigned e1 = NO_EDGE;
                if (p < g.P) {
                    int2 pr = g.pairs[p];
                    int lw = g.link_w ? g.link_w[p] : (int)q.dimer_w;
                    float bond = (abs(lw) == (int)q.dimer_w) ? dimer_link : 1.0f - __expf(-2.0f * beta * fabsf((float)lw) * 0.001f);
                    bool satisfied = lw > 0 ? s[pr.x] == s[pr.y] : s[pr.x] != s[pr.y];
                    if (satisfied && urand(rng) < bond) e1 = edge(pr.x, pr.y);
                }
                push_edges(e1, NO_EDGE, elist, ecount, n, parent);
            }
            __syncthreads();
            {
                int m = min(*ecount, n);
                for (int k = threadIdx.x; k < m; k += blockDim.x) {
                    unsigned e = elist[k];
                    unite_v2(parent, (int)(e >> 16), (int)(e & 0xffffu));
                }
            }
#endif
            __syncthreads();
            PT(0);
            // The list lived in acc: zero it for the accumulate, two barriers on.
            for (int i = threadIdx.x; i < n; i += blockDim.x) acc[i] = 0;
            if (threadIdx.x == 0) *ecount = 0;
#ifdef QUIP_OTSUKA_SWEEP_V4
            if (threadIdx.x == 0) *qhead = 0;
#endif
            // Round-robin pointer jumping: one jump per owned node per pass, so
            // every chain in the block halves at once. Only the owner writes
            // parent[i], always an ancestor, and roots are fixed in this pass.
#if defined(QUIP_OTSUKA_SWEEP_V4) && !defined(V4_NO_COMPRESS)
            compress_v4(parent, s, n);
#elif defined(V3SIGN)
            // Same jumps on masked values; each owner also tags its sign bit.
            for (bool again = true; again;) {
                again = false;
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    int v = ((volatile int*)parent)[i];
                    int pi = v & V3M, pp = ((volatile int*)parent)[pi] & V3M;
                    if (pp != pi) again = true;
                    int nv = (pp != pi ? pp : pi) | sbit(s[i]) | V3TAG;
                    if (nv != v) ((volatile int*)parent)[i] = nv;
                }
            }
#else
            for (bool again = true; again;) {
                again = false;
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    int pi = ((volatile int*)parent)[i], pp = ((volatile int*)parent)[pi];
                    if (pp != pi) { ((volatile int*)parent)[i] = pp; again = true; }
                }
            }
#endif
            __syncthreads();
            PT(1);
            uint32_t salt = mix32((uint32_t)sweep * 0x9e3779b9U + blockIdx.x * 0x85ebca6bU);
            for (int round = 0; round < q.rounds; ++round) {
                const bool last = round + 1 == q.rounds;
                if (round > 0) {
                    for (int i = threadIdx.x; i < n; i += blockDim.x) acc[i] = 0;
                    __syncthreads();
                }
                // Two nodes per pass, loads interleaved; integer sums, same totals.
#ifdef V5_ACC_PIPE
                // Opt-in (-DV5_ACC_PIPE): one node per step, the thread's next
                // node's row in flight while this one sums. Same nodes per
                // thread; red.add is integer, so acc and every draw match.
                if (w16) {
                    const int bd = blockDim.x;
                    int u = threadIdx.x;
                    int r = u < n ? (ld32(par_sh + 4u * u) & V3M) : -1;
                    bool iv = r >= 0 && (one_round || (int)(mix32(r ^ salt) % q.rounds) == round);
                    int4 w[V5_KMAX]; int d, hh;
                    load16_1((const int4*)g.weak_ell, g.weak_deg, g.h, n, kw, u, iv, w, d, hh);
                    while (u < n) {
                        const int un = u + bd;
                        const int rn = un < n ? (ld32(par_sh + 4u * un) & V3M) : -1;
                        const bool ivn = rn >= 0 && (one_round || (int)(mix32(rn ^ salt) % q.rounds) == round);
                        int4 wn[V5_KMAX]; int dn, hn;
                        load16_1((const int4*)g.weak_ell, g.weak_deg, g.h, n, kw, un, ivn, wn, dn, hn);
                        if (iv) {
                            int f = hh + wsum16(w, d, (const int4*)g.weak_ell, n, u, par_sh, r, wt01, wt23);
                            if (f) red_add(acc_sh + 4u * r, s[u] * f);
                        }
                        u = un; r = rn; iv = ivn; d = dn; hh = hn;
#pragma unroll
                        for (int k = 0; k < V5_KMAX; ++k) w[k] = wn[k];
                    }
                } else
#endif
                for (int u = threadIdx.x; u < n; u += 2 * blockDim.x) {
                    int ub = u + blockDim.x;
#ifdef V3SIGN
                    int ra = parent[u] & V3M, rb = ub < n ? parent[ub] & V3M : -1;
#else
                    int ra = parent[u], rb = ub < n ? parent[ub] : -1;
#endif
                    bool ia = one_round || (int)(mix32(ra ^ salt) % q.rounds) == round;
                    bool ib = rb >= 0 && (one_round || (int)(mix32(rb ^ salt) % q.rounds) == round);
                    int fa = ia ? g.h[u] : 0, fb = ib ? g.h[ub] : 0;
#if defined(QUIP_OTSUKA_SWEEP_V6)
                    if (w6) {
                        fa += field6s_sh((const int4*)g.weak_ell, wrows6, g.n, kw, wh.z, wh.w, u, ia, par_sh, ra);
                        fb += field6s_sh((const int4*)g.weak_ell, wrows6, g.n, kw, wh.z, wh.w, ub, ib, par_sh, rb);
                    } else
#endif
#if defined(V5_ELL16)
                    if (w16) {
                        fa += field16s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, wt01, wt23, u, ia, par_sh, ra);
                        fb += field16s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, wt01, wt23, ub, ib, par_sh, rb);
                    } else {
                        fa += field4s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, u, ia, par_sh, ra);
                        fb += field4s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, ub, ib, par_sh, rb);
                    }
#elif defined(V4_SHADDR)
                    fa += field4s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, u, ia, par_sh, ra);
                    fb += field4s_sh((const int4*)g.weak_ell, g.weak_deg, g.n, kw, ub, ib, par_sh, rb);
#elif defined(V3SIGN)
                    fa += field4s((const int4*)g.weak_ell, g.weak_deg, g.n, kw, u, ia, parent, ra);
                    fb += field4s((const int4*)g.weak_ell, g.weak_deg, g.n, kw, ub, ib, parent, rb);
#else
                    field2(g.weak_ell, g.n, u, ia ? g.weak_deg[u] : 0, ib ? ub : u, ib ? g.weak_deg[ub] : 0,
                           s, parent, ra, rb, fa, fb);
#endif
#if defined(V4_SHADDR)
                    if (ia && fa) red_add(acc_sh + 4u * ra, s[u] * fa);
                    if (ib && fb) red_add(acc_sh + 4u * rb, s[ub] * fb);
#else
                    if (ia && fa) atomicAdd(&acc[ra], s[u] * fa);
                    if (ib && fb) atomicAdd(&acc[rb], s[ub] * fb);
#endif
                }
                __syncthreads();
                PT(2);
                for (int r = threadIdx.x; r < n; r += blockDim.x) {
#ifdef V3SIGN
                    if ((parent[r] & V3M) != r) continue;
#else
                    if (parent[r] != r) continue;
#endif
                    if (!one_round && (int)(mix32(r ^ salt) % q.rounds) != round) continue;
                    float delta = -2.0f * acc[r];
                    acc[r] = (urand(rng) < 1.0f / (1.0f + __expf(beta * delta * 0.001f))) ? 1 : 0;
                }
                __syncthreads();
                PT(3);
                for (int u = threadIdx.x; u < n; u += blockDim.x) {
#ifdef V3SIGN
                    int lu = parent[u] & V3M;
#else
                    int lu = parent[u];
#endif
#ifdef QUIP_OTSUKA_SWEEP_V6
                    if ((one_round || (int)(mix32(lu ^ salt) % q.rounds) == round) && acc[lu] == 1) { sn[u] = s[u]; s[u] = -s[u]; }
#else
                    if ((one_round || (int)(mix32(lu ^ salt) % q.rounds) == round) && acc[lu] == 1) s[u] = -s[u];
#endif
                    // Only this thread reads parent[u] in this phase: reset it for the next sweep.
                    if (last) parent[u] = u;
#ifdef V3SIGN
                    else parent[u] = lu | sbit(s[u]) | V3TAG;  // next round's accumulate sees the flip
#endif
                }
                __syncthreads();
                PT(4);
            }
            if (q.single_every > 0 && (sweep + 1) % q.single_every == 0) {
#ifdef QUIP_OTSUKA_SWEEP_V6
                if (m6) metropolis_v6(g, s, sn, beta, rng, mh6.x, mh6.z, mh6.w, s_sh);
                else
#endif
#ifdef V5_MPREF
                if (m16) metropolis_v5(g, s, beta, rng, mh.x, (unsigned)mh.z, (unsigned)mh.w, s_sh);
                else
#endif
                metropolis_v2(g, s, beta, rng);
            }
            PT(5);
        }
        PT_PRINT
#else
        PT_DECL
        for (int sweep = sweep0; sweep < sweep0 + nsweeps; ++sweep) {
            float beta = beta_at(q, sweep);
            float bt = beta * q.tile_w * 0.001f;
            float z = __expf(-2.0f * bt);
            float w0 = z * z * z * z;
            float zero_ice = w0, single_ice = w0 + 4.0f * (z - w0) / 3.0f;
            float zero_defect = __expf(-6.0f * bt);
            float dimer_link = 1.0f - __expf(-2.0f * beta * q.dimer_w * 0.001f);

            for (int i = threadIdx.x; i < n; i += blockDim.x) parent[i] = i;
            __syncthreads();
            for (int t = threadIdx.x; t < g.T; t += blockDim.x) {
                int4 nd = g.tiles[t]; char4 ga = g.gauge[t];
#ifdef QUIP_OTSUKA_FAST
                // Same draws and unions as the array form below, without local memory.
                {
                unsigned up = (s[nd.x] * ga.x > 0 ? 1u : 0u) | (s[nd.y] * ga.y > 0 ? 2u : 0u)
                            | (s[nd.z] * ga.z > 0 ? 4u : 0u) | (s[nd.w] * ga.w > 0 ? 8u : 0u);
                unsigned dn = up ^ 15u;
                int np = __popc(up), nn = 4 - np;
                if (np == 0 || np == 4) continue;
                float r = urand(rng);
                float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                if (g.tile_w && (float)g.tile_w[t] != q.tile_w) {
                    float tbt = beta * (float)g.tile_w[t] * 0.001f;
                    float tz = __expf(-2.0f * tbt);
                    tz0 = tz * tz * tz * tz;
                    tsi = tz0 + 4.0f * (tz - tz0) / 3.0f;
                    tzd = __expf(-6.0f * tbt);
                }
                if (np == 2) {
                    if (r < tz0) continue;
                    if (r < tsi) {
                        uint32_t b = (uint32_t)xs(rng);
                        unite(parent, tile_pick(nd, up, b & 1), tile_pick(nd, dn, (b >> 1) & 1));
                    } else {
                        unsigned ch = (unsigned)(xs(rng) & 1);
                        unite(parent, tile_pick(nd, up, 0), tile_pick(nd, dn, ch));
                        unite(parent, tile_pick(nd, up, 1), tile_pick(nd, dn, 1 - ch));
                    }
                } else if (r >= tzd) {
                    uint32_t b = (uint32_t)xs(rng);
                    unite(parent, tile_pick(nd, up, (b & 0xffff) % np), tile_pick(nd, dn, (b >> 16) % nn));
                }
                continue;
                }
#endif
                int node[4] = {nd.x, nd.y, nd.z, nd.w};
                int gg[4] = {ga.x, ga.y, ga.z, ga.w};
                int pos[4], neg[4], np = 0, nn = 0;
                for (int k = 0; k < 4; ++k) {
                    if (s[node[k]] * gg[k] > 0) pos[np++] = node[k]; else neg[nn++] = node[k];
                }
                if (np == 0 || np == 4) continue;
                float r = urand(rng);
                float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                if (g.tile_w && (float)g.tile_w[t] != q.tile_w) {
                    float tbt = beta * (float)g.tile_w[t] * 0.001f;
                    float tz = __expf(-2.0f * tbt);
                    tz0 = tz * tz * tz * tz;
                    tsi = tz0 + 4.0f * (tz - tz0) / 3.0f;
                    tzd = __expf(-6.0f * tbt);
                }
                if (np == 2) {
                    if (r < tz0) continue;
                    if (r < tsi) {
                        uint32_t b = (uint32_t)xs(rng);
                        unite(parent, pos[b & 1], neg[(b >> 1) & 1]);
                    } else {
                        int ch = (int)(xs(rng) & 1);
                        unite(parent, pos[0], neg[ch]);
                        unite(parent, pos[1], neg[1 - ch]);
                    }
                } else if (r >= tzd) {
                    uint32_t b = (uint32_t)xs(rng);
                    unite(parent, pos[(b & 0xffff) % np], neg[(b >> 16) % nn]);
                }
            }
            for (int p = threadIdx.x; p < g.P; p += blockDim.x) {
                int2 pr = g.pairs[p];
                int lw = g.link_w ? g.link_w[p] : (int)q.dimer_w;
                float bond = (abs(lw) == (int)q.dimer_w) ? dimer_link : 1.0f - __expf(-2.0f * beta * fabsf((float)lw) * 0.001f);
                bool satisfied = lw > 0 ? s[pr.x] == s[pr.y] : s[pr.x] != s[pr.y];
                if (satisfied && urand(rng) < bond) unite(parent, pr.x, pr.y);
            }
            __syncthreads();
            PT(0);
#ifdef QUIP_OTSUKA_FAST
            for (int i = threadIdx.x; i < n; i += blockDim.x) ((volatile pid_t_*)parent)[i] = (pid_t_)find_root(parent, i);
#else
            for (int i = threadIdx.x; i < n; i += blockDim.x) atomicExch(&parent[i], find_root(parent, i));
#endif
            __syncthreads();
            PT(1);
            uint32_t salt = mix32((uint32_t)sweep * 0x9e3779b9U + blockIdx.x * 0x85ebca6bU);
            for (int round = 0; round < q.rounds; ++round) {
                for (int i = threadIdx.x; i < n; i += blockDim.x) acc[i] = 0;
                __syncthreads();
                for (int base = 0; base < n; base += blockDim.x / LANES) {
                    int u = base + threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1);
                    int a = 0, lu = -1;
                    if (u < n) {
                        lu = parent[u];
                        if ((int)(mix32(lu ^ salt) % q.rounds) == round) {
                            if (sub == 0) a = g.h[u];
                            for (int e = sub, d = g.weak_deg[u]; e < d; e += LANES) {
                                unsigned int w = (unsigned int)__ldg(&g.weak_ell[e * g.n + u]);
                                int v = (unsigned)w >> 16;
                                if (parent[v] != lu) a += (int)(short)(w & 0xffffu) * s[v];
                            }
                        }
                    }
                    for (int o = 1; o < LANES; o <<= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                    if (sub == 0 && a) atomicAdd(&acc[lu], s[u] * a);
                }
                __syncthreads();
                PT(2);
                // Decide on roots; acc[root] = 1 marks a flip.
                for (int r = threadIdx.x; r < n; r += blockDim.x) {
                    if (parent[r] != r) continue;
                    if ((int)(mix32(r ^ salt) % q.rounds) != round) continue;
                    float delta = -2.0f * acc[r];
                    acc[r] = (urand(rng) < 1.0f / (1.0f + __expf(beta * delta * 0.001f))) ? 1 : 0;
                }
                __syncthreads();
                PT(3);
                for (int u = threadIdx.x; u < n; u += blockDim.x) {
                    int lu = parent[u];
                    if ((int)(mix32(lu ^ salt) % q.rounds) == round && acc[lu] == 1) s[u] = -s[u];
                }
                __syncthreads();
                PT(4);
            }
            if (q.single_every > 0 && (sweep + 1) % q.single_every == 0) {
                metropolis(g, s, beta, rng, false, nullptr);
            }
            PT(5);
        }
        PT_PRINT
#endif
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) mine[i] = s[i];
    grng[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = rng;
}

// Local build (BudFox, 2026-11-13): model energy of every anneal's state, in
// milli, for population annealing. One block per anneal over device node order;
// each edge is seen from both ends, so the sum halves. Unused by the stock path.
extern "C" __global__ void quip_cluster_energy(
    int n_nodes, const int* all_deg, const int* all_ell, const int* h,
    const signed char* gspins, long long* out) {
    __shared__ unsigned long long total;
    if (threadIdx.x == 0) total = 0ULL;
    __syncthreads();
    const signed char* s = gspins + (size_t)blockIdx.x * n_nodes;
    long long local = 0;
    for (int u = threadIdx.x; u < n_nodes; u += blockDim.x) {
        int field = 0;
        for (int e = 0, d = all_deg[u]; e < d; ++e) {
            unsigned int w = (unsigned int)__ldg(&all_ell[e * n_nodes + u]);
            field += (int)(short)(w & 0xffffu) * s[w >> 16];
        }
        local += (long long)s[u] * (long long)field + 2LL * (long long)h[u] * (long long)s[u];
    }
    atomicAdd(&total, (unsigned long long)local);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = ((long long)total) / 2;
}
