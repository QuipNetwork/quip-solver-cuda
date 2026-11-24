// GPU generator and structural scorer for quip/twisted-ice-dimer@1 (the nonce -> instance construction of ising-generators).
//
// One block builds one instance: lane 0 runs the sequential parts (the ChaCha8 streams, Fisher-Yates shuffles, the augmenting-path
// matching of the spine tiles, the greedy pair walk) out of shared memory; all lanes then count the triangle classes of the
// coupling signs. The output is the S0 structural score, identical to `structural_score` of the canonical instance. It is only
// a screen: a nonce that passes is rebuilt by the canonical CPU generator, so a defect here can lose or add a candidate but never
// produces a wrong proof.

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

struct Tables {
    int L, R, C, E, N, T;      // left spines, right spines, candidate tiles, edges, nodes, triangles
    int weak, pairs;           // weak coupling magnitude, number of dimer pairs
    int sel_expected;          // size of the maximum matching (selected tiles)
    int pair_cap;              // capacity of the pair order array
    const u32* cand_edges;     // [C][6] edge positions of the six edges of a candidate tile
    const int* adj_off;        // [L + 1] CSR over left spines
    const u16* adj_r;          // right spine of each adjacency entry
    const u32* adj_c;          // candidate tile of each adjacency entry
    const u16* eu;             // [E] edge ends
    const u16* ev;
    const u16* tri;            // [T][3] edge positions of every triangle
    const double* fw;          // [9] score weights, means, deviations
    const double* fmu;
    const double* fsd;
};

struct Chacha {
    u32 key[8];
    u64 ctr;
    u32 blk[16];
    int idx;
};

__device__ __forceinline__ u32 rotl(u32 x, int r) { return (x << r) | (x >> (32 - r)); }

#define QR(a, b, c, d)                    \
    a += b; d ^= a; d = rotl(d, 16);      \
    c += d; b ^= c; b = rotl(b, 12);      \
    a += b; d ^= a; d = rotl(d, 8);       \
    c += d; b ^= c; b = rotl(b, 7);

__device__ void chacha_regen(Chacha& c) {
    u32 s[16], x[16];
    s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
#pragma unroll
    for (int i = 0; i < 8; ++i) s[4 + i] = c.key[i];
    s[12] = (u32)c.ctr; s[13] = (u32)(c.ctr >> 32); s[14] = 0; s[15] = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        QR(x[0], x[4], x[8], x[12]);
        QR(x[1], x[5], x[9], x[13]);
        QR(x[2], x[6], x[10], x[14]);
        QR(x[3], x[7], x[11], x[15]);
        QR(x[0], x[5], x[10], x[15]);
        QR(x[1], x[6], x[11], x[12]);
        QR(x[2], x[7], x[8], x[13]);
        QR(x[3], x[4], x[9], x[14]);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) c.blk[i] = x[i] + s[i];
    c.ctr += 1;
    c.idx = 0;
}

__device__ __forceinline__ void chacha_init(Chacha& c, const u32* key) {
#pragma unroll
    for (int i = 0; i < 8; ++i) c.key[i] = key[i];
    c.ctr = 0;
    c.idx = 16;
}

__device__ __forceinline__ u32 chacha_next(Chacha& c) {
    if (c.idx >= 16) chacha_regen(c);
    return c.blk[c.idx++];
}

// Unbiased draw in 0..n by rejection, as ising-generators `uniform`.
__device__ __forceinline__ u32 uniform_n(Chacha& c, u32 n) {
    const u32 r = (0u - n) % n;  // 2^32 mod n
    for (;;) {
        const u32 x = chacha_next(c);
        if (r == 0 || x < (0u - r)) return x % n;
    }
}

// Fisher-Yates in place: for i from len-1 down to 1, swap items[i] with items[uniform(i + 1)].
__device__ void shuffle_u16(Chacha& c, u16* items, int len) {
    for (int i = len - 1; i >= 1; --i) {
        const int j = (int)uniform_n(c, (u32)(i + 1));
        const u16 t = items[i]; items[i] = items[j]; items[j] = t;
    }
}

// The adjacency of left spine `left` (at most 4 entries) sorted by the rank of the right spine.
__device__ __forceinline__ int sorted_adj(const Tables& t, const u16* rank, int left, int* rs, u32* cs) {
    const int b = t.adj_off[left], n = t.adj_off[left + 1] - b;
    for (int k = 0; k < n; ++k) {
        int r = t.adj_r[b + k]; u32 cd = t.adj_c[b + k];
        int p = k;
        while (p > 0 && rank[rs[p - 1]] > rank[r]) { rs[p] = rs[p - 1]; cs[p] = cs[p - 1]; --p; }
        rs[p] = r; cs[p] = cd;
    }
    return n;
}

#define STACK 256

extern "C" __global__ void quip_gen_s0(
    int L, int R, int C, int E, int N, int T, int weak, int pairs, int sel_expected, int pair_cap,
    const u32* cand_edges, const int* adj_off, const u16* adj_r, const u32* adj_c, const u16* eu, const u16* ev, const u16* tri,
    const double* fw, const double* fmu, const double* fsd,
    const u32* seeds, int batch, unsigned char* scratch, double* score, int* status, unsigned long long* prof) {
    Tables t;
    t.L = L; t.R = R; t.C = C; t.E = E; t.N = N; t.T = T; t.weak = weak; t.pairs = pairs; t.sel_expected = sel_expected;
    t.pair_cap = pair_cap; t.cand_edges = cand_edges; t.adj_off = adj_off; t.adj_r = adj_r; t.adj_c = adj_c; t.eu = eu; t.ev = ev;
    t.tri = tri; t.fw = fw; t.fmu = fmu; t.fsd = fsd;
    extern __shared__ __align__(16) unsigned char smem[];
    // Shared layout: pair order | left order | rank | matched | stamp | selected bits | used bits | counters.
    u16* pa = (u16*)smem;
    u16* left_order = pa + ((t.pair_cap + 1) & ~1);
    u16* rank = left_order + ((t.L + 1) & ~1);
    short* matched = (short*)(rank + ((t.R + 1) & ~1));
    u16* stamp = (u16*)(matched + ((t.R + 1) & ~1));
    u32* selbits = (u32*)(stamp + ((t.R + 1) & ~1));
    u32* usedbits = selbits + ((t.C + 31) / 32);
    int* cnt = (int*)(usedbits + ((t.N + 31) / 32));  // [0..7] triangle classes, [8] weak positives, [9] status
    unsigned char* flags = scratch + (size_t)blockIdx.x * t.E;  // bit0 strong, bit1 non-positive
    const int tid = threadIdx.x;

    for (int inst = blockIdx.x; inst < batch; inst += gridDim.x) {
        const u32* key = seeds + (size_t)inst * 40;  // left, right, gauge, weak, dimer
        for (int e = tid; e < t.E; e += blockDim.x) flags[e] = 0;
        if (tid < 10) cnt[tid] = 0;
        __syncthreads();

        long long t_mark = clock64();
#define PHASE(k) do { long long now_ = clock64(); atomicAdd(&prof[k], (unsigned long long)(now_ - t_mark)); t_mark = now_; } while (0)
        if (tid == 0) {
            int st = 0;
            {
                Chacha c;
                chacha_init(c, key + 0);
                for (int i = 0; i < t.L; ++i) left_order[i] = (u16)i;
                shuffle_u16(c, left_order, t.L);
                chacha_init(c, key + 8);
                for (int i = 0; i < t.R; ++i) pa[i] = (u16)i;
                shuffle_u16(c, pa, t.R);
                for (int p = 0; p < t.R; ++p) rank[pa[p]] = (u16)p;
            }
            PHASE(0);
            for (int r = 0; r < t.R; ++r) { matched[r] = -1; stamp[r] = 0; }
            for (int w = 0; w < (t.C + 31) / 32; ++w) selbits[w] = 0;
            // Maximum matching of the left spines in `left_order`, augmenting paths over the right spines in rank order.
            int sl[STACK], sp[STACK];
            for (int p = 0; p < t.L && st == 0; ++p) {
                const u16 id = (u16)(p + 1);
                int depth = 1; sl[0] = left_order[p]; sp[0] = 0;
                while (depth > 0) {
                    const int left = sl[depth - 1], position = sp[depth - 1];
                    int rs[4]; u32 cs[4];
                    const int n = sorted_adj(t, rank, left, rs, cs);
                    if (position >= n) { --depth; continue; }
                    sp[depth - 1] = position + 1;
                    const int right = rs[position];
                    if (stamp[right] == id) continue;
                    stamp[right] = id;
                    const int owner = matched[right];
                    if (owner >= 0) {
                        if (depth >= STACK) { st = 4; break; }
                        sl[depth] = owner; sp[depth] = 0; ++depth;
                        continue;
                    }
                    for (int f = 0; f < depth; ++f) {
                        int rr[4]; u32 cc[4];
                        sorted_adj(t, rank, sl[f], rr, cc);
                        matched[rr[sp[f] - 1]] = (short)sl[f];
                    }
                    break;
                }
            }
            PHASE(1);
            int selected = 0;
            if (st == 0) {
                for (int r = 0; r < t.R; ++r) {
                    const int l = matched[r];
                    if (l < 0) continue;
                    const int b = t.adj_off[l], n = t.adj_off[l + 1] - b;
                    for (int k = 0; k < n; ++k) {
                        if (t.adj_r[b + k] == r) { const u32 cd = t.adj_c[b + k]; selbits[cd >> 5] |= 1u << (cd & 31); ++selected; break; }
                    }
                }
                if (selected != t.sel_expected) st = 2;
            }
            if (st == 0) {
                // Local gauges, in candidate order (the order of the selected tiles), and the strong tile couplings.
                Chacha g;
                chacha_init(g, key + 16);
                for (int cd = 0; cd < t.C; ++cd) {
                    if (!((selbits[cd >> 5] >> (cd & 31)) & 1u)) continue;
                    int gs[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) gs[k] = (chacha_next(g) & 1u) ? -1 : 1;
                    const u32* ce = t.cand_edges + (size_t)cd * 6;
                    flags[ce[0]] = 1 | ((gs[0] * gs[1] < 0) ? 2 : 0);
                    flags[ce[1]] = 1 | ((gs[0] * gs[2] < 0) ? 2 : 0);
                    flags[ce[2]] = 1 | ((gs[0] * gs[3] < 0) ? 2 : 0);
                    flags[ce[3]] = 1 | ((gs[1] * gs[2] < 0) ? 2 : 0);
                    flags[ce[4]] = 1 | ((gs[1] * gs[3] < 0) ? 2 : 0);
                    flags[ce[5]] = 1 | ((gs[2] * gs[3] < 0) ? 2 : 0);
                }
                PHASE(2);
                // Weak signs, one draw per non-tile edge in edge order.
                Chacha w;
                chacha_init(w, key + 24);
                int m = 0;
                for (int e = 0; e < t.E; ++e) {
                    if (flags[e] & 1) continue;
                    flags[e] = (chacha_next(w) & 1u) ? 2 : 0;
                    if (m < t.pair_cap) pa[m] = (u16)e;
                    ++m;
                }
                PHASE(3);
                if (m > t.pair_cap) st = 5;
                if (st == 0) {
                    // Dimer pairs: shuffle the non-tile edges and take edges whose ends are both free.
                    Chacha d;
                    chacha_init(d, key + 32);
                    shuffle_u16(d, pa, m);
                    for (int w2 = 0; w2 < (t.N + 31) / 32; ++w2) usedbits[w2] = 0;
                    int taken = 0;
                    for (int k = 0; k < m && taken < t.pairs; ++k) {
                        const int e = pa[k];
                        const int u = t.eu[e], v = t.ev[e];
                        if (((usedbits[u >> 5] >> (u & 31)) & 1u) || ((usedbits[v >> 5] >> (v & 31)) & 1u)) continue;
                        usedbits[u >> 5] |= 1u << (u & 31);
                        usedbits[v >> 5] |= 1u << (v & 31);
                        flags[e] = 1 | 2;
                        ++taken;
                    }
                    if (taken < t.pairs) st = 3;
                    PHASE(4);
                }
            }
            cnt[9] = st;
        }
        __syncthreads();

        if (tid == 0) t_mark = clock64();
        if (cnt[9] == 0) {
            int local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = tid; k < t.T; k += blockDim.x) {
                const unsigned char x = flags[t.tri[3 * k]], y = flags[t.tri[3 * k + 1]], z = flags[t.tri[3 * k + 2]];
                const int ns = (x & 1) + (y & 1) + (z & 1);
                const int nonpos = ((x >> 1) & 1) + ((y >> 1) & 1) + ((z >> 1) & 1);
                local[ns * 2 + ((nonpos & 1) == 0 ? 1 : 0)] += 1;
            }
            int wp = 0;
            for (int e = tid; e < t.E; e += blockDim.x) wp += (flags[e] & 3) == 0 ? 1 : 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int v = local[i];
                for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
                if ((tid & 31) == 0 && v) atomicAdd(&cnt[i], v);
            }
            for (int o = 16; o > 0; o >>= 1) wp += __shfl_down_sync(0xffffffffu, wp, o);
            if ((tid & 31) == 0 && wp) atomicAdd(&cnt[8], wp);
            __syncthreads();
            if (tid == 0) {
                double s = 0.0;
                const double m = (double)t.E;
                for (int i = 0; i < 9; ++i) s += t.fw[i] * ((double)cnt[i] / m - t.fmu[i]) / t.fsd[i];
                score[inst] = s;
                status[inst] = 0;
                PHASE(5);
            }
        } else if (tid == 0) {
            score[inst] = 0.0;
            status[inst] = cnt[9];
        }
        __syncthreads();
    }
}
