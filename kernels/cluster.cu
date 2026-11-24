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

__device__ __forceinline__ int find_root(int* p, int x) {
    while (true) {
        int y = *(volatile int*)&p[x];
        if (y == x) return x;
        int z = *(volatile int*)&p[y];
        if (z == y) return y;
        *(volatile int*)&p[x] = z;  // path halving: z is an ancestor of x
        x = z;
    }
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
extern "C" __global__ void quip_cluster_anneal(
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
    int* parent = smem;
    int* acc = parent + n;
    signed char* s = (signed char*)(acc + n);
    // Use CU-1 dynamic padding for the flag. A static int is rounded to
    // 16 bytes by the compiler, exceeding the four-byte static reservation.
    int* changed = (int*)(s + ((n + 3) & ~3));
    signed char* mine = gspins + (size_t)blockIdx.x * n;
    uint64_t rng = grng[(size_t)blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = mine[i];
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
        for (int sweep = sweep0; sweep < sweep0 + nsweeps; ++sweep) {
            float beta = beta_at(q, sweep);
            // beta_knee > 0 with knee_frac == 0 is the cluster-bond temperature factor.
            float lam = (q.beta_knee > 0.0f && q.knee_frac <= 0.0f) ? q.beta_knee : 1.0f;
            float cbeta = beta * lam;
            float bt = cbeta * q.tile_w * 0.001f;
            float z = __expf(-2.0f * bt);
            float w0 = z * z * z * z;
            float zero_ice = w0, single_ice = w0 + 4.0f * (z - w0) / 3.0f;
            float zero_defect = __expf(-6.0f * bt);
            float dimer_link = 1.0f - __expf(-2.0f * cbeta * q.dimer_w * 0.001f);

            const int skip = q.chunk >> 8;
            for (int i = threadIdx.x; i < n; i += blockDim.x) parent[i] = i;
            __syncthreads();
            int cw = -1; float cz0 = 0.0f, csi = 0.0f, czd = 0.0f;
            for (int t = threadIdx.x; (skip & 1) == 0 && t < g.T; t += blockDim.x) {
                int4 nd = g.tiles[t]; char4 ga = g.gauge[t];
                // Bit k of m is set when node k is positive in the tile gauge. Registers only.
                unsigned m = 0;
                m |= (s[nd.x] * (int)ga.x > 0) ? 1u : 0u;
                m |= (s[nd.y] * (int)ga.y > 0) ? 2u : 0u;
                m |= (s[nd.z] * (int)ga.z > 0) ? 4u : 0u;
                m |= (s[nd.w] * (int)ga.w > 0) ? 8u : 0u;
                int np = __popc(m), nn = 4 - np;
                if (np == 0 || np == 4) continue;
                float r = urand(rng);
                float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                if (g.tile_w && (float)g.tile_w[t] != q.tile_w) {
                    int tw = g.tile_w[t];
                    if (tw != cw) {
                        float tbt = cbeta * (float)tw * 0.001f;
                        float tz = __expf(-2.0f * tbt);
                        cz0 = tz * tz * tz * tz;
                        csi = cz0 + 4.0f * (tz - cz0) / 3.0f;
                        czd = __expf(-6.0f * tbt);
                        cw = tw;
                    }
                    tz0 = cz0; tsi = csi; tzd = czd;
                }
                const unsigned nm = (~m) & 15u;
                if (np == 2) {
                    if (r < tz0) continue;
                    // pos[0],pos[1] are the set bits of m in ascending order; neg likewise for nm.
                    const int p0 = __ffs(m) - 1, p1 = __ffs(m & (m - 1u)) - 1;
                    const int n0 = __ffs(nm) - 1, n1 = __ffs(nm & (nm - 1u)) - 1;
                    const int nodes[4] = {nd.x, nd.y, nd.z, nd.w};
                    if (r < tsi) {
                        uint32_t b = (uint32_t)xs(rng);
                        const int a = (b & 1) ? p1 : p0;
                        const int c = ((b >> 1) & 1) ? n1 : n0;
                        unite(parent, nodes[a], nodes[c]);
                    } else {
                        int ch = (int)(xs(rng) & 1);
                        unite(parent, nodes[p0], nodes[ch ? n1 : n0]);
                        unite(parent, nodes[p1], nodes[ch ? n0 : n1]);
                    }
                } else if (r >= tzd) {
                    uint32_t b = (uint32_t)xs(rng);
                    unsigned pm = m, qm = nm;
                    for (int i = (int)((b & 0xffff) % (unsigned)np); i > 0; --i) pm &= pm - 1u;
                    for (int i = (int)((b >> 16) % (unsigned)nn); i > 0; --i) qm &= qm - 1u;
                    const int nodes[4] = {nd.x, nd.y, nd.z, nd.w};
                    unite(parent, nodes[__ffs(pm) - 1], nodes[__ffs(qm) - 1]);
                }
            }
            int caw = -1; float cbond = 0.0f;
            for (int p = threadIdx.x; (skip & 2) == 0 && p < g.P; p += blockDim.x) {
                int2 pr = g.pairs[p];
                int lw = g.link_w ? g.link_w[p] : (int)q.dimer_w;
                int aw = abs(lw);
                if (aw != caw) {
                    cbond = (aw == (int)q.dimer_w) ? dimer_link : 1.0f - __expf(-2.0f * cbeta * (float)aw * 0.001f);
                    caw = aw;
                }
                float bond = cbond;
                bool satisfied = lw > 0 ? s[pr.x] == s[pr.y] : s[pr.x] != s[pr.y];
                if (satisfied && urand(rng) < bond) unite(parent, pr.x, pr.y);
            }
            __syncthreads();
            for (int i = threadIdx.x; (skip & 4) == 0 && i < n; i += blockDim.x) *(volatile int*)&parent[i] = find_root(parent, i);
            __syncthreads();
            uint32_t salt = mix32((uint32_t)sweep * 0x9e3779b9U + blockIdx.x * 0x85ebca6bU);
            for (int round = 0; (skip & 8) == 0 && round < q.rounds; ++round) {
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
                // Decide on roots; acc[root] = 1 marks a flip.
                for (int r = threadIdx.x; r < n; r += blockDim.x) {
                    if (parent[r] != r) continue;
                    if ((int)(mix32(r ^ salt) % q.rounds) != round) continue;
                    float delta = -2.0f * acc[r];
                    acc[r] = (urand(rng) < 1.0f / (1.0f + __expf(beta * delta * 0.001f))) ? 1 : 0;
                }
                __syncthreads();
                for (int u = threadIdx.x; u < n; u += blockDim.x) {
                    int lu = parent[u];
                    if ((int)(mix32(lu ^ salt) % q.rounds) == round && acc[lu] == 1) s[u] = -s[u];
                }
                __syncthreads();
            }
            if ((skip & 16) == 0 && q.single_every > 0 && (sweep + 1) % q.single_every == 0) {
                metropolis(g, s, beta, rng, false, nullptr);
            }
        }
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) mine[i] = s[i];
    grng[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = rng;
}

// Twice the energy in milli per anneal (permuted layout). Used to rank reads for resampling.
extern "C" __global__ void quip_cluster_energy(
    int n, const int* all_deg, const int* all_ell, const int* h,
    const signed char* gspins, int* out) {
    __shared__ int tot;
    const signed char* s = gspins + (size_t)blockIdx.x * n;
    if (threadIdx.x == 0) tot = 0;
    __syncthreads();
    int local = 0;
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
        int field = 2 * h[u];
        for (int e = 0, d = all_deg[u]; e < d; ++e) {
            unsigned int w = (unsigned int)__ldg(&all_ell[e * n + u]);
            field += (int)(short)(w & 0xffffu) * s[(unsigned)w >> 16];
        }
        local += s[u] * field;
    }
    atomicAdd(&tot, local);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = tot;
}

// Row i of out becomes row map[i] of in (one block per anneal, original permuted layout).
extern "C" __global__ void quip_cluster_resample(
    int n, const int* map, const signed char* in, signed char* out) {
    const signed char* src = in + (size_t)map[blockIdx.x] * n;
    signed char* dst = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// Houdayer isoenergetic cluster move (ICM) between pairs of reads at the same temperature: the connected domain of
// nodes where the two replicas differ (a randomly chosen one, never a percolating one) is swapped between them, so the
// pair's total energy is conserved and the move is always accepted. One block per pair; perm gives the pairing.
extern "C" __global__ void quip_cluster_icm(
    int n, const int* all_deg, const int* all_ell, const int* perm, signed char* gspins,
    unsigned int salt, int maxsize) {
    extern __shared__ int smem[];
    int* parent = smem;
    int* cnt = parent + n;  // cnt[0] = domain size, cnt[1] = chosen root (-1: none)
    signed char* A = gspins + (size_t)perm[2 * blockIdx.x] * n;
    signed char* B = gspins + (size_t)perm[2 * blockIdx.x + 1] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parent[i] = i;
    if (threadIdx.x == 0) { cnt[0] = 0; cnt[1] = -1; }
    __syncthreads();
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
        if (A[u] == B[u]) continue;
        for (int e = 0, d = all_deg[u]; e < d; ++e) {
            unsigned int w = (unsigned int)__ldg(&all_ell[e * n + u]);
            int v = (int)((unsigned)w >> 16);
            if (v > u && A[v] != B[v]) unite(parent, u, v);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) *(volatile int*)&parent[i] = find_root(parent, i);
    __syncthreads();
    if (threadIdx.x == 0) {
        int i0 = (int)(mix32(salt ^ (blockIdx.x * 0x9e3779b9U)) % (unsigned)n);
        for (int k = 0; k < n; ++k) {
            int i = i0 + k; if (i >= n) i -= n;
            if (A[i] != B[i]) { cnt[1] = parent[i]; break; }
        }
    }
    __syncthreads();
    int root = cnt[1];
    if (root < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (A[i] != B[i] && parent[i] == root) atomicAdd(&cnt[0], 1);
    __syncthreads();
    if (cnt[0] >= maxsize) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (A[i] != B[i] && parent[i] == root) { signed char t = A[i]; A[i] = B[i]; B[i] = t; }
    }
}
