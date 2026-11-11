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
        for (int pass = 0; pass < 256; ++pass) {
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
            for (int i = threadIdx.x; i < n; i += blockDim.x) atomicExch(&parent[i], find_root(parent, i));
            __syncthreads();
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
            if (q.single_every > 0 && (sweep + 1) % q.single_every == 0) {
                metropolis(g, s, beta, rng, false, nullptr);
            }
        }
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) mine[i] = s[i];
    grng[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = rng;
}
