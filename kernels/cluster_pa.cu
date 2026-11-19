// Local build (BudFox, 2026-11-13): device-side population annealing for the
// Otsuka cluster kernel (QUIP_OTSUKA_PA_DEVICE). A checkpoint is four launches
// on the lane's stream and no host synchronization:
//   quip_pa_energy_best  per-replica energy and lineage-best tracking
//   quip_pa_plan         systematic resampling picks and Houdayer pairing
//   quip_pa_gather       copy parents (state and lineage best) to the other buffer
//   quip_pa_houdayer     one Houdayer cluster swap per replica pair
// Spins are int8 in device node order. Adjacency is the anneal kernel's
// all_deg/all_ell ELL: slot e of node u at all_ell[e * n + u], neighbour in the
// high 16 bits and the coupling in milli in the low 16 bits.
// Kept apart from cluster.cu so the sweep kernel can change independently.

typedef unsigned long long pa_u64;

__device__ __forceinline__ pa_u64 pa_splitmix(pa_u64* state) {
    *state += 0x9E3779B97F4A7C15ULL;
    pa_u64 z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform below `n` by multiply-high, as the host Population::below.
__device__ __forceinline__ unsigned pa_below(pa_u64* state, unsigned n) {
    return (unsigned)__umul64hi(pa_splitmix(state), (pa_u64)n);
}

// The RNG state of one (checkpoint, stream) pair.
__device__ __forceinline__ pa_u64 pa_seed(pa_u64 seed, int checkpoint, int stream) {
    pa_u64 s = seed ^ 0x5EED0F0750CA00ULL;
    s ^= (pa_u64)(unsigned)checkpoint * 0xD1B54A32D192ED03ULL;
    s ^= (pa_u64)(unsigned)stream * 0xAEF17502108EF2D9ULL;
    pa_splitmix(&s);
    return s;
}

// One block per replica. Energy in milli exactly as quip_cluster_energy; a
// replica below its lineage best copies its state into `best`.
extern "C" __global__ void quip_pa_energy_best(
    int n, const int* all_deg, const int* all_ell, const int* h,
    const signed char* spins, long long* energy, signed char* best, long long* best_e) {
    __shared__ unsigned long long total;
    __shared__ int better;
    if (threadIdx.x == 0) total = 0ULL;
    __syncthreads();
    const signed char* s = spins + (size_t)blockIdx.x * n;
    long long local = 0;
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
        int field = 0;
        for (int e = 0, d = all_deg[u]; e < d; ++e) {
            unsigned int w = (unsigned int)__ldg(&all_ell[e * n + u]);
            field += (int)(short)(w & 0xffffu) * s[w >> 16];
        }
        local += (long long)s[u] * (long long)field + 2LL * (long long)h[u] * (long long)s[u];
    }
    atomicAdd(&total, (unsigned long long)local);
    __syncthreads();
    if (threadIdx.x == 0) {
        long long e = ((long long)total) / 2;
        energy[blockIdx.x] = e;
        better = e < best_e[blockIdx.x];
        if (better) best_e[blockIdx.x] = e;
    }
    __syncthreads();
    if (better) {
        signed char* b = best + (size_t)blockIdx.x * n;
        for (int u = threadIdx.x; u < n; u += blockDim.x) b[u] = s[u];
    }
}

// One block. Systematic resampling as the host `resample` (weights
// exp(-step * (E - E_min) / 1000), one uniform offset), then, with
// `houdayer`, a Fisher-Yates shuffle of the persistent pairing order.
// Threads compute the weights into `weight`; thread 0 sums and picks in slot
// order, so the picks equal the host's bit for bit.
extern "C" __global__ void quip_pa_plan(
    int slots, double step, int houdayer, pa_u64 seed, int checkpoint,
    const long long* energy, int* picks, int* order, double* weight) {
    __shared__ long long low_s;
    if (step > 0.0) {
        if (threadIdx.x == 0) {
            long long low = energy[0];
            for (int i = 1; i < slots; ++i) low = energy[i] < low ? energy[i] : low;
            low_s = low;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < slots; i += blockDim.x)
            weight[i] = exp(-step * (double)(energy[i] - low_s) / 1000.0);
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    pa_u64 rng = pa_seed(seed, checkpoint, 0);
    if (step > 0.0) {
        double total = 0.0;
        for (int i = 0; i < slots; ++i) total += weight[i];
        double stride = total / (double)slots;
        double u = (double)(pa_splitmix(&rng) >> 11) * (1.0 / 9007199254740992.0);
        double mark = u * stride;
        double reach = 0.0;
        int count = 0;
        for (int i = 0; i < slots; ++i) {
            reach += weight[i];
            while (mark < reach && count < slots) {
                picks[count++] = i;
                mark += stride;
            }
        }
        while (count < slots) picks[count++] = slots - 1;
    } else {
        for (int i = 0; i < slots; ++i) picks[i] = i;
    }
    if (houdayer) {
        for (int k = slots - 1; k >= 1; --k) {
            int j = (int)pa_below(&rng, (unsigned)(k + 1));
            int t = order[k];
            order[k] = order[j];
            order[j] = t;
        }
    }
}

// One block per destination replica: state and lineage best from its parent.
extern "C" __global__ void quip_pa_gather(
    int n, const int* picks,
    const signed char* src, signed char* dst,
    const signed char* best_src, signed char* best_dst,
    const long long* best_e_src, long long* best_e_dst) {
    int p = picks[blockIdx.x];
    const signed char* s = src + (size_t)p * n;
    const signed char* bs = best_src + (size_t)p * n;
    signed char* d = dst + (size_t)blockIdx.x * n;
    signed char* bd = best_dst + (size_t)blockIdx.x * n;
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
        d[u] = s[u];
        bd[u] = bs[u];
    }
    if (threadIdx.x == 0) best_e_dst[blockIdx.x] = best_e_src[p];
}

// One block per pair (order[2p], order[2p+1]). Picks a uniform node where the
// two replicas differ, grows its connected cluster of differing nodes
// (breadth first, level synchronous), and flips it in both replicas. The
// summed energy of the pair is unchanged. Dynamic shared memory: an int queue
// of n entries, then a seen bitmap of ceil(n / 32) words.
#define PA_HOUDAYER_THREADS 512
extern "C" __global__ void __launch_bounds__(PA_HOUDAYER_THREADS)
quip_pa_houdayer(int n, const int* all_deg, const int* all_ell, signed char* spins,
                 const int* order, pa_u64 seed, int checkpoint) {
    extern __shared__ int pa_smem[];
    int* queue = pa_smem;
    unsigned* seen = (unsigned*)(pa_smem + n);
    __shared__ int scan[PA_HOUDAYER_THREADS];
    __shared__ int head, tail, root;
    signed char* a = spins + (size_t)order[2 * blockIdx.x] * n;
    signed char* b = spins + (size_t)order[2 * blockIdx.x + 1] * n;
    int words = (n + 31) >> 5;
    for (int w = threadIdx.x; w < words; w += blockDim.x) seen[w] = 0u;
    // Count differing nodes over contiguous per-thread ranges.
    int per = (n + blockDim.x - 1) / blockDim.x;
    int lo = threadIdx.x * per;
    int hi = min(n, lo + per);
    int count = 0;
    for (int u = lo; u < hi; ++u) count += a[u] != b[u];
    scan[threadIdx.x] = count;
    __syncthreads();
    // Inclusive Hillis-Steele scan.
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int v = threadIdx.x >= off ? scan[threadIdx.x - off] : 0;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    int differ = scan[blockDim.x - 1];
    if (differ == 0) return;  // uniform across the block
    if (threadIdx.x == 0) {
        pa_u64 rng = pa_seed(seed, checkpoint, 1 + blockIdx.x);
        root = (int)pa_below(&rng, (unsigned)differ);  // rank of the root
    }
    __syncthreads();
    int rank = root;
    int before = scan[threadIdx.x] - count;
    __syncthreads();
    if (rank >= before && rank < before + count) {
        int k = rank - before;
        for (int u = lo; u < hi; ++u) {
            if (a[u] != b[u]) {
                if (k == 0) {
                    root = u;
                    break;
                }
                --k;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        queue[0] = root;
        seen[root >> 5] = 1u << (root & 31);
        head = 0;
        tail = 1;
    }
    __syncthreads();
    while (true) {
        int from = head, to = tail;
        // Every thread holds the same level bounds before any thread appends.
        __syncthreads();
        if (from >= to) break;
        for (int i = from + threadIdx.x; i < to; i += blockDim.x) {
            int u = queue[i];
            for (int e = 0, d = all_deg[u]; e < d; ++e) {
                int v = (int)(((unsigned)__ldg(&all_ell[e * n + u])) >> 16);
                if (a[v] != b[v]) {
                    unsigned bit = 1u << (v & 31);
                    if (!(atomicOr(&seen[v >> 5], bit) & bit)) queue[atomicAdd(&tail, 1)] = v;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) head = to;
        __syncthreads();
    }
    for (int i = threadIdx.x; i < tail; i += blockDim.x) {
        int u = queue[i];
        a[u] = -a[u];
        b[u] = -b[u];
    }
}
