// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA BLOCK GIBBS SAMPLER - SELF-FEEDING PERSISTENT KERNEL
// ==============================================================================
// Architecture: self-feeding persistent kernel with 3-slot rotating buffers.
// Each nonce owns blocks_per_nonce blocks (static assignment).
// Blocks process reads via per-nonce work queue, then barrier.
// After barrier, last block marks slot COMPLETE and finds next
// READY slot. Host uploads new models to free slots while kernel runs.
//
// Color loop inside kernel with __syncthreads() between colors.
// Spin state in shared memory (unpacked, 1 byte/spin) - same as Metal.

// Largest N this kernel's shared state supports. The host supplies it through
// NVRTC as -D QUIP_MAX_NODES=<n>; the default matches the historical fixed
// size so a plain compile is unchanged. This array is per block and lives in
// shared memory, so the device's shared-memory-per-block limit caps it: on
// sm_86 that is 49152 bytes less the 8 taken by s_chunk and s_arrival.
#ifndef QUIP_MAX_NODES
#define QUIP_MAX_NODES 4800
#endif

// Dense updates always use the launch-sized prefix and per-block spill.
#if defined(QUIP_GIBBS_DENSE) && !defined(QUIP_GIBBS_SPILL)
#define QUIP_GIBBS_SPILL
#endif

// One coefficient type serves both h and J. The default expands to the
// original spelling so the mining instantiation preprocesses unchanged.
#if defined(QUIP_GIBBS_COEF_I32)
typedef int coef_t;
#elif defined(QUIP_GIBBS_COEF_F32)
typedef float coef_t;
#else
#define coef_t signed char
#endif

// Spill sessions can have coefficient and sample slots larger than 2 GiB.
#ifdef QUIP_GIBBS_SPILL
#define GIBBS_OFFSET unsigned long long
#else
#define GIBBS_OFFSET int
#endif

// ==============================================================================
// NonceControl layout (flat int array, CTRL_STRIDE ints per nonce)
// ==============================================================================
// Slot states
#define SLOT_EMPTY    0
#define SLOT_READY    1
#define SLOT_ACTIVE   2
#define SLOT_COMPLETE 3

// Field offsets within each nonce's control block
#define CTRL_STRIDE       8
#define CTRL_SLOT_STATE_0 0
#define CTRL_SLOT_STATE_1 1
#define CTRL_SLOT_STATE_2 2
#define CTRL_ACTIVE_SLOT  3
#define CTRL_BLOCKS_DONE  4
#define CTRL_WORK_QUEUE   5
#define CTRL_EXIT_NOW     6
#define CTRL_GENERATION   7

// CTRL_EXIT_NOW carries EXIT_AFTER_MODEL (1) or EXIT_ABORT (2), see sa.cu.
// This kernel treats both as "leave after the current model": it never
// aborts mid-model.

// ==============================================================================
// xoshiro128** RNG
// ==============================================================================

__device__ __forceinline__ unsigned int rotl(
    unsigned int x, int k
) {
    return (x << k) | (x >> (32 - k));
}

struct Xoshiro128 {
    unsigned int s[4];
};

__device__ __forceinline__ unsigned int xoshiro128ss(
    Xoshiro128 &state
) {
    unsigned int result = rotl(state.s[1] * 5, 7) * 9;
    unsigned int t = state.s[1] << 9;

    state.s[2] ^= state.s[0];
    state.s[3] ^= state.s[1];
    state.s[1] ^= state.s[2];
    state.s[0] ^= state.s[3];
    state.s[2] ^= t;
    state.s[3] = rotl(state.s[3], 11);

    return result;
}

__device__ __forceinline__ void xoshiro128_init(
    Xoshiro128 &state, unsigned int seed,
    unsigned int block_id, unsigned int thread_id
) {
    unsigned long long z =
        (unsigned long long)seed + block_id * 65537ULL
        + thread_id * 2654435761ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    state.s[0] = (unsigned int)(z & 0xFFFFFFFF);
    state.s[1] = (unsigned int)(z >> 32);

    z = (unsigned long long)seed + block_id * 65537ULL
        + thread_id * 2654435761ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    state.s[2] = (unsigned int)(z & 0xFFFFFFFF);
    state.s[3] = (unsigned int)(z >> 32);

    if (state.s[0] == 0 && state.s[1] == 0
        && state.s[2] == 0 && state.s[3] == 0) {
        state.s[0] = 1;
    }
}

// ==============================================================================
// Bit-packing helpers (for output)
// ==============================================================================

__device__ __forceinline__ void set_spin_packed(
    int var, int spin, signed char* packed
) {
    int byte_idx = var >> 3;
    int bit_idx = var & 7;
    signed char bit = (spin < 0) ? 1 : 0;
    signed char mask = 1 << bit_idx;
    if (bit) {
        packed[byte_idx] |= mask;
    } else {
        packed[byte_idx] &= ~mask;
    }
}

// ==============================================================================
// Effective field from shared memory state
// h_eff = h_i + sum_j(J_ij * x_j)
// ==============================================================================

#ifdef QUIP_GIBBS_SPILL
// Both loads and stores use the same split, with one tail per block.
struct GibbsState {
    volatile signed char* prefix;
    volatile signed char* spill;
    int shared_spins;

    __device__ __forceinline__ volatile signed char& operator[](int var) const {
        return var < shared_spins ? prefix[var] : spill[var - shared_spins];
    }
};
#endif

#ifdef QUIP_GIBBS_DENSE
#ifdef QUIP_GIBBS_COEF_F32
typedef float field_t;
#else
typedef int field_t;
#endif

// Each block has 256 threads. Integer partials stay exact through both
// reduction stages; only the final field converts for the probability.
__device__ float compute_effective_field_dense(
    int var, int N, GibbsState state,
    const coef_t* matrix, GIBBS_OFFSET j_off,
    const coef_t* fields, GIBBS_OFFSET h_off,
    field_t* partials
) {
    field_t sum = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        sum += (field_t)__ldg(&matrix[j_off + (GIBBS_OFFSET)var * N + j])
             * (int)state[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) partials[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partials[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) sum += (field_t)__ldg(&fields[h_off + var]);
    }
    // Only thread zero consumes the result. The update barrier protects
    // partials and the new spin before the next node starts its reduction.
    return (float)sum;
}
#else
__device__ float compute_effective_field_shared(
    int var,
#ifdef QUIP_GIBBS_SPILL
    GibbsState shared_state,
#else
    volatile signed char* shared_state,
#endif
    const int* csr_row_ptr, int rp_off,
    const int* csr_col_ind, int ci_off,
    const coef_t* csr_J_vals, GIBBS_OFFSET j_off,
    const coef_t* h_vals, GIBBS_OFFSET h_off
) {
    int start = __ldg(&csr_row_ptr[rp_off + var]);
    int end = __ldg(&csr_row_ptr[rp_off + var + 1]);

#ifdef QUIP_GIBBS_COEF_I32
    // WIDE_MAX_FIELD bounds every integer partial sum below INT_MAX.
    int h_eff = __ldg(&h_vals[h_off + var]);
#else
    float h_eff = (float)__ldg(&h_vals[h_off + var]);
#endif

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = __ldg(&csr_col_ind[ci_off + p]);
#ifdef QUIP_GIBBS_COEF_F32
        float Jij = __ldg(&csr_J_vals[j_off + p]);
#else
        int Jij = (int)__ldg(&csr_J_vals[j_off + p]);
#endif
        int neighbor_spin = (int)shared_state[neighbor];
#ifdef QUIP_GIBBS_COEF_I32
        h_eff += Jij * neighbor_spin;
#else
        h_eff += (float)(Jij * neighbor_spin);
#endif
    }

#ifdef QUIP_GIBBS_COEF_I32
    return (float)h_eff;
#else
    return h_eff;
#endif
}

#endif

// ==============================================================================
// Gibbs update: P(spin = +1) = 1 / (1 + exp(2 * beta * h_eff))
// ==============================================================================

__device__ __forceinline__ int gibbs_sample(
    float h_eff, float beta, Xoshiro128 &rng
) {
    float prob_plus =
#ifdef QUIP_GIBBS_COEF_F32
        // The admitted field keeps 2*h_eff finite; 2*beta need not fit.
        1.0f / (1.0f + __expf((2.0f * h_eff) * beta));
#else
        1.0f / (1.0f + __expf(2.0f * beta * h_eff));
#endif
    unsigned int rand_val = xoshiro128ss(rng);
    float rand_norm = (float)rand_val / 4294967295.0f;
    return (rand_norm < prob_plus) ? 1 : -1;
}

// ==============================================================================
// Metropolis update: accept flip with P = min(1, exp(-beta * delta))
// ==============================================================================

__device__ __forceinline__ int metropolis_update(
    int current_spin, float h_eff, float beta,
    Xoshiro128 &rng
) {
    float delta = -2.0f * (float)current_spin * h_eff;

    if (delta <= 0.0f) {
        return -current_spin;
    }

    float prob = __expf(-delta * beta);
    unsigned int rand_val = xoshiro128ss(rng);
    float rand_norm = (float)rand_val / 4294967295.0f;
    return (rand_norm < prob) ? -current_spin : current_spin;
}

// ==============================================================================
// KERNEL: Self-feeding persistent kernel with 3-slot rotating buffers
// ==============================================================================
// Each nonce owns blocks_per_nonce blocks (static assignment).
// Blocks process reads via per-nonce work queue, then barrier.
// After barrier, last block marks slot COMPLETE and finds next
// READY slot. Host uploads new models to free slots while kernel
// runs. Kernel exits when no READY slot found or exit_now set.

extern "C" __global__ void cuda_gibbs_self_feeding(
    // Shared topology (same for all nonces)
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,

    // Color blocks (shared topology, tiled per nonce)
    const int* __restrict__ all_block_starts,
    const int* __restrict__ all_block_counts,
    const int* __restrict__ all_color_nodes,
    int num_colors,

    // Shared annealing schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Per-slot flat buffers (stride: see below)
    // J: indexed by (nonce_id * 3 + slot_id) * nnz + p
    coef_t* slot_J_vals,
    // h: indexed by (nonce_id * 3 + slot_id) * N + var
    coef_t* slot_h_vals,
    // samples: indexed by (nonce_id * 3 + slot_id)
    //          * reads_per_nonce * max_packed_size
    //          + read_idx * max_packed_size
    signed char* slot_samples,
    // energies: indexed by (nonce_id * 3 + slot_id)
    //           * reads_per_nonce + read_idx
    int* slot_energies,

    // Per-nonce control (flat int array, CTRL_STRIDE per nonce)
    volatile int* nonce_ctrl,

    // Config
    int num_nonces,
    int blocks_per_nonce,
    int reads_per_nonce,
    int N,
    int nnz,
    int max_packed_size,
    int chunks_per_model,
    int reads_per_chunk,
    unsigned int base_seed,
    int update_mode
#ifdef QUIP_GIBBS_SPILL
    , signed char* spin_spill,
    int gibbs_shared_spins
#endif
) {
#ifdef QUIP_GIBBS_SPILL
    extern __shared__ signed char spin_prefix[];
    GibbsState shared_state = {
        spin_prefix,
        spin_spill + (unsigned long long)blockIdx.x * (N - gibbs_shared_spins),
        gibbs_shared_spins
    };
#else
    __shared__ signed char shared_state[QUIP_MAX_NODES];
#endif
    __shared__ int s_chunk;
    __shared__ int s_arrival;
#ifdef QUIP_GIBBS_DENSE
    __shared__ field_t partials[8];
#endif

    int nonce_id = blockIdx.x / blocks_per_nonce;
    if (nonce_id >= num_nonces) return;

    int ctrl_base = nonce_id * CTRL_STRIDE;
    int color_base = nonce_id * num_colors;
    int local_block = blockIdx.x % blocks_per_nonce;

    // Lead block (local_block==0) claims the initial READY
    // slot; other blocks spin until leader signals via
    // generation counter.  This prevents multiple blocks
    // from claiming different slots.
    int active_slot;
    if (local_block == 0) {
        // Waits for a slot instead of giving up after one pass. The host's
        // slot uploads -- and the SLOT_READY write that publishes them -- run
        // on its transfer stream while this launch runs on the compute
        // stream, and the two are deliberately unordered, so the lead block
        // can start before its own cold-start slot has been marked READY.
        // Retiring the nonce on that first empty pass (generation = -1, which
        // also sends every non-lead block home) meant the host kept admitting
        // jobs no block was left to run: they never reach SLOT_COMPLETE and
        // the driver waits forever on a completion that cannot arrive. Same
        // failure QUI-828 fixed in the model loop's slot wait; this claim kept
        // the old give-up-immediately behavior. Only an explicit host
        // EXIT_NOW ends the wait, and teardown always raises it.
        active_slot = -1;
        if (threadIdx.x == 0) {
            while (true) {
                for (int s = 0; s < 3; s++) {
                    int old = atomicCAS(
                        (int*)&nonce_ctrl[ctrl_base + s],
                        SLOT_READY, SLOT_ACTIVE
                    );
                    if (old == SLOT_READY) {
                        active_slot = s;
                        break;
                    }
                }
                if (active_slot >= 0) {
                    nonce_ctrl[
                        ctrl_base + CTRL_ACTIVE_SLOT
                    ] = active_slot;
                    nonce_ctrl[
                        ctrl_base + CTRL_WORK_QUEUE
                    ] = 0;
                    nonce_ctrl[
                        ctrl_base + CTRL_BLOCKS_DONE
                    ] = 0;
                    __threadfence();
                    atomicExch(
                        (int*)&nonce_ctrl[
                            ctrl_base + CTRL_GENERATION],
                        1
                    );
                    break;
                }
                if (nonce_ctrl[
                        ctrl_base + CTRL_EXIT_NOW]) {
                    atomicExch(
                        (int*)&nonce_ctrl[
                            ctrl_base + CTRL_GENERATION],
                        -1
                    );
                    break;  // host ended the stream
                }
                __nanosleep(10000);  // 10us
            }
            s_chunk = active_slot;
        }
        __syncthreads();
        active_slot = s_chunk;
        if (active_slot < 0) return;
    } else {
        // Non-lead: thread 0 spins on generation, then
        // broadcasts active_slot to other threads.
        if (threadIdx.x == 0) {
            while (true) {
                int gen = nonce_ctrl[
                    ctrl_base + CTRL_GENERATION];
                if (gen > 0) { s_chunk = 1; break; }
                if (gen < 0) { s_chunk = -1; break; }
                __nanosleep(100);
            }
        }
        __syncthreads();
        if (s_chunk < 0) return;
        active_slot = nonce_ctrl[
            ctrl_base + CTRL_ACTIVE_SLOT];
    }

    int my_gen = 1;

    // === Model loop: process models until no READY slot ===
    while (true) {
        // Compute buffer offsets for this slot
        GIBBS_OFFSET slot_idx = nonce_id * 3 + active_slot;
        GIBBS_OFFSET j_off = slot_idx * nnz;
        GIBBS_OFFSET h_off = slot_idx * N;
        GIBBS_OFFSET sample_base = slot_idx * reads_per_nonce
                          * max_packed_size;
#ifndef QUIP_GIBBS_COEF_F32
        int energy_base = slot_idx * reads_per_nonce;
#endif

        unsigned int seed =
            base_seed + (unsigned int)(nonce_id * 3
                                       + active_slot);

        // === Work-steal loop: grab chunks of reads ===
        while (true) {
            if (threadIdx.x == 0) {
                s_chunk = atomicAdd(
                    (int*)&nonce_ctrl[
                        ctrl_base + CTRL_WORK_QUEUE],
                    1
                );
            }
            __syncthreads();

            if (s_chunk >= chunks_per_model) break;

            int read_start = s_chunk * reads_per_chunk;
            int read_end = read_start + reads_per_chunk;
            if (read_end > reads_per_nonce)
                read_end = reads_per_nonce;

            // Process each read in this chunk
            for (int read_idx = read_start;
                 read_idx < read_end; read_idx++) {

                // Init RNG
                Xoshiro128 rng;
                xoshiro128_init(
                    rng, seed,
                    (unsigned int)read_idx, threadIdx.x
                );

                // Phase 1: Init random spins
                for (int var = threadIdx.x; var < N;
                     var += blockDim.x) {
                    unsigned int r = xoshiro128ss(rng);
                    int spin = (r & 1) ? -1 : 1;
                    shared_state[var] =
                        (signed char)spin;
                }
                __syncthreads();

                // Phase 2: Chromatic Gibbs annealing
                for (int beta_idx = 0;
                     beta_idx < num_betas;
                     beta_idx++) {
                    float beta = __ldg(
                        &beta_schedule[beta_idx]);

                    for (int sweep = 0;
                         sweep < sweeps_per_beta;
                         sweep++) {

                        for (int color = 0;
                             color < num_colors;
                             color++) {
                            int bstart = __ldg(
                                &all_block_starts[
                                    color_base + color]);
                            int bcount = __ldg(
                                &all_block_counts[
                                    color_base + color]);
#ifdef QUIP_GIBBS_DENSE
                            for (int i = 0; i < bcount; ++i) {
#else
                            for (int i = threadIdx.x;
                                 i < bcount;
                                 i += blockDim.x) {
#endif
                                int var = __ldg(
                                    &all_color_nodes[
                                        bstart + i]);

#ifdef QUIP_GIBBS_DENSE
                                float h_eff = compute_effective_field_dense(
                                    var, N, shared_state, slot_J_vals, j_off,
                                    slot_h_vals, h_off, partials);
                                if (threadIdx.x == 0) {
#else
                                float h_eff =
                                    compute_effective_field_shared(
                                        var,
                                        shared_state,
                                        csr_row_ptr, 0,
                                        csr_col_ind, 0,
                                        slot_J_vals,
                                        j_off,
                                        slot_h_vals,
                                        h_off
                                    );
#endif

                                int new_spin;
                                if (update_mode == 0) {
                                    new_spin =
                                        gibbs_sample(
                                            h_eff,
                                            beta, rng);
                                } else {
                                    int cur = (int)
                                        shared_state[
                                            var];
                                    new_spin =
                                        metropolis_update(
                                            cur, h_eff,
                                            beta, rng);
                                }

                                shared_state[var] =
                                    (signed char)
                                    new_spin;
#ifdef QUIP_GIBBS_DENSE
                                }
                                __syncthreads();
#endif
                            }
                            __syncthreads();
                        }
                    }
                }

                // Real energies are scored from the original f64 graph on
                // the host. An integer output cannot represent them.
#ifndef QUIP_GIBBS_COEF_F32
                // Phase 3: Energy (warp reduction)
                float thread_energy = 0.0f;
                for (int var = threadIdx.x; var < N;
                     var += blockDim.x) {
                    int spin_i =
                        (int)shared_state[var];

                    thread_energy +=
                        (float)slot_h_vals[
                            h_off + var]
                        * (float)spin_i;

#ifdef QUIP_GIBBS_DENSE
                    int start = var * N;
                    int end = start + N;
#else
                    int start = __ldg(
                        &csr_row_ptr[var]);
                    int end = __ldg(
                        &csr_row_ptr[var + 1]);
#endif
                    for (int p = start; p < end;
                         ++p) {
#ifdef QUIP_GIBBS_DENSE
                        int j = p - start;
#else
                        int j = __ldg(
                            &csr_col_ind[p]);
#endif
                        if (j > var) {
                            int Jij = (int)
                                slot_J_vals[
                                    j_off + p];
                            int spin_j = (int)
                                shared_state[j];
                            thread_energy +=
                                (float)(Jij
                                    * spin_i
                                    * spin_j);
                        }
                    }
                }

                unsigned mask = __activemask();
                for (int offset = 16; offset > 0;
                     offset >>= 1) {
                    thread_energy +=
                        __shfl_down_sync(
                            mask, thread_energy,
                            offset);
                }

                if ((threadIdx.x & 31) == 0) {
                    atomicAdd(
                        &slot_energies[
                            energy_base + read_idx],
                        (int)thread_energy);
                }

                __syncthreads();
#endif

                // Phase 4: Bit-pack (thread 0)
                if (threadIdx.x == 0) {
                    int packed_size = (N + 7) / 8;
                    signed char* output =
                        &slot_samples[
                            sample_base
                            + read_idx
                              * max_packed_size];

                    for (int b = 0;
                         b < packed_size; b++) {
                        output[b] = 0;
                    }
                    for (int var = 0;
                         var < N; var++) {
                        int spin = (int)
                            shared_state[var];
                        set_spin_packed(
                            var, spin, output);
                    }
                }

                __syncthreads();
            }
        }

        // === Inter-block barrier (generation-based) ===
        if (threadIdx.x == 0) {
            s_arrival = atomicAdd(
                (int*)&nonce_ctrl[
                    ctrl_base + CTRL_BLOCKS_DONE],
                1
            );
        }
        __syncthreads();

        if (s_arrival + 1 == blocks_per_nonce) {
            // Last block: fence all prior writes so the host
            // sees energy/sample data before SLOT_COMPLETE.
            __threadfence();

            // Last block: transition state
            if (threadIdx.x == 0) {
                // Mark current slot COMPLETE
                nonce_ctrl[
                    ctrl_base + active_slot
                ] = SLOT_COMPLETE;

                // Reset barrier counter
                nonce_ctrl[
                    ctrl_base + CTRL_BLOCKS_DONE
                ] = 0;

                // Check exit
                if (nonce_ctrl[
                        ctrl_base + CTRL_EXIT_NOW]) {
                    __threadfence();
                    atomicExch(
                        (int*)&nonce_ctrl[
                            ctrl_base
                            + CTRL_GENERATION],
                        -1
                    );
                } else {
                    // Find next READY slot
                    int next_slot = -1;
                    for (int s = 0; s < 3; s++) {
                        int old = atomicCAS(
                            (int*)&nonce_ctrl[
                                ctrl_base + s],
                            SLOT_READY,
                            SLOT_ACTIVE
                        );
                        if (old == SLOT_READY) {
                            next_slot = s;
                            break;
                        }
                    }

                    if (next_slot >= 0) {
                        nonce_ctrl[
                            ctrl_base
                            + CTRL_ACTIVE_SLOT
                        ] = next_slot;
                        nonce_ctrl[
                            ctrl_base
                            + CTRL_WORK_QUEUE
                        ] = 0;
                        __threadfence();
                        atomicAdd(
                            (int*)&nonce_ctrl[
                                ctrl_base
                                + CTRL_GENERATION],
                            1
                        );
                    } else {
                        // No more work
                        __threadfence();
                        atomicExch(
                            (int*)&nonce_ctrl[
                                ctrl_base
                                + CTRL_GENERATION],
                            -1
                        );
                    }
                }
            }
        }

        // All blocks: spin on generation
        while (true) {
            int gen = nonce_ctrl[
                ctrl_base + CTRL_GENERATION];
            if (gen > my_gen) break;
            if (gen < 0) return;
            __nanosleep(100);
        }
        __syncthreads();
        my_gen++;

        // Read new active slot
        active_slot = nonce_ctrl[
            ctrl_base + CTRL_ACTIVE_SLOT];
    }
}
