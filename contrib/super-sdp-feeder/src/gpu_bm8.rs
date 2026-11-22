//! `quip-gpu-feeder v2.0`: 100% GPU-Native FastDimer Generation + SMEM-FP16 BM4/BM8 + Exact FP32 BM8_w30!
//!
//! Eliminates BOTH host CPU generation bottleneck AND RTX 3080Ti 6MB L2 cache bottleneck:
//!   1. Host uploads ONLY 5 ChaCha8 keys per salt (`160 B/salt`, 950x less PCIe than CSR upload!).
//!   2. Kernel 1 (`gpu_generate_dimer_batch`): Generates `B` salts in parallel directly in VRAM
//!      (bit-exact to CPU `FastDimer`, zero host memory allocations, zero PCIe CSR transfer!).
//!   3. Kernel 2A (`bm4_batch_sdp_smem_fp16`): Optional ultra-fast Rank-4 FP16 pre-filter
//!      with `s_u` + `s_stile` in SMEM (45 KB -> 2 blocks/SM) AND `mom`/`u_nxt` in registers
//!      (ZERO DRAM writes during iterations -> 30,000-50,000 salts/s on RTX 3080Ti!).
//!   4. Kernel 2B (`bm8_batch_sdp_smem_fp16`): Rank-8 FP16 BM-SDP with BOTH `s_u` (57.6 KB)
//!      and `s_stile` (32.8 KB) resident in Ampere dynamic SMEM (90.4 KB < 99 KB limit),
//!      turning 112 MB/salt of random DRAM misses into 20 TB/s on-chip SMEM hits!
//!   5. Kernel 3 (`bm8_batch_sdp_fp32_indexed`): Exact FP32 200-step `BM8_w30` (`return_u=true`)
//!      running ONLY on the ~2% surviving slot indices directly from VRAM!

use crate::engine::FastDimer;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

pub const MAX_N: usize = 3600;
pub const MAX_TILES: usize = 2048;
pub const MAX_INC: usize = 8192;
pub const MAX_WEAK_DIR: usize = 70000;

pub const MAX_NODES: usize = 4800;
pub const MAX_EDGES: usize = 44000;
pub const MAX_SIDE: usize = 2048;
pub const MAX_LENT: usize = 8192;
pub const MAX_CAND: usize = 8192;
pub const MAX_STACK: usize = 2048;
pub const MAX_SMEM_ORDER: usize = 38400;
pub const EDGE_WORDS: usize = (MAX_EDGES + 31) / 32;
pub const NODE_WORDS: usize = (MAX_NODES + 31) / 32;
pub const CAND_WORDS: usize = (MAX_CAND + 31) / 32;

const SMEM_GEN_BYTES: u32 = 16512; // 16.1 KB -> 6 blocks/SM (768 concurrent salts!) on RTX 4090 sm_89!
const SMEM_GEN_MAX_BYTES: u32 = 100352;
const SMEM_BM2_BYTES: u32 = (MAX_N * 4 + MAX_TILES * 4) as u32;     // 14400 + 8192  = 22592 B -> 4 blocks/SM on sm_89!
const SMEM_BM4_BYTES: u32 = (MAX_N * 2 * 4 + MAX_TILES * 2 * 4) as u32; // 28800 + 16384 = 45184 B
const SMEM_BM8_BYTES: u32 = (MAX_N * 4 * 4 + MAX_TILES * 4 * 4) as u32; // 57600 + 32768 = 90368 B
const SCRATCH_1TPS_U32: usize = 25600;
const MAX_1TPS_WORKERS: usize = 64; // Tiny 6 MB allocation (v4.2 6-blk/SM is 1.8x faster than 1tps!)

const GPU_FEEDER_V2_CUDA_SRC: &str = r#"
// ==================== Zero-header inline PTX FP16x2 SIMD ====================
__device__ __forceinline__ unsigned int pack_h2(float a, float b) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float2 unpack_h2(unsigned int r) {
    unsigned short lo = (unsigned short)(r & 0xFFFFu);
    unsigned short hi = (unsigned short)(r >> 16);
    float a, b;
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
    return make_float2(a, b);
}
__device__ __forceinline__ unsigned int add_h2(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ unsigned int mul_h2(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// ==================== Exact ChaCha8Rng matching ising-generators 0.0.15 ====================
struct ChaCha8 {
    unsigned int key[8];
    unsigned long long counter;
    unsigned int blk[16];
    int idx;
};

__device__ __forceinline__ unsigned int rotl32(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}

#define QR(s, a, b, c, d) \
    s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32(s[d], 16); \
    s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32(s[b], 12); \
    s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32(s[d], 8);  \
    s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32(s[b], 7);

__device__ __forceinline__ void chacha8_init(ChaCha8& r, const unsigned int* k8) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.key[i] = k8[i];
    r.counter = 0ULL;
    r.idx = 16;
}

__device__ void chacha8_regen(ChaCha8& r) {
    unsigned int s[16];
    s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
    #pragma unroll
    for (int i = 0; i < 8; i++) s[4 + i] = r.key[i];
    s[12] = (unsigned int)(r.counter & 0xFFFFFFFFULL);
    s[13] = (unsigned int)(r.counter >> 32);
    s[14] = 0u;
    s[15] = 0u;
    unsigned int st[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) st[i] = s[i];
    #pragma unroll
    for (int round = 0; round < 4; round++) {
        QR(s, 0, 4, 8, 12);  QR(s, 1, 5, 9, 13);  QR(s, 2, 6, 10, 14); QR(s, 3, 7, 11, 15);
        QR(s, 0, 5, 10, 15); QR(s, 1, 6, 11, 12); QR(s, 2, 7, 8, 13);  QR(s, 3, 4, 9, 14);
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) r.blk[i] = s[i] + st[i];
    r.counter += 1ULL;
    r.idx = 0;
}

__device__ __forceinline__ unsigned int chacha8_next(ChaCha8& r) {
    if (r.idx >= 16) chacha8_regen(r);
    return r.blk[r.idx++];
}

__device__ __forceinline__ unsigned int fast_uniform(
    ChaCha8& r, unsigned int n,
    const unsigned long long* __restrict__ div_limit,
    const unsigned long long* __restrict__ div_inv
) {
    unsigned long long lim = div_limit[n];
    unsigned long long inv = div_inv[n];
    unsigned long long n64 = (unsigned long long)n;
    while (true) {
        unsigned long long x = (unsigned long long)chacha8_next(r);
        if (x < lim) {
            unsigned long long q = __umul64hi(x, inv);
            return (unsigned int)(x - q * n64);
        }
    }
}

__device__ void fast_shuffle_u16(
    ChaCha8& r, unsigned short* arr, unsigned int len,
    const unsigned long long* __restrict__ div_limit,
    const unsigned long long* __restrict__ div_inv
) {
    if (len <= 1u) return;
    for (unsigned int i = len - 1u; i >= 1u; i--) {
        unsigned int j = fast_uniform(r, i + 1u, div_limit, div_inv);
        unsigned short tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

__device__ __forceinline__ void chacha8_block_at(
    const unsigned int* __restrict__ k8,
    unsigned long long counter,
    unsigned int out[16]
) {
    unsigned int s[16];
    s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
    #pragma unroll
    for (int i = 0; i < 8; i++) s[4 + i] = k8[i];
    s[12] = (unsigned int)(counter & 0xFFFFFFFFULL);
    s[13] = (unsigned int)(counter >> 32);
    s[14] = 0u;
    s[15] = 0u;
    unsigned int st[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) st[i] = s[i];
    #pragma unroll
    for (int round = 0; round < 4; round++) {
        QR(s, 0, 4, 8, 12);  QR(s, 1, 5, 9, 13);  QR(s, 2, 6, 10, 14); QR(s, 3, 7, 11, 15);
        QR(s, 0, 5, 10, 15); QR(s, 1, 6, 11, 12); QR(s, 2, 7, 8, 13);  QR(s, 3, 4, 9, 14);
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = s[i] + st[i];
}

__device__ __forceinline__ unsigned int valid_bitmask_32(int w, int total) {
    int rem = total - (w << 5);
    return (rem >= 32) ? 0xFFFFFFFFu : ((rem > 0) ? ((1u << rem) - 1u) : 0u);
}

// ==================== KERNEL 1 (v4.0): 2-Blocks/SM Compact-SMEM + 72MB-L2 FastDimer ====================
// Dynamic SMEM <= 47.4 KB (2 blocks/SM on RTX 4090 -> 2x parallel serial DFS/shuffle throughput!)
// `d_order_all` (~58 KB/block, 256 active blocks = 14.8 MB << 72 MB L2) stays 100% L2 resident!
extern "C" __global__ void gpu_generate_dimer_batch(
    int batch_size,
    // Fixed topology arrays (read-only, shared across all SMs -> 100% L2 resident)
    int nodes, int m_edges, int n_cand, int n_fixed, int n_left, int n_right, int n_lent,
    int pairs, int weak_milli,
    const unsigned int* __restrict__ edges_uv,         // [m_edges] packed u | (v << 16)
    const unsigned short* __restrict__ cand_nodes,     // [n_cand * 4]
    const unsigned short* __restrict__ cand_eids,      // [n_cand * 6]
    const unsigned short* __restrict__ fixed_cids,     // [n_fixed]
    const unsigned short* __restrict__ left_off,       // [n_left + 1]
    const unsigned short* __restrict__ left_ent_r,     // [n_lent]
    const unsigned short* __restrict__ left_ent_cid,   // [n_lent]
    const unsigned long long* __restrict__ div_limit,
    const unsigned long long* __restrict__ div_inv,
    // Per-salt inputs
    const unsigned int* __restrict__ keys_all,         // [B * 40] (5 keys x 8 u32)
    // L2-resident per-salt order scratch [B * max_smem_order]
    unsigned short* __restrict__ order_all,
    // Outputs written directly into GPU BM8 arena (write-only ~156 KB/salt!)
    int*            __restrict__ out_n,
    int*            __restrict__ out_nt,
    float*          __restrict__ out_base,
    unsigned short* __restrict__ out_tiles_all,        // [B * MAX_TILES * 4]
    signed char*    __restrict__ out_gauges_all,       // [B * MAX_TILES * 4]
    int*            __restrict__ out_woff_all,         // [B * (MAX_N + 1)]
    unsigned short* __restrict__ out_wnb_pk_all,       // [B * MAX_WEAK_DIR]
    // Constants
    int max_n, int max_tiles, int max_weak_dir, int max_smem_order
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    int tid = threadIdx.x; // 0..127

    extern __shared__ unsigned char smem_raw[];

    // Persistent across all 3 phases: [0 .. 1248)
    unsigned int* s_sel_bits  = (unsigned int*)(smem_raw + 0);       // 256 u32 = 1024 B
    unsigned int* s_keys      = (unsigned int*)(smem_raw + 1024);    // 40 u32  = 160 B
    unsigned long long* s_u64 = (unsigned long long*)(smem_raw + 1184); // 4 u64 = 32 B
    int* s_meta               = (int*)(smem_raw + 1216);             // 8 i32   = 32 B
    if (tid < 40) s_keys[tid] = keys_all[(size_t)b * 40 + tid];

    unsigned short* s_order    = order_all + (size_t)b * max_smem_order;
    unsigned short* out_tiles  = out_tiles_all + (size_t)b * max_tiles * 4;
    signed char*    out_gauges = out_gauges_all + (size_t)b * max_tiles * 4;
    int*            out_woff   = out_woff_all + (size_t)b * (max_n + 1);
    unsigned short* out_wnb_pk = out_wnb_pk_all + (size_t)b * max_weak_dir;

    // ==================== PHASE 1: Ultra-Compact SMEM (<16.1 KB -> 6 blocks/SM!) ====================
    // Read `left_off` directly from read-only constant L1/L2 cache (saves 3.8 KB SMEM!)
    #define A16(x) (((x) + 15) & ~15)
    int off = 1248;
    unsigned short* s_left_ord  = (unsigned short*)(smem_raw + off); off += A16(n_left * 2);
    unsigned short* s_right_ord = (unsigned short*)(smem_raw + off); off += A16(n_right * 2);
    unsigned short* s_rank      = (unsigned short*)(smem_raw + off); off += A16(n_right * 2);
    unsigned short* s_vis       = (unsigned short*)(smem_raw + off); off += A16(n_right * 2);
    unsigned int*   s_p1_flag   = (unsigned int*)(smem_raw + off);
    #undef A16
    // Store adj_r, adj_cid, stack_l, stack_p, p1_jbuf inside L2-resident s_order (dead during Phase 1!)
    unsigned short* s_adj_r     = s_order + 0;
    unsigned short* s_adj_cid   = s_adj_r + n_lent;
    unsigned short* s_stack_l   = s_adj_cid + n_lent;
    unsigned short* s_stack_p   = s_stack_l + n_left;
    unsigned short* s_p1_jbuf   = s_stack_p + n_left;

    // 1a. Parallel init left_ord, right_ord, adj_r, adj_cid
    for (int i = tid; i < n_left; i += 128)  s_left_ord[i] = (unsigned short)i;
    for (int i = tid; i < n_right; i += 128) s_right_ord[i] = (unsigned short)i;
    for (int i = tid; i < n_lent; i += 128) {
        s_adj_r[i] = left_ent_r[i];
        s_adj_cid[i] = left_ent_cid[i];
    }
    if (tid == 0) s_p1_flag[0] = 0u;
    __syncthreads();

    // 1b. Shuffle left_ord (parallel ChaCha8 + div_inv, SMEM swap)
    if (n_left > 1) {
        int swaps = n_left - 1;
        int n_blks = (swaps + 15) >> 4;
        for (int blk = tid; blk < n_blks; blk += 128) {
            unsigned int w16[16];
            chacha8_block_at(s_keys + 0, (unsigned long long)blk, w16);
            #pragma unroll
            for (int m = 0; m < 16; m++) {
                int k = (blk << 4) + m;
                if (k < swaps) {
                    unsigned int n = (unsigned int)(n_left - k);
                    unsigned long long x = (unsigned long long)w16[m];
                    if (x >= div_limit[n]) {
                        atomicOr(&s_p1_flag[0], 1u);
                    } else {
                        unsigned long long q = __umul64hi(x, div_inv[n]);
                        s_p1_jbuf[k] = (unsigned short)(x - q * (unsigned long long)n);
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (s_p1_flag[0] == 0u) {
                for (int k = 0; k < swaps; k++) {
                    int i = swaps - k;
                    int j = (int)s_p1_jbuf[k];
                    unsigned short tmp = s_left_ord[i];
                    s_left_ord[i] = s_left_ord[j];
                    s_left_ord[j] = tmp;
                }
            } else {
                ChaCha8 rng;
                chacha8_init(rng, s_keys + 0);
                for (unsigned int i = (unsigned int)swaps; i >= 1u; i--) {
                    unsigned int j = fast_uniform(rng, i + 1u, div_limit, div_inv);
                    unsigned short tmp = s_left_ord[i];
                    s_left_ord[i] = s_left_ord[j];
                    s_left_ord[j] = tmp;
                }
            }
            s_p1_flag[0] = 0u;
        }
        __syncthreads();
    }

    // 1c. Shuffle right_ord (parallel ChaCha8 + div_inv, SMEM swap)
    if (n_right > 1) {
        int swaps = n_right - 1;
        int n_blks = (swaps + 15) >> 4;
        for (int blk = tid; blk < n_blks; blk += 128) {
            unsigned int w16[16];
            chacha8_block_at(s_keys + 8, (unsigned long long)blk, w16);
            #pragma unroll
            for (int m = 0; m < 16; m++) {
                int k = (blk << 4) + m;
                if (k < swaps) {
                    unsigned int n = (unsigned int)(n_right - k);
                    unsigned long long x = (unsigned long long)w16[m];
                    if (x >= div_limit[n]) {
                        atomicOr(&s_p1_flag[0], 1u);
                    } else {
                        unsigned long long q = __umul64hi(x, div_inv[n]);
                        s_p1_jbuf[k] = (unsigned short)(x - q * (unsigned long long)n);
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (s_p1_flag[0] == 0u) {
                for (int k = 0; k < swaps; k++) {
                    int i = swaps - k;
                    int j = (int)s_p1_jbuf[k];
                    unsigned short tmp = s_right_ord[i];
                    s_right_ord[i] = s_right_ord[j];
                    s_right_ord[j] = tmp;
                }
            } else {
                ChaCha8 rng;
                chacha8_init(rng, s_keys + 8);
                for (unsigned int i = (unsigned int)swaps; i >= 1u; i--) {
                    unsigned int j = fast_uniform(rng, i + 1u, div_limit, div_inv);
                    unsigned short tmp = s_right_ord[i];
                    s_right_ord[i] = s_right_ord[j];
                    s_right_ord[j] = tmp;
                }
            }
        }
        __syncthreads();
    }

    // 1d. Invert right_ord -> s_rank (128-way parallel)
    for (int pos = tid; pos < n_right; pos += 128) {
        s_rank[s_right_ord[pos]] = (unsigned short)pos;
    }
    __syncthreads();

    // 2. Insertion-sort each left node's adjacency list by s_rank (128-way parallel!)
    for (int l = tid; l < n_left; l += 128) {
        int s = (int)left_off[l];
        int e = (int)left_off[l + 1];
        for (int i = s + 1; i < e; i++) {
            unsigned short kr_node = s_adj_r[i];
            unsigned short kcid = s_adj_cid[i];
            unsigned short kr = s_rank[kr_node];
            int j = i;
            while (j > s && s_rank[s_adj_r[j - 1]] > kr) {
                s_adj_r[j] = s_adj_r[j - 1];
                s_adj_cid[j] = s_adj_cid[j - 1];
                j--;
            }
            s_adj_r[j] = kr_node;
            s_adj_cid[j] = kcid;
        }
    }
    __syncthreads();

    // 3. Exact iterative DFS augmenting-path matching
    unsigned short* s_matched     = s_right_ord; // safe: s_right_ord dead
    unsigned short* s_matched_cid = s_rank;      // safe: s_rank dead
    for (int r = tid; r < n_right; r += 128) {
        s_matched[r] = 0xFFFFu;
        s_vis[r] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        unsigned short cur_vis = 0u;
        for (int li = 0; li < n_left; li++) {
            unsigned short top_l = s_left_ord[li];
            cur_vis++;
            int sp = 1;
            int top_s = (int)left_off[top_l];
            int top_e = (int)left_off[top_l + 1];
            int top_pos = 0;
            while (true) {
                if (top_s + top_pos >= top_e) {
                    sp--;
                    if (sp == 0) break;
                    top_l = s_stack_l[sp - 1];
                    top_pos = (int)s_stack_p[sp - 1];
                    top_s = (int)left_off[top_l];
                    top_e = (int)left_off[top_l + 1];
                    continue;
                }
                int idx = top_s + top_pos;
                int right = (int)s_adj_r[idx];
                top_pos++;
                if (s_vis[right] == cur_vis) continue;
                s_vis[right] = cur_vis;
                unsigned short owner = s_matched[right];
                if (owner != 0xFFFFu) {
                    s_stack_l[sp - 1] = top_l;
                    s_stack_p[sp - 1] = (unsigned short)top_pos;
                    top_l = owner;
                    top_s = (int)left_off[owner];
                    top_e = (int)left_off[owner + 1];
                    top_pos = 0;
                    sp++;
                    continue;
                }
                s_matched[right] = top_l;
                s_matched_cid[right] = s_adj_cid[idx];
                for (int f = 0; f < sp - 1; f++) {
                    int fl = (int)s_stack_l[f];
                    int fidx = (int)left_off[fl] + (int)s_stack_p[f] - 1;
                    int cr = (int)s_adj_r[fidx];
                    s_matched[cr] = (unsigned short)fl;
                    s_matched_cid[cr] = s_adj_cid[fidx];
                }
                break;
            }
        }
    }
    __syncthreads();

    // 4. Collect selected candidate tile bitmap into persistent s_sel_bits (128-way parallel)
    int n_cwords = (n_cand + 31) >> 5;
    for (int w = tid; w < 256; w += 128) s_sel_bits[w] = 0u;
    __syncthreads();
    for (int i = tid; i < n_fixed; i += 128) {
        unsigned int cid = (unsigned int)fixed_cids[i];
        atomicOr(&s_sel_bits[cid >> 5], 1u << (cid & 31u));
    }
    for (int r = tid; r < n_right; r += 128) {
        if (s_matched[r] != 0xFFFFu) {
            unsigned int cid = (unsigned int)s_matched_cid[r];
            atomicOr(&s_sel_bits[cid >> 5], 1u << (cid & 31u));
        }
    }
    __syncthreads();

    // ==================== PHASE 2: Ultra-Compact SMEM (<16.1 KB -> 6 blocks/SM!) ====================
    // Layout [1248 .. 17504) -> s_partner (9.6 KB) lives in L2-hot out_tiles (dead until Step 9!)
    unsigned int*   s_strong_bits = (unsigned int*)(smem_raw + 1248);   // 1376 u32 (5504 B -> 6752)
    unsigned int*   s_weak_neg    = (unsigned int*)(smem_raw + 6752);   // 1376 u32 (5504 B -> 12256)
    unsigned int*   s_used_bits   = (unsigned int*)(smem_raw + 12256);  // 152 u32  (608 B  -> 12864)
    unsigned short* s_partner     = out_tiles;                          // 4800 u16 in L2 (out_tiles is 8192 u16, dead until Step 9!)
    unsigned short* s_p2_cids     = out_tiles;                          // 2048 u16 in L2 (dead before Step 7b)
    unsigned short* s_p2_jbuf     = out_tiles;                          // 2048 u16 in L2 (dead before Step 7b)
    int*            s_p2_blk      = (int*)(smem_raw + 12864);           // 128 i32  (512 B  -> 13376)
    unsigned int*   s_p2_flag     = (unsigned int*)(smem_raw + 13376);  // 4 u32    (16 B   -> 13392)

    int n_ewords = (m_edges + 31) >> 5;
    for (int w = tid; w < n_ewords; w += 128) {
        s_strong_bits[w] = 0u;
        s_weak_neg[w] = 0u;
    }

    // 5a. Parallel compact s_sel_bits (2 words/thread) into s_p2_cids[0..n_tiles]
    int cw0 = tid * 2;
    int cw1 = cw0 + 1;
    unsigned int cb0 = (cw0 < n_cwords) ? (s_sel_bits[cw0] & valid_bitmask_32(cw0, n_cand)) : 0u;
    unsigned int cb1 = (cw1 < n_cwords) ? (s_sel_bits[cw1] & valid_bitmask_32(cw1, n_cand)) : 0u;
    s_p2_blk[tid] = __popc(cb0) + __popc(cb1);
    __syncthreads();
    for (int off = 1; off < 128; off <<= 1) {
        int v = (tid >= off) ? s_p2_blk[tid - off] : 0;
        __syncthreads();
        s_p2_blk[tid] += v;
        __syncthreads();
    }
    int n_tiles = s_p2_blk[127];
    if (n_tiles > max_tiles) {
        if (tid == 0) out_n[b] = -1;
        return;
    }
    {
        int pos = (tid > 0) ? s_p2_blk[tid - 1] : 0;
        unsigned int m0 = cb0;
        while (m0 != 0u) {
            int bit = __ffs(m0) - 1; m0 &= (m0 - 1u);
            s_p2_cids[pos++] = (unsigned short)((cw0 << 5) + bit);
        }
        unsigned int m1 = cb1;
        while (m1 != 0u) {
            int bit = __ffs(m1) - 1; m1 &= (m1 - 1u);
            s_p2_cids[pos++] = (unsigned short)((cw1 << 5) + bit);
        }
    }
    __syncthreads();

    // 5b. Parallel gauges (keys + 16) + strong_bits marking (128-way parallel!)
    int n_gwords = n_tiles * 4;
    int n_gblks = (n_gwords + 15) >> 4;
    for (int blk = tid; blk < n_gblks; blk += 128) {
        unsigned int w16[16];
        chacha8_block_at(s_keys + 16, (unsigned long long)blk, w16);
        int base = blk << 4;
        #pragma unroll
        for (int m = 0; m < 16; m++) {
            if (base + m < n_gwords) {
                out_gauges[base + m] = (w16[m] & 1u) == 0u ? (signed char)1 : (signed char)-1;
            }
        }
    }
    for (int ti = tid; ti < n_tiles; ti += 128) {
        int ebase = (int)s_p2_cids[ti] * 6;
        #pragma unroll
        for (int k = 0; k < 6; k++) {
            unsigned int eid = (unsigned int)cand_eids[ebase + k];
            atomicOr(&s_strong_bits[eid >> 5], 1u << (eid & 31u));
        }
    }
    __syncthreads();

    // 6a. Parallel non-strong edge compaction into s_order[0..order_len]
    int w_per = (n_ewords + 127) >> 7;
    int w_lo = tid * w_per;
    int w_hi = w_lo + w_per;
    if (w_hi > n_ewords) w_hi = n_ewords;
    int my_ecnt = 0;
    for (int w = w_lo; w < w_hi; w++) {
        my_ecnt += __popc((~s_strong_bits[w]) & valid_bitmask_32(w, m_edges));
    }
    s_p2_blk[tid] = my_ecnt;
    __syncthreads();
    for (int off = 1; off < 128; off <<= 1) {
        int v = (tid >= off) ? s_p2_blk[tid - off] : 0;
        __syncthreads();
        s_p2_blk[tid] += v;
        __syncthreads();
    }
    int order_len = s_p2_blk[127];
    if (order_len > max_smem_order) {
        if (tid == 0) out_n[b] = -1;
        return;
    }
    {
        int pos = (tid > 0) ? s_p2_blk[tid - 1] : 0;
        for (int w = w_lo; w < w_hi; w++) {
            unsigned int bits = (~s_strong_bits[w]) & valid_bitmask_32(w, m_edges);
            while (bits != 0u) {
                int bit = __ffs(bits) - 1; bits &= (bits - 1u);
                s_order[pos++] = (unsigned short)((w << 5) + bit);
            }
        }
    }
    __syncthreads();

    // 6b. Parallel weak signs (keys + 24)
    int n_wblks = (order_len + 15) >> 4;
    for (int blk = tid; blk < n_wblks; blk += 128) {
        unsigned int w16[16];
        chacha8_block_at(s_keys + 24, (unsigned long long)blk, w16);
        int base = blk << 4;
        #pragma unroll
        for (int m = 0; m < 16; m++) {
            int k = base + m;
            if (k < order_len && (w16[m] & 1u) != 0u) {
                unsigned int eid = (unsigned int)s_order[k];
                atomicOr(&s_weak_neg[eid >> 5], 1u << (eid & 31u));
            }
        }
    }
    if (tid == 0) {
        s_meta[2] = 0;       // k_cur
        s_u64[0] = 0ULL;     // rng_word_pos (always 16-aligned at top of chunk loop!)
    }
    __syncthreads();

    // 7a. Chunked 128-thread parallel j[k] precomputation + pure SMEM Fisher-Yates!
    if (order_len > 1) {
        int total_swaps = order_len - 1;
        while (true) {
            int k_cur = s_meta[2];
            if (k_cur >= total_swaps) break;
            unsigned long long rng_pos = s_u64[0];
            int cnt = total_swaps - k_cur;
            if (cnt > 2048) cnt = 2048;

            if (tid == 0) s_p2_flag[0] = 0u;
            __syncthreads();

            int lk_base = tid << 4;
            if (lk_base < cnt) {
                unsigned int w16[16];
                chacha8_block_at(s_keys + 32, (rng_pos >> 4) + (unsigned long long)tid, w16);
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    int lk = lk_base + m;
                    if (lk < cnt) {
                        unsigned int n = (unsigned int)(order_len - (k_cur + lk));
                        unsigned long long x = (unsigned long long)w16[m];
                        if (x >= div_limit[n]) {
                            atomicOr(&s_p2_flag[0], 1u);
                        } else {
                            unsigned long long q = __umul64hi(x, div_inv[n]);
                            s_p2_jbuf[lk] = (unsigned short)(x - q * (unsigned long long)n);
                        }
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                if (s_p2_flag[0] == 0u) {
                    int idx = order_len - 1 - k_cur;
                    for (int lk = 0; lk < cnt; lk++, idx--) {
                        int j = (int)s_p2_jbuf[lk];
                        unsigned short tmp = s_order[idx];
                        s_order[idx] = s_order[j];
                        s_order[j] = tmp;
                    }
                    s_meta[2] = k_cur + cnt;
                    s_u64[0] = rng_pos + (unsigned long long)cnt;
                } else {
                    // Exact fallback on <0.05% rejection chunk, re-aligning rng_pos to multiple of 16!
                    ChaCha8 rng;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) rng.key[i] = s_keys[32 + i];
                    rng.counter = rng_pos >> 4;
                    rng.idx = 16;
                    int k = k_cur;
                    int k_end = k_cur + cnt;
                    unsigned long long cur_pos = rng_pos;
                    while (k < k_end || (k < total_swaps && (cur_pos & 15ULL) != 0ULL)) {
                        unsigned int idx = (unsigned int)(order_len - 1 - k);
                        unsigned int j = fast_uniform(rng, idx + 1u, div_limit, div_inv);
                        unsigned short tmp = s_order[idx];
                        s_order[idx] = s_order[j];
                        s_order[j] = tmp;
                        k++;
                        cur_pos = ((rng.counter - 1ULL) << 4) + (unsigned long long)rng.idx;
                    }
                    s_meta[2] = k;
                    s_u64[0] = cur_pos;
                }
            }
            __syncthreads();
        }
    }

    // 7b. Greedy dimer pairs (100% SMEM + single packed u32 edges_uv load!)
    int n_nwords = (nodes + 31) >> 5;
    for (int w = tid; w < n_nwords; w += 128) s_used_bits[w] = 0u;
    __syncthreads();

    if (tid == 0) {
        int n_pairs_found = 0;
        for (int i = 0; i < order_len; i++) {
            if (n_pairs_found >= pairs) break;
            int eid = (int)s_order[i];
            unsigned int uv = edges_uv[eid];
            unsigned int u = uv & 0xFFFFu;
            unsigned int v = uv >> 16;
            unsigned int um = 1u << (u & 31u);
            unsigned int vm = 1u << (v & 31u);
            if ((s_used_bits[u >> 5] & um) || (s_used_bits[v >> 5] & vm)) continue;
            s_used_bits[u >> 5] |= um;
            s_used_bits[v >> 5] |= vm;
            s_partner[u] = (unsigned short)v;
            s_partner[v] = (unsigned short)u;
            s_strong_bits[eid >> 5] |= (1u << (eid & 31));
            n_pairs_found++;
        }
        s_meta[0] = n_pairs_found;
    }
    __syncthreads();
    if (s_meta[0] < pairs) {
        if (tid == 0) out_n[b] = -1;
        return;
    }

    // ==================== PHASE 3: Contract, Tiles & 128-Way Weak CSR (<16.1 KB SMEM -> 6 blocks/SM!) ====================
    // s_order is 100% DEAD after Step 7b -> reuse as L2-resident s_rep (9.6 KB) + L2-resident s_woff (15.4 KB)!
    unsigned short* s_rep = s_order;

    // 8. Contract pairs -> s_rep (L2-resident!)
    for (int u = tid; u < nodes; u += 128) s_rep[u] = 0xFFFFu;
    __syncthreads();

    if (tid == 0) {
        int n_super = 0;
        for (int u = 0; u < nodes; u++) {
            if (s_rep[u] != 0xFFFFu) continue;
            unsigned short id = (unsigned short)n_super++;
            s_rep[u] = id;
            if ((s_used_bits[u >> 5] >> (u & 31)) & 1u) {
                s_rep[s_partner[u]] = id;
            }
        }
        s_meta[1] = n_super;
    }
    __syncthreads();
    int n_super = s_meta[1];
    if (n_super > max_n || n_super >= 32768) {
        if (tid == 0) out_n[b] = -1;
        return;
    }

    // s_used_bits & s_partner (out_tiles) are now 100% DEAD!
    // Put s_p3_cids in L2 right after s_rep (offset 4800 u16 -> 6848 u16), s_woff at offset 8192 u16 (aligned)!
    unsigned short* s_p3_cids  = s_order + 4800;                      // 2048 u16 in L2
    int*            s_p3_blk9  = (int*)(smem_raw + 12256);            // 128 i32  (512 B -> 12768)

    // 9. Fill contracted tiles (128-way parallel!)
    s_p3_blk9[tid] = __popc(cb0) + __popc(cb1);
    __syncthreads();
    for (int off = 1; off < 128; off <<= 1) {
        int v = (tid >= off) ? s_p3_blk9[tid - off] : 0;
        __syncthreads();
        s_p3_blk9[tid] += v;
        __syncthreads();
    }
    {
        int pos = (tid > 0) ? s_p3_blk9[tid - 1] : 0;
        unsigned int m0 = cb0;
        while (m0 != 0u) {
            int bit = __ffs(m0) - 1; m0 &= (m0 - 1u);
            s_p3_cids[pos++] = (unsigned short)((cw0 << 5) + bit);
        }
        unsigned int m1 = cb1;
        while (m1 != 0u) {
            int bit = __ffs(m1) - 1; m1 &= (m1 - 1u);
            s_p3_cids[pos++] = (unsigned short)((cw1 << 5) + bit);
        }
    }
    __syncthreads();
    for (int ti = tid; ti < n_tiles; ti += 128) {
        int nb = (int)s_p3_cids[ti] * 4;
        int ob = ti * 4;
        out_tiles[ob + 0] = s_rep[cand_nodes[nb + 0]];
        out_tiles[ob + 1] = s_rep[cand_nodes[nb + 1]];
        out_tiles[ob + 2] = s_rep[cand_nodes[nb + 2]];
        out_tiles[ob + 3] = s_rep[cand_nodes[nb + 3]];
    }
    __syncthreads();

    // s_woff (3840 i32 = 15,360 B) lives in L2-resident s_order + 8192 (Ada L2 hardware atomics!)
    // Peak SMEM in Phase 3 = 12256 + 512 + 1024 = 13,792 B (< 16.1 KB -> 6 blocks/SM!)
    int*       s_woff   = (int*)(s_order + 8192);         // 3840 i32 in L2 (ends at 15872 u16 << 38400!)
    int*       s_p3_blk = (int*)(smem_raw + 12256);       // 128 i32  (512 B  -> 12768)
    long long* s_sw     = (long long*)(smem_raw + 12768); // 128 i64  (1024 B -> 13792)
    for (int i = tid; i < 3840; i += 128) s_woff[i] = 0;
    __syncthreads();

    // 10. 128-way parallel Weak CSR in SMEM (sweep non-strong edges by word!)
    long long local_sw = 0LL;
    for (int w = tid; w < n_ewords; w += 128) {
        unsigned int wmask = (~s_strong_bits[w]) & valid_bitmask_32(w, m_edges);
        unsigned int nmask = s_weak_neg[w];
        while (wmask != 0u) {
            int bit = __ffs(wmask) - 1; wmask &= (wmask - 1u);
            int eid = (w << 5) + bit;
            unsigned int uv = edges_uv[eid];
            int a = (int)s_rep[uv & 0xFFFFu];
            int c = (int)s_rep[uv >> 16];
            if (a == c) {
                local_sw += ((nmask >> bit) & 1u) ? -(long long)weak_milli : (long long)weak_milli;
            } else {
                atomicAdd(&s_woff[a + 1], 1);
                atomicAdd(&s_woff[c + 1], 1);
            }
        }
    }
    s_sw[tid] = local_sw;
    __syncthreads();

    // Reduce self_weak (128 -> 1)
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) s_sw[tid] += s_sw[tid + s];
        __syncthreads();
    }

    // 128-thread parallel prefix sum of s_woff[0..3840) (30 items/thread) + coalesced out_woff store!
    {
        int base = tid * 30;
        int lsum = 0;
        #pragma unroll
        for (int k = 0; k < 30; k++) {
            lsum += s_woff[base + k];
            s_woff[base + k] = lsum;
        }
        s_p3_blk[tid] = lsum;
        __syncthreads();
        for (int off = 1; off < 128; off <<= 1) {
            int v = (tid >= off) ? s_p3_blk[tid - off] : 0;
            __syncthreads();
            s_p3_blk[tid] += v;
            __syncthreads();
        }
        int carry = (tid > 0) ? s_p3_blk[tid - 1] : 0;
        #pragma unroll
        for (int k = 0; k < 30; k++) {
            int idx = base + k;
            int val = s_woff[idx] + carry;
            s_woff[idx] = val;
            if (idx <= n_super) out_woff[idx] = val;
        }
    }
    __syncthreads();

    int tot_wdir = s_woff[n_super];
    if (tot_wdir > max_weak_dir) {
        if (tid == 0) out_n[b] = -1;
        return;
    }

    // Pass 2: 128-way parallel scatter into out_wnb_pk using SMEM s_woff atomic counters!
    for (int w = tid; w < n_ewords; w += 128) {
        unsigned int wmask = (~s_strong_bits[w]) & valid_bitmask_32(w, m_edges);
        unsigned int nmask = s_weak_neg[w];
        while (wmask != 0u) {
            int bit = __ffs(wmask) - 1; wmask &= (wmask - 1u);
            int eid = (w << 5) + bit;
            unsigned int uv = edges_uv[eid];
            unsigned short a = s_rep[uv & 0xFFFFu];
            unsigned short c = s_rep[uv >> 16];
            if (a == c) continue;
            unsigned short neg_bit = ((nmask >> bit) & 1u) ? 0x8000u : 0u;
            int pa = atomicAdd(&s_woff[a], 1);
            out_wnb_pk[pa] = c | neg_bit;
            int pc = atomicAdd(&s_woff[c], 1);
            out_wnb_pk[pc] = a | neg_bit;
        }
    }

    if (tid == 0) {
        out_n[b] = n_super;
        out_nt[b] = n_tiles;
        out_base[b] = (float)(-(double)pairs + (double)s_sw[0] / 1000.0);
    }
}

__device__ void fast_shuffle_u16_soa(
    ChaCha8& r, unsigned short* arr, unsigned int len, int lane,
    const unsigned long long* __restrict__ div_limit,
    const unsigned long long* __restrict__ div_inv
) {
    if (len <= 1u) return;
    for (unsigned int i = len - 1u; i >= 1u; i--) {
        unsigned int j = fast_uniform(r, i + 1u, div_limit, div_inv);
        size_t pi = ((size_t)i << 5) + (size_t)lane;
        size_t pj = ((size_t)j << 5) + (size_t)lane;
        unsigned short tmp = arr[pi];
        arr[pi] = arr[pj];
        arr[pj] = tmp;
    }
}

// ==================== KERNEL 1-v5.1: Warp-Coalesced SoA 1-Thread-Per-Salt FastDimer ====================
// Every warp (32 lanes) interleaves its scratch arrays `arr[(idx << 5) + lane]` for 100% coalesced 128B L1/L2 lines!
// Launches across ALL 128 SMs (4096..8192 threads = 1..2 warps/SM), ZERO SMEM, ZERO barriers, ZERO atomics!
extern "C" __global__ void gpu_dimer_1tps(
    int batch_size,
    int nodes, int m_edges, int n_cand, int n_fixed, int n_left, int n_right, int n_lent,
    int pairs, int weak_milli,
    const unsigned int* __restrict__ edges_uv,
    const unsigned short* __restrict__ cand_nodes,
    const unsigned short* __restrict__ cand_eids,
    const unsigned short* __restrict__ fixed_cids,
    const unsigned short* __restrict__ left_off,
    const unsigned short* __restrict__ left_ent_r,
    const unsigned short* __restrict__ left_ent_cid,
    const unsigned long long* __restrict__ div_limit,
    const unsigned long long* __restrict__ div_inv,
    const unsigned int* __restrict__ keys_all,
    unsigned int* __restrict__ scratch_all,        // [N_WARPS * 32 * 25600 u32] warp-interleaved SoA!
    int*            __restrict__ out_n,
    int*            __restrict__ out_nt,
    float*          __restrict__ out_base,
    unsigned short* __restrict__ out_tiles_all,
    signed char*    __restrict__ out_gauges_all,
    int*            __restrict__ out_woff_all,
    unsigned short* __restrict__ out_wnb_pk_all,
    int*            __restrict__ out_toff_all,
    unsigned short* __restrict__ out_inc_tid_all,
    signed char*    __restrict__ out_inc_g_all,
    int max_n, int max_tiles, int max_inc, int max_weak_dir, int max_smem_order
) {
    int wid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int warp_id = wid >> 5;
    int lane = wid & 31;
    // Each warp owns 32 * 102,400 B = 3,276,800 B, partitioned into warp-interleaved SoA slices!
    unsigned char* ww = (unsigned char*)(scratch_all + (size_t)warp_id * (25600ULL * 32ULL));
    #define SI(idx) (((size_t)(idx) << 5) + (size_t)lane)

    // Non-overlapping warp-SoA partitions (each offset multiplied by 32 lanes!):
    unsigned int*   sel_bits    = (unsigned int*)(ww + 0ULL * 32ULL);
    // Phase 1
    unsigned short* left_ord    = (unsigned short*)(ww + 1024ULL * 32ULL);
    unsigned short* right_ord   = (unsigned short*)(ww + 5120ULL * 32ULL);
    unsigned short* rank_arr    = (unsigned short*)(ww + 9216ULL * 32ULL);
    unsigned short* vis         = (unsigned short*)(ww + 13312ULL * 32ULL);
    unsigned short* adj_r       = (unsigned short*)(ww + 17408ULL * 32ULL);
    unsigned short* adj_cid     = (unsigned short*)(ww + 33792ULL * 32ULL);
    unsigned short* stack_l     = (unsigned short*)(ww + 50176ULL * 32ULL);
    unsigned short* stack_p     = (unsigned short*)(ww + 54272ULL * 32ULL);
    // Phase 2
    unsigned int*   strong_bits = (unsigned int*)(ww + 1024ULL * 32ULL);
    unsigned int*   weak_neg    = (unsigned int*)(ww + 6528ULL * 32ULL);
    unsigned int*   used_bits   = (unsigned int*)(ww + 12032ULL * 32ULL);
    unsigned short* partner     = (unsigned short*)(ww + 12640ULL * 32ULL);
    unsigned short* order       = (unsigned short*)(ww + 22240ULL * 32ULL);
    // Phase 3 (reuses dead order)
    unsigned short* rep         = (unsigned short*)(ww + 22240ULL * 32ULL);
    int*            woff_tmp    = (int*)(ww + 31840ULL * 32ULL);

    int n_cwords = (n_cand + 31) >> 5;
    int n_ewords = (m_edges + 31) >> 5;
    int n_nwords = (nodes + 31) >> 5;

    for (int b = wid; b < batch_size; b += stride) {
        const unsigned int* kb = keys_all + (size_t)b * 40ULL;
        unsigned short* out_tiles   = out_tiles_all + (size_t)b * max_tiles * 4;
        signed char*    out_gauges  = out_gauges_all + (size_t)b * max_tiles * 4;
        int*            out_woff    = out_woff_all + (size_t)b * (max_n + 1);
        unsigned short* out_wnb_pk  = out_wnb_pk_all + (size_t)b * max_weak_dir;
        int*            out_toff    = out_toff_all + (size_t)b * (max_n + 1);
        unsigned short* out_inc_tid = out_inc_tid_all + (size_t)b * max_inc;
        signed char*    out_inc_g   = out_inc_g_all + (size_t)b * max_inc;

        // 1. Shuffle left_ord & right_ord (coalesced init!)
        for (int i = 0; i < n_left; i++) left_ord[SI(i)] = (unsigned short)i;
        {
            ChaCha8 rng;
            chacha8_init(rng, kb + 0);
            fast_shuffle_u16_soa(rng, left_ord, (unsigned int)n_left, lane, div_limit, div_inv);
        }
        for (int i = 0; i < n_right; i++) right_ord[SI(i)] = (unsigned short)i;
        {
            ChaCha8 rng;
            chacha8_init(rng, kb + 8);
            fast_shuffle_u16_soa(rng, right_ord, (unsigned int)n_right, lane, div_limit, div_inv);
        }
        for (int pos = 0; pos < n_right; pos++) rank_arr[SI(right_ord[SI(pos)])] = (unsigned short)pos;

        // 2. Copy & insertion-sort adjacency by rank_arr (coalesced copy!)
        for (int i = 0; i < n_lent; i++) {
            adj_r[SI(i)] = left_ent_r[i];
            adj_cid[SI(i)] = left_ent_cid[i];
        }
        for (int l = 0; l < n_left; l++) {
            int s = (int)left_off[l];
            int e = (int)left_off[l + 1];
            for (int i = s + 1; i < e; i++) {
                unsigned short kr_node = adj_r[SI(i)];
                unsigned short kcid = adj_cid[SI(i)];
                unsigned short kr = rank_arr[SI(kr_node)];
                int j = i;
                while (j > s && rank_arr[SI(adj_r[SI(j - 1)])] > kr) {
                    adj_r[SI(j)] = adj_r[SI(j - 1)];
                    adj_cid[SI(j)] = adj_cid[SI(j - 1)];
                    j--;
                }
                adj_r[SI(j)] = kr_node;
                adj_cid[SI(j)] = kcid;
            }
        }

        // 3. Kuhn DFS matching
        unsigned short* matched     = right_ord;
        unsigned short* matched_cid = rank_arr;
        for (int r = 0; r < n_right; r++) {
            matched[SI(r)] = 0xFFFFu;
            vis[SI(r)] = 0u;
        }
        unsigned short cur_vis = 0u;
        for (int li = 0; li < n_left; li++) {
            unsigned short top_l = left_ord[SI(li)];
            cur_vis++;
            int sp = 1;
            int top_s = (int)left_off[top_l];
            int top_e = (int)left_off[top_l + 1];
            int top_pos = 0;
            while (true) {
                if (top_s + top_pos >= top_e) {
                    sp--;
                    if (sp == 0) break;
                    top_l = stack_l[SI(sp - 1)];
                    top_pos = (int)stack_p[SI(sp - 1)];
                    top_s = (int)left_off[top_l];
                    top_e = (int)left_off[top_l + 1];
                    continue;
                }
                int idx = top_s + top_pos;
                int right = (int)adj_r[SI(idx)];
                top_pos++;
                if (vis[SI(right)] == cur_vis) continue;
                vis[SI(right)] = cur_vis;
                unsigned short owner = matched[SI(right)];
                if (owner != 0xFFFFu) {
                    stack_l[SI(sp - 1)] = top_l;
                    stack_p[SI(sp - 1)] = (unsigned short)top_pos;
                    top_l = owner;
                    top_s = (int)left_off[owner];
                    top_e = (int)left_off[owner + 1];
                    top_pos = 0;
                    sp++;
                    continue;
                }
                matched[SI(right)] = top_l;
                matched_cid[SI(right)] = adj_cid[SI(idx)];
                for (int f = 0; f < sp - 1; f++) {
                    int fl = (int)stack_l[SI(f)];
                    int fidx = (int)left_off[fl] + (int)stack_p[SI(f)] - 1;
                    int cr = (int)adj_r[SI(fidx)];
                    matched[SI(cr)] = (unsigned short)fl;
                    matched_cid[SI(cr)] = adj_cid[SI(fidx)];
                }
                break;
            }
        }

        // 4. Collect selected candidate tile bitmap (coalesced clear!)
        for (int w = 0; w < n_cwords; w++) sel_bits[SI(w)] = 0u;
        for (int i = 0; i < n_fixed; i++) {
            unsigned int cid = (unsigned int)fixed_cids[i];
            sel_bits[SI(cid >> 5)] |= (1u << (cid & 31u));
        }
        for (int r = 0; r < n_right; r++) {
            if (matched[SI(r)] != 0xFFFFu) {
                unsigned int cid = (unsigned int)matched_cid[SI(r)];
                sel_bits[SI(cid >> 5)] |= (1u << (cid & 31u));
            }
        }

        // 5. Gauges + strong_bits (coalesced clear!)
        for (int w = 0; w < n_ewords; w++) {
            strong_bits[SI(w)] = 0u;
            weak_neg[SI(w)] = 0u;
        }
        int n_tiles = 0;
        {
            ChaCha8 rng_g;
            chacha8_init(rng_g, kb + 16);
            for (int w = 0; w < n_cwords; w++) {
                unsigned int m = sel_bits[SI(w)] & valid_bitmask_32(w, n_cand);
                while (m != 0u) {
                    int bit = __ffs(m) - 1; m &= (m - 1u);
                    if (n_tiles >= max_tiles) { n_tiles = max_tiles + 1; break; }
                    int cid = (w << 5) + bit;
                    int ob = n_tiles * 4;
                    out_gauges[ob + 0] = (chacha8_next(rng_g) & 1u) == 0u ? (signed char)1 : (signed char)-1;
                    out_gauges[ob + 1] = (chacha8_next(rng_g) & 1u) == 0u ? (signed char)1 : (signed char)-1;
                    out_gauges[ob + 2] = (chacha8_next(rng_g) & 1u) == 0u ? (signed char)1 : (signed char)-1;
                    out_gauges[ob + 3] = (chacha8_next(rng_g) & 1u) == 0u ? (signed char)1 : (signed char)-1;
                    int ebase = cid * 6;
                    #pragma unroll
                    for (int k = 0; k < 6; k++) {
                        unsigned int eid = (unsigned int)cand_eids[ebase + k];
                        strong_bits[SI(eid >> 5)] |= (1u << (eid & 31u));
                    }
                    n_tiles++;
                }
                if (n_tiles > max_tiles) break;
            }
        }
        if (n_tiles > max_tiles) { out_n[b] = -1; continue; }

        // 6. Non-strong edge compaction + weak signs (coalesced sweep!)
        int order_len = 0;
        {
            ChaCha8 rng_w;
            chacha8_init(rng_w, kb + 24);
            for (int w = 0; w < n_ewords; w++) {
                unsigned int bits = (~strong_bits[SI(w)]) & valid_bitmask_32(w, m_edges);
                unsigned int nmask = 0u;
                while (bits != 0u) {
                    int bit = __ffs(bits) - 1; bits &= (bits - 1u);
                    if (order_len >= max_smem_order) { order_len = max_smem_order + 1; break; }
                    order[SI(order_len)] = (unsigned short)((w << 5) + bit);
                    order_len++;
                    if ((chacha8_next(rng_w) & 1u) != 0u) nmask |= (1u << bit);
                }
                weak_neg[SI(w)] = nmask;
                if (order_len > max_smem_order) break;
            }
        }
        if (order_len > max_smem_order) { out_n[b] = -1; continue; }

        // 7a. Fisher-Yates shuffle of order[0..order_len]
        {
            ChaCha8 rng_p;
            chacha8_init(rng_p, kb + 32);
            fast_shuffle_u16_soa(rng_p, order, (unsigned int)order_len, lane, div_limit, div_inv);
        }

        // 7b. Greedy dimer pairs
        for (int w = 0; w < n_nwords; w++) used_bits[SI(w)] = 0u;
        int n_pairs_found = 0;
        for (int i = 0; i < order_len; i++) {
            if (n_pairs_found >= pairs) break;
            int eid = (int)order[SI(i)];
            unsigned int uv = edges_uv[eid];
            unsigned int u = uv & 0xFFFFu;
            unsigned int v = uv >> 16;
            unsigned int um = 1u << (u & 31u);
            unsigned int vm = 1u << (v & 31u);
            if ((used_bits[SI(u >> 5)] & um) || (used_bits[SI(v >> 5)] & vm)) continue;
            used_bits[SI(u >> 5)] |= um;
            used_bits[SI(v >> 5)] |= vm;
            partner[SI(u)] = (unsigned short)v;
            partner[SI(v)] = (unsigned short)u;
            strong_bits[SI(eid >> 5)] |= (1u << (eid & 31u));
            n_pairs_found++;
        }
        if (n_pairs_found < pairs) { out_n[b] = -1; continue; }

        // 8. Contract pairs -> rep[0..nodes] (coalesced!)
        for (int u = 0; u < nodes; u++) rep[SI(u)] = 0xFFFFu;
        int n_super = 0;
        for (int u = 0; u < nodes; u++) {
            if (rep[SI(u)] != 0xFFFFu) continue;
            unsigned short id = (unsigned short)n_super++;
            rep[SI(u)] = id;
            if ((used_bits[SI(u >> 5)] >> (u & 31)) & 1u) {
                rep[SI(partner[SI(u)])] = id;
            }
        }
        if (n_super > max_n || n_super >= 32768) { out_n[b] = -1; continue; }

        // 9. Fill contracted out_tiles
        {
            int ti = 0;
            for (int w = 0; w < n_cwords; w++) {
                unsigned int m = sel_bits[SI(w)] & valid_bitmask_32(w, n_cand);
                while (m != 0u) {
                    int bit = __ffs(m) - 1; m &= (m - 1u);
                    int nb = ((w << 5) + bit) * 4;
                    int ob = ti * 4;
                    out_tiles[ob + 0] = rep[SI(cand_nodes[nb + 0])];
                    out_tiles[ob + 1] = rep[SI(cand_nodes[nb + 1])];
                    out_tiles[ob + 2] = rep[SI(cand_nodes[nb + 2])];
                    out_tiles[ob + 3] = rep[SI(cand_nodes[nb + 3])];
                    ti++;
                }
            }
        }

        // 10. Weak CSR (2 passes, coalesced!)
        for (int i = 0; i <= n_super; i++) woff_tmp[SI(i)] = 0;
        long long self_weak = 0LL;
        for (int w = 0; w < n_ewords; w++) {
            unsigned int wmask = (~strong_bits[SI(w)]) & valid_bitmask_32(w, m_edges);
            unsigned int nmask = weak_neg[SI(w)];
            while (wmask != 0u) {
                int bit = __ffs(wmask) - 1; wmask &= (wmask - 1u);
                int eid = (w << 5) + bit;
                unsigned int uv = edges_uv[eid];
                int a = (int)rep[SI(uv & 0xFFFFu)];
                int c = (int)rep[SI(uv >> 16)];
                if (a == c) {
                    self_weak += ((nmask >> bit) & 1u) ? -(long long)weak_milli : (long long)weak_milli;
                } else {
                    woff_tmp[SI(a + 1)]++;
                    woff_tmp[SI(c + 1)]++;
                }
            }
        }
        int run = 0;
        for (int i = 0; i <= n_super; i++) {
            run += woff_tmp[SI(i)];
            woff_tmp[SI(i)] = run;
            out_woff[i] = run;
        }
        if (run > max_weak_dir) { out_n[b] = -1; continue; }
        for (int w = 0; w < n_ewords; w++) {
            unsigned int wmask = (~strong_bits[SI(w)]) & valid_bitmask_32(w, m_edges);
            unsigned int nmask = weak_neg[SI(w)];
            while (wmask != 0u) {
                int bit = __ffs(wmask) - 1; wmask &= (wmask - 1u);
                int eid = (w << 5) + bit;
                unsigned int uv = edges_uv[eid];
                unsigned short a = rep[SI(uv & 0xFFFFu)];
                unsigned short c = rep[SI(uv >> 16)];
                if (a == c) continue;
                unsigned short neg_bit = ((nmask >> bit) & 1u) ? 0x8000u : 0u;
                out_wnb_pk[woff_tmp[SI(a)]++] = c | neg_bit;
                out_wnb_pk[woff_tmp[SI(c)]++] = a | neg_bit;
            }
        }

        // 11. Tile incidence CSR
        for (int i = 0; i <= n_super; i++) woff_tmp[SI(i)] = 0;
        int n_inc = n_tiles * 4;
        for (int idx = 0; idx < n_inc; idx++) {
            woff_tmp[SI((int)out_tiles[idx] + 1)]++;
        }
        int trun = 0;
        for (int i = 0; i <= n_super; i++) {
            trun += woff_tmp[SI(i)];
            woff_tmp[SI(i)] = trun;
            out_toff[i] = trun;
        }
        for (int idx = 0; idx < n_inc; idx++) {
            int pos = woff_tmp[SI((int)out_tiles[idx])]++;
            out_inc_tid[pos] = (unsigned short)(idx >> 2);
            out_inc_g[pos] = out_gauges[idx];
        }

        out_n[b] = n_super;
        out_nt[b] = n_tiles;
        out_base[b] = (float)(-(double)pairs + (double)self_weak / 1000.0);
    }
    #undef SI
}

// ==================== Helper: Build tile incidence CSR in SMEM -> global ====================
__device__ __forceinline__ void build_tile_csr(
    int tid, int n, int nt,
    const unsigned short* __restrict__ tiles,
    const signed char* __restrict__ gauges,
    int* __restrict__ toff,
    unsigned short* __restrict__ inc_tid,
    signed char* __restrict__ inc_g,
    int* s_fill,
    int* s_blk
) {
    for (int i = tid; i < 4096; i += 256) s_fill[i] = 0;
    __syncthreads();
    int n_inc = nt * 4;
    for (int idx = tid; idx < n_inc; idx += 256) {
        atomicAdd(&s_fill[(int)tiles[idx] + 1], 1);
    }
    __syncthreads();
    int base = tid * 16;
    int lsum = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        lsum += s_fill[base + k];
        s_fill[base + k] = lsum;
    }
    s_blk[tid] = lsum;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        int v = (tid >= off) ? s_blk[tid - off] : 0;
        __syncthreads();
        s_blk[tid] += v;
        __syncthreads();
    }
    int carry = (tid > 0) ? s_blk[tid - 1] : 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int idx = base + k;
        int val = s_fill[idx] + carry;
        s_fill[idx] = val;
        if (idx <= n) toff[idx] = val;
    }
    __syncthreads();
    for (int idx = tid; idx < n_inc; idx += 256) {
        int u = (int)tiles[idx];
        int pos = atomicAdd(&s_fill[u], 1);
        inc_tid[pos] = (unsigned short)(idx >> 2);
        inc_g[pos] = gauges[idx];
    }
    __syncthreads();
}

// ==================== KERNEL 2-BM2: 4-Block/SM Ultra-Fast Rank-2 (S^1) FP16x2 Pre-Filter ====================
// Dynamic SMEM: 22,592 B (s_u[3600 u32] + s_stile[2048 u32]) -> 4 blocks/SM on RTX 4090 sm_89!
extern "C" __global__ void bm2_batch_sdp_smem_fp16(
    const int* __restrict__ active_slots,
    int skip_build_csr,
    const int* __restrict__ n_arr,
    const int* __restrict__ nt_arr,
    const float* __restrict__ base_const_arr,
    float weak_milli,
    const unsigned long long* __restrict__ seed_arr,
    const unsigned short* __restrict__ tiles_all,
    const signed char* __restrict__ g_all,
    const int* __restrict__ woff_all,
    const unsigned short* __restrict__ wnb_pk_all,
    int* __restrict__ toff_all,
    unsigned short* __restrict__ inc_tid_all,
    signed char* __restrict__ inc_g_all,
    float* __restrict__ out_energy,
    int steps, float sw0, float sw1, float weak_mul, float lr0,
    int max_n, int max_tiles, int max_inc, int max_weak_dir
) {
    int out_idx = blockIdx.x;
    int b = active_slots ? active_slots[out_idx] : out_idx;
    int tid = threadIdx.x;
    int n = n_arr[b];
    int nt = nt_arr[b];
    if (n <= 0 || n > max_n || nt > max_tiles) {
        if (tid == 0) out_energy[out_idx] = 1e9f;
        return;
    }

    const unsigned short* tiles  = tiles_all + (size_t)b * max_tiles * 4;
    const signed char* gauges    = g_all + (size_t)b * max_tiles * 4;
    const int* woff              = woff_all + (size_t)b * (max_n + 1);
    const unsigned short* wnb_pk = wnb_pk_all + (size_t)b * max_weak_dir;
    int* toff                    = toff_all + (size_t)b * (max_n + 1);
    unsigned short* inc_tid      = inc_tid_all + (size_t)b * max_inc;
    signed char* inc_g           = inc_g_all + (size_t)b * max_inc;

    extern __shared__ unsigned char dyn_smem[];
    if (!skip_build_csr) {
        build_tile_csr(tid, n, nt, tiles, gauges, toff, inc_tid, inc_g,
                       (int*)dyn_smem, (int*)(dyn_smem + 4096 * sizeof(int)));
    }

    unsigned int* s_u     = (unsigned int*)dyn_smem;                        // [MAX_N] (14,400 B)
    unsigned int* s_stile = (unsigned int*)(dyn_smem + max_n * sizeof(unsigned int)); // [MAX_TILES] (8,192 B)

    unsigned long long seed0 = seed_arr[b] ^ 0xA5A55A5A12345678ULL;
    unsigned int r_mom[15];
    unsigned int r_unxt[15];

    #pragma unroll
    for (int k = 0; k < 15; k++) {
        int i = tid + k * 256;
        r_mom[k] = 0u;
        if (i < n) {
            unsigned long long st = seed0 ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            float v0 = (float)((st * 0x2545F4914F6CDD1DULL) >> 40) * (1.f / 16777216.f) * 2.f - 1.f;
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            float v1 = (float)((st * 0x2545F4914F6CDD1DULL) >> 40) * (1.f / 16777216.f) * 2.f - 1.f;
            float inv = rsqrtf(v0*v0 + v1*v1 + 1e-12f);
            s_u[i] = pack_h2(v0 * inv, v1 * inv);
        }
    }
    __syncthreads();

    const float inv_steps = 1.f / (float)(steps > 1 ? steps : 1);
    const float mom_beta = 0.75f;
    const float w_scale = weak_milli * weak_mul * 0.001f;

    for (int iter = 0; iter < steps; iter++) {
        float prog = (float)iter * inv_steps;
        float strong_w = sw0 + (sw1 - sw0) * prog;
        unsigned int sw_h2 = pack_h2(strong_w, strong_w);
        float lr = lr0 * (1.f - 0.80f * prog);

        for (int t = tid; t < nt; t += 256) {
            unsigned int u0 = s_u[tiles[t * 4 + 0]];
            unsigned int u1 = s_u[tiles[t * 4 + 1]];
            unsigned int u2 = s_u[tiles[t * 4 + 2]];
            unsigned int u3 = s_u[tiles[t * 4 + 3]];
            unsigned int m0 = (gauges[t * 4 + 0] < 0) ? 0x80008000u : 0u;
            unsigned int m1 = (gauges[t * 4 + 1] < 0) ? 0x80008000u : 0u;
            unsigned int m2 = (gauges[t * 4 + 2] < 0) ? 0x80008000u : 0u;
            unsigned int m3 = (gauges[t * 4 + 3] < 0) ? 0x80008000u : 0u;
            unsigned int sx = add_h2(add_h2(u0 ^ m0, u1 ^ m1), add_h2(u2 ^ m2, u3 ^ m3));
            s_stile[t] = mul_h2(sx, sw_h2);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 15; k++) {
            int i = tid + k * 256;
            if (i >= n) break;
            unsigned int g_x = 0u;
            int tp0 = toff[i], tp1 = toff[i + 1];
            for (int p = tp0; p < tp1; p++) {
                unsigned int stv = s_stile[inc_tid[p]];
                unsigned int sm = (inc_g[p] < 0) ? 0x80008000u : 0u;
                g_x = add_h2(g_x, stv ^ sm);
            }
            unsigned int w_x = 0u;
            int wp0 = woff[i], wp1 = woff[i + 1];
            for (int p = wp0; p < wp1; p++) {
                unsigned short pk = wnb_pk[p];
                unsigned int nu = s_u[pk & 0x7FFFu];
                unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
                w_x = add_h2(w_x, nu ^ sm);
            }
            float2 ga = unpack_h2(g_x);
            float2 wa = unpack_h2(w_x);
            float gx = ga.x + w_scale * wa.x;
            float gy = ga.y + w_scale * wa.y;

            float2 ua = unpack_h2(s_u[i]);
            float dot = gx * ua.x + gy * ua.y;
            float2 ma = unpack_h2(r_mom[k]);
            ma.x = mom_beta * ma.x - lr * (gx - dot * ua.x);
            ma.y = mom_beta * ma.y - lr * (gy - dot * ua.y);
            r_mom[k] = pack_h2(ma.x, ma.y);

            float nx = ua.x + ma.x, ny = ua.y + ma.y;
            float inv = rsqrtf(nx*nx + ny*ny + 1e-12f);
            r_unxt[k] = pack_h2(nx * inv, ny * inv);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 15; k++) {
            int i = tid + k * 256;
            if (i < n) s_u[i] = r_unxt[k];
        }
        __syncthreads();
    }

    float local_e = 0.f;
    for (int t = tid; t < nt; t += 256) {
        unsigned int u0 = s_u[tiles[t * 4 + 0]], u1 = s_u[tiles[t * 4 + 1]];
        unsigned int u2 = s_u[tiles[t * 4 + 2]], u3 = s_u[tiles[t * 4 + 3]];
        unsigned int m0 = (gauges[t * 4 + 0] < 0) ? 0x80008000u : 0u;
        unsigned int m1 = (gauges[t * 4 + 1] < 0) ? 0x80008000u : 0u;
        unsigned int m2 = (gauges[t * 4 + 2] < 0) ? 0x80008000u : 0u;
        unsigned int m3 = (gauges[t * 4 + 3] < 0) ? 0x80008000u : 0u;
        float2 sa = unpack_h2(add_h2(add_h2(u0 ^ m0, u1 ^ m1), add_h2(u2 ^ m2, u3 ^ m3)));
        local_e += 0.5f * (sa.x*sa.x + sa.y*sa.y - 4.f);
    }
    float local_w = 0.f;
    const float w_unit = weak_milli * 0.001f;
    for (int i = tid; i < n; i += 256) {
        unsigned int w_x = 0u;
        int wp0 = woff[i], wp1 = woff[i + 1];
        for (int p = wp0; p < wp1; p++) {
            unsigned short pk = wnb_pk[p];
            unsigned int nu = s_u[pk & 0x7FFFu];
            unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
            w_x = add_h2(w_x, nu ^ sm);
        }
        float2 wa = unpack_h2(w_x);
        float2 ua = unpack_h2(s_u[i]);
        local_w += w_unit * (ua.x*wa.x + ua.y*wa.y);
    }
    __syncthreads();
    double* ssum = (double*)s_stile;
    ssum[tid] = (double)local_e + 0.5 * (double)local_w;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_energy[out_idx] = (float)((double)base_const_arr[b] + ssum[0]);
}

// ==================== KERNEL 2A: Ultra-Fast SMEM-FP16 + Register-Mom Rank-4 BM-SDP ====================
// Dynamic SMEM: 45,184 B (s_u[3600*2 u32] + s_stile[2048*2 u32]) -> 2 blocks/SM on Ampere!
// ZERO global reads/writes for u, stile, mom, u_nxt during iterations!
extern "C" __global__ void bm4_batch_sdp_smem_fp16(
    const int* __restrict__ active_slots, // NULL -> slot = blockIdx.x
    int skip_build_csr,                   // 1 if 1tps/BM2 already built toff/inc_tid/inc_g!
    const int* __restrict__ n_arr,
    const int* __restrict__ nt_arr,
    const float* __restrict__ base_const_arr,
    float weak_milli,
    const unsigned long long* __restrict__ seed_arr,
    const unsigned short* __restrict__ tiles_all,
    const signed char* __restrict__ g_all,
    const int* __restrict__ woff_all,
    const unsigned short* __restrict__ wnb_pk_all,
    int* __restrict__ toff_all,
    unsigned short* __restrict__ inc_tid_all,
    signed char* __restrict__ inc_g_all,
    float* __restrict__ out_energy,
    int steps, float sw0, float sw1, float weak_mul, float lr0,
    int max_n, int max_tiles, int max_inc, int max_weak_dir
) {
    int out_idx = blockIdx.x;
    int b = active_slots ? active_slots[out_idx] : out_idx;
    int tid = threadIdx.x;
    int n = n_arr[b];
    int nt = nt_arr[b];
    if (n <= 0 || n > max_n || nt > max_tiles) {
        if (tid == 0) out_energy[out_idx] = 1e9f;
        return;
    }

    const unsigned short* tiles  = tiles_all + (size_t)b * max_tiles * 4;
    const signed char* gauges    = g_all + (size_t)b * max_tiles * 4;
    const int* woff              = woff_all + (size_t)b * (max_n + 1);
    const unsigned short* wnb_pk = wnb_pk_all + (size_t)b * max_weak_dir;
    int* toff                    = toff_all + (size_t)b * (max_n + 1);
    unsigned short* inc_tid      = inc_tid_all + (size_t)b * max_inc;
    signed char* inc_g           = inc_g_all + (size_t)b * max_inc;

    extern __shared__ unsigned char dyn_smem[];
    // Pass 0: build tile CSR reusing dyn_smem (skipped if 1tps/BM2 already built it!)
    if (!skip_build_csr) {
        build_tile_csr(tid, n, nt, tiles, gauges, toff, inc_tid, inc_g,
                       (int*)dyn_smem, (int*)(dyn_smem + 4096 * sizeof(int)));
    }

    uint2* s_u     = (uint2*)dyn_smem;                        // [MAX_N] (28,800 B)
    uint2* s_stile = (uint2*)(dyn_smem + max_n * sizeof(uint2)); // [MAX_TILES] (16,384 B)

    unsigned long long seed0 = seed_arr[b] ^ 0xA5A55A5A12345678ULL;
    // Register-resident FP16 momentum & next-u for up to 15 owned nodes per thread (15*256=3840 >= 3600)!
    uint2 r_mom[15];
    uint2 r_unxt[15];

    #pragma unroll
    for (int k = 0; k < 15; k++) {
        int i = tid + k * 256;
        r_mom[k] = make_uint2(0u, 0u);
        if (i < n) {
            unsigned long long st = seed0 ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
            float v[4];
            float n2 = 0.f;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
                unsigned long long h = st * 0x2545F4914F6CDD1DULL;
                float val = (float)(h >> 40) * (1.f / 16777216.f) * 2.f - 1.f;
                v[r] = val; n2 += val * val;
            }
            float inv = rsqrtf(n2 + 1e-12f);
            s_u[i] = make_uint2(pack_h2(v[0]*inv, v[1]*inv), pack_h2(v[2]*inv, v[3]*inv));
        }
    }
    __syncthreads();

    const float inv_steps = 1.f / (float)(steps > 1 ? steps : 1);
    const float mom_beta = 0.75f;
    const float w_scale = weak_milli * weak_mul * 0.001f;

    for (int iter = 0; iter < steps; iter++) {
        float prog = (float)iter * inv_steps;
        float strong_w = sw0 + (sw1 - sw0) * prog;
        unsigned int sw_h2 = pack_h2(strong_w, strong_w);
        float lr = lr0 * (1.f - 0.80f * prog);

        // Pass 1: Tile sums completely in SMEM using native f16x2 SIMD!
        for (int t = tid; t < nt; t += 256) {
            uint2 u0 = s_u[tiles[t * 4 + 0]];
            uint2 u1 = s_u[tiles[t * 4 + 1]];
            uint2 u2 = s_u[tiles[t * 4 + 2]];
            uint2 u3 = s_u[tiles[t * 4 + 3]];
            unsigned int m0 = (gauges[t * 4 + 0] < 0) ? 0x80008000u : 0u;
            unsigned int m1 = (gauges[t * 4 + 1] < 0) ? 0x80008000u : 0u;
            unsigned int m2 = (gauges[t * 4 + 2] < 0) ? 0x80008000u : 0u;
            unsigned int m3 = (gauges[t * 4 + 3] < 0) ? 0x80008000u : 0u;
            unsigned int sx = add_h2(add_h2(u0.x ^ m0, u1.x ^ m1), add_h2(u2.x ^ m2, u3.x ^ m3));
            unsigned int sy = add_h2(add_h2(u0.y ^ m0, u1.y ^ m1), add_h2(u2.y ^ m2, u3.y ^ m3));
            s_stile[t] = make_uint2(mul_h2(sx, sw_h2), mul_h2(sy, sw_h2));
        }
        __syncthreads();

        // Pass 2: Native f16x2 SMEM gather + u16 weak SpMV (4x fewer instructions!) + register momentum!
        #pragma unroll
        for (int k = 0; k < 15; k++) {
            int i = tid + k * 256;
            if (i >= n) break;
            unsigned int g_x = 0u, g_y = 0u;
            int tp0 = toff[i], tp1 = toff[i + 1];
            for (int p = tp0; p < tp1; p++) {
                uint2 stv = s_stile[inc_tid[p]];
                unsigned int sm = (inc_g[p] < 0) ? 0x80008000u : 0u;
                g_x = add_h2(g_x, stv.x ^ sm);
                g_y = add_h2(g_y, stv.y ^ sm);
            }
            unsigned int w_x = 0u, w_y = 0u;
            int wp0 = woff[i], wp1 = woff[i + 1];
            for (int p = wp0; p < wp1; p++) {
                unsigned short pk = wnb_pk[p];
                uint2 nu = s_u[pk & 0x7FFFu];
                unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
                w_x = add_h2(w_x, nu.x ^ sm);
                w_y = add_h2(w_y, nu.y ^ sm);
            }
            float2 ga = unpack_h2(g_x), gb = unpack_h2(g_y);
            float2 wa = unpack_h2(w_x), wb = unpack_h2(w_y);
            float gx = ga.x + w_scale * wa.x;
            float gy = ga.y + w_scale * wa.y;
            float gz = gb.x + w_scale * wb.x;
            float gw = gb.y + w_scale * wb.y;

            uint2 ui = s_u[i];
            float2 ua = unpack_h2(ui.x), ub = unpack_h2(ui.y);
            float dot = gx*ua.x + gy*ua.y + gz*ub.x + gw*ub.y;

            float2 ma = unpack_h2(r_mom[k].x), mb = unpack_h2(r_mom[k].y);
            ma.x = mom_beta * ma.x - lr * (gx - dot * ua.x);
            ma.y = mom_beta * ma.y - lr * (gy - dot * ua.y);
            mb.x = mom_beta * mb.x - lr * (gz - dot * ub.x);
            mb.y = mom_beta * mb.y - lr * (gw - dot * ub.y);
            r_mom[k] = make_uint2(pack_h2(ma.x, ma.y), pack_h2(mb.x, mb.y));

            float nx = ua.x + ma.x, ny = ua.y + ma.y, nz = ub.x + mb.x, nw = ub.y + mb.y;
            float inv = rsqrtf(nx*nx + ny*ny + nz*nz + nw*nw + 1e-12f);
            r_unxt[k] = make_uint2(pack_h2(nx*inv, ny*inv), pack_h2(nz*inv, nw*inv));
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 15; k++) {
            int i = tid + k * 256;
            if (i < n) s_u[i] = r_unxt[k];
        }
        __syncthreads();
    }

    // Fast FP32 energy accumulation (eliminates 1/64-rate FP64 stall on Ada sm_89!) + FP64 block reduction
    float local_e = 0.f;
    for (int t = tid; t < nt; t += 256) {
        uint2 u0 = s_u[tiles[t * 4 + 0]], u1 = s_u[tiles[t * 4 + 1]];
        uint2 u2 = s_u[tiles[t * 4 + 2]], u3 = s_u[tiles[t * 4 + 3]];
        unsigned int m0 = (gauges[t * 4 + 0] < 0) ? 0x80008000u : 0u;
        unsigned int m1 = (gauges[t * 4 + 1] < 0) ? 0x80008000u : 0u;
        unsigned int m2 = (gauges[t * 4 + 2] < 0) ? 0x80008000u : 0u;
        unsigned int m3 = (gauges[t * 4 + 3] < 0) ? 0x80008000u : 0u;
        float2 sa = unpack_h2(add_h2(add_h2(u0.x ^ m0, u1.x ^ m1), add_h2(u2.x ^ m2, u3.x ^ m3)));
        float2 sb = unpack_h2(add_h2(add_h2(u0.y ^ m0, u1.y ^ m1), add_h2(u2.y ^ m2, u3.y ^ m3)));
        local_e += 0.5f * (sa.x*sa.x + sa.y*sa.y + sb.x*sb.x + sb.y*sb.y - 4.f);
    }
    float local_w = 0.f;
    const float w_unit = weak_milli * 0.001f;
    for (int i = tid; i < n; i += 256) {
        unsigned int w_x = 0u, w_y = 0u;
        int wp0 = woff[i], wp1 = woff[i + 1];
        for (int p = wp0; p < wp1; p++) {
            unsigned short pk = wnb_pk[p];
            uint2 nu = s_u[pk & 0x7FFFu];
            unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
            w_x = add_h2(w_x, nu.x ^ sm);
            w_y = add_h2(w_y, nu.y ^ sm);
        }
        float2 wa = unpack_h2(w_x), wb = unpack_h2(w_y);
        uint2 ui = s_u[i];
        float2 ua = unpack_h2(ui.x), ub = unpack_h2(ui.y);
        local_w += w_unit * (ua.x*wa.x + ua.y*wa.y + ub.x*wb.x + ub.y*wb.y);
    }
    __syncthreads();
    double* ssum = (double*)s_stile;
    ssum[tid] = (double)local_e + 0.5 * (double)local_w;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_energy[out_idx] = (float)((double)base_const_arr[b] + ssum[0]);
}

// ==================== KERNEL 2B: Full Rank-8 SMEM-FP16 BM-SDP ====================
// Dynamic SMEM: 90,368 B (s_u[3600*4 u32] = 57.6 KB + s_stile[2048*4 u32] = 32.8 KB) < 99 KB Ampere!
// ALL random weak-edge gathers hit 20 TB/s on-chip SMEM!
extern "C" __global__ void bm8_batch_sdp_smem_fp16(
    const int* __restrict__ active_slots, // NULL -> slot = blockIdx.x
    int skip_build_csr,                   // 1 if BM4 already built toff/inc_tid/inc_g!
    const int* __restrict__ n_arr,
    const int* __restrict__ nt_arr,
    const float* __restrict__ base_const_arr,
    float weak_milli,
    const unsigned long long* __restrict__ seed_arr,
    const unsigned short* __restrict__ tiles_all,
    const signed char* __restrict__ g_all,
    const int* __restrict__ woff_all,
    const unsigned short* __restrict__ wnb_pk_all,
    int* __restrict__ toff_all,
    unsigned short* __restrict__ inc_tid_all,
    signed char* __restrict__ inc_g_all,
    uint4* __restrict__ mom_h_all,        // [MAX_ACTIVE * MAX_N] (51 KB/block, L2-hot FP16!)
    uint4* __restrict__ unxt_h_all,       // [MAX_ACTIVE * MAX_N] (51 KB/block, L2-hot FP16!)
    float* __restrict__ out_energy,
    int steps, float sw0, float sw1, float weak_mul, float lr0,
    int max_n, int max_tiles, int max_inc, int max_weak_dir
) {
    int out_idx = blockIdx.x;
    int b = active_slots ? active_slots[out_idx] : out_idx;
    int tid = threadIdx.x;
    int n = n_arr[b];
    int nt = nt_arr[b];
    if (n <= 0 || n > max_n || nt > max_tiles) {
        if (tid == 0) out_energy[out_idx] = 1e9f;
        return;
    }

    const unsigned short* tiles  = tiles_all + (size_t)b * max_tiles * 4;
    const signed char* gauges    = g_all + (size_t)b * max_tiles * 4;
    const int* woff              = woff_all + (size_t)b * (max_n + 1);
    const unsigned short* wnb_pk = wnb_pk_all + (size_t)b * max_weak_dir;
    int* toff                    = toff_all + (size_t)b * (max_n + 1);
    unsigned short* inc_tid      = inc_tid_all + (size_t)b * max_inc;
    signed char* inc_g           = inc_g_all + (size_t)b * max_inc;

    // Indexed by out_idx (compact!) so active subset fits 100% in L2!
    uint4* mom_h  = mom_h_all + (size_t)out_idx * max_n;
    uint4* unxt_h = unxt_h_all + (size_t)out_idx * max_n;

    extern __shared__ unsigned char dyn_smem[];
    if (!skip_build_csr) {
        build_tile_csr(tid, n, nt, tiles, gauges, toff, inc_tid, inc_g,
                       (int*)dyn_smem, (int*)(dyn_smem + 4096 * sizeof(int)));
    }

    uint4* s_u     = (uint4*)dyn_smem;                           // [MAX_N] (57,600 B)
    uint4* s_stile = (uint4*)(dyn_smem + max_n * sizeof(uint4)); // [MAX_TILES] (32,768 B)

    unsigned long long seed0 = seed_arr[b] ^ 0xA5A55A5A12345678ULL;
    const uint4 zu4 = make_uint4(0u, 0u, 0u, 0u);
    for (int i = tid; i < n; i += 256) {
        unsigned long long st = seed0 ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
        float v[8];
        float n2 = 0.f;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            unsigned long long h = st * 0x2545F4914F6CDD1DULL;
            float val = (float)(h >> 40) * (1.f / 16777216.f) * 2.f - 1.f;
            v[r] = val; n2 += val * val;
        }
        float inv = rsqrtf(n2 + 1e-12f);
        s_u[i] = make_uint4(
            pack_h2(v[0]*inv, v[1]*inv), pack_h2(v[2]*inv, v[3]*inv),
            pack_h2(v[4]*inv, v[5]*inv), pack_h2(v[6]*inv, v[7]*inv)
        );
        mom_h[i] = zu4;
    }
    __syncthreads();

    const float inv_steps = 1.f / (float)(steps > 1 ? steps : 1);
    const float mom_beta = 0.75f;
    const float w_scale = weak_milli * weak_mul * 0.001f;

    for (int iter = 0; iter < steps; iter++) {
        float prog = (float)iter * inv_steps;
        float strong_w = sw0 + (sw1 - sw0) * prog;
        unsigned int sw_h2 = pack_h2(strong_w, strong_w);
        float lr = lr0 * (1.f - 0.80f * prog);

        // Pass 1: Tile sums in SMEM using native f16x2 SIMD!
        for (int t = tid; t < nt; t += 256) {
            uint4 u0 = s_u[tiles[t * 4 + 0]], u1 = s_u[tiles[t * 4 + 1]];
            uint4 u2 = s_u[tiles[t * 4 + 2]], u3 = s_u[tiles[t * 4 + 3]];
            unsigned int m0 = (gauges[t * 4 + 0] < 0) ? 0x80008000u : 0u;
            unsigned int m1 = (gauges[t * 4 + 1] < 0) ? 0x80008000u : 0u;
            unsigned int m2 = (gauges[t * 4 + 2] < 0) ? 0x80008000u : 0u;
            unsigned int m3 = (gauges[t * 4 + 3] < 0) ? 0x80008000u : 0u;
            s_stile[t] = make_uint4(
                mul_h2(add_h2(add_h2(u0.x ^ m0, u1.x ^ m1), add_h2(u2.x ^ m2, u3.x ^ m3)), sw_h2),
                mul_h2(add_h2(add_h2(u0.y ^ m0, u1.y ^ m1), add_h2(u2.y ^ m2, u3.y ^ m3)), sw_h2),
                mul_h2(add_h2(add_h2(u0.z ^ m0, u1.z ^ m1), add_h2(u2.z ^ m2, u3.z ^ m3)), sw_h2),
                mul_h2(add_h2(add_h2(u0.w ^ m0, u1.w ^ m1), add_h2(u2.w ^ m2, u3.w ^ m3)), sw_h2)
            );
        }
        __syncthreads();

        // Pass 2: Native f16x2 SMEM gather + u16 weak SpMV (4x fewer instructions!)
        for (int i = tid; i < n; i += 256) {
            unsigned int gx_h = 0u, gy_h = 0u, gz_h = 0u, gw_h = 0u;
            int tp0 = toff[i], tp1 = toff[i + 1];
            for (int p = tp0; p < tp1; p++) {
                uint4 stv = s_stile[inc_tid[p]];
                unsigned int sm = (inc_g[p] < 0) ? 0x80008000u : 0u;
                gx_h = add_h2(gx_h, stv.x ^ sm);
                gy_h = add_h2(gy_h, stv.y ^ sm);
                gz_h = add_h2(gz_h, stv.z ^ sm);
                gw_h = add_h2(gw_h, stv.w ^ sm);
            }
            unsigned int wx_h = 0u, wy_h = 0u, wz_h = 0u, ww_h = 0u;
            int wp0 = woff[i], wp1 = woff[i + 1];
            for (int p = wp0; p < wp1; p++) {
                unsigned short pk = wnb_pk[p];
                uint4 nu = s_u[pk & 0x7FFFu];
                unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
                wx_h = add_h2(wx_h, nu.x ^ sm);
                wy_h = add_h2(wy_h, nu.y ^ sm);
                wz_h = add_h2(wz_h, nu.z ^ sm);
                ww_h = add_h2(ww_h, nu.w ^ sm);
            }
            float2 gx2=unpack_h2(gx_h), gy2=unpack_h2(gy_h), gz2=unpack_h2(gz_h), gw2=unpack_h2(gw_h);
            float2 wx2=unpack_h2(wx_h), wy2=unpack_h2(wy_h), wz2=unpack_h2(wz_h), ww2=unpack_h2(ww_h);
            float g0=gx2.x+w_scale*wx2.x, g1=gx2.y+w_scale*wx2.y;
            float g2=gy2.x+w_scale*wy2.x, g3=gy2.y+w_scale*wy2.y;
            float g4=gz2.x+w_scale*wz2.x, g5=gz2.y+w_scale*wz2.y;
            float g6=gw2.x+w_scale*ww2.x, g7=gw2.y+w_scale*ww2.y;

            uint4 ui = s_u[i];
            float2 ux=unpack_h2(ui.x), uy=unpack_h2(ui.y), uz=unpack_h2(ui.z), uw=unpack_h2(ui.w);
            float dot = g0*ux.x+g1*ux.y+g2*uy.x+g3*uy.y+g4*uz.x+g5*uz.y+g6*uw.x+g7*uw.y;

            uint4 mi = mom_h[i];
            float2 mx=unpack_h2(mi.x), my=unpack_h2(mi.y), mz=unpack_h2(mi.z), mw=unpack_h2(mi.w);
            mx.x = mom_beta*mx.x - lr*(g0 - dot*ux.x); mx.y = mom_beta*mx.y - lr*(g1 - dot*ux.y);
            my.x = mom_beta*my.x - lr*(g2 - dot*uy.x); my.y = mom_beta*my.y - lr*(g3 - dot*uy.y);
            mz.x = mom_beta*mz.x - lr*(g4 - dot*uz.x); mz.y = mom_beta*mz.y - lr*(g5 - dot*uz.y);
            mw.x = mom_beta*mw.x - lr*(g6 - dot*uw.x); mw.y = mom_beta*mw.y - lr*(g7 - dot*uw.y);
            mom_h[i] = make_uint4(pack_h2(mx.x, mx.y), pack_h2(my.x, my.y), pack_h2(mz.x, mz.y), pack_h2(mw.x, mw.y));

            float n0=ux.x+mx.x, n1=ux.y+mx.y, n2=uy.x+my.x, n3=uy.y+my.y;
            float n4=uz.x+mz.x, n5=uz.y+mz.y, n6=uw.x+mw.x, n7=uw.y+mw.y;
            float inv = rsqrtf(n0*n0+n1*n1+n2*n2+n3*n3+n4*n4+n5*n5+n6*n6+n7*n7 + 1e-12f);
            unxt_h[i] = make_uint4(pack_h2(n0*inv, n1*inv), pack_h2(n2*inv, n3*inv), pack_h2(n4*inv, n5*inv), pack_h2(n6*inv, n7*inv));
        }
        __syncthreads();
        for (int i = tid; i < n; i += 256) s_u[i] = unxt_h[i];
        __syncthreads();
    }

    // Fast FP32 energy reduction + FP64 block sum (eliminates 1/64-rate FP64 stall!)
    float local_e = 0.f;
    for (int t = tid; t < nt; t += 256) {
        uint4 u0=s_u[tiles[t*4+0]], u1=s_u[tiles[t*4+1]], u2=s_u[tiles[t*4+2]], u3=s_u[tiles[t*4+3]];
        unsigned int m0=(gauges[t*4+0]<0)?0x80008000u:0u, m1=(gauges[t*4+1]<0)?0x80008000u:0u;
        unsigned int m2=(gauges[t*4+2]<0)?0x80008000u:0u, m3=(gauges[t*4+3]<0)?0x80008000u:0u;
        float2 sx=unpack_h2(add_h2(add_h2(u0.x^m0,u1.x^m1),add_h2(u2.x^m2,u3.x^m3)));
        float2 sy=unpack_h2(add_h2(add_h2(u0.y^m0,u1.y^m1),add_h2(u2.y^m2,u3.y^m3)));
        float2 sz=unpack_h2(add_h2(add_h2(u0.z^m0,u1.z^m1),add_h2(u2.z^m2,u3.z^m3)));
        float2 sw=unpack_h2(add_h2(add_h2(u0.w^m0,u1.w^m1),add_h2(u2.w^m2,u3.w^m3)));
        local_e += 0.5f * (sx.x*sx.x+sx.y*sx.y+sy.x*sy.x+sy.y*sy.y+sz.x*sz.x+sz.y*sz.y+sw.x*sw.x+sw.y*sw.y - 4.f);
    }
    float local_w = 0.f;
    const float w_unit = weak_milli * 0.001f;
    for (int i = tid; i < n; i += 256) {
        unsigned int wx_h=0u, wy_h=0u, wz_h=0u, ww_h=0u;
        int wp0 = woff[i], wp1 = woff[i + 1];
        for (int p = wp0; p < wp1; p++) {
            unsigned short pk = wnb_pk[p];
            uint4 nu = s_u[pk & 0x7FFFu];
            unsigned int sm = (pk & 0x8000u) ? 0x80008000u : 0u;
            wx_h = add_h2(wx_h, nu.x ^ sm); wy_h = add_h2(wy_h, nu.y ^ sm);
            wz_h = add_h2(wz_h, nu.z ^ sm); ww_h = add_h2(ww_h, nu.w ^ sm);
        }
        float2 wx=unpack_h2(wx_h), wy=unpack_h2(wy_h), wz=unpack_h2(wz_h), ww=unpack_h2(ww_h);
        uint4 ui = s_u[i];
        float2 ux=unpack_h2(ui.x), uy=unpack_h2(ui.y), uz=unpack_h2(ui.z), uw=unpack_h2(ui.w);
        local_w += w_unit * (ux.x*wx.x+ux.y*wx.y+uy.x*wy.x+uy.y*wy.y+uz.x*wz.x+uz.y*wz.y+uw.x*ww.x+uw.y*ww.y);
    }
    __syncthreads();
    double* ssum = (double*)s_stile;
    ssum[tid] = (double)local_e + 0.5 * (double)local_w;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_energy[out_idx] = (float)((double)base_const_arr[b] + ssum[0]);
}

// ==================== KERNEL 3: Exact FP32 BM8_w30 (top ~2% survivors only, return_u!) ====================
extern "C" __global__ void bm8_batch_sdp_fp32_indexed(
    const int* __restrict__ active_slots, // [K]
    const int* __restrict__ n_arr,
    const int* __restrict__ nt_arr,
    const float* __restrict__ base_const_arr,
    float weak_milli,
    const unsigned long long* __restrict__ seed_arr,
    unsigned long long seed_add,
    const unsigned short* __restrict__ tiles_all,
    const signed char* __restrict__ g_all,
    const int* __restrict__ woff_all,
    const unsigned short* __restrict__ wnb_pk_all,
    const int* __restrict__ toff_all,
    const unsigned short* __restrict__ inc_tid_all,
    const signed char* __restrict__ inc_g_all,
    float4* __restrict__ u_a_all,         // [K * MAX_N * 2] compact!
    float4* __restrict__ u_b_all,
    float4* __restrict__ mom_all,
    float4* __restrict__ stile_all,
    float* __restrict__ out_energy,       // [K]
    int steps, float sw0, float sw1, float weak_mul, float lr0,
    int max_n, int max_tiles, int max_inc, int max_weak_dir
) {
    int out_idx = blockIdx.x;
    int b = active_slots[out_idx];
    int tid = threadIdx.x;
    int n = n_arr[b];
    int nt = nt_arr[b];
    if (n <= 0 || n > max_n || nt > max_tiles) {
        if (tid == 0) out_energy[out_idx] = 1e9f;
        return;
    }

    const unsigned short* tiles  = tiles_all + (size_t)b * max_tiles * 4;
    const signed char* gauges    = g_all + (size_t)b * max_tiles * 4;
    const int* woff              = woff_all + (size_t)b * (max_n + 1);
    const unsigned short* wnb_pk = wnb_pk_all + (size_t)b * max_weak_dir;
    const int* toff              = toff_all + (size_t)b * (max_n + 1);
    const unsigned short* inc_tid= inc_tid_all + (size_t)b * max_inc;
    const signed char* inc_g     = inc_g_all + (size_t)b * max_inc;

    float4* u_cur = u_a_all + (size_t)out_idx * max_n * 2;
    float4* u_nxt = u_b_all + (size_t)out_idx * max_n * 2;
    float4* mom   = mom_all + (size_t)out_idx * max_n * 2;
    float4* stile = stile_all + (size_t)out_idx * max_tiles * 2;

    unsigned long long seed0 = (seed_arr[b] + seed_add) ^ 0xA5A55A5A12345678ULL;
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = tid; i < n; i += 256) {
        unsigned long long st = seed0 ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
        float v[8]; float n2 = 0.f;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            unsigned long long h = st * 0x2545F4914F6CDD1DULL;
            float val = (float)(h >> 40) * (1.f / 16777216.f) * 2.f - 1.f;
            v[r] = val; n2 += val * val;
        }
        float inv = rsqrtf(n2 + 1e-12f);
        u_cur[i*2+0] = make_float4(v[0]*inv, v[1]*inv, v[2]*inv, v[3]*inv);
        u_cur[i*2+1] = make_float4(v[4]*inv, v[5]*inv, v[6]*inv, v[7]*inv);
        mom[i*2+0] = z4; mom[i*2+1] = z4;
    }
    __syncthreads();

    const float inv_steps = 1.f / (float)(steps > 1 ? steps : 1);
    const float mom_beta = 0.75f;
    const float w_scale = weak_milli * weak_mul * 0.001f;

    for (int iter = 0; iter < steps; iter++) {
        float prog = (float)iter * inv_steps;
        float strong_w = sw0 + (sw1 - sw0) * prog;
        float lr = lr0 * (1.f - 0.80f * prog);

        for (int t = tid; t < nt; t += 256) {
            int t0=(int)tiles[t*4+0]*2, t1=(int)tiles[t*4+1]*2, t2=(int)tiles[t*4+2]*2, t3=(int)tiles[t*4+3]*2;
            float g0=(float)gauges[t*4+0], g1=(float)gauges[t*4+1], g2=(float)gauges[t*4+2], g3=(float)gauges[t*4+3];
            float4 a0=u_cur[t0], a1=u_cur[t1], a2=u_cur[t2], a3=u_cur[t3];
            float4 b0=u_cur[t0+1], b1=u_cur[t1+1], b2=u_cur[t2+1], b3=u_cur[t3+1];
            stile[t*2+0] = make_float4(strong_w*(g0*a0.x+g1*a1.x+g2*a2.x+g3*a3.x), strong_w*(g0*a0.y+g1*a1.y+g2*a2.y+g3*a3.y),
                                       strong_w*(g0*a0.z+g1*a1.z+g2*a2.z+g3*a3.z), strong_w*(g0*a0.w+g1*a1.w+g2*a2.w+g3*a3.w));
            stile[t*2+1] = make_float4(strong_w*(g0*b0.x+g1*b1.x+g2*b2.x+g3*b3.x), strong_w*(g0*b0.y+g1*b1.y+g2*b2.y+g3*b3.y),
                                       strong_w*(g0*b0.z+g1*b1.z+g2*b2.z+g3*b3.z), strong_w*(g0*b0.w+g1*b1.w+g2*b2.w+g3*b3.w));
        }
        __syncthreads();

        for (int i = tid; i < n; i += 256) {
            float4 ga = z4, gb = z4;
            int tp0 = toff[i], tp1 = toff[i + 1];
            for (int p = tp0; p < tp1; p++) {
                int t2 = (int)inc_tid[p] * 2;
                float sg = (float)inc_g[p];
                float4 sa = stile[t2], sb = stile[t2 + 1];
                ga.x+=sg*sa.x; ga.y+=sg*sa.y; ga.z+=sg*sa.z; ga.w+=sg*sa.w;
                gb.x+=sg*sb.x; gb.y+=sg*sb.y; gb.z+=sg*sb.z; gb.w+=sg*sb.w;
            }
            float4 wa = z4, wb = z4;
            int wp0 = woff[i], wp1 = woff[i + 1];
            for (int p = wp0; p < wp1; p++) {
                unsigned short pk = wnb_pk[p];
                int nb2 = (int)(pk & 0x7FFFu) * 2;
                float sgn = (pk & 0x8000u) ? -1.f : 1.f;
                float4 na = u_cur[nb2], nb = u_cur[nb2 + 1];
                wa.x+=sgn*na.x; wa.y+=sgn*na.y; wa.z+=sgn*na.z; wa.w+=sgn*na.w;
                wb.x+=sgn*nb.x; wb.y+=sgn*nb.y; wb.z+=sgn*nb.z; wb.w+=sgn*nb.w;
            }
            ga.x+=w_scale*wa.x; ga.y+=w_scale*wa.y; ga.z+=w_scale*wa.z; ga.w+=w_scale*wa.w;
            gb.x+=w_scale*wb.x; gb.y+=w_scale*wb.y; gb.z+=w_scale*wb.z; gb.w+=w_scale*wb.w;

            int i2 = i * 2;
            float4 ua = u_cur[i2], ub = u_cur[i2 + 1];
            float dot = ga.x*ua.x+ga.y*ua.y+ga.z*ua.z+ga.w*ua.w+gb.x*ub.x+gb.y*ub.y+gb.z*ub.z+gb.w*ub.w;
            float4 ma = mom[i2], mb = mom[i2 + 1];
            ma.x=mom_beta*ma.x-lr*(ga.x-dot*ua.x); ma.y=mom_beta*ma.y-lr*(ga.y-dot*ua.y);
            ma.z=mom_beta*ma.z-lr*(ga.z-dot*ua.z); ma.w=mom_beta*ma.w-lr*(ga.w-dot*ua.w);
            mb.x=mom_beta*mb.x-lr*(gb.x-dot*ub.x); mb.y=mom_beta*mb.y-lr*(gb.y-dot*ub.y);
            mb.z=mom_beta*mb.z-lr*(gb.z-dot*ub.z); mb.w=mom_beta*mb.w-lr*(gb.w-dot*ub.w);
            mom[i2] = ma; mom[i2 + 1] = mb;

            float4 nva = make_float4(ua.x+ma.x, ua.y+ma.y, ua.z+ma.z, ua.w+ma.w);
            float4 nvb = make_float4(ub.x+mb.x, ub.y+mb.y, ub.z+mb.z, ub.w+mb.w);
            float inv = rsqrtf(nva.x*nva.x+nva.y*nva.y+nva.z*nva.z+nva.w*nva.w+nvb.x*nvb.x+nvb.y*nvb.y+nvb.z*nvb.z+nvb.w*nvb.w+1e-12f);
            u_nxt[i2]   = make_float4(nva.x*inv, nva.y*inv, nva.z*inv, nva.w*inv);
            u_nxt[i2+1] = make_float4(nvb.x*inv, nvb.y*inv, nvb.z*inv, nvb.w*inv);
        }
        __syncthreads();
        float4* tmp = u_cur; u_cur = u_nxt; u_nxt = tmp;
    }

    if ((steps & 1) != 0) {
        float4* dst = u_a_all + (size_t)out_idx * max_n * 2;
        for (int idx = tid; idx < n * 2; idx += 256) dst[idx] = u_cur[idx];
        __syncthreads();
    }

    float local_e = 0.f;
    for (int t = tid; t < nt; t += 256) {
        int t0=(int)tiles[t*4+0]*2, t1=(int)tiles[t*4+1]*2, t2=(int)tiles[t*4+2]*2, t3=(int)tiles[t*4+3]*2;
        float g0=(float)gauges[t*4+0], g1=(float)gauges[t*4+1], g2=(float)gauges[t*4+2], g3=(float)gauges[t*4+3];
        float4 a0=u_cur[t0], a1=u_cur[t1], a2=u_cur[t2], a3=u_cur[t3];
        float4 b0=u_cur[t0+1], b1=u_cur[t1+1], b2=u_cur[t2+1], b3=u_cur[t3+1];
        float s0=g0*a0.x+g1*a1.x+g2*a2.x+g3*a3.x, s1=g0*a0.y+g1*a1.y+g2*a2.y+g3*a3.y;
        float s2=g0*a0.z+g1*a1.z+g2*a2.z+g3*a3.z, s3=g0*a0.w+g1*a1.w+g2*a2.w+g3*a3.w;
        float s4=g0*b0.x+g1*b1.x+g2*b2.x+g3*b3.x, s5=g0*b0.y+g1*b1.y+g2*b2.y+g3*b3.y;
        float s6=g0*b0.z+g1*b1.z+g2*b2.z+g3*b3.z, s7=g0*b0.w+g1*b1.w+g2*b2.w+g3*b3.w;
        local_e += 0.5f * (s0*s0+s1*s1+s2*s2+s3*s3+s4*s4+s5*s5+s6*s6+s7*s7 - 4.f);
    }
    float local_w = 0.f;
    const float w_unit = weak_milli * 0.001f;
    for (int i = tid; i < n; i += 256) {
        float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
        int wp0 = woff[i], wp1 = woff[i + 1];
        for (int p = wp0; p < wp1; p++) {
            unsigned short pk = wnb_pk[p];
            int nb2 = (int)(pk & 0x7FFFu) * 2;
            float sgn = (pk & 0x8000u) ? -1.f : 1.f;
            float4 na = u_cur[nb2], nb = u_cur[nb2 + 1];
            a0+=sgn*na.x; a1+=sgn*na.y; a2+=sgn*na.z; a3+=sgn*na.w;
            b0+=sgn*nb.x; b1+=sgn*nb.y; b2+=sgn*nb.z; b3+=sgn*nb.w;
        }
        float4 ua = u_cur[i*2], ub = u_cur[i*2+1];
        local_w += w_unit * (ua.x*a0+ua.y*a1+ua.z*a2+ua.w*a3+ub.x*b0+ub.y*b1+ub.z*b2+ub.w*b3);
    }
    __shared__ double ssum[256];
    ssum[tid] = (double)local_e + 0.5 * (double)local_w;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_energy[out_idx] = (float)((double)base_const_arr[b] + ssum[0]);
}
"#;

#[derive(Clone)]
pub struct Bm8GpuInput {
    pub n_super: i32,
    pub n_tiles: i32,
    pub base_const: f32,
    pub weak_milli: f32,
    pub seed: u64,
    pub tiles: Vec<u16>,
    pub gauges: Vec<i8>,
    pub weak_off: Vec<i32>,
    pub weak_nb_packed: Vec<u16>,
}

pub struct GpuTopology {
    pub nodes: i32,
    pub m_edges: i32,
    pub n_cand: i32,
    pub n_fixed: i32,
    pub n_left: i32,
    pub n_right: i32,
    pub n_lent: i32,
    pub pairs: i32,
    pub weak_milli: i32,
    pub gen_smem_bytes: u32,
    d_edges_uv: CudaSlice<u32>,
    d_cand_nodes: CudaSlice<u16>,
    d_cand_eids: CudaSlice<u16>,
    d_fixed_cids: CudaSlice<u16>,
    d_left_off: CudaSlice<u16>,
    d_left_ent_r: CudaSlice<u16>,
    d_left_ent_cid: CudaSlice<u16>,
    d_div_limit: CudaSlice<u64>,
    d_div_inv: CudaSlice<u64>,
}

impl GpuTopology {
    pub fn upload(stream: &Arc<CudaStream>, fd: &FastDimer) -> Result<Self, String> {
        if fd.nodes > MAX_NODES
            || fd.edges.len() > MAX_EDGES
            || fd.cand_tiles.len() > MAX_CAND
            || fd.n_left > MAX_SIDE
            || fd.n_right > MAX_SIDE
            || fd.left_entries.len() > MAX_LENT
        {
            return Err(format!(
                "GpuTopology shape overflow: nodes={} edges={} cand={} left={} right={} lent={}",
                fd.nodes, fd.edges.len(), fd.cand_tiles.len(), fd.n_left, fd.n_right, fd.left_entries.len()
            ));
        }
        let edges_uv: Vec<u32> = fd.edges.iter().map(|&(u, v)| (u as u32) | ((v as u32) << 16)).collect();
        let mut cand_nodes = Vec::with_capacity(fd.cand_tiles.len() * 4);
        let mut cand_eids = Vec::with_capacity(fd.cand_tiles.len() * 6);
        for pt in &fd.cand_tiles {
            for &x in &pt.nodes { cand_nodes.push(x as u16); }
            for &x in &pt.eids { cand_eids.push(x as u16); }
        }
        let fixed_cids: Vec<u16> = fd.fixed_cids.iter().map(|&x| x as u16).collect();
        let left_off: Vec<u16> = fd.left_off.iter().map(|&x| x as u16).collect();
        let left_ent_r: Vec<u16> = fd.left_entries.iter().map(|&(r, _)| r as u16).collect();
        let left_ent_cid: Vec<u16> = fd.left_entries.iter().map(|&(_, c)| c as u16).collect();
        let div_limit: Vec<u64> = fd.div_table.iter().map(|d| d.limit).collect();
        let div_inv: Vec<u64> = fd.div_table.iter().map(|d| d.inv).collect();

        let a16 = |x: usize| (x + 15) & !15;
        let mut p1 = 1248usize;
        p1 += a16(fd.n_left * 2);
        p1 += a16(fd.n_right * 2);
        p1 += a16(fd.n_right * 2);
        p1 += a16(fd.n_right * 2);
        p1 += 16;
        let p1_l2_u16 = fd.left_entries.len() * 2 + fd.n_left * 2 + fd.n_left.max(fd.n_right);
        if p1_l2_u16 > MAX_SMEM_ORDER {
            return Err(format!("p1_l2_u16={p1_l2_u16} exceeds MAX_SMEM_ORDER={MAX_SMEM_ORDER}"));
        }
        let gen_smem_bytes = (p1 as u32).max(SMEM_GEN_BYTES);
        if gen_smem_bytes > SMEM_GEN_MAX_BYTES {
            return Err(format!("gen_smem_bytes={gen_smem_bytes} exceeds {SMEM_GEN_MAX_BYTES}"));
        }
        eprintln!(
            "  GpuTopology: n_left={} n_right={} n_lent={} p1_smem={}B -> launch gen_smem={}B (6 blocks/SM!)",
            fd.n_left, fd.n_right, fd.left_entries.len(), p1, gen_smem_bytes
        );

        Ok(Self {
            nodes: fd.nodes as i32,
            m_edges: fd.edges.len() as i32,
            n_cand: fd.cand_tiles.len() as i32,
            n_fixed: fd.fixed_cids.len() as i32,
            n_left: fd.n_left as i32,
            n_right: fd.n_right as i32,
            n_lent: fd.left_entries.len() as i32,
            pairs: fd.pairs as i32,
            weak_milli: fd.weak,
            gen_smem_bytes,
            d_edges_uv: stream.clone_htod(&edges_uv).map_err(|e| e.to_string())?,
            d_cand_nodes: stream.clone_htod(&cand_nodes).map_err(|e| e.to_string())?,
            d_cand_eids: stream.clone_htod(&cand_eids).map_err(|e| e.to_string())?,
            d_fixed_cids: stream.clone_htod(&fixed_cids).map_err(|e| e.to_string())?,
            d_left_off: stream.clone_htod(&left_off).map_err(|e| e.to_string())?,
            d_left_ent_r: stream.clone_htod(&left_ent_r).map_err(|e| e.to_string())?,
            d_left_ent_cid: stream.clone_htod(&left_ent_cid).map_err(|e| e.to_string())?,
            d_div_limit: stream.clone_htod(&div_limit).map_err(|e| e.to_string())?,
            d_div_inv: stream.clone_htod(&div_inv).map_err(|e| e.to_string())?,
        })
    }
}

pub struct GpuSharedRuntime {
    pub ctx: Arc<CudaContext>,
    pub arch: i32,
    f_gen: CudaFunction,
    f_gen_1tps: CudaFunction,
    f_bm2: CudaFunction,
    f_bm4: CudaFunction,
    f_bm8: CudaFunction,
    f_w30: CudaFunction,
    topos: std::sync::Mutex<HashMap<([u8; 32], [u8; 32]), Arc<GpuTopology>>>,
}

impl GpuSharedRuntime {
    pub fn init(device: usize) -> Result<Arc<Self>, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("CudaContext::new({device}): {e}"))?;
        let _ = ctx.set_blocking_synchronize();
        unsafe { ctx.disable_event_tracking(); }
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("cc major: {e}"))?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("cc minor: {e}"))?;
        let arch = major * 10 + minor;

        let ptx = compile_ptx_with_opts(
            GPU_FEEDER_V2_CUDA_SRC,
            CompileOptions {
                options: vec![
                    format!("--gpu-architecture=compute_{arch}"),
                    "--use_fast_math".into(),
                ],
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc compile (sm_{arch}): {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {e}"))?;
        let f_gen = module.load_function("gpu_generate_dimer_batch").map_err(|e| e.to_string())?;
        let f_gen_1tps = module.load_function("gpu_dimer_1tps").map_err(|e| e.to_string())?;
        let f_bm2 = module.load_function("bm2_batch_sdp_smem_fp16").map_err(|e| e.to_string())?;
        let f_bm4 = module.load_function("bm4_batch_sdp_smem_fp16").map_err(|e| e.to_string())?;
        let f_bm8 = module.load_function("bm8_batch_sdp_smem_fp16").map_err(|e| e.to_string())?;
        let f_w30 = module.load_function("bm8_batch_sdp_fp32_indexed").map_err(|e| e.to_string())?;

        f_gen
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_GEN_MAX_BYTES as i32,
            )
            .map_err(|e| format!("set_attr gen smem={SMEM_GEN_MAX_BYTES}: {e}"))?;
        f_bm2
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_BM2_BYTES as i32,
            )
            .map_err(|e| format!("set_attr bm2 smem={SMEM_BM2_BYTES}: {e}"))?;
        f_bm4
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_BM4_BYTES as i32,
            )
            .map_err(|e| format!("set_attr bm4 smem={SMEM_BM4_BYTES}: {e}"))?;
        f_bm8
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_BM8_BYTES as i32,
            )
            .map_err(|e| format!("set_attr bm8 smem={SMEM_BM8_BYTES}: {e}"))?;

        eprintln!(
            "quip-gpu-feeder v5.0: CUDA sm_{arch} ready (1-Thread-Per-Salt L2-FastDimer + 4-blk/SM BM2[{SMEM_BM2_BYTES}B] + f16x2-SIMD BM4/BM8 + Lazy-DtoH BM8_w30)"
        );
        Ok(Arc::new(Self {
            ctx,
            arch,
            f_gen,
            f_gen_1tps,
            f_bm2,
            f_bm4,
            f_bm8,
            f_w30,
            topos: std::sync::Mutex::new(HashMap::new()),
        }))
    }

    pub fn get_or_upload_topo(
        &self,
        stream: &Arc<CudaStream>,
        fd: &FastDimer,
    ) -> Result<Arc<GpuTopology>, String> {
        let key = (fd.order_digest, fd.descriptor_digest);
        if let Some(t) = self.topos.lock().unwrap().get(&key).cloned() {
            return Ok(t);
        }
        let built = Arc::new(GpuTopology::upload(stream, fd)?);
        let mut map = self.topos.lock().unwrap();
        Ok(map.entry(key).or_insert_with(|| Arc::clone(&built)).clone())
    }
}

/// One independent GPU pipeline arena (1 CUDA stream, ~290 MB VRAM for max_batch=512).
/// Run 2 arenas on 2 dedicated worker threads to overlap CPU/PCIe/GPU seamlessly!
pub struct GpuArena {
    pub max_batch: usize,
    pub max_w30: usize,
    rt: Arc<GpuSharedRuntime>,
    pub stream: Arc<CudaStream>,
    // Tiny host pinned staging (keys=80KB, seeds=4KB, slots=2KB, out=2KB)
    h_keys: PinnedHostSlice<u32>,
    h_seeds: PinnedHostSlice<u64>,
    h_slots: PinnedHostSlice<i32>,
    h_out: PinnedHostSlice<f32>,
    h_n: PinnedHostSlice<i32>,
    // Device inputs
    d_keys: CudaSlice<u32>,
    d_seeds: CudaSlice<u64>,
    d_slots: CudaSlice<i32>,
    // Device generated graph arena + L2-resident d_order & 1tps persistent scratch
    d_order: CudaSlice<u16>,
    d_1tps_scratch: CudaSlice<u32>,
    d_n: CudaSlice<i32>,
    d_nt: CudaSlice<i32>,
    d_base: CudaSlice<f32>,
    d_tiles: CudaSlice<u16>,
    d_gauges: CudaSlice<i8>,
    d_woff: CudaSlice<i32>,
    d_wnb_pk: CudaSlice<u16>,
    d_toff: CudaSlice<i32>,
    d_inc_tid: CudaSlice<u16>,
    d_inc_g: CudaSlice<i8>,
    // Device BM8 SMEM-FP16 momentum/unxt scratch (uint4 = 4*u32)
    d_mom_h: CudaSlice<u32>,
    d_unxt_h: CudaSlice<u32>,
    // Device BM8_w30 exact FP32 scratch (compact: max_w30 slots!)
    d_w30_ua: CudaSlice<f32>,
    d_w30_ub: CudaSlice<f32>,
    d_w30_mom: CudaSlice<f32>,
    d_w30_stile: CudaSlice<f32>,
    d_out: CudaSlice<f32>,
}

impl GpuArena {
    pub fn new(rt: Arc<GpuSharedRuntime>, max_batch: usize, max_w30: usize) -> Result<Self, String> {
        let stream = rt.ctx.new_stream().map_err(|e| e.to_string())?;
        let ctx = &rt.ctx;
        let b = max_batch;
        let w = max_w30.min(max_batch);

        Ok(Self {
            max_batch: b,
            max_w30: w,
            rt: Arc::clone(&rt),
            h_keys: unsafe { ctx.alloc_pinned::<u32>(b * 40) }.map_err(|e| e.to_string())?,
            h_seeds: unsafe { ctx.alloc_pinned::<u64>(b) }.map_err(|e| e.to_string())?,
            h_slots: unsafe { ctx.alloc_pinned::<i32>(b) }.map_err(|e| e.to_string())?,
            h_out: unsafe { ctx.alloc_pinned::<f32>(b) }.map_err(|e| e.to_string())?,
            h_n: unsafe { ctx.alloc_pinned::<i32>(b) }.map_err(|e| e.to_string())?,
            d_keys: stream.alloc_zeros::<u32>(b * 40).map_err(|e| e.to_string())?,
            d_seeds: stream.alloc_zeros::<u64>(b).map_err(|e| e.to_string())?,
            d_slots: stream.alloc_zeros::<i32>(b).map_err(|e| e.to_string())?,
            d_order: stream.alloc_zeros::<u16>(b * MAX_SMEM_ORDER).map_err(|e| e.to_string())?,
            d_1tps_scratch: stream.alloc_zeros::<u32>(MAX_1TPS_WORKERS * SCRATCH_1TPS_U32).map_err(|e| e.to_string())?,
            d_n: stream.alloc_zeros::<i32>(b).map_err(|e| e.to_string())?,
            d_nt: stream.alloc_zeros::<i32>(b).map_err(|e| e.to_string())?,
            d_base: stream.alloc_zeros::<f32>(b).map_err(|e| e.to_string())?,
            d_tiles: stream.alloc_zeros::<u16>(b * MAX_TILES * 4).map_err(|e| e.to_string())?,
            d_gauges: stream.alloc_zeros::<i8>(b * MAX_TILES * 4).map_err(|e| e.to_string())?,
            d_woff: stream.alloc_zeros::<i32>(b * (MAX_N + 1)).map_err(|e| e.to_string())?,
            d_wnb_pk: stream.alloc_zeros::<u16>(b * MAX_WEAK_DIR).map_err(|e| e.to_string())?,
            d_toff: stream.alloc_zeros::<i32>(b * (MAX_N + 1)).map_err(|e| e.to_string())?,
            d_inc_tid: stream.alloc_zeros::<u16>(b * MAX_INC).map_err(|e| e.to_string())?,
            d_inc_g: stream.alloc_zeros::<i8>(b * MAX_INC).map_err(|e| e.to_string())?,
            d_mom_h: stream.alloc_zeros::<u32>(b * MAX_N * 4).map_err(|e| e.to_string())?,
            d_unxt_h: stream.alloc_zeros::<u32>(b * MAX_N * 4).map_err(|e| e.to_string())?,
            d_w30_ua: stream.alloc_zeros::<f32>(w * MAX_N * 8).map_err(|e| e.to_string())?,
            d_w30_ub: stream.alloc_zeros::<f32>(w * MAX_N * 8).map_err(|e| e.to_string())?,
            d_w30_mom: stream.alloc_zeros::<f32>(w * MAX_N * 8).map_err(|e| e.to_string())?,
            d_w30_stile: stream.alloc_zeros::<f32>(w * MAX_TILES * 8).map_err(|e| e.to_string())?,
            d_out: stream.alloc_zeros::<f32>(b).map_err(|e| e.to_string())?,
            stream,
        })
    }

    /// Generate `bsz` salts directly on GPU from their 5 ChaCha8 keys (`bsz * 40` u32s).
    /// Returns `n_super` per slot (`-1` means `failed_fast`).
    pub fn generate_batch(
        &mut self,
        topo: &GpuTopology,
        keys_flat: &[u32],
        seeds: &[u64],
    ) -> Result<Vec<i32>, String> {
        let bsz = seeds.len();
        assert!(bsz <= self.max_batch && keys_flat.len() == bsz * 40);
        self.h_keys.as_mut_slice().map_err(|e| e.to_string())?[..bsz * 40].copy_from_slice(keys_flat);
        self.h_seeds.as_mut_slice().map_err(|e| e.to_string())?[..bsz].copy_from_slice(seeds);
        self.stream
            .memcpy_htod(&self.h_keys.as_slice().map_err(|e| e.to_string())?[..bsz * 40], &mut self.d_keys.slice_mut(..bsz * 40))
            .map_err(|e| e.to_string())?;
        self.stream
            .memcpy_htod(&self.h_seeds.as_slice().map_err(|e| e.to_string())?[..bsz], &mut self.d_seeds.slice_mut(..bsz))
            .map_err(|e| e.to_string())?;

        let bsz_i = bsz as i32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;
        let max_smem_order = MAX_SMEM_ORDER as i32;

        let cfg = LaunchConfig {
            grid_dim: (bsz as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: topo.gen_smem_bytes,
        };
        {
            let mut l = self.stream.launch_builder(&self.rt.f_gen);
            l.arg(&bsz_i)
                .arg(&topo.nodes)
                .arg(&topo.m_edges)
                .arg(&topo.n_cand)
                .arg(&topo.n_fixed)
                .arg(&topo.n_left)
                .arg(&topo.n_right)
                .arg(&topo.n_lent)
                .arg(&topo.pairs)
                .arg(&topo.weak_milli)
                .arg(&topo.d_edges_uv)
                .arg(&topo.d_cand_nodes)
                .arg(&topo.d_cand_eids)
                .arg(&topo.d_fixed_cids)
                .arg(&topo.d_left_off)
                .arg(&topo.d_left_ent_r)
                .arg(&topo.d_left_ent_cid)
                .arg(&topo.d_div_limit)
                .arg(&topo.d_div_inv)
                .arg(&self.d_keys)
                .arg(&mut self.d_order)
                .arg(&mut self.d_n)
                .arg(&mut self.d_nt)
                .arg(&mut self.d_base)
                .arg(&mut self.d_tiles)
                .arg(&mut self.d_gauges)
                .arg(&mut self.d_woff)
                .arg(&mut self.d_wnb_pk)
                .arg(&max_n)
                .arg(&max_tiles)
                .arg(&max_weak_dir)
                .arg(&max_smem_order);
            unsafe { l.launch(cfg) }.map_err(|e| format!("gen launch: {e}"))?;
        }
        {
            let h_n = self.h_n.as_mut_slice().map_err(|e| e.to_string())?;
            self.stream.memcpy_dtoh(&self.d_n.slice(..bsz), &mut h_n[..bsz]).map_err(|e| e.to_string())?;
        }
        self.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(self.h_n.as_slice().map_err(|e| e.to_string())?[..bsz].to_vec())
    }

    /// Generate `bsz` salts using persistent L2-resident 1-thread-per-salt FastDimer (v5.0)!
    /// Also builds `d_toff/d_inc_tid/d_inc_g` non-atomically -> BM2/BM4/BM8 skip build_tile_csr!
    pub fn generate_batch_1tps(
        &mut self,
        topo: &GpuTopology,
        keys_flat: &[u32],
        seeds: &[u64],
        num_workers: usize,
    ) -> Result<Vec<i32>, String> {
        let bsz = seeds.len();
        assert!(bsz <= self.max_batch && keys_flat.len() == bsz * 40);
        self.h_keys.as_mut_slice().map_err(|e| e.to_string())?[..bsz * 40].copy_from_slice(keys_flat);
        self.h_seeds.as_mut_slice().map_err(|e| e.to_string())?[..bsz].copy_from_slice(seeds);
        self.stream
            .memcpy_htod(&self.h_keys.as_slice().map_err(|e| e.to_string())?[..bsz * 40], &mut self.d_keys.slice_mut(..bsz * 40))
            .map_err(|e| e.to_string())?;
        self.stream
            .memcpy_htod(&self.h_seeds.as_slice().map_err(|e| e.to_string())?[..bsz], &mut self.d_seeds.slice_mut(..bsz))
            .map_err(|e| e.to_string())?;

        let bsz_i = bsz as i32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_inc = MAX_INC as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;
        let max_smem_order = MAX_SMEM_ORDER as i32;

        let workers = num_workers.clamp(32, MAX_1TPS_WORKERS).min(bsz.max(32));
        let block_x = 32u32;
        let grid_x = ((workers as u32) + block_x - 1) / block_x;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut l = self.stream.launch_builder(&self.rt.f_gen_1tps);
            l.arg(&bsz_i)
                .arg(&topo.nodes)
                .arg(&topo.m_edges)
                .arg(&topo.n_cand)
                .arg(&topo.n_fixed)
                .arg(&topo.n_left)
                .arg(&topo.n_right)
                .arg(&topo.n_lent)
                .arg(&topo.pairs)
                .arg(&topo.weak_milli)
                .arg(&topo.d_edges_uv)
                .arg(&topo.d_cand_nodes)
                .arg(&topo.d_cand_eids)
                .arg(&topo.d_fixed_cids)
                .arg(&topo.d_left_off)
                .arg(&topo.d_left_ent_r)
                .arg(&topo.d_left_ent_cid)
                .arg(&topo.d_div_limit)
                .arg(&topo.d_div_inv)
                .arg(&self.d_keys)
                .arg(&mut self.d_1tps_scratch)
                .arg(&mut self.d_n)
                .arg(&mut self.d_nt)
                .arg(&mut self.d_base)
                .arg(&mut self.d_tiles)
                .arg(&mut self.d_gauges)
                .arg(&mut self.d_woff)
                .arg(&mut self.d_wnb_pk)
                .arg(&mut self.d_toff)
                .arg(&mut self.d_inc_tid)
                .arg(&mut self.d_inc_g)
                .arg(&max_n)
                .arg(&max_tiles)
                .arg(&max_inc)
                .arg(&max_weak_dir)
                .arg(&max_smem_order);
            unsafe { l.launch(cfg) }.map_err(|e| format!("gen_1tps launch: {e}"))?;
        }
        {
            let h_n = self.h_n.as_mut_slice().map_err(|e| e.to_string())?;
            self.stream.memcpy_dtoh(&self.d_n.slice(..bsz), &mut h_n[..bsz]).map_err(|e| e.to_string())?;
        }
        self.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(self.h_n.as_slice().map_err(|e| e.to_string())?[..bsz].to_vec())
    }

    /// Run 4-block/SM Rank-2 SMEM-FP16 BM2 pre-filter on all `bsz` slots.
    pub fn run_bm2_all(
        &mut self,
        bsz: usize,
        skip_build_csr: bool,
        weak_milli: f32,
        steps: i32,
        lr0: f32,
    ) -> Result<Vec<f32>, String> {
        let null_ptr: u64 = 0;
        let skip_i: i32 = if skip_build_csr { 1 } else { 0 };
        let sw0 = 0.20f32;
        let sw1 = 1.00f32;
        let weak_mul = 2.0f32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_inc = MAX_INC as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;
        let cfg = LaunchConfig {
            grid_dim: (bsz as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: SMEM_BM2_BYTES,
        };
        {
            let mut l = self.stream.launch_builder(&self.rt.f_bm2);
            l.arg(&null_ptr)
                .arg(&skip_i)
                .arg(&self.d_n)
                .arg(&self.d_nt)
                .arg(&self.d_base)
                .arg(&weak_milli)
                .arg(&self.d_seeds)
                .arg(&self.d_tiles)
                .arg(&self.d_gauges)
                .arg(&self.d_woff)
                .arg(&self.d_wnb_pk)
                .arg(&mut self.d_toff)
                .arg(&mut self.d_inc_tid)
                .arg(&mut self.d_inc_g)
                .arg(&mut self.d_out)
                .arg(&steps)
                .arg(&sw0)
                .arg(&sw1)
                .arg(&weak_mul)
                .arg(&lr0)
                .arg(&max_n)
                .arg(&max_tiles)
                .arg(&max_inc)
                .arg(&max_weak_dir);
            unsafe { l.launch(cfg) }.map_err(|e| format!("bm2 launch: {e}"))?;
        }
        {
            let h_out = self.h_out.as_mut_slice().map_err(|e| e.to_string())?;
            self.stream.memcpy_dtoh(&self.d_out.slice(..bsz), &mut h_out[..bsz]).map_err(|e| e.to_string())?;
        }
        self.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(self.h_out.as_slice().map_err(|e| e.to_string())?[..bsz].to_vec())
    }

    /// Run SMEM-FP16 BM4 on all `bsz` slots or a filtered subset `slots`.
    pub fn run_bm4_smem(
        &mut self,
        slots: Option<&[i32]>,
        bsz_all: usize,
        skip_build_csr: bool,
        weak_milli: f32,
        steps: i32,
        lr0: f32,
    ) -> Result<Vec<f32>, String> {
        let count = match slots {
            Some(s) => {
                if s.is_empty() { return Ok(Vec::new()); }
                self.h_slots.as_mut_slice().map_err(|e| e.to_string())?[..s.len()].copy_from_slice(s);
                self.stream
                    .memcpy_htod(&self.h_slots.as_slice().map_err(|e| e.to_string())?[..s.len()], &mut self.d_slots.slice_mut(..s.len()))
                    .map_err(|e| e.to_string())?;
                s.len()
            }
            None => bsz_all,
        };
        let null_ptr: u64 = 0;
        let skip_i: i32 = if skip_build_csr { 1 } else { 0 };
        let sw0 = 0.20f32;
        let sw1 = 1.00f32;
        let weak_mul = 2.0f32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_inc = MAX_INC as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;
        let cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: SMEM_BM4_BYTES,
        };
        {
            let mut l = self.stream.launch_builder(&self.rt.f_bm4);
            if slots.is_some() {
                l.arg(&self.d_slots);
            } else {
                l.arg(&null_ptr);
            }
            l.arg(&skip_i)
                .arg(&self.d_n)
                .arg(&self.d_nt)
                .arg(&self.d_base)
                .arg(&weak_milli)
                .arg(&self.d_seeds)
                .arg(&self.d_tiles)
                .arg(&self.d_gauges)
                .arg(&self.d_woff)
                .arg(&self.d_wnb_pk)
                .arg(&mut self.d_toff)
                .arg(&mut self.d_inc_tid)
                .arg(&mut self.d_inc_g)
                .arg(&mut self.d_out)
                .arg(&steps)
                .arg(&sw0)
                .arg(&sw1)
                .arg(&weak_mul)
                .arg(&lr0)
                .arg(&max_n)
                .arg(&max_tiles)
                .arg(&max_inc)
                .arg(&max_weak_dir);
            unsafe { l.launch(cfg) }.map_err(|e| format!("bm4 launch: {e}"))?;
        }
        {
            let h_out = self.h_out.as_mut_slice().map_err(|e| e.to_string())?;
            self.stream.memcpy_dtoh(&self.d_out.slice(..count), &mut h_out[..count]).map_err(|e| e.to_string())?;
        }
        self.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(self.h_out.as_slice().map_err(|e| e.to_string())?[..count].to_vec())
    }

    /// Run SMEM-FP16 BM4 on all `bsz` generated slots (back-compat wrapper).
    pub fn run_bm4_all(
        &mut self,
        bsz: usize,
        weak_milli: f32,
        steps: i32,
        lr0: f32,
    ) -> Result<Vec<f32>, String> {
        self.run_bm4_smem(None, bsz, false, weak_milli, steps, lr0)
    }

    /// Run SMEM-FP16 BM8 on either all `bsz` slots (`slots = None`) or a filtered subset (`slots = Some(&[..])`).
    pub fn run_bm8_smem(
        &mut self,
        slots: Option<&[i32]>,
        bsz_all: usize,
        skip_build_csr: bool,
        weak_milli: f32,
        steps: i32,
        lr0: f32,
    ) -> Result<Vec<f32>, String> {
        let count = match slots {
            Some(s) => {
                if s.is_empty() { return Ok(Vec::new()); }
                self.h_slots.as_mut_slice().map_err(|e| e.to_string())?[..s.len()].copy_from_slice(s);
                self.stream
                    .memcpy_htod(&self.h_slots.as_slice().map_err(|e| e.to_string())?[..s.len()], &mut self.d_slots.slice_mut(..s.len()))
                    .map_err(|e| e.to_string())?;
                s.len()
            }
            None => bsz_all,
        };
        let null_ptr: u64 = 0;
        let skip_i: i32 = if skip_build_csr { 1 } else { 0 };
        let sw0 = 0.20f32;
        let sw1 = 1.00f32;
        let weak_mul = 2.0f32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_inc = MAX_INC as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;
        let cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: SMEM_BM8_BYTES,
        };
        {
            let mut l = self.stream.launch_builder(&self.rt.f_bm8);
            if slots.is_some() {
                l.arg(&self.d_slots);
            } else {
                l.arg(&null_ptr);
            }
            l.arg(&skip_i)
                .arg(&self.d_n)
                .arg(&self.d_nt)
                .arg(&self.d_base)
                .arg(&weak_milli)
                .arg(&self.d_seeds)
                .arg(&self.d_tiles)
                .arg(&self.d_gauges)
                .arg(&self.d_woff)
                .arg(&self.d_wnb_pk)
                .arg(&mut self.d_toff)
                .arg(&mut self.d_inc_tid)
                .arg(&mut self.d_inc_g)
                .arg(&mut self.d_mom_h)
                .arg(&mut self.d_unxt_h)
                .arg(&mut self.d_out)
                .arg(&steps)
                .arg(&sw0)
                .arg(&sw1)
                .arg(&weak_mul)
                .arg(&lr0)
                .arg(&max_n)
                .arg(&max_tiles)
                .arg(&max_inc)
                .arg(&max_weak_dir);
            unsafe { l.launch(cfg) }.map_err(|e| format!("bm8 smem launch: {e}"))?;
        }
        {
            let h_out = self.h_out.as_mut_slice().map_err(|e| e.to_string())?;
            self.stream.memcpy_dtoh(&self.d_out.slice(..count), &mut h_out[..count]).map_err(|e| e.to_string())?;
        }
        self.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(self.h_out.as_slice().map_err(|e| e.to_string())?[..count].to_vec())
    }

    /// Run exact FP32 BM8_w30 (`steps=200, weak_mul=3.0, lr0=0.15, seed+29`) on surviving slots,
    /// reading back ONLY `energies` first and lazily copying `u_vec` ONLY for slots selected by `select_fn`
    /// (~1-2 survivors out of 41 -> 40x less PCIe DtoH + zero wasted allocations!).
    pub fn run_w30_fp32_filtered<F>(
        &mut self,
        slots: &[i32],
        n_per_slot: &[i32],
        weak_milli: f32,
        mut select_fn: F,
    ) -> Result<Vec<(i32, f32, Vec<[f32; 8]>)>, String>
    where
        F: FnMut(&[f32]) -> Vec<usize>,
    {
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = Vec::new();
        let seed_add: u64 = 29;
        let steps = 200i32;
        let sw0 = 0.20f32;
        let sw1 = 1.00f32;
        let weak_mul = 3.0f32;
        let lr0 = 0.15f32;
        let max_n = MAX_N as i32;
        let max_tiles = MAX_TILES as i32;
        let max_inc = MAX_INC as i32;
        let max_weak_dir = MAX_WEAK_DIR as i32;

        for chunk in slots.chunks(self.max_w30) {
            let k = chunk.len();
            self.h_slots.as_mut_slice().map_err(|e| e.to_string())?[..k].copy_from_slice(chunk);
            self.stream
                .memcpy_htod(&self.h_slots.as_slice().map_err(|e| e.to_string())?[..k], &mut self.d_slots.slice_mut(..k))
                .map_err(|e| e.to_string())?;
            let cfg = LaunchConfig {
                grid_dim: (k as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            {
                let mut l = self.stream.launch_builder(&self.rt.f_w30);
                l.arg(&self.d_slots)
                    .arg(&self.d_n)
                    .arg(&self.d_nt)
                    .arg(&self.d_base)
                    .arg(&weak_milli)
                    .arg(&self.d_seeds)
                    .arg(&seed_add)
                    .arg(&self.d_tiles)
                    .arg(&self.d_gauges)
                    .arg(&self.d_woff)
                    .arg(&self.d_wnb_pk)
                    .arg(&self.d_toff)
                    .arg(&self.d_inc_tid)
                    .arg(&self.d_inc_g)
                    .arg(&mut self.d_w30_ua)
                    .arg(&mut self.d_w30_ub)
                    .arg(&mut self.d_w30_mom)
                    .arg(&mut self.d_w30_stile)
                    .arg(&mut self.d_out)
                    .arg(&steps)
                    .arg(&sw0)
                    .arg(&sw1)
                    .arg(&weak_mul)
                    .arg(&lr0)
                    .arg(&max_n)
                    .arg(&max_tiles)
                    .arg(&max_inc)
                    .arg(&max_weak_dir);
                unsafe { l.launch(cfg) }.map_err(|e| format!("w30 launch: {e}"))?;
            }
            {
                let h_out = self.h_out.as_mut_slice().map_err(|e| e.to_string())?;
                self.stream.memcpy_dtoh(&self.d_out.slice(..k), &mut h_out[..k]).map_err(|e| e.to_string())?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())?;
            let energies = self.h_out.as_slice().map_err(|e| e.to_string())?[..k].to_vec();
            let chosen = select_fn(&energies);
            for j in chosen {
                if j >= k { continue; }
                let slot = chunk[j];
                let n = n_per_slot[slot as usize] as usize;
                let off = j * MAX_N * 8;
                let u_flat = self
                    .stream
                    .clone_dtoh(&self.d_w30_ua.slice(off..off + n * 8))
                    .map_err(|e| e.to_string())?;
                let mut u_vec = Vec::with_capacity(n);
                for i in 0..n {
                    let p = i * 8;
                    u_vec.push([
                        u_flat[p], u_flat[p + 1], u_flat[p + 2], u_flat[p + 3],
                        u_flat[p + 4], u_flat[p + 5], u_flat[p + 6], u_flat[p + 7],
                    ]);
                }
                results.push((slot, energies[j], u_vec));
            }
        }
        Ok(results)
    }

    /// Run exact FP32 BM8_w30 returning all `(energy, u_vec)` (used by `--bench` verification).
    pub fn run_w30_fp32(
        &mut self,
        slots: &[i32],
        n_per_slot: &[i32],
        weak_milli: f32,
    ) -> Result<Vec<(f32, Vec<[f32; 8]>)>, String> {
        let all = self.run_w30_fp32_filtered(slots, n_per_slot, weak_milli, |es| (0..es.len()).collect())?;
        Ok(all.into_iter().map(|(_, e, u)| (e, u)).collect())
    }

    /// Debug helper: read back generated graph for slot `b` to verify 100% byte-exact match with CPU!
    pub fn readback_generated_slot(&self, b: usize) -> Result<Bm8GpuInput, String> {
        let n = self.stream.clone_dtoh(&self.d_n.slice(b..b + 1)).map_err(|e| e.to_string())?[0];
        let nt = self.stream.clone_dtoh(&self.d_nt.slice(b..b + 1)).map_err(|e| e.to_string())?[0];
        let base_const = self.stream.clone_dtoh(&self.d_base.slice(b..b + 1)).map_err(|e| e.to_string())?[0];
        let seed = self.stream.clone_dtoh(&self.d_seeds.slice(b..b + 1)).map_err(|e| e.to_string())?[0];
        let tb = b * MAX_TILES * 4;
        let tiles = self.stream.clone_dtoh(&self.d_tiles.slice(tb..tb + nt as usize * 4)).map_err(|e| e.to_string())?;
        let gauges = self.stream.clone_dtoh(&self.d_gauges.slice(tb..tb + nt as usize * 4)).map_err(|e| e.to_string())?;
        let ob = b * (MAX_N + 1);
        let weak_off = self.stream.clone_dtoh(&self.d_woff.slice(ob..ob + n as usize + 1)).map_err(|e| e.to_string())?;
        let nw = weak_off[n as usize] as usize;
        let wb = b * MAX_WEAK_DIR;
        let weak_nb_packed = self.stream.clone_dtoh(&self.d_wnb_pk.slice(wb..wb + nw)).map_err(|e| e.to_string())?;
        Ok(Bm8GpuInput {
            n_super: n,
            n_tiles: nt,
            base_const,
            weak_milli: 45.0,
            seed,
            tiles,
            gauges,
            weak_off,
            weak_nb_packed,
        })
    }
}
