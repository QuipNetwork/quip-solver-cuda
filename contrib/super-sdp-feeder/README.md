# Super-SDP Feeder: GPU-Native FastDimer + Burer-Monteiro SDP + Goemans-Williamson Screening for `quip/twisted-ice-dimer`

Written for: Quip miners, researchers, and maintainers who want to read, benchmark, or reuse a geometric-relaxation screening pipeline alongside downstream CUDA solvers (such as `native-two-pass` or `otsuka`) without repeating our experiments.

This directory is a **self-contained reference crate** (`contrib/super-sdp-feeder/`). It touches **zero upstream files** and builds standalone on top of `ising-generators` `0.0.15` and `cudarc` `0.19` (`cuda-12090` NVRTC pin, matching the repository contract). The full empirical record—including negative and null results—is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Core Idea

Instead of screening candidate PoW salts with short stochastic annealing or plain BP alone, this feeder explores a **continuous geometric relaxation + rounding** approach tailored to the `twisted-ice-dimer` factor structure:

1. **Zero-PCIe GPU-native instance generation (`gpu_generate_dimer_batch`)**:
   - The host computes only the 5 protocol ChaCha8 stream keys per salt (`160 B/salt`, ~950x less PCIe traffic than uploading CSR graphs) and generates full canonical instances bit-exact to CPU `FastDimer` directly in VRAM (16.1 KB dynamic SMEM -> up to 6 blocks/SM on sm_89).
2. **SMEM-FP16 Rank-4 Burer-Monteiro SDP pre-filter (`bm4_batch_sdp_smem_fp16`)**:
   - Keeps both `s_u` and `s_stile` in dynamic shared memory with momentum in registers (zero DRAM writes during inner iterations), cheaply discarding ~84% of weak salts.
3. **SMEM-FP16 Rank-8 Burer-Monteiro SDP filter (`bm8_batch_sdp_smem_fp16`)**:
   - Runs 60-step Rank-8 BM-SDP entirely in dynamic SMEM (90.4 KB < 99 KB limit on Ampere/Ada/Blackwell), retaining only the top ~1.2% of candidates.
4. **Indexed Exact FP32 Rank-8 BM-SDP (`bm8_batch_sdp_fp32_indexed`)**:
   - Runs 200-step exact FP32 Rank-8 BM-SDP (`w = 30`) solely on the ~1.2% surviving slots in VRAM and exports their continuous 8D embedding vectors `u_i` back to host.
5. **CPU `Super-Stage-0e` scoring + seed export (`stage_e_super`)**:
   - Single-pass 8x8 Jacobi eigendecomposition of the BM-SDP covariance matrix to obtain top spectral directions.
   - Combines:
     - `v3` reinforced K4-BP (4 eigenvector-seeded + 4 unseeded replicas x 120 updates),
     - `DeepBP` (`deep_reps = 4`, `deep_iters = 160` + 1-flip local polish, unscaled ramp schedule, top-3 mean), and
     - `GW` Goemans-Williamson hyperplane rounding (`gw_trials = 20`: 8 principal eigenvectors + 12 random hyperplanes + 1-flip polish, top-5 mean).
   - Emits top-ranked salt indices, composite scores, and quantized 8D `BmGpuSeed` payloads over the `QCF1` TCP binary protocol so downstream solvers can immediately warm-start from the SDP eigenvectors.

## Summary of Measured Results

Full ablation tables and methodology are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

- **Offline controlled A/B on 819 real Aglais nonces** (`twisted-ice-dimer`, `m=12, t=4, weak=45, pairs=1350`):
  - `v3fast` (previous Stage-0d production baseline): `AUC90 = 0.6865`, Top-50% recall `60/85 (70.6%)`.
  - **`Super-Stage-0e` (`4x160 + GW20`, unscaled ramp)**: **`AUC90 = 0.7705` (`+0.0840`)**, Top-50% recall **`73/85 (85.9%, +21.7% relative)`**.
- **Live 90-minute Aglais steady-state on 4x RTX 4090 (`2 Feeder : 2 Solver`, 56-core host)**:
  - Coordinator screening throughput: **`81.7k-85.7k salts/s`** (vs `72.0k salts/s` baseline, `+13.4%`).
  - Downstream solver consumption: **`20.9 salts/s`** (`gpu_wait = 0.0-0.1 ms/s`).
  - Deep-tail hit rate `<= -5093`: **`6.0/h`** (vs `4.9/h` baseline, `+22%`); `<= -5094`: `2.0/h`.

## Wire Protocol (`QCF1`)

`quip-super-sdp-feeder` runs as a lightweight TCP screening service consumed by one or more downstream solver workers:

- **Handshake**: Client sends `b"QCF1"` + `u16_le(token_len)` + `token` + `u16_le(name_len)` + `name`; server replies `b"OKAY"`.
- **Client -> Server opcodes**:
  - `OP_SET_TOPO (1)`: Registers `order_digest[32]`, `descriptor_digest[32]`, raw family params, node IDs, and edge pairs; server builds and caches `FastDimer` + per-topology adaptive rolling gates.
  - `OP_SUBMIT_CHUNK (2)`: Streams a lease nonce chunk (`lease_id`, `chunk_id`, `start_idx`, `count <= 8192`, `pool_cutoff`, `count x 32B nonces`).
  - `OP_CANCEL_LEASE (3)`: Cancels queued work for a finished/expired `lease_id`.
  - `OP_PING (4)`: Keep-alive (`OP_PONG (11)`).
- **Server -> Client opcode**:
  - `OP_CHUNK_RESULT (10)`: Returns surviving salt indices, composite scores, fast-fail flags, and optional quantized `BmGpuSeed` eigenvector payloads.

## Building & Running

```bash
cd contrib/super-sdp-feeder
cargo build --release
```

### 1. Bit-exactness verification & throughput benchmark

Verifies GPU-generated `FastDimer` graphs bit-for-bit against CPU `FastDimer` and benchmarks all 4 GPU stages:

```bash
./target/release/quip-super-sdp-feeder --device 0 --bench /path/to/topology.bin --n-salts 20000
```

### 2. Offline controlled A/B evaluation on real solved nonces

Evaluates `v3fast`, `v3`, `Super-Stage-0e`, and `v3+GW20` on a TSV of `(nonce_hex, solved_energy_milli)` records:

```bash
./target/release/quip-super-sdp-feeder --device 0 --bp-threads 24 \
  --eval-topo /path/to/topology.bin \
  --eval-tsv /path/to/real_nonces.tsv
```

### 3. Production multi-GPU deployment

See [`run.sh.example`](run.sh.example) for a verified 2-Feeder (`GPU 0, 1`, `nice -n 5`, 24 BP threads each on a 56-core host) : 2-Solver (`GPU 2, 3`) configuration.

## Status, Stated Plainly

- **Draft, reference, not for merge as is.** Lives cleanly under `contrib/super-sdp-feeder/` with zero edits to upstream files so maintainers and researchers can inspect, benchmark, or port individual kernels (e.g., the SMEM-FP16 BM-SDP or GPU-native FastDimer generator) into upstream miners if useful.
- **Scope**: Tuned and validated specifically for `quip/twisted-ice-dimer` (`m=12, t=4, weak=45, pairs=1350`). Requires sm_80+ GPUs with >= 96 KB opt-in dynamic shared memory for the Rank-8 SMEM kernel (tested on RTX 3080 Ti, RTX 4090, RTX 5090).
- **License**: `AGPL-3.0-or-later`, matching `quip-solver-cuda`.
