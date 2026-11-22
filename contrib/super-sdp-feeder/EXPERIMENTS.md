# Experiments Log: Burer-Monteiro SDP + Goemans-Williamson Screening (`contrib/super-sdp-feeder`)

This document records the quantitative benchmarks, ablations, live production telemetry, and **negative / regressed experiments** observed while building `quip-super-sdp-feeder` on the Aglais testnet (`quip/twisted-ice-dimer`, `m=12, t=4, weak=45, pairs=1350`, 4,560 nodes, 41,205 edges).

## 1. Conventions & Evidence Grades

- **Energies**: Ising ground-state energies (milli-units / 1000). Lower is better.
- **Metrics**:
  - `AUC90`: ROC-AUC distinguishing solved nonces with true downstream energy `<= -5090.0` from higher-energy nonces on the same topology.
  - `Top-50% Recall`: Fraction of `<= -5090.0` nonces retained when keeping the top 50% of BM-SDP survivors.
- **Grades**:
  - **S (Strong)**: Controlled paired evaluation on real solved Aglais nonces (`n = 819`) plus multi-hour live production confirmation.
  - **M (Moderate)**: Clean live telemetry comparison over a 90-minute steady-state window.
  - **N (Negative / Regressed)**: Tested and deliberately rejected; documented here so others do not repeat the same trap.

---

## 2. Controlled Offline A/B Benchmark (`n = 819` Live Aglais Nonces) — Grade: S

All methods were evaluated on the exact same 819 real post-BM30 candidate nonces collected from live Aglais rounds (of which **85 nonces** solved to `<= -5090.0` under a downstream `native-two-pass` solver):

| Screening Stage-0e Variant | Description | `AUC90` | Top-50% Recall (`<= -5090`) | Delta vs `v3fast` |
|---|---|---:|---:|---:|
| `v3fast` (v4.1 baseline) | 4x120 BM-seeded K4-BP only | `0.6865` | `60 / 85 (70.6%)` | baseline |
| `v5.1` (ramp-scaled `4x160 + GW20`) | `DeepBP` with ramp scaled by `deep_iters / 220`, top-2 mean | `0.7545` | — | `+0.0680` (**rejected vs v5.0**) |
| **`v5.0` (`Super-Stage-0e`, production)** | **`v3` (8x120) + `DeepBP` (4x160 unscaled ramp, top-3 mean) + `GW20` + 1-flip polish** | **`0.7705`** | **`73 / 85 (85.9%)`** | **`+0.0840` (`+21.7%` recall)** |

### Key Takeaways

1. **Geometric SDP + GW rounding adds genuine orthogonal signal over BP alone**: Combining the continuous Rank-8 BM-SDP relaxation (`w = 30`) with Goemans-Williamson hyperplane rounding (`8` principal eigenvectors + `12` random hyperplanes + 1-flip local polish) and deep K4-BP raised `AUC90` by `+0.0840` and captured `+21.7%` more `<= -5090` nonces in the top half.
2. **Why `v5.0` (unscaled ramp) beat `v5.1` (scaled ramp) — Grade: N (negative result)**:
   - We initially hypothesized that running `deep_iters = 160` with unscaled `ramps = [90, 130, 170, 200]` was a bug because replicas 2 and 3 never reached the `rho = 0.95` ceiling before iteration 160.
   - Scaling the ramp down proportionally (`ramps * (deep_iters / 220.0)`) and switching from top-3 to top-2 mean **regressed** `AUC90` from `0.7705` down to `0.7545`.
   - **Root cause**: Under a gentler reinforcement ramp (`rho < 0.85` on the slower replicas), reinforced K4-BP avoids premature freezing into shallow local basins, yielding better energy separation across instances. We reverted `stage_e_super` to the unscaled ramp + top-3 mean.

---

## 3. Live Aglais Steady-State Production Comparison (4x RTX 4090) — Grade: M

Measured over a continuous 90-minute steady-state window on a 4x RTX 4090 / 56-physical-core EPYC host running **2 Feeder GPUs (GPU 0, 1) : 2 Solver GPUs (GPU 2, 3)**:

| Metric | `2:2 v4.1` (`v3fast`) | `2:2 v5.0` (`Super-Stage-0e`) | Change |
|---|---:|---:|---:|
| Coordinator salt throughput | `72.0k salts/s` | **`81.7k - 85.7k salts/s`** | **`+13.4%`** |
| Downstream solver consumption | `19.2 salts/s` | **`20.9 salts/s`** | **`+9.1%`** |
| Solver `gpu_wait` starvation | `0.0 - 0.2 ms/s` | **`0.0 - 0.1 ms/s`** | zero starvation |
| Live deep-tail hits `<= -5092` | — | `24` (`16.0 / hr`) | — |
| Live deep-tail hits `<= -5093` | `4.9 / hr` | **`6.0 / hr` (`9` in 90m)** | **`+22%`** |
| Live deep-tail hits `<= -5094` | — | `3` (`2.0 / hr`) | — |
| Stale drops / submission errors | `0` | `0` | clean |

---

## 4. Negative & Diagnostic Findings (What Did NOT Work) — Grade: N

1. **`3 Feeder : 1 Solver` split regressed badly on a 56-core container**:
   - **Setup**: Tested 3x Feeder (`--bp-threads 34` each = `102` BP threads) feeding 1x Solver on a host whose `lscpu` exposed 128 logical CPUs but only **56 online physical cores** (`0-55` online, `56-127` offline).
   - **Result**: Severe 1.82x CPU oversubscription doubled downstream `solve_ms` (`350 ms -> 750 ms`), while the single solver's `window=2` lease cap throttled overall coordinator throughput down to `54k-62k salts/s` (`-25%` vs `2:2`).
   - **Fix**: Returned to `2:2` with `--bp-threads 24` per feeder (`48 / 56` physical cores) and `nice -n 5` on feeders so solver host threads never starve.
2. **Confounder alert when monitoring raw `-5088` counts in live production**:
   - After deploying `v5.0`, our raw rate of full solves landing at `<= -5088` initially appeared to drop (`0.250% -> 0.147%`).
   - Before modifying code, we inspected the live energy histogram and ran the 819-nonce controlled A/B above. The apparent drop was **not** a screening regression: adaptive live target tracking (`QUIP_SUPER_LIVE_TARGET=1`) had automatically tightened downstream `probe_cutoff` from `-5085.2` to `-5086.5` as chain difficulty moved to `-5093.8..-5095`, skipping `-5087/-5088` candidates while increasing `<= -5093` wins by `+22%`.
3. **Bit-exactness & memory-bandwidth bottleneck lessons**:
   - Earlier CPU-generation feeders saturated host memory bandwidth and PCIe uploads (`~150 KB/salt` CSR). Moving `FastDimerV3` graph construction directly onto the GPU (`160 B/salt` ChaCha8 stream keys) plus keeping Rank-4 (`22.6 KB`) and Rank-8 (`90.4 KB`) BM-SDP state resident in dynamic SMEM was the single largest throughput multiplier, unlocking `>40k salts/s` per RTX 4090 even before Stage-0e.
