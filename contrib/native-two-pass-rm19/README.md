# Native two-pass, single-GPU build ("rm19"): a miner for `quip/twisted-ice-dimer` that runs on one GPU

Written for: Quip miners and maintainers who want to try this build on one GPU, know which settings to use, and test their own ideas against it.

This directory is a **self-contained crate** (`contrib/native-two-pass-rm19/`). It does not touch any upstream file. It is a two-file solver (`solver.rs` plus the CUDA kernels in `kernel.cu`) that builds on the native two-pass miner from PR #5 and adds the pieces that matter when **one GPU and one CPU have to do everything**: salt screening on the CPU with a small GPU rescoring step, a faster salt generator, a restart-ladder CPU refiner for solves that land just above the round target, and settings that keep the lease pipeline moving. There are no helper machines, brokers or remote screeners in this build.

## What is different from PR #5

| Piece | What it does | Env / config |
|---|---|---|
| **CPU salt screen + GPU rescoring** | `QUIP_GEN_THREADS` CPU threads generate salts and ridge-score them (the 20-count structural score of PR #5). Only the best `QUIP_RESCORE_FRAC` (3 %) are sent to the *same GPU* for a rank-8 relaxation (a small kernel, about 225 µs per salt scored), and ranked by the combined key (ridge z-score + rank-8 z-score). | `QUIP_GPU_RESCORE=1`, `QUIP_RESCORE_FRAC`, `QUIP_GG_BATCH` |
| **Small candidate pool** | Only the best `QUIP_POOL` candidates wait for a solve engine; the rest are dropped at once. A lease is finished when its salts are accounted for, so a small pool keeps leases short and lets the coordinator grow them. | `QUIP_POOL=32` |
| **Restart-ladder refiner** | The in-process CPU refiner (disagreement-basin repair of the finished population) runs three short independent restarts, keeps their champions and a few distinct elites, then spends its last slice on the union. Same total CPU budget as the plain refiner. Runs on a solve within `QUIP_REFINE_MARGIN_MILLI` of the live round target; a 16 s follow-up when the gap is under 1.0, a second when under 0.15. | `QUIP_REFINE*` |
| **Faster salt generation** | A bit-exact speed-up of the generator (patch to `ising-generators` 0.0.15, see `patches/`): per-topology tables built once, one 32-bit modulo per draw. 27 % less generation time per salt, same output (instance digest and couplings identical on 20,000 salts). | applied by `setup.sh` |
| **GPU cascade screening (optional, off)** | The whole screening cascade on the solving GPU. Included for experiments; **not recommended** (see EXPERIMENTS.md E2). | `QUIP_GPU_SCREEN=1` |

## Status, stated plainly

- **Draft, reference, not for merge as is.** It lives under `contrib/` so it can be read and tried without conflicting with anything.
- Built against `quip-solver-core` / `quip-protocol` / `quip-proto` **0.0.15** (pinned, `Cargo.lock` included). Not rebased onto current `main`.
- Tested on an RTX 5070 Ti laptop (WSL2), an RTX 4090 (Linux, Docker miner image) and an RTX 4060 (WSL2). GPU only. Results may not transfer to other GPUs.
- Nothing here has been reviewed by the upstream maintainers. Use at your own risk.
- **Our evidence is small.** Wins are rare; the numbers below and in EXPERIMENTS.md are offline paired tests and short live windows, with sample sizes stated. Please read them as "worth trying", not as proof.

## Build and run

```sh
cd contrib/native-two-pass-rm19
./setup.sh                     # fetches ising-generators 0.0.15, applies patches/ising-generators-0.0.15.patch, enables the [patch] section
cargo +1.97.1 build --release  # binary: target/release/quip-cuda-native-two-pass  (no CUDA toolkit needed; kernels JIT through NVRTC)
./target/release/quip-cuda-native-two-pass --capabilities | head -c 100   # sanity check
```

Point the miner's `[cuda.0] binary` at a launcher like `run.sh.example`. Always check the config loaded: `--print-search-config --search-config configs/example.json`. A mistyped or missing config silently runs slow defaults.

## Recommended settings (single GPU)

`run.sh.example` is the full launcher. What each setting is and why:

| Setting | Value | Notes |
|---|---|---|
| `QUIP_TWO_PASS_REFINEMENT` | `native-cluster-rebase` | required |
| `--search-config` | `configs/example.json` | the funnel (stage budgets, cutoffs, the 61-number ridge model). `screen_target_milli` should follow the live round threshold. |
| `QUIP_PARALLEL_STREAMS` | 6 on an RTX 4090 / 5070 Ti | concurrent solve engines. **Smaller cards: start lower.** We run 4 on an RTX 4060 and are still measuring 1-3; a stream test on the 4060 is not in this PR. |
| `QUIP_GEN_THREADS` | CPU threads minus about 8 | 22 on 30 vCPUs, 12-16 on 24 threads, 6 on 12. The CPU runs near saturation; the engines need some of it. |
| `QUIP_GPU_RESCORE` / `QUIP_RESCORE_FRAC` / `QUIP_GG_BATCH` | `1` / `0.03` / `512` | rank-8 rescoring of the top 3 % on the solving GPU. Needs about 200 MB of GPU memory at batch 512. |
| `QUIP_POOL` | `32` | **important.** With 256, leases finished slowly and the coordinator kept them small (see E4). |
| `QUIP_LEASE_ACTIVE` | `3` | leases worked at once |
| `QUIP_REROLL_SLOTS`, `QUIP_FOCUS`, `QUIP_FOCUS_PAR`, `QUIP_FOCUS_FRAC` | `2`, `1`, `4`, `0.4` | carried over from PR #5's tuning |
| `QUIP_REFINE` / `_SECONDS` / `_MARGIN_MILLI` / `_MAX` | `1` / `8` / `2500` / `2` | margin 2500 refines solves up to 2.5 above the target; 2000 also works, with fewer attempts |
| `QUIP_GPU_SCREEN` | unset (off) | do not turn on for the solving GPU (E2) |

**Memory.** A 12-CPU / 8 GB machine was tight: the miner used about 3.4 GB and the validator about 2.7 GB. If you run the node under WSL, raise the WSL memory limit; swap pressure slows everything down.

**How to judge a setting.** The coordinator sizes leases from the throughput it has measured, so after every miner restart leases start at 4,500 salts and take about 20 minutes to grow. Compare windows of at least 30 minutes after that, never the first minutes. Count full solves per hour and draws at -5088 / -5090 / -5092 or better per hour, not wins: wins are too rare to separate two settings in a day.

## Largest refiner gains we recorded (live, ladder build)

Single solves where the in-process refiner took a finished solve to a deeper state. These are the biggest ones we noted while running the build for about 13.5 hours on an RTX 4090 (23:27 to 13:02 UTC); they are not an exhaustive list. The machine has since been shut down, so these come from log lines we recorded at the time, not from a fresh pass over the log.

| Start energy | After refiner | Gain | What followed |
|---|---|---|---|
| -5092.485 | -5093.925, then -5095.185 after the 16 s follow-up | **2.70** (1.44 + 1.26) | block won 5 s later (proof -5096, target -5095) |
| -5091.715 | -5093.925 | **2.21** (22 separate improvements in one pass) | block won two minutes later (proof -5094, target -5094) |
| -5090.255 | -5091.945 | **1.69** | not near the target |
| -5091.495 | -5093.045 | **1.55** | not near the target |
| -5093.855 | -5094.755 | **0.90** | block won 7 s later (proof -5095, target -5095) |
| -5094.015 | -5094.755 | **0.74** | block won 7 s later (proof -5095, target -5095) |
| -5093.725 | -5093.885 (second 16 s follow-up) | 0.16 | the solve had missed the target by 0.021; the follow-up crossed it, block won |

On the laptop, with the previous refiner, one solve gained 2.11 and won a block. A "gain" is how many energy units lower the refiner's state is than the solve's best. Most gains we saw were between 0.1 and 1.0, so the table is the tail, not the typical case.

Offline, the typical effect is smaller (96 paired runs per budget): mean gain 0.84 for the plain refiner and 0.92 for the ladder at 8 s, 0.87 and 1.00 at 16 s.

## Testing your own ideas

Offline refiner tests need no GPU and take minutes: `--refine-test INSTANCE ARCHIVE --refine-seconds 8 --refine-seed N` runs the in-process refiner on a finished population (an archive produced by `--replay-canonical`). `tools/refine_ab_run.sh`, `tools/refine_ab_analyze.py` and `tools/verify_energies.py` run two builds on the same instances, archives and seeds, compare the energies, and recompute every returned energy independently. The runs are deterministic for a given binary, seed and CPU budget, so a pair of builds can be compared with a few hundred runs. For ranking or screening ideas, a set of salts labelled with their cold-stage energy is enough to compute the recall of the best salts in the top 0.5 / 1 / 2 / 3 %.

`--gpu-screen-test FAMILY_DUMP.json NONCES.txt` runs the in-process GPU screener on a nonce list and prints `index key` for the survivors, for checking a change against another implementation.

## Pointers

- Knobs: `SearchConfig` and its `validate()` in `solver.rs`; the lease loop is `fn lease_local`.
- Screening: `mod cpu_screen` (the ridge score), `mod gpu_screen` (generator, rank-4 / rank-8 kernels, in-process rescoring). The screening kernels sit at the end of `kernel.cu` between `==SCREEN_*==` markers, behind an `#ifdef` that is never defined, and are compiled as separate NVRTC modules.
- Refiner: `mod refine_native`; the restart ladder is the first loop in `refine_native::run`, and the follow-up passes (`QUIP_REFINE_EXTRA_*`: 16 s when the gap is under 1.0, a second when under 0.15, at most `QUIP_REFINE_EXTRA_TRIES`) are in the code that calls it.
- Experiments and null results: `EXPERIMENTS.md`.

## Not covered here

The source still contains optional multi-machine code from our earlier setup (a hub/worker split via `--worker-listen` / `QUIP_REMOTE`). It is off unless used and is neither documented nor tested here; the settings above do not touch it.

## License

AGPL-3.0-or-later, same as upstream. See the repository `LICENSE`. The solver source for any build you run as a network service must be made available under the same terms. The generator patch is a patch to the Apache-2.0 `ising-generators` crate.
