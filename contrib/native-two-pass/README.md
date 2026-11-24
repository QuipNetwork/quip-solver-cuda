# Native two-pass: a belief-propagation + cluster-rebase GPU miner for `quip/twisted-ice-dimer`

Written for: Quip miners and maintainers who want to read, try or reuse this build without repeating our experiments.

This directory is a **self-contained crate** (`contrib/native-two-pass/`). It does not touch any upstream file. It is a two-file solver (`solver.rs` plus the CUDA kernels in `kernel.cu`) built on `quip-solver-core` 0.0.15, in the same way the other miners in this repository are. The honest record of everything we tried, including null results, is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## What the solver does

Instead of annealing, it runs loopy belief propagation with reinforcement on the instance's factor graph (4 K4 tiles per region, strong ±1000 and weak ±45 layers), then repairs and fuses the best states.

| Stage | What it does |
|---|---|
| Cold search | 192 replicas (42 min-sum and the rest joint-region), 30,000 updates, a slow reinforcement ramp (`ramp_base` 15000, `ramp_step` 5000) |
| Closure, auxiliary, pair refinement | Restore tile and pair constraints, short targeted refinements |
| Record archive closure | Keeps the best states per charge cell and closes them |
| Seeded stages, native-cluster rebase | Two seeded passes, then an exact-window / cluster rebase of the best states |
| CPU fusion | Combines donors on disjoint windows (`fusion_width`, `fusion_donors`) |

Selection and time allocation (all opt-in through the config, an empty config runs the plain path):

| Knob group | What it does |
|---|---|
| `probe0_*` | Stage 0: a short probe (700 updates, 96 replicas) on every salt; an adaptive rolling window keeps the best 5 % |
| `probe_updates`, `probe_cutoff_milli` | Stage 1: a 3000-update probe; only salts at or below the cutoff get the full solve |
| `reroll_*`, `screen_target_milli` | Adaptive re-rolls: salts whose first full solve is near the round threshold are solved again with fresh seeds, up to 24 times, stopping when the chance of reaching the target falls below `reroll_min_p` |
| `QUIP_PARALLEL_STREAMS` | Number of concurrent solve engines (also what the solver declares as its stream width) |
| `weak_scale` | Scales the ±45 weak couplings in the search model only. Reported energies stay exact |

`configs/example.json` is the configuration we run. `run.sh.example` shows the launcher.

## Status, stated plainly

- **Draft, reference, not for merge as is.** It lives under `contrib/` so it can be read and tried without conflicting with anything.
- Built against `quip-solver-core` / `quip-protocol` / `quip-proto` **0.0.15** (pinned, `Cargo.lock` included). It has **not** been rebased onto current `main`.
- Tested on an RTX 5070 Ti laptop (WSL2) and an RTX 5090 (native Linux, Docker miner image). GPU only. Results may not transfer to other GPUs.
- Nothing here has been reviewed by the upstream maintainers. Use at your own risk.
- It builds with `cargo +1.97.1 build --release` (Rust 1.97, as in `rust-toolchain.toml`). Building does not need the CUDA toolkit. Kernels are JIT-compiled through NVRTC at run time.

## Build and run

```sh
cd contrib/native-two-pass
cargo build --release          # binary: target/release/quip-cuda-native-two-pass
```

Point the miner's `[cuda.0] binary` at a launcher that exports `QUIP_TWO_PASS_REFINEMENT=native-cluster-rebase` (see `run.sh.example`) and passes `--search-config configs/example.json`. Check the parsed settings with `--print-search-config`. Keep `screen_target_milli` at the current round threshold: it only controls how much extra work a near-miss salt gets and never changes reported energies.

Two practical warnings: a mistyped or missing config silently runs the default settings (which is slow), so always check with `--print-search-config`; and `probe0_updates` must be a multiple of 100, at least 500 and below `probe_updates`.

## Speed settings

`loop_factor_bp` (the cold and stage-0 kernel, ~85% of GPU time) takes the fast sweep path by default: the same arithmetic in the same order as the generic loops, with wide loads, register-resident junction tables and the next edge colour prefetched during the barrier wait. Best energies are bit-identical to the generic path.

| Environment variable | Effect |
|---|---|
| `QUIP_FBP_NT` | Threads per block of `loop_factor_bp`: `512` (default), `256` or `128`. At 256 two blocks share an SM. Results are bit-identical for every value: the kernel emulates the 512-thread random streams, and its reductions are integer or xor, so they are exact in any order. |
| `QUIP_FBP_CARVEOUT` | Preferred shared-memory carveout in percent (default 100) when `QUIP_FBP_NT` is below 512. |

Defining `QUIP_NO_FAST_SWEEPS` at the top of `kernel.cu` compiles only the generic sweep loops, for comparison.

## Pointers

- Knobs: `SearchConfig` and its `validate()` in `solver.rs`.
- Per-source-line profile notes for the main kernel `loop_factor_bp`: `EXPERIMENTS.md`, section 4.2.
- Kernels: `kernel.cu` (`loop_factor_bp` is the cold and stage-0 kernel).

## License

AGPL-3.0-or-later, same as upstream. See the repository `LICENSE`. The solver source for any build you run as a network service must be made available under the same terms.
