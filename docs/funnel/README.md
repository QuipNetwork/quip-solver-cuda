# Funnel: a staged-selection GPU Otsuka miner for `quip/twisted-ice-dimer`

Written for: Quip miners and maintainers who want to read, try or reuse the changes without repeating our experiments. (Internal name of this build in our notes: `G3V6L3E`.)

This branch is **`quip-solver-cuda` plus one squashed commit** (author: BudFox, squashed from 51 local commits). It is the source of the build we ran and verified on a live network. The honest record of everything we tried, including what failed and the exact settings, is in [`EXPERIMENTS.md`](EXPERIMENTS.md) (settings: section 3.3).

## What "Funnel" means

Most of the work of finding a deep result is deciding where to spend GPU time. Funnel pushes many candidate instances through cheap filters and spends long anneals only on the few that survive:

| Stage | What it does | Main knobs |
|---|---|---|
| Stage 0 | CPU screen of every salt, keeps the best 1 % into a best-first pool | `QUIP_OTSUKA_STAGE0_*`, `QUIP_OTSUKA_POOL` |
| Stage 1 | Short GPU anneal of pool candidates, promotes about 7 % | `QUIP_OTSUKA_STAGE1_*` |
| Full anneal | 524k sweeps, 64 replicas, population annealing with Houdayer; V6 sweep kernel with class-split gathers and a union pass | `QUIP_OTSUKA_SWEEP_*`, `QUIP_OTSUKA_UNION_SV` |
| Polish, stage 2 | CPU polish; re-anneal of unusually deep first runs | `QUIP_OTSUKA_CPU_POLISH`, `QUIP_OTSUKA_STAGE2_*` |
| Deep lane, ease booster | Hold the best results of a round and deepen them; boost the round best when it is close to the threshold | `QUIP_OTSUKA_DEEP_*`, `QUIP_OTSUKA_EASE_*` |

(Our notes call the three groups "G3" for the selection pipeline and weak-layer scale 1.5, "V6" for the kernel, and "L3E" for three deep lanes plus the ease booster.)

The code also contains features that are **off** in Funnel because they did not help or failed on the live network (for example the beta-knee schedule, the V4/V5 kernels, Houdayer moves and tile/pair weight changes). They are left in so the experiments can be reproduced; see `EXPERIMENTS.md` before enabling any of them.

## Status, stated plainly

- This is the verified form. It is built on upstream commit `1fceca7` and has **not** been rebased onto current upstream `main`. Upstream has moved on since. A rebase is a separate piece of work and must be re-validated on a GPU before it can be called verified.
- The only differences between the build we validated and this branch are formatting (`cargo fmt`), comment text (an internal reference to another miner was reworded), a few `#[allow(clippy::...)]` attributes and one hex literal regrouped with underscores (same value), a comment in one test fixture, and git metadata. Our helper scripts and benchmark tools are not included: set up your own launcher to suit your machine.
- Tested on one machine: a single desktop RTX 5090 under WSL2 with Docker. Results may not transfer to other GPUs. Use at your own risk; nothing here has been reviewed by the upstream maintainers.
- `src/cluster/tuning.rs` documents every knob and its default. By design an unset environment runs the stock upstream path; we have not audited every commit for exceptions, so diff against upstream if you need a guarantee.

## Build and run

```sh
cargo build --release --bin quip-cuda-otsuka        # toolchain from rust-toolchain.toml
```

Point the miner's `[cuda.0] binary` at the built `quip-cuda-otsuka` (or at a small launcher of your own that exports the `QUIP_OTSUKA_*` variables from `EXPERIMENTS.md` section 3.3 and then `exec`s the binary). Every knob can be overridden from the environment. To fall back to the stock path inside the same binary, unset the `QUIP_OTSUKA_*` variables (for the screen, `QUIP_OTSUKA_STAGE0_KEEP=1` solves every salt).

## Pointers

- Knob reference: `src/cluster/tuning.rs`.
- Scheduler pieces: `src/cluster/sampler/local.rs`, `stage1.rs`, `stage2.rs`, `deep.rs`, `ease.rs`, `pool.rs`; screen: `src/family_screen.rs` and `src/family_screen/`.

## License

AGPL-3.0, same as upstream. See `LICENSE`. The solver source for any build you run as a network service must be made available under the same terms.
