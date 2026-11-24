# Helper: the PR #5 solver as a solve and probe server

| file | what |
|---|---|
| [`pr5-serve.patch`](pr5-serve.patch) | our patch to `contrib/native-two-pass/solver.rs` of PR #5 at commit `97faa92` |
| [`PROTOCOL.md`](PROTOCOL.md) | the line protocol between the miner and the helper, v1 |
| [`serve-full.json`](serve-full.json) | search config of the full-solve helper (`QUIP_OTSUKA_BP_HELPER`) |
| [`serve-probe.json`](serve-probe.json) | the same plus `probe_updates: 3000`: the probe helper (`QUIP_OTSUKA_BP_PROBE_HELPER`) |

PR #5 is not merged, so this folder carries a patch, not a copy of the crate. The patched `solver.rs` is a derivative of
PR #5's code and stays under its licence, AGPL-3.0-or-later; the SPDX header of the file is unchanged by the patch.

What the patch adds, all opt-in:

- `--serve`: open the GPU once, compile the kernels, print `{"ready":true,"proto":1,"nodes_max":N}`, then answer one solve
  request per stdin line on stdout (the canonical instance is passed as a file path). A request's `shift` selects an
  independent draw (the solver's own trajectory-shift mechanism, kept separate from the config's `trajectory_shift`), and
  `{"cancel":id}` stops the solve in flight. stderr carries the logs. Without `--serve` the miner, replay and probe paths
  run as before.
- `struct_cutoff`, `struct_streams` (search config): a CPU structural prefilter inside the native miner (E83). Unset =
  unchanged. Not used by the helper configs here.

In `serve-probe.json`, `probe_cutoff_milli: -9000000000` is a cut no probe can clear, so the probe helper answers with the
probe's result and never continues into the full solve.

Apply: at the root of a checkout of PR #5's head, commit `97faa929da83af19807c66ccc18ddd01f6f2e337`,
`git apply pr5-serve.patch` (or `patch -p1 < pr5-serve.patch`); then `cargo build --release` in
`contrib/native-two-pass`. `git apply --check` passes on that commit. The helper used in our measurements was built that
way from `97faa92` (the patch here differs from it only in four comment lines).
