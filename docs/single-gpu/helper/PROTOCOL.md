# Helper protocol (v1)

Our miner (a patched `quip-cuda-otsuka`, not included in this pull request) keeps doing selection (fast score, keep set,
cumulative probe ladder). For the puzzle the ladder picks, instead of its own long anneal it asks a helper process
(the PR #5 native two-pass solver in the `--serve` mode added by `pr5-serve.patch`) for a solution, checks the energy itself and pushes it.

`pr5-serve.patch` applies to PR #5's head at commit `97faa929da83af19807c66ccc18ddd01f6f2e337` (checked with
`git apply --check` from the root of a checkout of that commit; it touches `contrib/native-two-pass/solver.rs` only).
The probe re-order and the probe gate (`../probe-gate/`) use a second helper process of the same binary with a probe
search config (`probe_updates: 3000`).

## Processes
- The miner spawns ONE helper child at first use and keeps it for the life of the process.
- Helper command line (a single string from the miner's setting `QUIP_OTSUKA_BP_HELPER`, split on whitespace, no shell):
  `<binary> --search-config <cfg.json> --serve [--device N]`
- Helper environment: cleared by the miner, then only: `PATH`, `LD_LIBRARY_PATH`, `HOME`, `QUIP_CUDA_CACHE`,
  `QUIP_TWO_PASS_REFINEMENT=native-cluster-rebase`, `RUST_LOG`, and every variable whose name starts with `NVIDIA_` or `CUDA_`.
  (No session token, no API tokens.)
- stdin/stdout are pipes carrying the protocol, one JSON object per line (`\n` terminated, UTF-8, no pretty printing).
  stderr is inherited (helper logs go to the container log). The helper must never write anything but protocol lines to stdout.

## Helper -> miner, once after start-up (GPU opened, kernels compiled)
`{"ready":true,"proto":1,"nodes_max":5000}`

## Miner -> helper
- Solve: `{"id":<u64>,"instance":"<absolute path of a canonical instance JSON file>","shift":<u64, optional, default 0>,"max_reads":<integer 1..64, optional, default 8>}`
  - The canonical instance file has exactly the keys `load_canonical_instance` reads in the helper:
    `descriptor_digest`, `topology_order_digest_bytes`, `nonce_bytes`, `node_ids`, `edges`, `h_milli`, `j_milli`,
    `offset_milli`, `metadata`, `instance_digest_bytes` (writer: `write_instance` in our miner's `bp_helper.rs`, not included here).
  - `shift` = k asks for an independent draw: the helper runs the solve under its trajectory shift for draw k
    (the same mechanism its re-rolls use: `TrajectoryScope(k * 0x9E3779B97F4A7C15)`); 0 = the normal first draw.
  - One solve at a time. A second solve request before the first has been answered is an error (answered `ok:false`).
- Cancel: `{"cancel":<id>}`: stop the in-flight solve with that id as soon as possible. Unknown/finished ids are ignored.
- EOF on stdin: finish (cancel anything in flight) and exit 0.

## Helper -> miner, one line per solve request
- Success: `{"id":<u64>,"ok":true,"energy_milli":<i64 best>,"reads":[{"energy_milli":<i64>,"spins":"<string>"}, ...],"solve_ms":<u64>,"stage":"<the helper's stage label>","screened_out":<bool>}`
  - `reads` sorted by energy ascending, at most `max_reads`, distinct states preferred.
  - `spins`: one ASCII character per node, in the order of the instance's `node_ids`/`h_milli` arrays: `'1'` = spin +1, `'0'` = spin -1.
  - `energy_milli` is the exact energy of that spin vector on the instance's own h/J/edges, in milli-units, WITHOUT `offset_milli` added.
- Failure: `{"id":<u64>,"ok":false,"cancelled":<bool>,"error":"<short text>"}`

## Miner side rules
- Before a solve the miner writes the instance file to `QUIP_OTSUKA_BP_DIR` (default `/dev/shm/quip-bp`, created 0700), named
  `<64-hex nonce>.json`, and deletes it after the answer.
- The miner recomputes the energy of every returned read on its own graph and drops any read whose energy does not match
  (logs an error). If no read survives, the helper died, the answer is `ok:false`, or `QUIP_OTSUKA_BP_TIMEOUT_S` (default 180)
  passes, the miner falls back to its own long anneal for that puzzle (unless the lease has stopped).
- When the lease stops while a solve is in flight the miner sends `cancel`, waits up to 5 s for the answer line, and goes on.
- With `QUIP_OTSUKA_BP_HELPER` unset the miner behaves exactly as before (same code path, same seeds, same log lines).
