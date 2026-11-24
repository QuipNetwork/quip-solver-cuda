# Probe gate: a short probe of the final solver between selection and the full solve

A reference description of the probe re-order (E86) and the probe gate (E88) as they ran in our miner. Experiment numbers
refer to [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

## Where it sits

```
salts ──► selection (any selector) ──► last N candidates ──► probe each (3,000 updates, 1 read)
                                                                   │
                                          re-order by verified probe energy, best first
                                                                   │
                                  walk the order (at most top-K): full solve while probe ≤ limit,
                                  stop at the first candidate above it; none clears ──► next selection
```

In our miner the selector is a fast structural score followed by a cumulative GPU probe ladder; the full solve is the PR #5
solver run as a helper process ([`../helper/`](../helper/)). The probe is the same solver binary with a short search config
(`probe_updates: 3000`), run as a second helper process. Nothing in the gate depends on how the N candidates were chosen.

## Algorithm

```
input: candidates c[0..N) in the selector's order (c[0] = selector's best), limit L (or none), top_k

best[j] = none for all j
for j in 0..N:
    if lease stopped: cancelled; break
    r = probe(c[j])                      # 3,000 updates, max_reads 1, shift 0
    if r is a read:                      # re-score the read on our own graph; keep it only if the energy matches
        best[j] = r.energy               # offset included
    if r failed (error, no verified read): leave best[j] = none, go on
    if helper down (cannot start, died, killed after timeout): down; break

order = 0..N
if not down:
    order = stable sort of 0..N by (best[j] is none, best[j])   # best first; ties and failures keep selector order,
                                                                # failures last
# with top_k > N, positions N.. follow unprobed, in selector order

if L is none or cancelled:            solve_n = top_k
elif down or all best[j] are none:    solve_n = top_k                # gate not applied: no verified probe
else:                                 solve_n = number of leading entries of order[0..top_k]
                                                with best[j] verified and best[j] <= L
full-solve order[0..solve_n] in turn (each pushed when done)
the rest: never pushed
```

Probe results are never pushed. A probe helper that is down gives the selector's order (one warning per outage, respawn
at most once a minute). Before probing, the miner waits out any cancelled full solve still running in the main helper, and
before the full solves it waits out any cancelled probe, so the two helpers never use the GPU at the same time.

The decision logic is two pure functions in the patches (`order_by_probe`, `solve_cut` in `bp_helper.rs`); the core of
`solve_cut` is:

```rust
let k = order
    .iter()
    .take(topk)
    .take_while(|&&j| po.best.get(j).copied().flatten().is_some_and(|e| e <= limit))
    .count();
```

## Knobs

All are environment variables (or lines of the miner's settings file). Unset = off; with all of them unset the miner runs
as before the patches.

| knob | meaning | default | value in our live runs |
|---|---|---|---|
| `QUIP_OTSUKA_BP_HELPER` | command of the full-solve helper (required) | unset | PR #5 solver, `--serve`, [`serve-full.json`](../helper/serve-full.json) |
| `QUIP_OTSUKA_BP_PROBE_HELPER` | command of the probe helper; turns the re-order on | unset | same binary, probe config ([`serve-probe.json`](../helper/serve-probe.json): `probe_updates: 3000`) |
| `QUIP_OTSUKA_BP_PROBE_N` | candidates probed, 2..=16 | 4 | 4 (8 was break-even, E86) |
| `QUIP_OTSUKA_BP_PROBE_TIMEOUT_S` | per probe; past it the probe helper is killed as hung | 30 | default |
| `QUIP_OTSUKA_BP_SOLVE_MAX` | gate limit on the verified probe, milli-units, offset included; unparsable = off | unset | `-5083000` after E88's out-of-sample check, `-5084000` since E96 |
| `QUIP_OTSUKA_BP_TOPK` | full solves per selection (from an earlier patch) | 1 | 2 (E85) |

The limit is in milli-units because the protocol is; −5,083.0 units = `-5083000`.

## Log lines

| message | fields | when |
|---|---|---|
| `gpu-pa bp probe configured` | `cmd`, `n`, `timeout_s`, `solve_max` (number, or `"off"` when set but unparsable) | once per process |
| `gpu-pa ladder winner` | `salt`, `ms`, `nonce` | before probing |
| `gpu-pa bp probe` | `salt`, `best`, `ms`, `lrank`, `helper_ms`, `stage` | each verified probe |
| `gpu-pa bp probe failed; the salt keeps its ladder position after the probed ones` | `salt`, `lrank`, `reason` | probe failed, helper up |
| `gpu-pa bp probe read rejected` | `why` | a returned read did not verify |
| `gpu-pa bp probe cancelled: the lease stopped` | `salt`, `ms` | stop during a probe |
| `gpu-pa bp probe order` | `salts`, `bests`, `lranks`, `ms` | after probing, in solve order |
| `gpu-pa bp probe skipped; ladder order` | `reason`, `salts`, `ms` | helper down |
| `gpu-pa bp probe helper unavailable; the ladder order is used until it is back` | `reason` | once per outage |
| `gpu-pa bp probe helper back; probing again` | | after an outage |
| `gpu-pa bp solve skipped: best probe above the limit` | `best`, `limit`, `salt`, `probes` | gate cut everything |
| `gpu-pa bp solve limit not applied: no probe verified; solving in ladder order` | `limit`, `probes` | every probe failed |
| `gpu-pa screen solve` | as before, plus `rank` (position in probe order) and `lrank` (selector rank) | each full solve |

## Measured effect (from EXPERIMENTS.md)

- **Probe as predictor (E86, bench, 120 live picks).** 3,000-update probe against the full solve of the same puzzle: r 0.754
  (3.16 s on the shared GPU); an independent seed gives r 0.788. Best of 4 by probe, then full solve: −5,087.25 against
  −5,085.23 for a random one of the 4 (+2.0, 80 % of the oracle's −5,087.77).
- **Re-order live (E86).** 66 solves: mean −5,086.20 against −5,085.04 before (+1.16); ≤ −5,090 9.1 % against 2.1 %. At
  245 solves: −5,086.31, ≤ −5,090 7.3 %. Probes 2.2–2.3 s each, ~9 s per set of 4. Probing 8 instead of 4 (150 solves)
  was break-even.
- **Fit (E88, 308 live full solves).** full ≈ −954.4 + 0.813 × probe, residual sd 1.66. No full solve with a probe in
  (−5,084, −5,083] (51) or above −5,082 (112) reached −5,090.
- **Limit chosen in sample, checked out of sample (E88).** −5,084.0, chosen on those 308, would have dropped a win on the
  next 85 solves; −5,083.0 kept 4 of 4 results ≤ −5,090 there. On all 393: −5,083.0 keeps 24 of 28 (≤ −5,090), 10 of 11
  (≤ −5,091), 5 of 5 (≤ −5,092).
- **Gate live (E88, 6.3 h, 418 ladders).** 61 % of full solves skipped; 66.5 ladders/h (≈ 54 before), cycle median 36.0 s;
  per hour ≤ −5,090 2.86, ≤ −5,091 1.91, ≤ −5,092 0.95, against 3.4 and 0.96 at ≤ −5,090 / ≤ −5,091 before: about ×2 at
  ≤ −5,091, flat at ≤ −5,090. Rank-1 full solves (all past the gate) mean −5,087.85.
- **Tighter limit later (E96).** −5,084.0 together with more re-draws near the bar: neutral over 13.7 h.

## The patches

| file | from | what |
|---|---|---|
| `0001-probe-reorder.patch` | commit `ea9b0f9` (patch 42, E86) | probe helper, `order_by_probe`, re-order before the top-K full solves; tests; three options for the protocol-level fake helper the tests use |
| `0002-probe-gate.patch` | commit `676cfcf` (patch 43, E88) | `QUIP_OTSUKA_BP_SOLVE_MAX`, `solve_cut`, the skip and not-applied log lines; tests |

They are `git format-patch` output of our unpublished branch, which is a patch series on `v0.3.6-rc5` (`cbf899d`); they
apply on top of its commit `b829f0c` (patch 41: helper re-draws and top-K), which itself needs the helper process wiring of
patch 40 (`bp_helper.rs`, ~900 lines) and the probe ladder in `screen_lease.rs`. **They are for reading, not for `git am`
on `main`**: `main` has neither file. Comments that name `patch-gpu-pa-4x.py` refer to our internal patch scripts and can
be ignored. The full branch can be shared on request.
