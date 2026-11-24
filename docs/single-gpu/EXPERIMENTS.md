# Experiments: single-GPU mining on twisted-ice-dimer

Every experiment we ran on the `quip/twisted-ice-dimer` proof of work between 2026-11-11 and 2026-11-23, in the order we
ran them, including the ones that failed. The failures are the larger part and probably the more useful one.

> **Note for this pull request.** This record was written alongside our miner branch (a patch series on
> `v0.3.6-rc5`, `cbf899d`), which is **not included here**. Where it names patch numbers, knobs, tools or tests, those
> live in that branch; it can be shared on request. The only code carried in this pull request is the helper patch to
> PR #5 (`helper/pr5-serve.patch`, with its two search configs) and, for reading, the two commits of the probe re-order
> and the probe gate (patches 42 and 43 below) in `probe-gate/`. Apart from this note and the file references it
> qualifies, the text is unchanged.

## Reading the numbers

- **Energies** are in units (the chain's milli-energy divided by 1,000), offset included. Lower is better. "Deeper" means
  lower. On the recipe mined since 2026-11-14 (topology 0x2d130ce6, 4,560 nodes, 41,205 edges, weak couplings ±45,
  1,350 dimer pairs) the strong layer alone has a floor of −4,888; winning states sit near −5,084 … −5,097.
- **Bar**: the energy a result must reach to win the round. It starts at the previous winner's energy, holds for 100
  blocks, then eases. It moved from about −5,070 (2026-11-13) to about −5,093.5 (2026-11-22).
- **Puzzle / salt**: each salt defines one puzzle instance (the canonical generator turns the salt's nonce into couplings).
  "Selection" is choosing which salts to solve; "solver" is how deep a chosen puzzle is solved.
- **Hardware**: our node = one laptop RTX 5090 (82 SMs, 175 W limit) and 24 CPU threads in the same machine. "Shared GPU"
  means a bench ran while the live miner used the same GPU; such timings are slower than a free GPU, and the live
  miner's rate dropped during benches (those windows were excluded from live comparisons).
- **How we judged**: offline benches on fixed puzzle sets (replayed nonces, deterministic per nonce and seed), then live
  A/B windows: one change at a time, compared on the mean of the full solves and on the share at or under a tail
  threshold close to the bar, over 60 to 450 solves. Wins are reported but are too rare to judge by. Each code change was
  first gated for identity: with its knob unset, the new binary had to reproduce the previous binary's log lines on a
  fixed lease (timings masked).
- **Verdicts**: *kept* (live and kept), *rejected* (measured, no gain or a loss), *neutral* (live, no measurable effect,
  left on), *bench only* (a study, nothing to deploy), *not run*.
- **Knobs** are `QUIP_OTSUKA_*` environment variables or lines in the miner's settings file (commit 1). Patch numbers
  refer to the commits of our branch (not included here):

| patches | commit topic |
|---|---|
| 1–12 | population anneal on the GPU, research knobs, settings file |
| 13–17 | pair-worm polish, early abandon, deep re-solve |
| 18–28 | research modes (elite re-anneal, tempering, worm moves, resample variants) |
| 29–30 | wrong-length read fix, polish on deep jobs only |
| 31 | lease-local salt screen (E65–E67) |
| 32–34 | batched probe kernel and probe ladder (E70, E71) |
| 35 | fast score (E72) |
| 36–38 | batched-probe host work, prepare cache (E73) |
| 39 | cumulative ladder (E79) |
| 40 | helper-process solve (E84) |
| 41 | re-draw rule, push on target, top-K (E85) |
| 42 | helper probe re-order (E86) |
| 43 | probe gate (E88) |
| 44 | record hold (E90) |
| 45 | staged fast score, early exit, score-pick probe slot (E98) |

## Summary

| ID | Experiment | Verdict | Takeaway |
|---|---|---|---|
| E63 | Readiness: offline harness on the new family | bench only | Generator verified; first (wrong) reading that selection does not pay |
| E64a | Stock solvers on the new chain | bench only | Stock GPU Otsuka −5,030 in 33 s; bar ≈ −5,070 |
| E64b | Population anneal on the GPU (beta range, resampling, Houdayer) | kept | −4,986 → −5,036 at equal size |
| E64c | Cluster weights 250 and weak scale 1.5 | kept | +10 to +17 units |
| E64d | Schedule, ramp, strength, kernel-round knobs | rejected | No gain from any |
| E64e | Deep re-solve of promising salts + pair-worm polish (v2) | kept, later replaced | Best of round −5,067 → −5,072 |
| E64f | Early abandon | rejected | r 0.73 at 1/8 of the anneal, +1.5 to +2.4 simulated only |
| E64g | Funnel model of scan → deep | bench only | No stage arrangement beats −5,072.3 |
| E64h | Elite re-anneal from earlier runs' best states | rejected | Same plateau, no edge at equal compute |
| E64i | Decoding winners' on-chain solutions | bench only | Proofs carry full states; winners hold fewer violations |
| E64j | Plateau characterisation, repairs, tempering | rejected | Search failure, not a model bias; no repair pays |
| E64k | Outside review fixes and resample-energy variants | rejected | One host-energy bug fixed; no gain |
| E64l | Defect-directed path moves in the population step | rejected | Moves work locally, the anneal erases them |
| E64m | Smaller second deep stage (v3) | rejected | Freed no throughput |
| E64n | Cheap first check instead of the scan | rejected | Correlation equal, end-to-end race no gain |
| E64o | Wrong-length reads after a topology switch | kept (fix) | Drop malformed reads before the coordinator |
| E65 | Lease-local salt screen, structural score, two-step selection (v4) | kept | First structural score; v4 = screen → probe → one long anneal |
| E66 | Only the best lease waits for the GPU | kept | Leases stopped unsolved 85 % → 18 % |
| E67 | Generation turn released by the last generator thread | kept | Screening 167–300 → 701–837 salts/s |
| E68 | Parity of the upstream population-anneal PR against our build | bench only | Equal within noise on 96 salts |
| E69 | Field profile and probe-ladder bench | bench only | Short GPU probes separate winners' puzzles (AUC 0.72–0.96) |
| E70 | Batched probe kernel | kept | Identical energies, ~10× probe throughput |
| E71 | Probe ladder in the lease path | kept | +1.7 units mean, ≤ −5,086 share ×3 |
| E72 | Fast score from cached topology tables | kept | Scoring ×2.9–4.7, identical picks |
| E73 | Batched-probe host work and prepare cache | kept | First rung 47 s → 15.5–17.4 s |
| E74 | First rung of 2,048 | not run | — |
| E75 | Depth against tries on fixed puzzles | bench only | One long anneal beats four short ones |
| E76 | Pair strength in the anneal | rejected | Moves violations around, no gain |
| E77 | Are winners' puzzles special? | bench only | ~8 units better than a random score-top puzzle; ladder recovered ~3 |
| E78 | Which probe separates winners' puzzles per GPU-second | bench only | One cheap probe is mostly noise; averaging and summing rungs works |
| E79 | Cumulative probe ladder | kept | +1.13 ± 0.40 units |
| E80 | Fewer reads, more puzzles in the long-anneal slot | rejected | Time is per sweep, not per read |
| E81 | Long anneal at double depth | parked | n = 2, no gain |
| E82 | The PR #5 solver on our puzzle sets | bench only | 2.2–2.6 units deeper than our long anneal in ~1/6 of the time |
| E83 | Our structural score inside the PR #5 miner | rejected | Its serial instance generation caps selection at 32 salts/s |
| E84 | Hybrid: our selection, PR #5 solver as a helper process | kept | +2.43 units, 52.6 solves/h vs 29 |
| E85 | Re-draw rule, push on target, top-K | top-2 break-even, re-draws neutral | Rank 2 worth about half a rank-1 solve |
| E86 | Helper probe as judge of the final candidates | kept | ≤ −5,090 share 2.1 % → 9.1 %; probe 8 = break-even |
| E87 | Concurrent solve engines | bench only | ×1.72 at 4 engines, ×1.82 at 6 |
| E88 | Probe gate on the full solve | kept | Full ≈ −954.4 + 0.813 × probe; ≤ −5,091 per hour ×2 |
| E89 | Deep-limit bench | bench only | ×8.9 solver work = +0.39 units; floors near −5,096 |
| E90 | Record hold | kept (no hold seen) | Rare trigger by design |
| E91 | Ladder audit with the helper probe as judge | bench only | Ladder keeps deep puzzles; order in the top 48 is near noise |
| E92 | Cheap-probe fidelity bench | not run | — |
| E93 | PR #5's native selection against our ladder | bench only | 0 of 274 PR #5 probes ≤ −5,084; ours 19.4/h live |
| E94 | Other miners' winning puzzles under our solver | bench only | Same energies to the third decimal: the edge is selection |
| E94b | The record puzzle under our solver | bench only | First draw 0.090 from the record |
| E95 | Winners-based study of the first-stage score | bench only | AUC 0.9645; no cheap score does materially better |
| E96 | More re-draws near the bar, tighter solve limit | neutral | No measurable change in 13.7 h |
| E97 | Fast-score tail density and scorer throughput | bench only | Winner density ×400 at 10⁻⁵–10⁻⁴; scorer CPU-bound |
| E98 | Faster fast score and a score-pick probe slot | rejected | ×1.05 scorer; score picks probe 3 units shallower |
| E99 | PR #6 GPU instance generator on one shared GPU | bench only | Exact; ~5,000 salts/s per 10 % of mining given up |
| E100 | PR #4 funnel selection against our ladder | bench only | No measurable difference in pick quality (+0.39 units [−1.07, +1.93]) |

---

## E63 — Readiness: an offline harness on the new family (2026-11-11 … 11-13)

**Question.** The proof of work was announced to move to `quip/twisted-ice-dimer`. How do solve time, salt-to-salt spread and
short pre-solves behave on it, before the chain switches?

**Method.** Built the upstream coordinator 0.3.8-rc2 and `quip-cuda-otsuka` 0.3.6-rc1 (later rc4) with the canonical
generator; CPU runs of the CPU Otsuka solver on generated puzzles; a GPU smoke run on the shared GPU.

**Numbers.** Generator verified against 121 of 121 test vectors, 17–20 ms per salt on one core. GPU smoke: 2.2 s + 6.7 ms
per sweep per salt with the GPU shared. Salt effect ≈ 23 % of one solve's variance (sd 3.2 against 5.8 units at
64 × 1,024). Structural features predicted a salt's solved energy at r ≤ 0.23 (≈ 1 unit); a short pre-solve ≈ 3 units.
The CPU Otsuka solver beat the GPU Otsuka kernel per budget on our node; on CPU, depth won (8 × 4,096 ≈ 64 × 1,024 at
3× the rate).

**Verdict.** Bench only.

**Implication.** The conclusion drawn then, "selection does not pay on this family", rested on 96 salts and was **wrong**
at the selection ratios the field later used (E65 onwards: puzzle choice turned out to be worth ~17 units). Small-sample
salt studies understate selection because the tail that matters is 1 in 10⁴–10⁶.

## E64 — New chain: stock solvers and the GPU population anneal (2026-11-13 … 11-15)

The chain was reset to the new family on 2026-11-13. Stock software on our node could not reach the bar. E64 covers the
solver work of the first two days, in sub-entries.

### E64a — Stock solvers (bench only)

On the new-chain graph, GPU shared: stock GPU Otsuka 256 × 4,096 −4,986 (2.3 s), 256 × 65,536 −5,030 (33 s); `cuda-msa`
−5,003 (31 s); CPU Otsuka 64 × 4,096 −5,030 (58 core-seconds), 16 × 65,536 −5,049 (best −5,062, ~230 s per core);
the CPU dimer solver −4,948; pair-worm polish on top +2 to +6. Bar at the time ≈ −5,070 (rising as rounds were won).
Per-salt sd ≈ 5 (instance ≈ 3.3, solver ≈ 4).

Cause of the weak GPU result, found by reading the code: a family job is annealed over the generic SA beta range
(~0.044 → 105) instead of the family range (1, 12) the CPU solver uses, and every read is an independent anneal, while the
CPU solver resamples its population and takes Houdayer cluster moves. CPU ablation at 64 × 4,096: independent reads
−5,008, resampling only −5,026, Houdayer only −5,008, both −5,031.

### E64b — Population anneal on the GPU (patches 1–3; kept)

**Change.** Family beta range for jobs that name none (`QUIP_OTSUKA_BETA`, `stock` = upstream), population resampling and
Houdayer moves on the host between kernel launches (`QUIP_OTSUKA_RESAMPLES`, `QUIP_OTSUKA_HOUDAYER`), several sweeps per
launch (`QUIP_OTSUKA_CHUNK`).

**Numbers** (GPU shared with production): 256 × 4,096 stock −4,986 (2.3 s) → family beta −5,009 → + resampling 256
−5,026 → + Houdayer −5,036 (4.6 s). 64 × 16,384 −5,045 (6.8 s); 256 × 16,384 −5,050 (13.5 s); 256 × 65,536 −5,050
(55 s, one salt). Beta top 6–8 slightly better than 12 (default became 1, 8); a cold quench tail (`QUIP_OTSUKA_TAIL`)
added nothing; 256 population steps are enough. Stock needed 33 s for −5,030.

**Implication.** Upstream later adopted its own population anneal (PR #2); on current `main` this is the default.

### E64c — Cluster-draw weights and annealed weak scale (patches 7–8; kept)

**Change.** `QUIP_OTSUKA_TILE_MILLI=250`, `QUIP_OTSUKA_PAIR_MILLI=250` (upstream hard-coded 1,000 although its own comment
names 250 as the research default) and `QUIP_OTSUKA_WEAK_SCALE=1.5` (the device anneals on weak couplings × 1.5; reported
energies stay the true ones).

**Numbers** (shared GPU): 64 × 4,096 −5,046 in 1.75 s (was −5,028); 128 × 8,192 −5,055 (3.7 s); 160 × 16,384 −5,057
(7 s; was −5,047 to −5,050); 256 × 16,384 −5,060 (10.5 s); 160 × 65,536 −5,061 (25 s, plateau). Best of a 600 s round
≈ −5,066 against a bar of −5,077. Re-checked on the later recipe: weak scale 1.5 −5,057.8 against 2.0 −5,056.9;
weights 125 / 250 / 500 within 1.2. Retune at 512 × 16,384 on 6 fixed salts: weak scale 1.5 −5,062.9, 1.25 −5,060.4,
1.75 −5,057.7.

**Verdict.** Kept in every later settings file.

### E64d — Knobs measured without gain (patches 2, 6, 9, 10, 11; rejected)

Beta schedule knots (`QUIP_OTSUKA_SCHED`), weak-scale ramps (`QUIP_OTSUKA_WEAK_RAMP`), separate effective pair or tile
strength (`QUIP_OTSUKA_PAIR_EFF`, `QUIP_OTSUKA_TILE_EFF`), cluster weights 125 or 60, weak scale 1.75 or 2, Houdayer 0 or 4
rounds, 64 or 1,024 resamples, kernel cluster rounds 1 or 4 (`QUIP_OTSUKA_ROUNDS`; 1 is 7 % faster at the same quality),
single-spin period 2 or 4 (`QUIP_OTSUKA_SINGLE_EVERY`), thermal cycling. Population against sweeps follows one smooth curve:
about +3 units per doubling of compute. The knobs stay in the code for reproducibility; all are off when unset.

### E64e — Deep re-solve and pair-worm polish (patches 13, 15–17; kept as v2, later replaced by E65)

**Change.** Every salt gets a 160 × 8,192 scan; salts in the best 12 % of recent scans get 2 runs at 512 × 16,384, the
best 40 % of those 6 more (`QUIP_OTSUKA_DEEP=0.12,2,512,16384,24`, `QUIP_OTSUKA_DEEP_MORE=6,0.4`); the 8 lowest states get
0.3 s of upstream pair-worm polish (`QUIP_OTSUKA_PW_TOP=8`, `QUIP_OTSUKA_PW_BUDGET_S=0.3`; needs the patched
`quip-solver-cpu`).

**Why.** Puzzles differ (instance sd ≈ 3.3 of 5 total; one salt scored −5,072 to −5,075 in every configuration), more
attempts add only ~1 unit per doubling, so concentrating compute on easy salts was the lever.

**Numbers.** Bench: selected salts −5,057 → −5,063 … −5,070; polish +1.2 for 0.3 s. Live v1 (scan only, 0.44 salts/s):
mean −5,053.5, sd 5, best per round −5,065 to −5,071, bar −5,076.5 to −5,079. Live v2 over 28 rounds: mean best per round
−5,072.9 (v1 −5,067), mean gap to the bar 5.7, closest 0.4. First win on the chain (2026-11-14, −5,079.6) came from a salt
that scanned −5,065.4, deep stage 1 −5,077.8, stage 2 −5,079.4, polish −5,079.6. Live tail over 5,827 v2 results:
1 in 26 ≤ −5,070, 1 in 99 ≤ −5,075, 1 in 486 ≤ −5,078. On the later recipe the pair-worm schedule falls back to a
greedy pass (~16 ms) and is worth nothing; it stays configured but inert.

### E64f — Early abandon (patch 14; rejected)

**Question.** Can a population be stopped early when it will not finish well? (`QUIP_OTSUKA_ABANDON`.)

**Numbers.** The population mean at 1/8 of the anneal predicts the final at r 0.73, at 1/2 r 0.88; a simulation gives only
+1.5 to +2.4 units on best-of-round. Not adopted.

### E64g — Funnel model (bench only)

A simulation calibrated to live (deep-run mean −5,059.8, instance sd 3.3, scan noise 3.7, deep noise 3.0) found no
arrangement of scan and deep stages or quantiles that beats −5,072.3 best-of-round: the funnel was at its optimum and the
remaining ~6 units had to come from per-run quality or from far more salts.

### E64h — Elite re-anneal (patches 18–20; rejected)

**Change.** `QUIP_OTSUKA_ELITE`: restart a population anneal from the best state of each earlier run of the same salt at
beta b0, optionally for several rounds with fresh runs mixed in.

**Numbers.** One round over the best of 5 runs (256 × 8,192): b0 1.5 +1.8, **2 +2.6**, 2.5 +0.5, 3 +0.2, 4.5 +0.25. Rounds
stack from weak starts (best of 5 −5,055.8 → −5,064.5 after 8 rounds) but from strong starts gain little (two 512 × 16,384
runs then rounds: −5,058.6 → −5,062.2 and −5,065.0 → −5,066.2). At equal compute: 8 runs of 512 × 16,384 −5,063.0 and
−5,067.9 (182 / 170 s) against 2 runs + elite rounds −5,062.6 and −5,066.7 (132 / 126 s).

**Implication.** It reaches the same ≈ −5,064 plateau of an average salt somewhat cheaper; it does not pass it.

### E64i — Decoding winners' solutions (bench only)

Winning proofs carry the full solution in the family's compact witness codec. Decoding them (the validator's own codec;
our tool `windec`, not included here) and rebuilding the instance from the salt gives the winner's state; decoded energies equal the
chain's in every case. First 10 proofs (−5,077.6 to −5,080.7): 2–7 strong violations, weak −314.6 to −324.4, while our deep
states had about the same weak energy with ~14 violations. **Correction made later the same day:** that compared winners'
best-of-round on their easiest salt with our average deep run; the like-for-like gap was 6–7 units (−5,079 against
−5,072.5), not ~9 violations. Also closed: exact search for zero-violation states in 140–700-node balls (none) and exact
full-energy re-solve of 60–250-node balls (+0.2).

### E64j — Plateau characterisation (patches 22–25; rejected)

Deep runs 512 × 16,384 on 5 fixed salts: weak scale 1.5 −5,061.3 (15.8 violations, weak −322.9); 1.0 −5,058.1 (7.6,
−303.3); pair weight 1,000 −5,059.6; full weights −5,055.5; slow cooling through beta 2–4 −5,061.6 (12.8, −317.2). 8× compute
as 2,048 × 32,768 (110 s per salt) +1.8 at scale 1.5 (−5,063.2) and nothing at scale 1.0. Winners' states are lower even in
our annealed energy (about −5,237 against −5,223), so this is a search failure, not a bias of the weak scale. Failed ways
past it: 10 s of pair-worm polish (16.0 → 14.2 violations, +1.0); exact local repair (infeasible within 400-node balls);
global integer-program repair with optional violations (62 flips, 18 → 14 violations, +1.5, then stuck); two-phase anneal
with a frozen strong layer (weak stays −300); parallel tempering from a random start (`QUIP_OTSUKA_PT`): 512 × 16,384
−5,036.6, 512 × 65,536 −5,048.5 (exchange acceptance 0.5–0.7, only 256 exchange steps).

**Correction (2026-11-16, E65):** "8× compute adds only +1.8" scaled the *reads* axis. Scaling *sweeps* is what closes the gap:
one 256 × 262,144 anneal lands 1.4 units short of a leading miner on its own winning puzzles, against 7.2 short for
scan + 8 deep runs.

### E64k — Outside-review fixes and resample-energy variants (patches 21, 26, 27; rejected)

An outside review of the code found one real bug in bench-only paths: the host's annealed energy took the strong layer at
the cluster weight (1.375 × strong + 1.5 × weak instead of strong + 1.5 × weak); fixed in patch 26 (it only affected
`QUIP_OTSUKA_RESAMPLE_EFF` and tempering, not the production path). It also noted that at beta 8 the cluster graph
percolates (largest component ~3,950 of 4,575 spins), so cluster flips cannot repair defects. Tests after the fix (5 salts,
512 × 16,384, baseline −5,061.3): resampling on the annealed energy −5,056.2 (22.8 violations); selection energy
strong + x × weak (`QUIP_OTSUKA_RESAMPLE_SCALE`) x 0.75 −5,060.0 (11.4 violations, weak −312.8), 0.5 −5,059.9, 0.25 −5,056.9;
tempering started from the annealed population −5,058.9 and −5,063.3; best state re-captured after Houdayer
(`QUIP_OTSUKA_BEST_AFTER`) no gain. Before the fix: `RESAMPLE_EFF` +0.5 and +0.9 on 14 salts each (noise); with it scale
1.75 −5,054.7, 2.0 −5,048.7. A final descent on the true energy: −5,051.4 against −5,053.1.

### E64l — Defect-directed path moves (patch 28; rejected)

**Change.** `QUIP_OTSUKA_WORM="b_lo,b_hi,attempts,maxlen,noise"`: open-path moves aimed at defects in the host population
step, Metropolis acceptance at the step's beta, 24 host threads (a port of the review's CPU prototype, which had improved one
salt's best by 1.4).

**Numbers.** The move works locally: −8.7 units per replica per step at beta 2, falling to −0.03 by beta 8. End results on
5 salts (baseline −5,060.2 to −5,061.3): window 2–4 × 10 attempts −5,060.7; 2–8 × 10 −5,059.4; 4–8 × 20 length 128
−5,059.7; 2–4 × 100 length 128 −5,060.6 (+10 s); 6.5–8 × 150 length 256 noise 0.8 −5,059.4 (+24 s). Violations unchanged
(13–15.6). The anneal washes the gains out.

### E64m — Smaller second deep stage, v3 (settings only; rejected)

`QUIP_OTSUKA_DEEP_MORE=6,0.08` (best 8 % instead of 40 %) because the second stage used ~58 % of GPU time for +0.4 to +1.6
units. Over 9 h: 536 scans/h against v2's 543/h and 62 deep re-solves/h against 65/h: no throughput freed (two concurrent
jobs already filled the GPU), only the second stage's gain lost. **Lesson:** measure throughput before assuming a stage is on
the critical path.

### E64n — Cheap first check instead of the scan (patch 30; rejected)

Correlation with a deep 512 × 16,384 run on 60 salts: live 160 × 8,192 scan (2.4 s) 0.53; 160 × 2,048 (0.91 s) 0.52;
32 × 4,096 (0.75 s) 0.53; 64 × 2,048 0.46; 64 × 512 0.34; 160 × 1,024 0.31. A parametric model predicted about +1 unit; an
end-to-end race (3 × 600 s rounds each) gave −5,075.4 against −5,076.3 for live: no gain, because deep runs (~20 s each)
cap both arms at 5–6 per round. `QUIP_OTSUKA_PW_DEEP_ONLY` (polish only deep jobs) was added for it and left off.
**Lesson:** race end to end before believing a funnel model.

### E64o — Wrong-length reads after a topology switch (patch 29; kept)

On 2026-11-14 the chain switched its default topology (to 4,560 nodes) in the middle of a round; the coordinator release we
ran encoded our 4,575-spin reads against the new topology and every submission of that round failed with
`family witness: length`. The root cause is upstream coordinator behaviour; the miner-side guard drops (and logs) any read
whose length differs from the job's node count, so one bad read can no longer sink a whole submission.

## E65 — Lease-local salt screen and the structural score (patch 31; 2026-11-15/16; kept as v4)

**Question.** Winners' salts are little-endian counters, so the gap between consecutive wins of one miner gives a lower bound
on how many salts it goes through: ~126, ~440–800 and (p75) up to 8,000 salts/s for the three leading miners of the time,
against our 0.44 solved/s. Does screening many salts cheaply and solving only the best help us?

**Method.** Phase 1: 304 decoded winners against 2,000 random puzzles (generator reproduces chain puzzles exactly) and 3,862
of our live results. Phase 2: lease-local `sample_lease` behind `QUIP_OTSUKA_SCREEN`: a CPU generator pool inside the miner
scores its own lease's salts and sends only the best to the GPU. Phase 3: 600 s bench rounds on fixed seeds, A = the v2
stock path, B = screen.

**Numbers.**
- Features: winners have fewer free nodes (in no tile and no pair, −1.23 sd) and fewer tiled pairs (−1.06 sd); score
  free + 0.42 × tiled pairs puts held-out winners at −1.25 sd. Within our own live results these features correlate only
  0.12–0.22 (range-restricted labels). GPU micro-probes cost ≥ 0.1 s per salt then: too slow to screen with.
- Two more cheap terms: f_mean (mean weak energy inside each tile over its six ice states, pair partners merged) and m2
  (the same between tiles sharing a variable), winners −1.17 and −1.09 sd, independent of the counts. Combined score
  **free + 0.773 × tiled pairs + 20.8 × f_mean + 22.2 × m2** (`QUIP_OTSUKA_SCREEN_W=0,1,0.773,20.8,22.2`): held-out
  winners at −2.00 sd (was −1.22), 38 % of winners beyond random's best 1 % (was 17 %). This is the score every later
  stage uses, and other miners adopted it as a pre-screen once published.
- Races (6 pairs, first score): B ahead in 6 of 6, mean +2.9, best −5,079.6, never −5,080. Combined score, wide tail:
  −5,080.7 mean against v2's −5,074.4 (+6.3). Production-like (tail 1, 30,000-salt leases): +3.4.
- Like-for-like replay: on 16 winners' puzzles we scored −5,080.4 (theirs −5,088.1), on 16 random puzzles −5,063.3: **puzzle
  choice ≈ 17 units, solver ≈ 8.** On winners' puzzles our states had 7.1 violations at weak −206.5 (theirs 2.9, −206.0).
  Our state and theirs differ in 1,200–2,200 spins forming one connected piece: different basins, no local repair.
- Long sweeps (replay of 10 winning puzzles of a leading miner): one 256 × 262,144 anneal (150–210 s) lands 1.4 short of the
  winner on average (−0.6 to +2.8; best −5,087.8, beats the winner on one puzzle), scan + 8 deep runs 7.2 short.
- Screen → one long anneal on the best candidate: 262,144 sweeps −5,080.4 mean; 65,536 −5,078.1; 524,288 −5,079.0. No
  better than the screen + v2 funnel: our screened candidates are harder than the winners' puzzles.
- Two-step selection (screen → short GPU solve of the top N → one 256 × 262,144 anneal on the best): probe 32 × 4,096 × 16
  −5,081.4 mean; probe 160 × 8,192 × 8 −5,082.3 (+5.2 at 300 s over v2). Deployed as v4.
- Live v4: 5 wins in 99 rounds (5.1 %) against about 1 in 600 on v2; mean gap to the winner 5.6.

**Production hazards found.** The coordinator connection gives a stopped lease thread 1 s to return or fails fatally; the GPU solve needs a
cancel check and every lock wait must be stop-aware (`QUIP_OTSUKA_SCREEN_SELFSTOP_S` measures it: 2–28 ms). Concurrent
leases each spawned a 20-thread pool until leases were made to take turns. The coordinator sizes leases from salts completed
per second, so a long GPU tail per lease shrinks leases (use tail 1). The deep quantile must be < 1. Pair detection must test
for K4 membership (it missed ~130 of 1,350 pairs at first).

**Verdict.** Kept (v4). **Implication:** on this family the screen, not the solver, set most of the gap.

## E66 — Only the best lease waits for the GPU (2026-11-16; kept)

**Question.** Live screening ran at ~300 salts/s against ~1,100 on the bench. Why?

**Finding.** In v4 every lease waited for the GPU after screening and kept its lease credit, so generation stalled;
80–90 % of leases were stopped at round end with their candidates unsolved.

**Change.** `QUIP_OTSUKA_SCREEN_HOLDERS=1`: only the lease holding the best candidates waits; others return as soon as they
are screened, and a better lease replaces the holder.

**Numbers.** First 15 minutes: leases stopped unsolved ~85 % → 5 of 28 (18 %), 4 holder hand-offs, screening 331 salts/s
(about unchanged: the next bottleneck is E67). The drive bench hands out one lease at a time, so this effect shows only live.

## E67 — Generation turn released by the last generator thread (2026-11-16; kept)

**Finding.** The one-lease-at-a-time generation lock was released only when the lease's GPU loop iterated, so a lease that
started GPU work before its own generation ended held it through minutes of probes and solves (a lease waited 593 s with 0
salts screened). **Change:** the turn is a flag released by the last worker thread the moment generation ends; waits are
stop-aware. **Numbers:** 701 salts/s screened in the first 15 minutes (E66 hour 167, v4 ~300), coordinator-measured rate
1,223 salts/s, leases 73,393 salts (were ~9,000); 837 salts/s over the next hour. Gate after 100 rounds: 5 wins in 99 (5.1 %).

## E68 — Parity of the upstream population-anneal PR against our build (2026-11-16; bench only)

Our build (patches 1–31) against a build of the upstream PR that carried the population anneal, on 96 fixed lease salts at
160 × 4,096: means −5,052.8 (ours) against −5,052.5 and −5,052.9 (two PR builds). Not bit-identical (different random
streams), equal within noise. No ledger row was written at the time; numbers from the bench outputs.

## E69 — Field profile and the probe-ladder bench (2026-11-17; bench only)

**Question.** After a dry spell (0 wins in 102 rounds while one multi-GPU miner took 63 %), how does the leading miner select
puzzles, and can one GPU do the same?

**Method.** Chain scan of 649 proofs since the recipe change, puzzles regenerated and scored; replay benches: 60 of the
leader's winning puzzles, 80 ordinary puzzles from the best 4 % of our score, 40 random; and a pool of 10,638 random
puzzles through score (best 480) → 32 × 1,024 → best 48 at 64 × 4,096 → best 12 at 160 × 8,192 → 4 long anneals,
against 4 random score-passers.

**Numbers.** From 2026-11-17 the leader's winners sit in the best 5.8 % of our published four-term score (50 of 52): the
score became its pre-screen (keep ≈ 4 %). Its salt streams ran ≈ 1,800 + 1,800 + 1,000 salts/s. Probes, leader's puzzles
against ordinary score-passers: 16 × 256 gap 6.0 units (AUC 0.72), 32 × 1,024 9.2 (0.90), 64 × 4,096 10.1 (0.95),
160 × 8,192 10.5 (0.96). Within a group the rungs are almost uncorrelated (0.1–0.3), so rungs stack. Probes are
deterministic per nonce. Time per salt 0.47 s + 0.83 ms per sweep (shared), the same for 16 to 256 reads. Pool test:
ladder picks −5,080.7, controls −5,077.0, the leader's puzzles under the same anneal ≈ −5,086.5. A 10.6k pool held none of
that grade (about 1 in 30–50k).

**Implication.** Selection by short GPU solves works, but a single-puzzle probe was too dear for one card.

## E70 — Batched probe kernel (patch 32; kept)

**Question.** One small probe leaves most of the GPU idle (one launch saturates near 128 blocks). Can several puzzles share a
launch sequence?

**Change.** A second kernel entry, a copy of the anneal kernel that takes each block's puzzle arrays from an offset table and
salts the cluster rounds with the block index inside its puzzle; host-side `run_batch` and `probe_batch`, no population step
(`QUIP_OTSUKA_BATCH=K` for replay). Production paths and the single-puzzle kernel are untouched.

**Numbers.** Identity: 181 of 181 puzzles identical energies at batches of 8, 16 and 32 against the un-batched run.
Throughput (16 × 1,024, shared): 2.0 → 19–21 salts/s. Separation of winners' puzzles from score-passers without the host
step: 16 × 512 AUC 0.88 at 29 salts/s; 16 × 1,024 0.88 at 21/s; 16 × 4,096 0.93 at 7.1/s; 32 × 4,096 0.97 at 3.5/s;
64 × 8,192 0.99 at 0.97/s (the old 160 × 8,192 population probe: 0.96 at 0.14/s). Same nonce and sweep count share random
streams across read counts (8 × 1,024 against 16 × 1,024 correlate 0.78), so rungs should differ in sweeps.

## E71 — Probe ladder in the lease path (patches 33–34; kept)

**Change.** `QUIP_OTSUKA_SCREEN_LADDER="n1,r1,s1,w1,…"`: the holder's best n1 candidates get batched probes, the best n2 the
next rung, and so on; the survivor gets the long anneal. The batch probe synchronises every 8 launches so a stopped lease
drains quickly.

**Numbers.** Bench race (3 × 600 s): live −5,077.9 mean, ladder −5,079.2. Pool test on 90,000 scored puzzles: ladder
−5,082.5 mean, control −5,081.0. First rung: 16 × 256 as good as 16 × 512 (AUC 0.91 against 0.88, 36 against 29 salts/s);
8 × 256 0.85 at 57/s; 4 × 256 0.81. Stop latency follows the number of held candidates: 64 ms with 48, ~125 ms with 1,024,
600–730 ms with 4,096 and the ladder (limit 1 s), so keep was set to 1,024. Live (141 long anneals in 6.4 h): mean −5,081.9
against −5,080.2 (+1.7), ≤ −5,086 7.1 % against 2.4 %, ≤ −5,087 3.5 % against 1.0 %. At 358 anneals: mean −5,081.8, 6.7 %
≤ −5,086. The last rung predicts the anneal at r 0.36. Live, the first rung was CPU-bound: 21 s for 1,024 probes where the
GPU needs ~6 s (→ E73).

## E72 — Fast score from cached topology tables (patch 35; kept)

**Question.** Every regular winner by then selected with our published score, and winner-grade puzzles concentrate at its top
(the best 0.1 % held ~31 % of them). Our scoring was CPU-bound at ~780–860 salts/s on 20 threads (≈ 24 ms of thread time
per salt, ~80 % of it building the puzzle). Can a salt be scored without building it?

**Change.** Profiling the generator: 67 % is candidate-tile enumeration, which depends only on the topology. `fastgen`
(a new module of `ising-generators` 0.0.14, a local patch not included here) caches the topology tables once per lease and replays the
salt's random streams (shuffles → matching → gauge → weak → dimers) to return the four score terms without building the
instance. `QUIP_OTSUKA_SCREEN_FAST=1`: a salt is built by the unchanged upstream generator and re-scored by the unchanged
code only when it would enter the keep set, or is one of the first `QUIP_OTSUKA_SCREEN_FAST_CHECK` (8) of a lease; a
difference over 1 milli turns the fast path off for the process.

**Numbers.** Four terms identical on 23,186 of 23,186 salts; 2.38 ms against 22.9 ms per salt (one Apple M1 Max core).
On our node: scoring phase of a 60,000-salt lease 141 s → 30 s (425 → 2,000 salts/s with production competing for the CPU);
rungs 1–4 and the winner salt identical; 2,487 salts/s on a 2,000,000-salt lease (v6 ~860/s); soak 300,000 scored, 11,346
built, 0 mismatches. Live: ~4,000 salts/s scored (was ~860), 2.9–4.7× depending on contention, identical picks. Half of the
remaining time is building the ~1,024 × (1 + ln(N/1,024)) salts that pass through the keep set.

**Limit.** Later measurements (E97, E98) put the live rate at ~4,100–4,735 salts/s with all 24 threads busy; per salt 83 % of
the time is the f_mean / m2 loops, which are memory-latency bound.

## E73 — Batched-probe host work and prepare cache (patches 36–38; kept)

**Change.** `QUIP_OTSUKA_BATCH_TIMING=1` timers; the batched probe takes the lowest read by the kernel-side energy and computes
the protocol energy for that read only (always on, checked every probe; `QUIP_OTSUKA_BATCH_ECHECK=1` runs the old full
pass); `QUIP_OTSUKA_PREP_CACHE=1` caches the per-topology colouring, permutation and row layout, checking the first 32
cached builds of a process against a fresh build.

**Numbers** (ms per 1,024 candidates, generation finished): before — prep 6,974, host 803, upload 287, gpu 10,434,
post 2,431, energy 10,078 (rung 31.9 s); after the energy change — energy 485, post 724 (15.6 s); after the cache — prep
1,073 (11.8 s; ladder 23.3 s against 30.5 s). With generation running: rung 1 47.1 s → 15.5–17.4 s, ladder 56–69 s →
27.9–30.1 s. All gates identical (rungs, winner salt, soak anneal −5,084.935); 0 energy or cache mismatches. The first rung
is now ~75 % GPU time.

## E74 — First rung of 2,048 (not run)

Planned after E73 (a first rung twice as wide in the time 1,024 took), after re-measuring the stop path at keep 2,048. Not
run; E79 changed the ladder's shape instead.

## E75 — Depth against tries on fixed puzzles (bench only)

Nine winning puzzles of one miner (wins at ≈ −5,090), 256 reads: four runs at ~65,536 sweeps each (different streams):
single-run mean −5,082.0 (34 runs), best of four per puzzle −5,084.7; one 262,144-sweep anneal −5,086.5. No short run reached
−5,090; long runs reached it in 5 of 36. **Depth beats tries.**

## E76 — Pair strength in the anneal (rejected)

Winning states of one miner break 0.25 dimer pairs on average, ours 3.0 on the same puzzles (tile violations 3.1 against
3.3). Long anneal 256 × 262,144, mean energy / tile violations / broken pairs / weak energy: baseline −5,086.53 / 3.33 /
3.00 / −211.2; `PAIR_EFF=1500` −5,086.71 / 4.22 / 0.89 / −208.9; `PAIR_EFF=2000` −5,085.14 / 3.89 / 0.33 / −205.6;
`PAIR_MILLI=1000` −5,085.75 / 2.89 / 2.44 / −208.4 (the winner on the same puzzles: −5,090.56). Stronger pairs keep pairs
intact but the cost moves to tiles and weak energy; run-to-run sd ≈ 2 units, so all four are equal within noise.

## E77 — Are winners' puzzles special? (bench only)

Our long anneal averaged −5,086.5 on winners' puzzles but −5,081.5 on our own ladder picks. With 100,000 fresh random
puzzles as reference (score cuts 0.26 % = 1,012.7, 1 % = 1,025.3, 4 % = 1,040.8): winners' median score percentile 0.16 %
to 0.84 % for the four score-screening miners (one miner without a score screen: 9 %). Batched probe 16 × 256: 215 winners
−4,979.0 (sd 5.1), 260 clean top-0.26 % −4,972.4 (sd 5.6), 100 from the 0.26–4 % band −4,969.7. Single-puzzle 64 × 8,192:
59 winners −5,067.6 (sd 4.7), 40 clean top −5,058.4 (sd 4.1); 90 % of winners inside the best 10 % of clean top. Long anneal:
17 winners' puzzles −5,086.5, 9 clean top −5,078.2, live ladder picks −5,081.5.

**Finding.** Winners' puzzles are ~8 units better than a random score-top puzzle and a deep probe sees it (~2 sd); our ladder
recovered ~3 of the 8. **Trap:** a first control set accidentally contained 186 winners' puzzles (72 % of the control);
always build random references from fresh salts.

## E78 — Which probe separates winners' puzzles per GPU-second? (bench only)

60 winners' puzzles against 100 clean top-0.26 % controls, separation d′ = gap / pooled sd, cost per puzzle on the shared GPU.
Batched: 16 × 256 d′ 1.29 (22 ms), 16 × 1,024 1.65 (47 ms), 32 × 4,096 2.18 (281 ms), 64 × 8,192 2.32 (1,111 ms).
Single-puzzle pipeline at the same depths: 1.15 (709 ms), 1.61 (1,260 ms), 1.78 (3,518 ms), 2.26 (6,474 ms): no better,
6–30× dearer. **One cheap probe is mostly noise:** repeat-to-repeat correlation on the same puzzle 0.17. Mean of k repeats of
16 × ~256: d′ 0.91 / 1.43 / 1.74 / 2.38 for k = 1 / 2 / 4 / 8 (172 ms at k = 8, as good as 64 × 8,192 at a sixth of the
time); the minimum over 64 or 128 reads in one probe does not help (1.37, 1.28). Sum of standardised probes over the four
rungs: d′ 2.66. A one-factor model fitted to these numbers (loadings 0.41 / 0.47 / 0.63 / 0.75) reproduced the live rung
statistics and put the live pick at ~2.2–2.4 sd of probe quality against ~3.1 for winners' puzzles.

**Implication.** The kernel was right, the ladder's shape was wrong: it cut to 10 % on one noisy probe and forgot earlier
rungs.

## E79 — Cumulative probe ladder (patch 39; kept)

**Change.** `QUIP_OTSUKA_SCREEN_LADDER_CUM=1`: each rung's energies are standardised over the puzzles probed at that rung; a
puzzle ranks on the weighted sum of its z-scores over all rungs so far (`QUIP_OTSUKA_SCREEN_LADDER_W`); repeated rungs draw
fresh random streams. Shape (v9): 1,024 → 512 → 256 at 16 × 256, 128 → 48 at 16 × 1,024, 16 at 32 × 4,096, 4 at
64 × 8,192; weights 0.5, 0.5, 0.5, 0.6, 0.6, 1.0, 1.7. (E88 later dropped the 7th rung.)

**Numbers.** Switch off: all rungs and the pick identical to the previous binary. Switch on: 7 rungs in 34.5 s against 22.2 s
(shared GPU). Model estimate +2 units. Live: 61 anneals mean −5,082.76 against −5,081.63 (81 anneals), **+1.13 ± 0.40**;
after 17.4 h, 407 anneals: mean −5,082.78, sd 2.45, 8.4 % ≤ −5,086, 4.2 % ≤ −5,087, two wins.

## E80 — Fewer reads, more puzzles in the long-anneal slot (rejected)

18 puzzles, 262,144 sweeps, shared GPU: 64 reads −5,083.16 (~148 s), 128 reads −5,084.91 (~157 s), 256 reads −5,085.66
(~210 s). A quarter of the reads costs 2.5 units (worse on 17 of 18) and still takes 70 % of the time: the anneal's time is
per sweep, not per read. Gain per doubling 1.75 then 0.75, so 512 reads was not run.

## E81 — Long anneal at double depth (parked)

256 × 524,288, stopped after 2 of 18 puzzles to free the GPU: −5,091.595 and −5,087.385 against −5,092.405 and −5,089.855 at
262,144 (~417 s per puzzle shared). No gain on two puzzles; not a verdict. The PR #4 write-up reports 524k → 1M sweeps worth
only −0.9 at 64 reads.

## E82 — The PR #5 belief-propagation solver on our puzzle sets (bench only)

**Method.** PR #5 at commit 97faa92, built unmodified, run in an isolated container (no network, only the bench folder
mounted), one full solve per puzzle with its published settings minus screens and re-rolls, one engine, shared GPU.

**Numbers.** 18 fixed puzzles: −5,088.25 against our long anneal −5,085.65, **−2.59 units (SE 0.48)**, better on 17 of 18,
35 s against ~210 s. Our last 120 live picks: −5,085.23 (sd 2.44) against −5,083.00, −2.22 (SE 0.17), better on 105 of 120,
correlation 0.72; ≤ −5,088 18 against 2. 60 winners' puzzles: −5,088.69, sd 2.02, 14 of 60 ≤ −5,090 (our live picks: 2 of
120). 88 score-only picks (best 0.26 %, no ladder): −5,079.38: **our ladder is worth ~5.9 units over the score alone under
this solver**, winners' puzzles another 3.5. Its published configuration with its own screen, driven by our coordinator for
420 s on 4 engines: 805 salts (1.9/s), 96 reached its stage 1, 2 full solves (−5,083.285, −5,085.825).

**Correction.** Early reports put a full PR #5 solve at ~0.5 s; those were stage-0 rejects. A full solve takes ~29 s on our GPU
(~35 s shared).

## E83 — Our structural score inside the PR #5 miner (rejected)

`struct_cutoff` / `struct_streams` (patch 01 to the PR #5 crate, under `helper/`): our score as a CPU prefilter in front of its
probes. Best 1 %, 900 s, our coordinator: 0 errors, but only 32 salts/s reach the score (the core library generates instances
serially), 266 passed, 7 full solves (28/h) at mean −5,084.2, engines ~45 % idle. Selection volume is the limit (our miner
scores ~4,000/s). Superseded by E84.

## E84 — Hybrid: our selection, the PR #5 solver as a helper process (patch 40; kept)

**Change.** `QUIP_OTSUKA_BP_HELPER=<command>`: a long-lived helper (the PR #5 solver with a `--serve` mode we added, protocol in
`helper/PROTOCOL.md`) solves the ladder's pick. The miner writes the canonical instance, sends one request line, re-scores
every returned read on its own graph and pushes only reads whose energy matches. Any failure falls back to the long anneal.

**Gates.** Helper unset: all ladder lines and the pick identical to the previous binary; helper alone on 18 puzzles: best
energies identical to the standalone runs, cancel answered in 0.03 s; lease stopped mid-solve: miner returned in 125 ms;
wrong helper path: fallback works; no instance files left behind.

**Numbers (live, first 100 solves).** Mean **−5,085.21** (sd 2.49) against −5,082.78 over 407 long anneals (**+2.43**, as the
bench predicted); ≤ −5,086 37 % (was 8.4 %), ≤ −5,087 25 % (was 4.2 %); **52.6 solves/h** against 29; helper solve median
30.7 s, ladder 28.0 s; 0 fallbacks, 0 energy mismatches. A win 71 minutes after the switch.

## E85 — Re-draw rule, push on target, top-K (patch 41; top-2 break-even, re-draws neutral)

**Change.** `QUIP_OTSUKA_BP_REDRAW_RULE=mean` (re-draw the same puzzle with a new random shift only while the mean of its draws
stays under the threshold); `QUIP_OTSUKA_BP_REDRAW_PUSH=1` (push as soon as the best clears the coordinator's submit target:
a lease accepts one push per salt, so "push first, improve later" cannot reach the coordinator); `QUIP_OTSUKA_BP_TOPK=k`
(also solve ranks 2..k of the last rung).

**Numbers.** 12.3 h on top-2, 903 solves, 432 pairs: rank 1 n = 471, mean −5,085.04, ≤ −5,090 2.1 %; rank 2 n = 432,
−5,084.35, ≤ −5,090 0.9 %; paired rank 2 − rank 1 = +0.61 (se 0.15), rank 2 better in 42 %. A rank-2 solve is worth about
half a rank-1 solve in the tail and costs 30 s against 57 s for a fresh cycle: break-even, so top-2 stayed and top-4 was not
run. (An early reading, 3 wins in 3 h all from rank-2 picks, did not hold up.) Draws on one puzzle stay within ~0.3 units
(e.g. −5,091.0 / −5,090.8 / −5,090.7), so re-draws add little; kept because they are rare and stop early.

**Implication.** With rank 2 better in 42 % of ladders, picking the better of the two *before* the full solve is worth more
than solving both (→ E86).

## E86 — The helper's own short probe as judge (patch 42; kept)

**Bench.** On our 120 live picks: short run against full result of the same puzzle, shared GPU: 1,500 updates (1.85 s) r 0.754;
**3,000 updates (3.16 s) r 0.754**; 6,000 (5.57 s) r 0.781; 3,000 with an independent seed r 0.788 (the signal is the puzzle,
not a shared stream). Best of k by the 3,000-update probe, then full solve: k = 2 −5,086.3 (random −5,085.23); **k = 4 −5,087.25
(+2.0, 80 % of the oracle −5,087.77)**; k = 8 −5,088.10 (oracle −5,088.72); P(full ≤ −5,090): 1.7 % → 6.6 % (k = 4) → 12.9 %
(k = 8).

**Change.** `QUIP_OTSUKA_BP_PROBE_HELPER=<command>`: a second helper process (same binary, probe configuration) probes the best
`QUIP_OTSUKA_BP_PROBE_N` final candidates; each probe's best read is verified, the candidates are re-ordered by probe energy,
the top-K full solves follow that order. Probe results are never pushed.

**Numbers (live).** 66 solves: mean **−5,086.20** against v12 rank 1's −5,085.04 (+1.16); ≤ −5,089 16.7 % (was 4.2 %); ≤ −5,090
9.1 % (was 2.1 %); best −5,094.755 (the deepest winning energy on the chain at that time). At 245 solves: mean −5,086.31,
≤ −5,090 7.3 %, ≤ −5,091 2.0 %. Probes 2.2–2.3 s each, ~9 s per set of 4. Probe mean by ladder rank −5,080.55 / −5,079.70 /
−5,078.86 / −5,078.31: the ladder order still carries information.

**Probe 8 (v15).** 150 solves: mean −5,086.56 against −5,086.31; ≤ −5,090 6.7 % against 7.3 %; 40.5 solves/h against ~47;
per hour ≤ −5,090 2.7 against 3.4, ≤ −5,091 1.6 against 0.9: **break-even**, not kept. By ladder rank (69 ladders): rank 1
probes ≤ −5,086 in 7.2 %, ranks 5–8 in 0–1.4 %.

## E87 — Concurrent solve engines (bench only)

The PR #5 miner under our coordinator, every salt one full solve, 300 s per setting, shared GPU: 1 / 2 / 4 / 6 engines =
78 / 112 / 135 / 142 full solves per hour (×1.00 / 1.44 / 1.72 / 1.82); per-solve wall time 46 / 64 / 108 / 152 s.
Concurrency is worth ≤ ×1.7 on the solve part on this card. Not adopted: E88 made the solve part smaller instead.

## E88 — Full solve only when the probe is deep enough (patch 43; kept)

**Finding (308 live full solves).** **full ≈ −954.4 + 0.813 × probe, residual sd 1.66.** Results ≤ −5,090 / −5,091 / −5,092 by
the pick's probe energy: probe ≤ −5,086 (n 24): 13 / 6 / 4; (−5,086, −5,085] (23): 5 / 0 / 0; (−5,085, −5,084] (38):
2 / 1 / 0; (−5,084, −5,083] (51): 0 / 0 / 0; (−5,083, −5,082] (59): 4 / 1 / 0; above −5,082 (112): 0 / 0 / 0. No round in 197
had been won above −5,090, so ~72 % of the 30 s full solves could not win.

**Change.** `QUIP_OTSUKA_BP_SOLVE_MAX=<milli>`: a candidate is full-solved only when its verified probe is at or under the
limit; the walk stops at the first one above it; nothing clears = no solve, next ladder. No verified probe = limit not
applied. The 7th ladder rung was dropped (it only ordered the four the probe re-orders anyway).

**Model** (331 ladders): cycle 66.6 s → 41.5 s; ×1.45 / ×1.49 / ×1.60 results per hour at ≤ −5,090 / −5,091 / −5,092.

**Out-of-sample check of the limit.** The first limit, −5,084.0, was chosen on those 308 solves. On the 85 solves made after
it was chosen, −5,084.0 kept 25 % of the solves but only 2 of 4 results ≤ −5,090 and would have dropped a win (probe
−5,083.965, full −5,092.175); −5,083.0 kept 49 % of the solves and 4 of 4. On all 393 solves: −5,083.0 keeps 24 of 28
(≤ −5,090), 10 of 11 (≤ −5,091), 5 of 5 (≤ −5,092); −5,084.0 keeps 22 / 8 / 4. The live limit went to −5,083.0.

**Numbers (live, 6.3 h, 418 ladders).** 61 % of solves skipped; 66.5 ladders/h (was ≈ 54), cycle median 36.0 s; rank 1 mean
−5,087.85; per hour ≤ −5,090 2.86, ≤ −5,091 1.91, ≤ −5,092 0.95 against v14 3.4 / 0.96 at ≤ −5,090 / −5,091: **the deep end
up about ×2 at ≤ −5,091, flat at ≤ −5,090**; best probe per set ≤ −5,084 in 24.3 % of sets. Rank 2 (n 21) mean −5,086.63,
none ≤ −5,090. A win at −5,095.545 in this window (target of −5,095 reached once). Kept.

**Later (E96):** −5,084.0 again, after the bar moved (see E96).

## E89 — Deep-limit bench: do our deepest puzzles go deeper with more effort? (bench only)

**Part 1 (live log, no GPU).** 150 full solves on 116 puzzles, 15 solved more than once: within-puzzle sd 1.21 units; best of
4 beats the first draw by +0.21 on average (max +3.33); one puzzle gave −5,092.375 on three separate draws; first draws across
puzzles sd 2.10: puzzle choice counts about twice as much as re-drawing. Result does not depend on solve time (corr ≈ 0).

**Part 2 (bench, 12 puzzles rebuilt from logged nonces; all digests match the live helper).** Arms: A = live config, 4 new
draws; B = main stage 30,000 → 100,000 updates (×3.3 work); C = B + 512 replicas (×8.9 work).

| puzzle | live | new draws (A) | ×3.3 | ×8.9 |
|---|---|---|---|---|
| our deepest (−5,095.545 live) | −5,095.545 (rerun identical) | −5,095.255 / −5,096.065 / −5,095.545 / −5,095.545 | −5,096.085 | −5,096.085 |
| 2 | −5,090.4 / −5,091.6 / −5,093.8 | −5,092.2 / −5,093.2 / −5,092.4 / −5,092.1 | −5,092.055 | −5,092.825 |
| 3 | −5,092.375 (×3) | −5,092.375 | −5,092.375 | −5,092.375 |
| control | −5,087.985 | (skipped by budget guard) | −5,088.865 | −5,088.995 |

Summary: best of 4 beats one run by +0.49; ×3.3 work +0.13; **×8.9 work +0.39**; at equal GPU time repeats ≥ longer runs. The
30 s solve is already within ~0.5 units of each puzzle's floor for this solver. A rough tail estimate (labelled rough): the
chain record was ~2.5 units below the median round winner after ~1,700 rounds of the whole field, ~0.34 units per e-fold of
search, so 4 more units would take ~e¹² times all search so far. **Verdict:** no "minutes on the best puzzle" mode; depth is set
by which puzzle is solved.

## E90 — Record hold (patch 44; kept, no hold observed)

**Change.** `QUIP_OTSUKA_BP_RECORD`, `_RECORD_NEAR`, `_RECORD_DRAWS`: when a result clears the submit target and is within NEAR
above the record energy without beating it, draw the same puzzle up to DRAWS more times, stop as soon as the record is beaten,
then push the best. Why: in E89 our −5,095.545 puzzle was pushed at its first draw while one fresh draw in five gave
−5,096.065, deeper than the chain record of the time. Cost: a rival may take the round during the 30–90 s hold (~1 in 6 at
~8-minute rounds); expected trigger about once a day.

**Gates** (fake helpers): knobs unset identical to the previous binary; deeper than the record → push now; inside the window →
hold, early stop when a draw beats the record; no draw beats it → best of all pushed; outside → push now; helper death →
first draw pushed; helper hang → killed at the timeout, first draw pushed; stop during a hold → ended at the stop.

**Result.** Live from 2026-11-21 23:26Z; no hold triggered in the windows we read. Not a measured gain.

## E91 — Ladder audit with the helper probe as judge (bench only)

**Question.** Only 8.6 % of live ladders handed the probe a salt ≤ −5,086. Does the ladder discard deep puzzles, or do the
1,024 not contain them?

**Method.** Three ladders of the live shape reproduced with our code on synthetic 60,000-salt leases; 128 helper probes per
ladder across strata (final 4, ranks 5–16, 17–48, 16 each cut at rungs 1–3, 32 not kept).

**Numbers.** Probe mean / best by stratum: final 4 −5,078.41 / −5,083.85; ranks 5–16 −5,077.24 / −5,082.77; ranks 17–48
−5,076.57 / −5,083.40; cut at rung 3 −5,073.84 / −5,081.60; rung 2 −5,073.55 / −5,080.48; rung 1 −5,072.74 / −5,079.80; not
kept (fast rank 1,025–4,096) −5,072.19 / −5,080.38; not kept (> 4,096) −5,069.84 / −5,078.73. No probe ≤ −5,084 in 384 (so the
audit cannot speak about record-class puzzles). Rank correlation with the probe: rungs 1–3 +0.28 … +0.36, rungs 5–6 +0.35 …
+0.36; fast score +0.40 over all probed, +0.10 inside the kept 1,024, −0.02 inside the top 48. Oracle best of all probed
against best of the final 4: +0.00 / +2.69 / +1.75 (mean +1.48); in 2 of 3 ladders the deepest probe sat in ranks 17–48.
Re-shapes on the top 48 at 2.2 s per probe: live 43.3 s per cycle, mean best −5,081.88; probe all 16 69.7 s, same; 4 rungs +
probe top 16 65.1 s, −5,082.88; 4 rungs + probe all 48 145.5 s, −5,083.36: none pays. Live cross-check (408 probe sets): the
set's best probe came from ladder rank 1 / 2 / 3 / 4 in 139 / 102 / 93 / 74 sets.

**Implication.** Early rungs and the first-stage score do their job; the loss is the order inside the top 16–48, recoverable only
with a probe ~3× cheaper used as a pre-stage (→ E92).

## E92 — Cheap-probe fidelity bench (not run)

**Plan.** Re-probe the 144 top-48 salts of E91 with the standard configuration at a second shift (self-consistency ceiling) and
cheaper configurations (probe cost in the helper = updates × replicas: 1,500 × 192, 1,000 × 192, 1,500 × 96, 1,000 × 96, and
the PR #5 stage-0 probe 700 × 96); report time, rank correlation, and best-of-4 pick quality for cheap-probe-16 / -48 schemes.
Staged with a fake-helper run and a dry run; ~25.6 GPU-minutes. **Not run.** The question (does a probe at ~1/3 of the cost rank
the top 48 nearly as well?) is open.

## E93 — PR #5's native selection against our ladder on one GPU (bench only; run 2026-11-23)

**Question.** On one GPU, does PR #5's own selection (stage-0 probe on every salt → 3,000-update probe → full solve with
re-rolls) find deep puzzles as often as our fast score + ladder + helper probe?

**Method.** Two 10-minute drives of PR #5 as published (`configs/example.json`, 4 concurrent engines; GPU shared with the live
miner), each on one synthetic gate lease of 60,000 salts of the same family and topology, under the upstream coordinator's
drive mode in a throwaway container without network. Drive 1 = the example config with its probe cutoff off, so every salt
that passes stage 0 is probed; drive 2 = the example config exactly. Ours: the live miner over the same 6 h (live leases).
Both judged by the same standard 3,000-update probe. Bench rates were scaled to live-equivalent with one sharing factor,
×1.50 (24 standard probes on the bench: median 3.31 s against 2.21 s live), measured on single-engine probes.

**Numbers.**

| | salts | rate | reached the probe | probe mean | best probe | probes ≤ −5,084 |
|---|---|---|---|---|---|---|
| PR #5, cutoff off | 1,748 in 9.8 min | 2.96 salts/s | 151 | −5,072.952 | −5,083.145 | 0 |
| PR #5, as published | 1,357 in 9.9 min | 2.29 salts/s | 123 (1 full solve, −5,085.575) | −5,072.435 | −5,079.885 | 0 |
| ours, live, 5.99 h | — | — | 1,849 standard probes (308/h) | — | — | 116 (19.4/h); ≤ −5,086 24 (4.0/h); ≤ −5,088 2 |

Pooled PR #5: 274 probes, none at or below −5,084. Its stage 0 took a median 0.90–0.95 s per salt per engine and passed
~5 % of salts after the warm-up (93 of 1,689; 65 of 1,298); stage 0 against the probe on the passed salts r +0.73 (range-restricted). On the same gate-lease family our
final four averaged −5,078.4 (E91, n 12). Of the 3,105 salts PR #5 processed, 48 fall inside our keep-1,024.

**Caveats.** Published settings only (PR #5 tuned for its own hardware and salt stream may do better); ours measured live on
live leases, PR #5 on gate leases; one sharing factor assumed for its 4 engines (they may suffer more or less from sharing);
one 10-minute drive per arm, so only large differences can be read.

**Verdict.** Bench only. On one shared GPU PR #5's own selection handles 2.3–3.0 salts/s and reached no probe ≤ −5,084 in
274 tries, against 19.4 per hour for our pipeline. This supports using its solver behind our
selection (E84), not its selection.

## E94 / E94b — Other miners' winning puzzles under our solver (bench only)

**Question.** One miner won 17 of 25 rounds at −5,093.9 … −5,094.8 while we reached that depth about once a day. A deeper solver,
or deeper puzzles found more often?

**Method.** Rebuild winning puzzles from the proofs (the extrinsic carries the nonce and the witness; decoded energies equal the
tracker on 22 of 22, and each winner's own state reproduces the published energy on our rebuilt instance), then our standard
probe and two full solves each.

**E94 numbers.** The leading miner's 12 latest winning puzzles: mean gap −0.014 units (ours − published), 8 of 12 within 0.3,
0 of 12 short by more than 1; exact on two, deeper on three (e.g. −5,094.985 against −5,094.155). The record puzzle of the time:
−5,095.905 at the first shift = the published record exactly. **Same energies to the third decimal** on other miners' winning
puzzles: their solver reaches the same per-puzzle floors as ours, and the helper is deterministic per shift.
Our probe on those puzzles: median −5,088.885, 10 of 12 ≤ −5,086, 8 of 12 ≤ −5,088; our own live best-of-4 sets: median −5,082.5,
5.4 % ≤ −5,086, 1.1 % ≤ −5,088. Fast-score rank of that miner's 73 decoded wins in a 180,000-salt random pool: median 1.26 %,
p90 9.7 %, max 62 %; 42 of 73 inside a keep-1,024 of 60,000, 17 of 73 inside one of 400,000.

**E94b.** The new chain record (−5,097.095), 1 probe + 4 draws: −5,097.005 (0.090 from the record), −5,095.545, −5,096.085,
−5,095.185; probe −5,092.825. Nine control winners' puzzles from two miners: mean gap +0.42 with two draws against the winners'
best of an unknown number. That puzzle's fast score is 969.6, ≈ 1.5 × 10⁻⁶ of the score distribution.

**Verdict.** **Depth is set by selection, not the solver.** Nobody among the miners we checked has a deeper solver; the deep
results come from puzzles at the extreme of our own score, found by scoring far more salts.

## E95 — Winners-based study of the first-stage score (bench only)

**Method.** 1,501 decoded winning puzzles on the current topology, 45 features from the fast score's motif decomposition,
background 60,000 random salts; cross-validated by round and by owner.

**Numbers.** Current score (free + 0.773 · pair_tt + 20.8 · f_mean + 22.2 · m2): **AUC 0.9645**; winners kept at our pass rates
(all owners): 34 % at 0.26 %, 55 % at 1 %, 65 % at 1.7 %, 82 % at 5 %; for the leading miner 25 % [15, 33] at 0.26 %, 59 % at
1.7 %. Candidates at the 0.26 % pass rate: reweighted terms 0.328 (no gain: the published weights are right); +
`tile_shared_incidences` (a free by-product of the fast score) 0.376 (+0.038 [+0.016, +0.063]; leading miner −0.001); L1
logistic on 41 features 0.397 (+0.059 [+0.031, +0.089]); gradient boosting 0.332 (no nonlinear structure left). Rank
correlation with the standard probe: E91's kept 1,024 +0.10 → +0.24 with the added feature, but on 1,443 live final-4 probes
+0.13 → +0.12. A wider first cut buys winners slowly (0.26 % → 1 % admits ×1.6 winners for ×4 ladder load).

**Verdict.** No cheap structural score does materially better; the first-stage score was left as is. `tile_shared_incidences` is
the one free addition if the score stage is touched (×1.11 winners kept).

## E96 — More re-draws near the bar and a tighter solve limit (settings; neutral)

**Why.** 333 live solves: results ≤ −5,092.5 by probe band: ≤ −5,086 5 of 51; (−5,086, −5,085] 1 of 49; (−5,085, −5,084] 1 of
91; (−5,084, −5,083] 0 of 142 (43 % of full solves). Re-draws of first draws in (−5,091.5, −5,089.5]: n 37, final ≤ −5,092.5
in 3 (mean gain 0.46), ≈ 18 GPU-minutes per win-grade result against ≈ 100 for the pipeline as a whole.

**Change.** `QUIP_OTSUKA_BP_REDRAW=-5089500,8` (cap 3 → 8), `QUIP_OTSUKA_BP_SOLVE_MAX=-5084000`.

**Result.** 13.7 h: wins 4 of 121 rounds against 3 of 109 before; results ≤ −5,092.5 0.37/h against 0.50–0.64/h (n 5). Neutral;
left on. An earlier proposal to raise the re-draw threshold was withdrawn on this data.

## E97 — Fast-score tail density and scorer throughput (bench only)

**Part A.** Background 10,000,000 random salts. Score cuts: 10⁻⁶ 959.5; 10⁻⁵ 973.0; 10⁻⁴ 987.5; 10⁻³ 1,004.6; 0.26 % 1,012.6;
1 % 1,025.3; median 1,088.7.

Density ratio of winners (winners' share ÷ background share) by score quantile:

| quantile band | all 1,501 winners | winners of owners without a score screen (n 139) |
|---|---|---|
| 1–5 % | 6.8 | 8.1 |
| 0.26–1 % | 31 | 23 |
| 0.1–0.26 % | 77 | 54 |
| 10⁻⁴–10⁻³ | 150 | 56 [16, 104] |
| 10⁻⁵–10⁻⁴ | 400 [296, 503] | 400 [80, 800] |
| 10⁻⁶–10⁻⁵ | 1,777 [1,110, 2,517] | — |
| top 10⁻⁶ | 4,000 (6 winners) | — |

Live final-4 probes by score bin (3,605 probes): P(probe ≤ −5,086) 1.1–1.2 % for 0.26–5 %, 2.1–2.3 % for 10⁻⁴–0.26 %, 3.9 % for
10⁻⁵–10⁻⁴ (n 155), 5.0 % for 10⁻⁶–10⁻⁵ (n 40). Gain in winner-class puzzles entering the kept 1,024 if N× more salts are
scored (owners without a score screen): N = 3 ×1.4 [0.9, 2.1]; N = 10 ×4.0 [2.1, 5.9]; N = 30 ×6.0 [1.4, 10.9]. Inside the
final 4, ladder order beats score order (rank-1 ≤ −5,086: 4.0 % against 2.3 %).

**Part B.** Scorer CPU-bound: 20 generator threads on all 24 cores, ~4,735 salts/s. Per salt: tile stage 6 %, pair stage 11 %,
score stage 83 % (the f / m2 loops, ~69k weight lookups and ~173k edge lookups). GPU port estimate: ~70k RNG words and ~170k
random lookups per salt, ~250 KB of state per salt → 5–10k salts/s for a whole 5090, taking GPU time from the ladder and the
solver.

**Corrections.** (1) A leading miner's scoring rate was first estimated at 0.6–2.8 M salts/s per worker by dividing a cumulative
counter by one round; per-worker counters give ~6–11k salts/s per worker, ≈ 60k salts/s in total, about 13× ours. (2) Part B
estimated ×2–4 from a CPU rework (×2 from restructuring the f / m2 loops, ×1.3–2 from an early exit); E98 measured ×1.05 from
the restructuring and ≈ ×1.2 from the exit at today's cut: the m2 loop is memory-latency bound. (3) Part B's GPU estimate was
contradicted by PR #6's generator (E99): generation alone is ~22k salts/s on our card.

**Verdict.** Closed; led to E98.

## E98 — Faster fast score and a score-pick probe slot (patch 45; rejected, rolled back)

**Change.** `QUIP_OTSUKA_FAST_V2=1`: f / m2 in exact integer arithmetic with a merged group-pair table (bit-identical terms).
`QUIP_OTSUKA_FAST_EXIT=1|check`: skip m2 when the other terms plus a lower bound on the m2 term cannot beat the keep set's worst
score (bound −4.266: the 10 M-salt minimum −3.815 minus a margin; **empirical, not proven**); `check` computes m2 anyway and
counts wrong skips. Either knob also turns on a lock-free keep threshold. `QUIP_OTSUKA_BP_PROBE_SCORE=k`: the last k probe slots
go to the batch's best fast scores outside the ladder's picks. Why the slot: live, ladder rank 4 probed ≤ −5,086 in 0.3 % of
sets while score-extreme salts probed ≤ −5,086 in 3.9–5 % (E97).

**Gates.** Knobs unset: identical to the previous binary (14 and 32 log lines). All 60,000 per-salt scores identical for every
variant on two seeds; kept set, digests and all rung orders identical, including single-threaded; with the exit on, the
1,309 / 1,409 skipped salts were none of the kept set; 0 violations. 100,000-salt exactness: 0 different. Per salt 2.438 ms →
2.328 (V2, ×1.05) → 0.889 (V2 with m2 skipped, ×2.74, but only for skipped salts). The exit skipped only ~2.2 % of salts in the
gates because the bound must sit below the worst m2 ever seen; most of the saving came from the lock-free reject.

**Live (5.5 h, 464 probe sets).** Probe depth by slot: ladder rank 1 mean −5,080.23 (≤ −5,084 8.0 %, ≤ −5,086 1.1 %); rank 2
−5,079.43 (5.0 %, 1.5 %); rank 3 −5,079.19 (4.1 %, 1.1 %); **score pick −5,076.06 (0.4 %, 0.0 %)**, best −5,084.655; the set's
best probe came from ranks 1 / 2 / 3 / score in 180 / 127 / 115 / 42 sets. The ladder rank 4 it replaced had mean −5,078.91 and
3.2 % ≤ −5,084. Scorer: 81,764,375 salts in 5.5 h = 4,131 salts/s (no visible gain); exit check: would skip 7.7 %,
**violations 0** over 81.8 M salts.

**Why the slot failed.** The 3.9–5 % deep-probe rate for score-extreme salts came only from salts the ladder had *also* chosen
(a selection bias noted at the time), and the density ratios of E97 scale up a tiny base rate, so even the top-scoring salt of a
400,000-salt batch is rarely winner-class; at that stage the rung energies carry far more than the score. In the gates, all
three score picks had already been dropped by the ladder at rungs 2–4.

**Verdict.** Rejected and rolled back; the knobs stay in the code, inert when unset.

**Fixed after the run.** The m2 group-pair table of the staged path (65,536 slots, linear probing) had no full-table guard:
on a topology with 65,536 or more distinct group pairs a lookup would never end (impossible on the current topology, 41,205
edges). Commit `fix(fastgen): never let the staged group-pair table fill` on our branch makes the table refuse the key that would take its
last empty slot; that salt is then scored by the original stage (same terms). Unit test
`fastgen::e98_tests::group_pair_table_never_fills`. The binary that ran live did not have this guard.

**The early-exit bound is empirical.** The lower bound −4.266 on m2 is the 10 M-salt offline minimum (−3.815) minus a margin, not a
proven bound: 0 wrong skips in 10 M salts offline and 0 in 81.8 M live salts in `check` mode. A salt below it would be
dropped wrongly.

## E99 — PR #6 GPU instance generator on one shared GPU (bench only)

**Question.** PR #6 (draft) builds the canonical instance on the GPU; its description reports > 40k salts/s per GPU. On a single GPU every
generated salt is GPU time taken from the ladder, probes and solves. What is the generation rate on our card with the miner
running, and what does it cost?

**Method.** The PR's 8 files at commit e130011c, hashes pinned; read for file, network and process access before building;
built offline in a throwaway container; run in a throwaway container with no network, all capabilities dropped and only the
bench folders mounted. A ~200-line local patch to its `main.rs` added a duty-cycled generation loop, a larger self-check and an
instance dump (our bench harness, not their code; not included here).

**Numbers.**
- **Exact:** GPU = the crate's CPU port on 4,096 of 4,096 salts (and 1,024 of 1,024 for a one-thread-per-salt variant); GPU =
  the canonical chain instances on **96 of 96**.
- Generation-only rate with the miner running: 12,966 / 21,565 / 29,674 salts/s at batch 512 / 2,048 / 8,192 (15 s each);
  **21,871 salts/s over 300 s at full duty**; **5,676 salts/s at 25 % duty**; the crate's own short bench printed 36,610 salts/s
  (0.67 s); the one-thread-per-salt variant ~400–475 salts/s. Its SDP stages alone: rank 4 116,647 salts/s, rank 8 32,267.
- Cost to the live miner: full duty — rungs +100 … +139 %, probes +60 %, solves +83 % → **≈ 49 % of live throughput lost**;
  25 % duty — rungs +10 … +19 %, probes +9 %, solves −1 % → **≈ 10 % lost**.
- Trade: 4,460 (full duty) to 5,434 (25 % duty) generated salts/s per 10 % of mining given up — about our whole CPU scoring rate,
  for generation only (the score on top costs more).

**Verdict.** No GPU first stage on a single-GPU machine: 10 % of mining buys at most a doubling of the scored pool (≈ ×1.2–1.4
winner-class puzzles by E97, less the 10 %), break-even within the uncertainty. For machines with a GPU to spare the generator is a
real gain (≈ 22–37k exact instances/s on our card).

## E100 — PR #4 funnel selection against our ladder

**Question.** Is our selection layer better than the funnel already published in PR #4, on one GPU?

**Method.** PR #4 at commit 77bfd821, hashes pinned, built offline in a throwaway container and run with no network and only the
bench folder mounted. A local patch (+101 / −5 lines) logs its stage-1 decisions and skips its full anneal, so the funnel runs in
picks-only mode. Both selections on the same three 1,000,000-salt gate leases, a 200 s GPU window each, back to back with the
order alternating, the GPU shared with our live miner throughout. Funnel picks = the puzzles its stage 1 promotes (after its
first 32 decisions); ours = the ladder's final 4. Every pick judged by one standard helper probe (3,000 updates, one read).

**Numbers.**

| | picks | mean probe | median | best | ≤ −5,084 | picks per GPU-min | busy GPU per pick |
|---|---|---|---|---|---|---|---|
| Funnel (PR #4) | 34 | −5,079.48 | −5,079.68 | −5,084.20 | 1 | 4.23 | 5.0 s |
| ours | 16 | −5,079.09 | −5,079.65 | −5,083.12 | 0 | 1.29 | 4.4 s |

- Mean probe, ours − Funnel: **+0.39 units, bootstrap 95 % [−1.07, +1.93]** (negative would mean ours deeper).
- The pick rates are not comparable. Our scorer and ladder host work ran on the cores the live miner left over: 779–1,262 salts/s
  scored against ~4,700 live, rung 1 at 33–60 s against 9 s, so 1–2 ladders finished per window; live, the same pipeline
  delivers about 4.5 picks per minute. The funnel screened ~400 salts/s on about 5 cores and was not CPU-limited in the same
  way; it was tuned by its author for a desktop card.
- The two selections choose different puzzles: 1 puzzle picked by both; 15 of the funnel's 34 picks fall inside our fast
  score's top 1,024 of the range it screened (median percentile 0.7–2.3 %); 9 of our 14 picks inside its range pass its stage 0.
- Load on the live miner: rungs +120 … +160 % during funnel segments, +25 … +60 % during ours.

**Verdict.** No measurable difference in pick quality at this sample (16 and 34 picks; one probe at or below −5,084 in 50). Our
selection is not shown to be better than the funnel's; the funnel reaches the same probe depth from about a fifth of the salts.
The stages after selection (probe gate, helper solve, re-draws) would apply to either selector's picks.

---

## What we would tell another single-GPU miner

1. **Selection sets the depth.** On other miners' winning puzzles our solver reproduces their energies to the third decimal (E94,
   E94b); more solver effort on a good puzzle adds ≤ 0.5 units (E89). Spend effort on finding puzzles.
2. **A cheap structural score is a strong first stage** (AUC 0.96, E95) and is close to what cheap structure can give; winner
   density keeps rising toward its extreme (×400 at 10⁻⁵–10⁻⁴, E97), so scored salts per second matter. On one machine the
   scorer is CPU-bound and memory-latency bound (E97, E98).
3. **One short probe is mostly noise** (repeat correlation 0.17); averaging repeats and summing standardised probes over rungs
   works (E78, E79). Batch many puzzles per launch (E70).
4. **Judge the last few candidates with a short run of the final solver** and solve only those whose probe is deep enough
   (E86, E88). The probe-to-full fit was linear with sd 1.66; re-check a limit out of sample (E88).
5. **On one GPU, everything competes for the same card**: a second GPU job halved live throughput (E99); concurrent engines give
   ≤ ×1.7 (E87); cutting a stage that was not on the critical path freed nothing (E64m).
6. **Race end to end** before trusting a funnel model (E64n, E65), and build random controls from fresh salts (E77).
