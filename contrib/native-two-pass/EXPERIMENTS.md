# Experiments log: what we tried while tuning the native two-pass solver for `quip/twisted-ice-dimer`

This is the record behind `contrib/native-two-pass`. It includes null results and the exact way each number was measured, so nobody has to repeat them. Everything is from one team, one problem family (m=12, t=4, weak 45, pairs 1350, 4560 nodes, 41,205 edges) and a few days of work. Treat every grade below accordingly.

## 1. Conventions and evidence grades

* Energies are in Ising units (milli-units / 1000). **Lower is better.** "Paired" means the same instances and the same config except for one change; the difference is reported as mean (SE) and the count of instances that improved.
* Grades: **S** strong (large n or large effect, replicated), **M** moderate (consistent but small n or one clean test), **W** weak or inconclusive.
* Rule we used before installing a change on the live miner: at least 0.5 units better with SE below 0.3, and no instance clearly worse. Several things below were installed on a weaker basis at the owner's request; they are marked.
* Chance hit rate scales steeply with depth: about x2.9 per energy unit near the threshold (a rival's estimate that matched ours). A 0.5-unit gain is worth about +40 % hits.
* Instance energy: `E = Es + Ew`, where `Es` is the strong layer (2000 per violated strong motif, about 3540 violated motifs at winning depth) and `Ew` the weak layer (about -205 000 to -217 000 milli). Across decoded winning solutions the two nearly cancel (each extra violation costs about as much weak energy as it saves), so the number of violated motifs is not a useful target.

## 2. Setup and runtime facts that matter

* Instances regenerate exactly from the topology with `ising-generators` 0.0.15 (`m=12, t=4, weak=45, pairs=1350`); the instance nonce is `derive_nonce(last_proof_block_hash, miner_account, salt)`, so work cannot be reused across rounds or accounts.
* The strong layer is identical in size for every salt (11,964 strong edges), but **which** edges are strong changes with the salt.
* On-chain witnesses decode with the family witness codec and the instance nonce taken from the proof's nonce field with its bytes reversed. We used that only to read winning solutions' energies and composition.
* GPU work is dominated by one kernel, `loop_factor_bp` (cold search, stage-0 probe).

## 3. What mattered, in order of measured impact

1. **Throughput engineering.** (a) Several solve engines in parallel and the declared stream width set to the engine count: completed salts per second 0.61 to 1.08 on a 5090 with 3 streams, then about x1.5 more attempts per second after the width fix. (b) A two-stage screen with a cheaper stage 0 (96 replicas, 700 updates; stage 0 only, the full solve keeps 192 replicas): solves per second 2.12 to 4.10, median solve time 641 to 308 ms. [M]
2. **Screen quality was verified on real chain data.** On 34 decoded winning instances the screen (stage 0 top 5 % and stage 1 at or below -5079) kept 97 %, while passing about 1 % of 100 random salts. [M, n=34 and 100]
3. **Adaptive re-rolls.** In a 12-hour live log, re-rolls improved 62 % of the salts that got a full solve, by 0.64 units on average, and one threshold-crossing block came from a re-roll. Draws 1 and 2 carry most of the gain; the cap of 24 costs little because few salts run that deep. [M]
4. **`weak_scale` 1.2** (details in 4.1). Modest and not cleanly separated from noise. [W to M]
5. **Slow cold ramp** (cold 30,000 updates, `ramp_base` 15000, `ramp_step` 5000). [W, see 4.1]

## 4. Experiment log

### 4.1 Solver algorithm and parameters

| ID | Hypothesis / change | How measured | Result | Verdict |
|---|---|---|---|---|
| A01 | Longer / slower cold ramp | Paired, 10 nonces x several seeds | One run: -1.33 (SE 0.28); an earlier run on two nonces: -1.01 (SE 0.58) then +0.16 (SE 0.45), pooled about -0.4 (SE 0.4) | Installed; evidence mixed [W to M] |
| A02 | Shorter cold stage (3000 updates) | Paired | Same quality (-0.01, SE 0.38), -25 % time. Later replaced by A01 | Superseded |
| A03 | Shorter seeded stages (2000 updates) | Paired | +0.97 worse (SE 0.58) | REJECTED |
| A04 | Cheaper stage 0 (fewer replicas / updates) | Rank correlation of stage-0 energy with the final energy, then live | 96 replicas x 700 updates: 0.707 (192 x 1000: 0.74, 48 x 700: 0.64). A 96-replica *full* solve is worse (+0.83, SE 0.67) so only stage 0 uses fewer | KEPT for stage 0 only [M] |
| A05 | `weak_scale` (scale the +-45 couplings in the search model only) | Paired on 12 decoded winning instances, current pipeline | 0.8: +0.55 (SE 0.51). 1.2: -0.64 (SE 0.29, 8/12). 1.3: -0.56 (SE 0.44, 9/12). 1.6: -0.18 (0.36). 2.0: +0.10 (0.48). On three further fresh instances: 1.1 -0.95, 1.2 -1.60, 1.3 -0.88 (n=3, very noisy). The best of six settings per instance was 1.2 or 1.3 on 10 of 12 (about 4 expected by chance) | Installed at 1.2 on weak evidence. An earlier test (before the pipeline changed) of 0.8 to 1.5 found no gain, so [W] |
| A06 | Weak scale spread across replicas {1.2, 1.3, 1.1, 1.4} | Paired on the same 12 | +0.19 worse than plain 1.2 (SE 0.33) | NULL |
| A07 | Weak scale schedule (1.6 to 1.2 or 0.9 to 1.2 over the first half of the cold stage) | Paired on the same 12 | +0.60 and +0.67 worse than plain 1.2 | NULL |
| A08 | Thermal messages in the cold stage plus temperature anneal | Paired | Cold stage -1.67 alone, but the full pipeline +0.25 (SE 0.41) at +40 % time | NULL |
| A09 | Deeper cold, first and second stages (3x updates) | Paired | About -0.2 for +40 % time | NULL |
| A10 | Fusion width / donors, cluster passes, arity, exact loops | Paired, n=16 | fusion 18/64: -0.28 (SE 0.24); passes 2 / arity 9: -0.39 (SE 0.66); exact loops no effect; arity 10 invalid | NULL |
| A11 | Lean pipelines (cold + closure only); no cluster rebase | Paired | Lean +0.83 worse; no-cluster loses on some nonces; after the slow ramp the later stages add only about 0.66 in total | Keep the full pipeline |
| A12 | Re-roll budget (cap 24, `reroll_min_p` 0.004, `reroll_within_milli` 9000) | 12-hour live draw log (1278 full solves, 4100 draws) | See section 3, item 3. No evidence for a different budget | KEPT |

### 4.2 Kernel and GPU engineering

Profile of `loop_factor_bp` (Nsight Compute, RTX 5070 Ti, stage-0 grid of 96 blocks and cold grid of 192, 512 threads, 109 registers, 28 KB dynamic shared memory):

* Latency bound: SM throughput 18 to 23 %, memory 26 to 34 %, DRAM 4 to 7 %, IPC 0.97, 76 % of cycles with no eligible warp, occupancy 33 % (one block per SM, limited by registers and by the shared-memory carve-out).
* Per source line: 28 % of stall samples are the block barrier at the end of the region sweeps (only about 350 regions per colour class, one per thread), 9.4 % and 7.3 % are the dependent loads in the junction cost, 10.8 % the edge sweep gather, 7 % the edge-sweep barrier. Shared-memory bank conflicts are about 2.7-way.

| ID | Change | Result | Verdict |
|---|---|---|---|
| K01 | `__launch_bounds__(512,2)` (two blocks per SM, 64 registers) | Bit-identical energies, 1.2 to 2x **slower** (spills) | REVERTED |
| K02 | Reorder edges inside each colour class to spread shared-memory banks (bank multiplicity 5.5 to 3.9, ideal 2.0) | Bit-identical, **no speedup** (2.38 s vs 2.36 s) | NULL, kept behind `QUIP_EDGE_BANKS` |
| K03 | Force-inline and unroll the junction cost and separator minimum | Bit-identical, no measurable speedup (the GPU was shared, timings noisy) | NULL |
| K04 | Idea not built: spread each region's 16 states across 16 lanes to remove the barrier stall | | open |

The live 5090 runs at its power limit with all stream worker threads spinning on GPU synchronisation, so it is GPU bound.

### 4.3 Where the deep tail comes from

* Same-salt draws correlate at 0.66 to 0.70. Between-salt SD is 1.5 units against 0.86 within a salt, so about two thirds of the variance is the instance floor. Many salts with several draws repeat the same best energy four or more times, so for them the solver already reaches the floor. [M]
* Draws are close to normal (mean -5084.2, SD 2.07 over 4100 draws); only 2 of 4100 reached -5090.
* On 12 decoded winning instances our solver's energies tie the winners' (mean ours minus theirs -0.19, 8 of 12 equal or better; the winners' numbers carry a winner's-curse bias). Per-instance solver quality is not the gap; the number of floor-quality instances solved per second is. [M]
* Cheap CPU-side predictors of instance depth (local-cluster exact bounds, cut and inner energies, edge sums) showed no signal against stage-0 energy or winnerness (AUC about 0.5). [W, n=134]

### 4.4 Algorithms that did not beat the solver

| Change | Result | Verdict |
|---|---|---|
| Seeded parallel tempering / ICM on two salts, 22 threads, 40 minutes | Nothing below the solver's best | NULL |
| ICM + PT from scratch | Stalls about 80 units above the solver | NULL |
| 26-spin exact window polish | 0 gains, even on a state 4.2 above its floor | NULL |
| Lin-Kernighan style ejection-chain beam search | +0.63 on 1 of 6 states, about +0.1 on average | NULL |
| Recombination of 8 parent states (class-level optimal) | Equals the best parent | NULL |
| Partition crossover of final states | No gain | NULL |
| Cosm (seeded: returns the incumbent; random start) | Random start is 105 to 120 units worse than BP | NULL |
| Instance features as a per-salt predictor | |r| at most 0.26 | NULL |

## 5. Methodology lessons

* Pair everything. Per-instance noise is about 1 to 3 units, so an unpaired comparison on a few instances means nothing.
* Selecting the instances on which a change "worked" and re-testing on those guarantees a nice-looking result. Re-test on all instances, or on fresh ones.
* A config key that fails validation can make the miner silently run default settings. Always check with `--print-search-config`.
* Timing on a GPU shared with other jobs drifts by 40 % or more. Interleave A and B runs.
* Count deep results per hour, not wins per hour: wins depend on the round threshold and timing as much as on the solver.

## 6. What we would try next, and what we would not repeat

Try next (ordered by our prior): the 16-lane region kernel (K04); a cheap instance-depth predictor that actually carries signal (we found none with local bounds); a persistent kernel to remove the barrier cost.

Do not repeat: weak-scale spreading or schedules (A06, A07), two blocks per SM at 64 registers (K01), bank-conflict reordering for speed (K02), shorter seeded stages (A03), PT / ICM / exact-window / crossover post-processing (4.4).

## 7. Honest limitations

* One problem family and parameter set, one chain, a few days. Different parameters or a runtime change can invalidate the thresholds and the screen.
* Small samples: most paired tests use 8 to 16 instances; the grades say so.
* Live block counts are far too noisy to separate small changes; the offline paired tests are the only reliable evidence we have, and several are marginal.
