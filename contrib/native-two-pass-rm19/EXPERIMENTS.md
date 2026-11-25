# Experiments log (single-GPU build)

What we tried, with sample sizes and how much to trust each result. Grades: **A** = paired test, same inputs, deterministic; **B** = live run, short, confounded; **C** = observation.
All energies are in energy units (milli-units / 1000), lower is better. "Round target" is the live threshold of the current round (about -5093 to -5095 in our runs).

## E1. Faster salt generation (grade A)
`ising-generators` 0.0.15 builds each salt's instance on the CPU. Per-topology tables (sorted edge list and index, tile tables) are built once, and `uniform()` does one 32-bit modulo instead of two 64-bit ones. Single thread, loaded machine: generation 1.5 ms to 1.1 ms per salt (-27 %), end to end 387 to 456 salts/s per thread (+18 %). Output unchanged: instance digest and a hash of all couplings identical on 20,000 salts.
Nothing else in generation was easy: what is left is tile selection (about 0.39 ms), the pair shuffle (0.24 ms) and the weak-edge draws (0.20 ms).

## E2. Full GPU screening cascade on the solving GPU (`QUIP_GPU_SCREEN=1`) (grade B, a clear negative)
The same GPU generates salts, runs rank-4 on all and rank-8 on the best 5 %. Standalone it screens 7,300 salts/s. Next to the solver the solver's long kernels hold the SMs: the screener got 200-700 salts/s at first, about 1,900 salts/s at best, while solves started per minute fell by about 45 %, and full solves per hour did not improve. A high-priority stream for the screening kernels stalled screening after a few batches. Not recommended. Screening should go on another device, or stay on the CPU.

## E3. CPU ridge screen + GPU rank-8 rescoring of the top 3 % (`QUIP_GPU_RESCORE=1`) (grade B)
CPU threads generate and ridge-score every salt; only the best 3 % are rescored on the GPU and ranked by the combined key. GPU cost is small (about 225 µs per rescored salt, i.e. a few percent of the GPU at several thousand salts/s). On an RTX 5070 Ti laptop: 22 full solves per hour in the first 25 minutes after settling, against 13 per hour for CPU-only screening at the same threads. Both are single windows of 25-46 minutes after a restart, so this is suggestive, not established.

## E4. Candidate pool size (grade B)
`QUIP_POOL=256` with rescoring left each lease's survivors waiting for a solve engine; the coordinator's measured salts/s stayed around 700 and leases stayed near 40,000 salts. At `QUIP_POOL=32` surplus candidates are dropped immediately: on the laptop the measured rate grew to about 4,700 salts/s with 284,000-salt leases, on the RTX 4090 to 6,000-8,000 salts/s with 400,000-1,000,000-salt leases. We did not run a clean A/B; the change came with a restart and the lease ramp itself takes about 20 minutes.

## E5. Restart-ladder refiner (grade A, offline)
12 salts x 8 seeds = 96 paired runs per budget, same instances, archives and seeds, GPU-free, deterministic. Previous refiner vs the ladder:

| Budget | Improved | Mean gain | Ladder deeper / shallower / tied | Runs reaching the chain-winning energy |
|---|---|---|---|---|
| 8 s | 84 vs 85 | 0.842 vs 0.923 (+0.081) | 31 / 11 / 54 | 68 vs 73 (5 gained, 0 lost) |
| 16 s | 84 vs 88 | 0.870 vs 1.003 (+0.134) | 32 / 5 / 59 | 69 vs 73 (4 gained, 0 lost) |

A small, repeatable gain at the same CPU cost. It does not by itself change how often a solve beats the target (that needs a solve within a few tenths of it). Live, the ladder build produced several large single gains (1.44 + 1.26 on one solve, 2.21 on another) but wins are too rare to attribute.

## E6. Refiner margin 2.0 to 2.5 (grade C)
More solves get a refine attempt. No measured effect on yield; the extra attempts cost 8 s of one core each.

## E7. Refiner variants tried and not adopted (grade A, offline)
Same 96-pair protocol against the ladder:
- **Conditional pockets** (bounded-width exact minimisation on weakly constrained sites, AVX2 row builder): mean gain +0.074 (8 s, 95 % interval -0.005..+0.167) and +0.046 (16 s); chain hits 72 to 73 and 74 to 74. About 24 % more passes in the same CPU time.
- **Polar domains** (adds a profile-elimination decoder and a rank-3 vector rounding on top): +0.077 (8 s, interval +0.003..+0.160) and +0.055 (16 s); chain hits 71 to 76 and 73 to 74. The extra hits come from solves that start 2.5 or more above the target, which the live refiner (margin 2.5) never works on; in the buckets it does work on (start 0-2.5 above target) chain hits were unchanged. About 37 % more passes.
Both returned energies verified exactly (every spin vector re-evaluated independently).

## E8. Wedge-grouped ridge features and a two-stage ridge (grade A, negative)
The 4-cycle counts (880,000 cycles, about 1 ns each) are 96 % of the ridge cost (0.93 ms). Grouping cycles by their diagonal gave identical features and the same speed. A ridge on a fixed fraction of the cycles correlates poorly with the full score (0.72 for half of the cycles, 0.48 for a quarter) and loses the best salts, so no two-stage ridge.

## E9. Platform notes (grade C)
- Under WSL2, the Linux `nvidia-smi` reports a fixed 1515 MHz SM clock on our laptop; the Windows `nvidia-smi.exe` shows the real clock.
- A 12-CPU / 8 GB WSL VM ran the miner at 3.4 GB and the validator at 2.7 GB and swapped heavily; raise the WSL memory limit.
- Salt generation and screening use all the CPU you give them; the solver engines also need CPU, so leave several threads free.

## E10. Largest live refiner gains (grade C, anecdotes)
See the table in README.md. The 4090 log showed 58 refiner improvements and no refiner failures over the 17-hour run (19:58 UTC to the shutdown at 13:02 UTC the next day); the ladder build was in place from 23:27 UTC, and 5 of the 58 came before it. Blocks won within seconds after a refiner improvement, with the winning proof's energy consistent with the refiner's output: 2370, 2371, 2372, 2383, 2407 and 2416, and 2458 two minutes after a 2.21 gain. Other blocks the same machine won (for example 2369, 2377, 2398, 2412, 2413, 2437, 2440, 2444, 2445) had no refiner event we could connect to them, and two (2454, 2455) had refiner events nearby whose energies did not match the proofs. We cannot say how many of the refiner-assisted blocks would have been won without it.

## Live reference (grade B, small)
An RTX 4090 (Linux, Docker), single machine, 22 generator threads + GPU rescoring + restart-ladder refiner: 3 of 25 decided rounds won in about 4.4 hours (12 %). Too few rounds to compare with anything; recorded so that the next person has a number to beat.
