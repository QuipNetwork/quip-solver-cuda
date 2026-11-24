# Single-GPU measurements on twisted-ice-dimer, and the probe gate

Measurements from mining `quip/twisted-ice-dimer` on one machine, offered as reference for upstream and for other
single-GPU miners. Docs and reference patches only: nothing that is compiled on `main` changes.

**What this contributes**

- **The probe gate** ([`probe-gate/`](probe-gate/)): a short run of the final solver on the last few candidates of any
  selector, used to re-order them and to skip the full solve when none is deep enough. It is the one technique here with a
  live A/B behind it that works on top of any selection. Description, knobs, log lines and the two commits that implemented
  it in our miner, as `git format-patch` output for reading.
- **The helper patch** ([`helper/`](helper/)): an opt-in `--serve` mode for the PR #5 solver, so another process can send
  it solve and probe requests over stdin/stdout, with the line protocol and the two search configs we ran.
- **The experiment record** ([`EXPERIMENTS.md`](EXPERIMENTS.md)): every experiment from E63 to E100 with question, method,
  numbers and verdict. Most are negative or neutral.

**What this does not contribute.** Our selection layer (fast structural score, cumulative GPU probe ladder) and the miner
wiring of the helper are not part of this pull request. A head-to-head against PR #4's funnel (E100) did not show our
selection to be better, so there is no case for sending it. The full implementation can be shared on request.

## Setup

- One laptop RTX 5090 (82 SMs, 175 W limit) and 24 CPU threads in the same machine.
- Our pipeline: a CPU structural score over each lease's salts, a cumulative probe ladder on the GPU over the best 1,024,
  then the PR #5 solver (commit `97faa92`, run as a helper process through [`helper/pr5-serve.patch`](helper/pr5-serve.patch))
  for the full solve (E84).
- Energies are in units, offset included; lower is better. A full solve takes ~29 s on our GPU (~35 s shared, E82).
- Benches ran with the GPU shared with our live miner; those windows were excluded from live comparisons.

## The probe gate

**What.** After selection, run a short probe of the final solver (3,000 updates, one read, ~2.2 s) on the last 4
candidates. Re-order them by the verified probe energy (E86). Walk that order and full-solve a candidate only while its
probe is at or below a limit (−5,083.0 after the out-of-sample check); if none clears it, skip the full solve and go
straight back to selecting (E88).

**Why.** In 308 live full solves, none with a probe in (−5,084, −5,083] (51 solves) or above −5,082 (112) reached
−5,090, and no round in 197 had been won above −5,090: ~72 % of the ~30 s full solves could not win (E88). A probe costs a
fraction of a full solve and predicts it well.

**Where it sits.** Between any selector and the full solve. Nothing in the gate depends on how the candidates were chosen,
so it applies equally to PR #4's funnel picks or to PR #5's own pipeline once the two are joined. Algorithm and knobs:
[`probe-gate/PROBE-GATE.md`](probe-gate/PROBE-GATE.md).

**Settings we ran.** 4 candidates probed (8 was break-even); probe = the same solver binary with `probe_updates: 3000`
([`helper/serve-probe.json`](helper/serve-probe.json)); limit −5,083.0 (later −5,084.0, E96); top-2 full solves (E85).

**Measured effect.**

| measurement | numbers | source |
|---|---|---|
| probe as predictor (bench, 120 live picks) | 3,000-update probe against the full solve of the same puzzle: r 0.754 (3.16 s shared); independent seed r 0.788 | E86 |
| best of 4 by probe, then full solve (bench) | −5,087.25 against −5,085.23 for a random one of the 4 (+2.0, 80 % of the oracle's −5,087.77) | E86 |
| re-order live, 66 solves | mean −5,086.20 against −5,085.04 (+1.16); ≤ −5,090 9.1 % against 2.1 % | E86 |
| re-order live, 245 solves | mean −5,086.31; ≤ −5,090 7.3 % | E86 |
| probe 8 instead of 4, 150 solves | per hour ≤ −5,090 2.7 against 3.4, ≤ −5,091 1.6 against 0.9: break-even | E86 |
| probe-to-full fit, 308 live full solves | full ≈ −954.4 + 0.813 × probe, residual sd 1.66 | E88 |
| limit check, 85 solves after it was chosen | −5,084.0 (chosen in sample) kept 2 of 4 results ≤ −5,090 and would have dropped a win; −5,083.0 kept 4 of 4 | E88 |
| limit −5,083.0 on all 393 solves | keeps 24 of 28 (≤ −5,090), 10 of 11 (≤ −5,091), 5 of 5 (≤ −5,092) | E88 |
| gate live, 6.3 h, 418 ladders | 61 % of full solves skipped; 66.5 ladders/h against ≈ 54; per hour ≤ −5,091 1.91 against 0.96, ≤ −5,090 2.86 against 3.4 | E88 |

So the gate roughly doubled the deep end (≤ −5,091) per hour and left ≤ −5,090 flat. Tightening the limit to −5,084.0
together with more re-draws near the bar was neutral over 13.7 h (E96).

**Limits.** The fit and the limit are specific to this solver config, this recipe and the bar of the time; a different
solver or config needs its own fit, and a limit chosen in sample should be re-checked out of sample (E88 shows why). The
probe and the full solve must not run on the GPU at the same time on one card; our miner waits out a cancelled job of
either helper before starting the other.

## Head-to-heads

**Ours against PR #5's own selection (E93).** PR #5 as published (`configs/example.json`, 4 engines), two 10-minute drives
on one shared GPU, against our live pipeline over 5.99 h, all judged by the same 3,000-update probe. PR #5: 2.3–3.0
salts/s, 274 probes, none at or below −5,084 (best −5,083.145). Ours: 1,849 probes, 116 at or below −5,084 (19.4 per
hour). Caveats: published settings only, gate leases against live leases, one sharing factor (×1.50) assumed for its 4
engines, one drive per arm. This supports running PR #5's solver behind a fast selector rather than its own selection on
one GPU.

**Ours against PR #4's funnel (E100).** PR #4 at `77bfd821` in picks-only mode, both selections on the same three
1,000,000-salt gate leases, 200 s GPU windows, order alternating, every pick judged by one standard probe. Funnel: 34
picks, mean −5,079.48. Ours: 16 picks, mean −5,079.09. Difference (ours − funnel) **+0.39 units, bootstrap 95 % interval
[−1.07, +1.93]**: no measurable difference. **Our selection is not shown to be better than PR #4's funnel**, which reaches
the same probe depth from about a fifth of the salts. The pick rates on that bench (1.29 against 4.23 picks per GPU-minute)
are not a valid comparison for our side: our CPU scorer and ladder host work ran on the cores the live miner left over
(779–1,262 salts/s scored against ~4,700 live; rung 1 at 33–60 s against 9 s), so only 1–2 ladders finished per window,
while live the same pipeline delivers about 4.5 picks per minute.

## Solver findings

- **More solver work adds little (E89).** On 12 puzzles rebuilt from live nonces: best of 4 draws beats one by +0.49 units;
  ×3.3 work +0.13; ×8.9 work +0.39. The ~30 s solve is already within ~0.5 units of each puzzle's floor for this solver.
- **Depth is set by selection (E94, E94b).** Other miners' winning puzzles rebuilt from their proofs and solved by our
  helper: on the leading miner's 12 latest, mean gap −0.014 units (ours − published), 8 of 12 within 0.3; the record puzzle
  of the time reproduced exactly (−5,095.905). The later record (−5,097.095) came within 0.090 on the first draw. Nobody
  among the miners we checked has a deeper solver; the deep results come from finding deeper puzzles.
- **The PR #5 solver against our own GPU anneal (E82).** 18 fixed puzzles: −5,088.25 against −5,085.65 (−2.59 units, SE
  0.48), better on 17 of 18, 35 s against ~210 s. Live as a helper behind our selection: +2.43 units and 52.6 solves/h
  against 29 (E84).

## What did not help

- **More re-draws near the bar and a tighter limit (E96):** neutral over 13.7 h.
- **Score-pick probe slot (E98):** score picks probed at mean −5,076.06 with 0.4 % ≤ −5,084, against −5,078.91 and 3.2 % for
  the ladder rank they replaced; the scorer rework gave ×1.05 and no visible live gain.
- **GPU instance generation on the same card (E99, PR #6's generator):** exact on 96 of 96 chain instances, but ≈ 49 % of
  live throughput lost at full duty and ≈ 10 % at 25 % duty (5,676 salts/s): break-even within the uncertainty on one GPU.
- **More CPU scoring (E97):** winner density rises ×400 at the top 10⁻⁵–10⁻⁴ of the score, but our scorer is CPU-bound at
  ~4,735 salts/s on all 24 cores, and its inner loops are memory-latency bound (E98).
- **Concurrent solve engines (E87):** ×1.72 at 4 engines, ×1.82 at 6.
- **Fewer reads per long anneal (E80):** a quarter of the reads cost 2.5 units and still took 70 % of the time.
- **PR #5 miner with our structural score in front (E83):** its serial instance generation caps selection at 32 salts/s.
- **Probe 8 instead of 4 (E86):** break-even.
- **Top-2 full solves (E85):** a rank-2 solve is worth about half a rank-1 solve: break-even.
- **A cheaper probe as a pre-stage (E92):** not run; open.

## Limits of these numbers

- One machine and one card. Benches ran with the GPU shared with the live miner, so bench timings are slower than a free
  GPU and the sharing factors are estimates.
- Live A/B windows of 60 to 450 solves per change; wins are too rare to judge by and are reported only as context.
- Head-to-heads are small (E93: one drive per arm; E100: 16 and 34 picks), so only large differences can be read.
- PR #4 and PR #5 were run with their published settings (PR #4 with a local logging patch for picks-only mode), which
  their authors tuned for their own hardware.

## Relation to open pull requests

- **PR #5** (native two-pass solver): [`helper/pr5-serve.patch`](helper/pr5-serve.patch) applies to its head `97faa92`
  (`git apply --check` passes) and touches `contrib/native-two-pass/solver.rs` only.
- **PR #4** (funnel selection): the gate needs only a list of candidates in the selector's order, so it can sit after the
  funnel's picks.
- The probe gate is the join between selection and the full solve. If upstream combines PR #4 and PR #5, that is where it
  would go.
