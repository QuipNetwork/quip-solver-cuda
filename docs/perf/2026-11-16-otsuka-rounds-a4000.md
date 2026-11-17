# Otsuka population anneal and lease screening on the A4000

With lease screening on, one RTX A4000 reaches about the same round energy as the
RTX 5090 Laptop GPU that reported about 6 % of Aglais rounds. Two A4000 cards on
one host add about 1.3 units per round. The estimated share is 4 to 6 % of rounds
for one card and 8 to 11 % for two cards. Without screening, the estimate drops
to about 0.1 to 0.2 %. The old settings, without the population anneal, reach no rounds.

The share estimate is an extrapolation from two published points, not a chain
measurement. The method section lists the assumptions.

## Results

Each round is 600 s on one card, with recipe `w45-p1350` on the
`tests/fixtures/aglais.spec.json` topology (4,577 nodes). Energies include the
instance offset. Lower is better.

| Mode | Rounds | Round best, mean | Round best, range | Work per round |
|---|---:|---:|---:|---|
| `stock` (settings before the population anneal) | 2 | −5,054.26 | −5,055.93 to −5,052.59 | 148 to 158 salts at 160 × 8,192 |
| `plain` (population anneal, every salt in order) | 6 | −5,107.48 | −5,112.57 to −5,104.36 | 119 to 127 salts at 160 × 8,192 |
| `screen` (population anneal and lease screening) | 6 | −5,120.01 | −5,123.40 to −5,117.13 | 658,149 to 691,244 salts scored, 24 probes, 2 long anneals |

The population anneal and the family defaults are worth about 53 units per round
at the same budget. Screening adds about 12.5 units over `plain`.

The screen result comes from two effects:

- The score selects better salts. A screened probe averages −5,101.1. A salt in
  order averages −5,095.7 (743 salts, standard deviation 4.5).
- The long anneal (256 × 262,144) gains about 18 units over a probe. It takes
  182 s on GPU 0 and 196 s on GPU 1. The best of each round came from a long anneal.

The expected best of k long anneals, resampled from the 12 measured:

| Long anneals per round | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Expected best | −5,118.77 | −5,120.18 | −5,120.96 | −5,121.51 |

One card fits two long anneals per round. Two cards give four, about 1.3 units lower.

## Estimated share of rounds

The chain graph has 4,560 nodes, and the fixture has 4,577, so absolute energies
do not compare. The estimate uses differences only, anchored on the two published
points of the QuipNetwork/quip-solver-cuda#3 author (5090 Laptop GPU, 600 s rounds, chain graph):

| Author setup | Round best | Share of rounds |
|---|---:|---:|
| Every salt at 160 × 8,192, best re-solved deeper | −5,074.4 | about 1 in 600 |
| Screening and the two-stage solve | −5,082.3 | about 6 % |

An exponential fit through these two points gives a share that changes by a factor
of e^0.454, about 1.57, per unit of energy. The estimates:

| Setup on this host | Energy relative to the author's screen | Estimated share |
|---|---:|---:|
| One A4000, `screen` | 0 to +0.8 (2 long anneals against the author's 2 or 3) | 4 to 6 % |
| Two A4000, `screen` | −1.3 to −0.5 | 8 to 11 % |
| One A4000, `plain` | Best of 124 salts against 250: +1.4 against the author's `plain` | about 0.1 % |
| Two A4000, `plain` | About the author's `plain` | about 0.2 % |
| `stock` | About +53 against `plain` | 0 |

The A4000 runs a probe in 4.0 s alone and the 5090 Laptop GPU in 2.4 s. The long
anneal takes about 3 minutes on both, so the long anneal, not the probe rate,
sets the screened result.

## Limits of the estimate

- Two points fix the energy-to-share curve. The real curve depends on the other
  miners, and it changes when they change.
- Shares are not additive. The #3 author runs the same code. Two miners that run
  it split the rounds that either one would win alone.
- Rounds are 600 s, as in the author's bench. A different round length moves the
  number of long anneals per round.
- Both cards throttled for most of each run (thermal slowdown, median SM clock
  1,485 MHz on GPU 0 and 1,395 MHz on GPU 1, up to 97 °C). The results describe
  sustained mining on this host.
- Screening shares the CPU with the host population step. With 14 generator
  threads per card on 32 hardware threads, a probe took 13.5 s. With 10 threads
  per card, it took 6.5 s, against 4.0 s alone. A two-card host must leave cores
  for the population step.

## Method

`examples/otsuka_round_bench.rs` generates canonical `quip/twisted-ice-dimer`
salts and solves them with `ClusterDevice::run_work_population`, the miner's
family path. `screen` mode keeps the best 48 scored salts, waits for 10,000
scored salts, then repeats: probe the best 8 at 160 × 8,192, then give the best
probe one long anneal. That is the lease screen with one holder. A solve that the
round deadline stops does not count.

Calibration: on the 2026-09-21 live topology (4,575 nodes) with recipe
`w62-p1200`, seven `plain` salts averaged −5,055.9 at 4.0 s each. The author
reports about −5,054 at 2.4 s on that graph.

Commands, run on 2026-11-16 under
`bench-evidence/cuda-sa-limitations/telemetry-run.sh`, with both cards in parallel:

```sh
otsuka_round_bench --mode screen --seconds 600 --rounds 3 --first-round 0 --threads 10  # GPU 0
otsuka_round_bench --mode screen --seconds 600 --rounds 3 --first-round 3 --threads 10  # GPU 1
otsuka_round_bench --mode plain --seconds 600 --rounds 3 --first-round 0   # GPU 0, then 3 on GPU 1
QUIP_OTSUKA_RESAMPLES=0 QUIP_OTSUKA_CHUNK=1 QUIP_OTSUKA_BETA=stock \
  otsuka_round_bench --mode stock --seconds 600 --rounds 1 --first-round 0  # GPU 0, then 3 on GPU 1
```

Hardware: two NVIDIA RTX A4000, AMD Ryzen 9 5950X (32 hardware threads).
`2026-11-16-otsuka-rounds-a4000.jsonl` holds one line per round, and
`2026-11-16-otsuka-rounds-a4000-solves.jsonl` holds one line per finished solve.
