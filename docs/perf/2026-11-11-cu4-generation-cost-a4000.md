# Family generation costs

Core 0.0.8 already overlaps canonical host generation with device work.
This work adds no GPU generator, and CU-1 preparation stays serial on the sampler thread.
CU-4 measures existing behavior without production changes.

The measured generation share depends on the job budget:

| Budget | Measured cases | Generation alone / job | Generation + coefficient H2D / job | Full frontend / job |
|---|---|---:|---:|---:|
| Mining floor: 64 reads, 256 sweeps | Three registered 4,800-node recipes | 13.39–13.65% | 13.47–13.73% | 21.22–21.79% |
| Mining ceiling: 256 reads, 2,048 sweeps | Three 4,800-node recipes and one 10,656-node case | 0.72–0.98% | 0.73–0.99% | 1.35–1.55% |

The measurements give host costs and upper bounds on hypothetical savings before overlap.
For the three registered recipes, the 10% opportunity threshold holds at the measured floor and fails at the measured ceiling.
The ceiling measures about 0.7–1.6% across generation and full frontend shares.
Generation alone is smaller than the full frontend, which also includes preparation and H2D.
Larger kernel times explain the smaller ceiling shares, independent of the observed clock decrease in the largest case.
No admitted measured case exceeds the alternative 50 ms saving condition.
A device generator would add kernel time. These bounds do not measure a device-generation speedup or new savings over existing overlap.

## Measurement environment

Physical GPU 1 is the only GPU used. Every GPU process has `CUDA_VISIBLE_DEVICES=1`.
Visible CUDA ordinal zero selects physical GPU 1. Commands run serially.
The GPU is NVIDIA RTX A4000, UUID `GPU-83b5eb66-e645-f1bb-23d6-95aa0ea03e09`, PCI `00000000:06:00.0`.
The driver is 615.71.10. The host CPU is AMD Ryzen 9 5950X. Rust and Cargo are 1.97.1.
The UTC measurement date is 2026-11-11.
The generation matrix runs on 2026-11-10 locally. The throughput sessions run on 2026-11-11 locally.

The registered low-budget recipes start at 65–68 C and finish at 69–73 C.
Their GPU phases last 0.88–0.89 seconds. Their reported SM clocks are 1,860–1,905 MHz.
The largest low-budget GPU phase lasts 3.054 seconds.
The recorded 10,656-node ceiling case starts at 93 C and 1,590 MHz and ends at 94 C and 1,485 MHz.
The high-temperature clock decrease is consistent with throttling, although the JSONL does not record throttle-reason flags.
That historical case takes 52.53 seconds, including a 50.810-second GPU phase, exceeding the 45-second limit.
The table retains this result. The new throughput runs each stay below the 45-second limit.
The longest individual complete-job sample is 4.874 seconds. The longest individual instrumented run is 4.622 seconds.
All 39 case logs report passing tests. The supervisor receives termination after the final row and passing result are present.

Source commits: `4698be367048a8f71f44a0a71e0574626d9db50f` is the API-default baseline.
`c4652f295d228144f7e5a8ee166e1fdcd865842d` adds mining budgets and the contract node endpoint.
The initial checkout is `fdb8dc76e5e2df7ea8ead88afa2eeb4798499702`.

Raw samples: [2026-11-11-cu4-generation-cost-a4000.jsonl](2026-11-11-cu4-generation-cost-a4000.jsonl).

## Method

The ignored `cluster::device::generation_cost::measure_generation_cost` test requires `long-tests`.
It uses quip-solver-core 0.0.8 and its canonical family generator.
The SDK exposes no public `FamilyLease` constructor.
To match `FamilyLease::generate`, the benchmark admits the saved descriptor and calls `generate_instance`.
The lease uses block hash `[1; 32]`, account `[2; 32]`, salt template `[3; 32]`, and counters from zero.
Nonce derivation and topology admission occur outside the generation timer.

Each valid size has 20 successful lease generations and 20 `PreparedGraph::from_instance` constructions, after warmup.
No salt skips occurred.
Each device-admitted size has 20 uploads, 20 coefficient uploads, five kernel sums, and five complete jobs.
All reported times are medians.
CUDA events measure the wrapper upload span for all 11 prepared buffers.
That span includes allocation and pageable-host enqueue gaps alongside the PCIe transfer.
The coefficient measurement copies five preallocated buffers: `weak_ell`, `all_ell`, `h`, `tile_w`, and `link_w`.
The ELL buffers also contain neighbor indices. Counting their full copy time gives an optimistic transfer bound.

Kernel time sums CUDA events around every production launch, including 256 greedy passes.
The event path uses `ClusterWork` and the existing kernel. It checks identical spins against uninstrumented `run_work`.
The complete-job timer includes generation, float conversion, CU-1, allocation/upload, schedule construction, launches, download, and final rescoring.
It excludes one-time lease admission, topology creation, JIT, and network transport.
Host generation and preparation use 20 distinct successful lease salts.
GPU measurements repeat the twentieth successful instance.
The baseline complete-job samples use the first five lease salts. The mining-budget samples repeat the twentieth salt.

`Front % = 100 * (generation + preparation + H2D) / total`.
`Save ms = generation + coefficient H2D`.
`Save % = 100 * Save ms / total`.
Preparation remains outside the saving bound. Device generation would add its own kernel time.

## Mining session floor: 64 reads, 256 sweeps

| Parameters m,t,weak,pairs | Nodes | Edges | Generation ms | Preparation ms | H2D ms | Coefficient H2D ms | Kernel ms | Total ms | Front % | Save ms | Save % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1,1,62,3` | 12 | 22 | 0.006 | 0.004 | 0.054 | 0.010 | 7.260 | 8.286 | 0.77 | 0.016 | 0.19 |
| `1,4,62,12` | 48 | 280 | 0.036 | 0.052 | 0.056 | 0.011 | 9.033 | 10.403 | 1.38 | 0.047 | 0.45 |
| `2,4,62,40` | 160 | 1224 | 0.223 | 0.202 | 0.058 | 0.012 | 10.690 | 12.674 | 3.81 | 0.236 | 1.86 |
| `4,4,62,144` | 576 | 5032 | 1.147 | 0.820 | 0.066 | 0.018 | 13.600 | 18.622 | 10.91 | 1.165 | 6.25 |
| `8,4,62,544` | 2176 | 20328 | 5.346 | 3.360 | 0.094 | 0.040 | 26.432 | 44.324 | 19.85 | 5.386 | 12.15 |
| `12,4,62,1200` | 4800 | 45864 | 13.452 | 7.867 | 0.150 | 0.077 | 56.944 | 98.520 | 21.79 | 13.529 | 13.73 |
| `12,4,62,1350` | 4800 | 45864 | 13.318 | 7.667 | 0.154 | 0.077 | 57.356 | 98.862 | 21.38 | 13.395 | 13.55 |
| `12,4,45,1200` | 4800 | 45864 | 13.103 | 7.509 | 0.144 | 0.077 | 56.509 | 97.823 | 21.22 | 13.180 | 13.47 |
| `16,4,62,2112` | 8448 | 81640 | 24.000 | 13.726 | 0.222 | 0.119 | 147.893 | 221.454 | 17.14 | 24.119 | 10.89 |
| `18,4,62,2664` | 10656 | 103368 | 32.113 | 18.235 | 0.257 | 0.146 | 174.489 | 269.653 | 18.77 | 32.259 | 11.96 |
| `13,7,62,2457` | 9828 | 151018 | 47.047 | 28.165 | 0.345 | 0.242 | 203.085 | 334.263 | 22.60 | 47.288 | 14.15 |
| `14,7,62,2842` | 11368 | 175182 | 58.425 | 33.225 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `32,1,62,2080` | 8320 | 32634 | 9.342 | 5.251 | 0.137 | 0.060 | 131.018 | 164.060 | 8.98 | 9.403 | 5.73 |
| `0,0,62,32768` | 131072 | 229376 | 75.892 | 15.965 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Host capacity and family admission

The API baseline below supplies host timings through z(32,7).
The mining table includes the adjacency-mode contract endpoint.
The card has 101,376 bytes of shared memory per block, which permits at most 11,261 nodes under CU-1.
The 11,368-node z(14,7) and the larger full lattices exceed that budget.
The 131,072-node adjacency instance exceeds the kernel index width of 65,536 nodes.
The full t=8 lattices at the schema endpoint fail family admission:

| Parameters m,t,weak,pairs | Nodes | Edges | Admission |
|---|---:|---:|---|
| `1,8,62,24` | 96 | 1072 | Code 13: degree exceeds 32 |
| `8,8,62,1088` | 4352 | 73424 | Code 13: degree exceeds 32 |
| `13,8,62,2808` | 11232 | 194224 | Code 13: degree exceeds 32 |
| `14,8,62,3248` | 12992 | 225296 | Code 13: degree exceeds 32 |
| `31,8,62,15624` | 62496 | 1106032 | Code 13: degree exceeds 32 |
| `32,8,62,16640` | 66560 | 1178576 | Code 13: degree exceeds 32 |

## One read and 64 sweeps

| Parameters m,t,weak,pairs | Nodes | Edges | Generation ms | Preparation ms | H2D ms | Coefficient H2D ms | Kernel ms | Total ms | Front % | Save ms | Save % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1,1,62,3` | 12 | 22 | 0.005 | 0.004 | 0.054 | 0.010 | 3.640 | 4.266 | 1.49 | 0.016 | 0.37 |
| `1,4,62,12` | 48 | 280 | 0.036 | 0.051 | 0.056 | 0.011 | 4.367 | 5.142 | 2.78 | 0.047 | 0.92 |
| `2,4,62,40` | 160 | 1224 | 0.221 | 0.203 | 0.058 | 0.012 | 5.358 | 6.461 | 7.46 | 0.233 | 3.61 |
| `4,4,62,144` | 576 | 5032 | 1.157 | 0.847 | 0.068 | 0.018 | 6.401 | 8.985 | 23.06 | 1.176 | 13.09 |
| `8,4,62,544` | 2176 | 20328 | 5.304 | 3.364 | 0.092 | 0.043 | 10.050 | 19.411 | 45.13 | 5.347 | 27.55 |
| `12,4,62,1200` | 4800 | 45864 | 13.298 | 7.707 | 0.148 | 0.079 | 15.190 | 36.394 | 58.13 | 13.377 | 36.76 |
| `12,4,62,1350` | 4800 | 45864 | 13.618 | 7.852 | 0.148 | 0.078 | 15.208 | 36.215 | 59.69 | 13.696 | 37.82 |
| `12,4,45,1200` | 4800 | 45864 | 13.300 | 7.718 | 0.146 | 0.081 | 15.313 | 35.852 | 59.03 | 13.381 | 37.32 |
| `16,4,62,2112` | 8448 | 81640 | 25.616 | 15.004 | 0.231 | 0.123 | 24.887 | 65.011 | 62.84 | 25.739 | 39.59 |
| `18,4,62,2664` | 10656 | 103368 | 32.307 | 18.613 | 0.266 | 0.150 | 28.598 | 81.055 | 63.15 | 32.456 | 40.04 |
| `19,4,62,2964` | 11856 | 115192 | 36.606 | 19.195 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `24,4,62,4704` | 18816 | 183912 | 61.624 | 37.733 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `32,4,62,8320` | 33280 | 327144 | 117.695 | 72.637 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `32,7,62,14560` | 58240 | 916566 | 389.302 | 280.276 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `32,1,62,2080` | 8320 | 32634 | 9.249 | 5.357 | 0.139 | 0.059 | 26.526 | 37.809 | 39.00 | 9.308 | 24.62 |

## Mining session ceiling: 256 reads, 2,048 sweeps

These measured results fall below the 10% opportunity threshold.
Generation alone is 0.72–0.98% of a complete job. Generation plus coefficient H2D is 0.73–0.99%.
The full frontend is 1.35–1.55%. Longer device work accounts for these smaller shares.
The environment section records the observed clock decrease for the 10,656-node case.

| Parameters m,t,weak,pairs | Nodes | Edges | Generation ms | Preparation ms | H2D ms | Coefficient H2D ms | Kernel ms | Total ms | Front % | Save ms | Save % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `12,4,62,1200` | 4800 | 45864 | 13.432 | 7.565 | 0.143 | 0.074 | 1252.908 | 1365.348 | 1.55 | 13.505 | 0.99 |
| `12,4,62,1350` | 4800 | 45864 | 13.463 | 7.649 | 0.138 | 0.071 | 1371.479 | 1488.955 | 1.43 | 13.534 | 0.91 |
| `12,4,45,1200` | 4800 | 45864 | 13.749 | 8.018 | 0.141 | 0.081 | 1490.763 | 1564.477 | 1.40 | 13.830 | 0.88 |
| `18,4,62,2664` | 10656 | 103368 | 33.475 | 28.625 | 0.292 | 0.159 | 4575.700 | 4622.234 | 1.35 | 33.634 | 0.73 |

## Assumptions

- Only twisted-ice-dimer revision 1 is in scope. This branch routes that revision through `quip-cuda-otsuka`.
- The main table uses the session floor of 64 reads and 256 sweeps. Each beta has one sweep. The beta range is automatic and the schedule is geometric.
- The sensitivity table uses the adaptation ceiling: 256 reads and 2,048 sweeps. Both use one requested solution and seed zero.
- The API baseline uses `SampleParams::default()`: one read and 64 sweeps. That baseline differs from the mining session budget.
- The size grid samples full Zephyr lattices from the smallest nonempty lattice to the parameter endpoints. The grid covers a sample of parameter sets.
- Full lattices with t=8 exceed the family degree limit of 32, despite schema acceptance of t=8. Their generation and device timings are N/A.
- Full z(32,7) reaches 58,240 nodes. Adjacency mode at parameters m=t=0 uses a ring of K4 motifs to reach the contract node ceiling of 131,072 nodes.
- Preparation timings on rejected models measure the attempted construction. N/A device cells represent admission failures, not zero cost.
- CUDA clocks and host CPU frequency remain under normal driver and system control. The saving bound is not a measured device-generation speedup.

## Rerun

Run one case at a time from the checkout. Let the card cool between high-budget cases.
The output appends one JSON record per case.

```sh
CUDA_VISIBLE_DEVICES=1 \
CU4_SHAPE=12,4,62,1200 \
CU4_BUDGET=mining-easy \
CU4_OUTPUT=docs/perf/cu4-rerun.jsonl \
cargo test --release --features long-tests --lib measure_generation_cost -- \
  --ignored --nocapture --test-threads=1
```

`CU4_SHAPE` lists m, t, weak milli, and pair count.
`CU4_BUDGET` accepts `mining-easy` (default), `mining-hard`, or `api-default`.
The `0,0,62,32768` case selects the 131,072-node adjacency fixture.
The JSONL records each case, schedule, commit, GPU facts, raw samples, and admission result.
The shape and budget columns in the tables define the recorded matrix.

## Checks

The requested Rust checks pass with zero compiler or lint warnings:

- `cargo fmt --all --check`
- `cargo clippy --release --all-targets --all-features -- -D warnings`
- `cargo clippy --release --all-targets -- -D warnings`

The record correction reruns these three checks. Both clippy commands report zero warnings.
Historical session verification records 438 passed and 264 ignored tests for the release suite.
The ignored total includes six library tests and 258 integration tests, confirmed through `--ignored --list` on existing default-feature binaries.
That earlier suite uses `RUST_TEST_THREADS=1` and `CUDA_VISIBLE_DEVICES=1`, and the record correction does not rerun it.

## Existing host generation overlap

Core 0.0.8 generates family instances on a Tokio runtime worker, separate from the `quip-sampler` thread.
The expander task can migrate between runtime workers at await points.
It calls canonical generation synchronously within each task poll.
It does not generate instances on `quip-sampler`.

The source chain in `quip-solver-core-0.0.8/src/` is:

- `lib.rs:187–191`: `generates_locally` defaults to false. The Otsuka sampler keeps this default.
- `session.rs:1999`: the miner creates the multithread Tokio runtime.
- `session.rs:1155`: prefetch is twice the live stream width. `lease_depth` uses that bound.
- `session.rs:1185`: the `quip-sampler` thread consumes the job receiver.
- `session.rs:1197`: a shared semaphore bounds unfinished lease salts across expanders.
- `session.rs:1507`: the session spawns `Expander::run` as a Tokio task.
- `lease.rs:698`: the expander acquires a slot before drawing each salt.
- `lease.rs:708`: `salt_graph` generates the next instance before sending its job.
- `lease.rs:636`: `family_graph` calls `FamilyLease::generate` and converts the instance into a graph and family context.
- `lease.rs:133`: `FamilyLease::generate` admits the saved descriptor and calls `generate_instance` at line 143.
- `lease.rs:750`: the expander sends the generated job to the sampler channel.

Within this crate, `src/cluster/sampler.rs:442` receives each generated `WarmStreamJob`.
Line 449 builds `PreparedGraph` on `quip-sampler`, immediately before the synchronous device work at line 450.
`src/cluster/prepare.rs:296` delegates family preparation to `from_instance`.
`src/cluster/sampler.rs:466` delegates stream width to SA, which resolves to 48 on the A4000.
Core permits 96 unfinished lease salts. This bound covers queued and active work.
This existing bound differs from a proposed new queue of one or two prepared jobs.
CU-4 adds no queue and changes no bound.

Canonical generation already overlaps GPU work. CU-1 preparation remains serial on the sampler thread.
The task explicitly requires stopping at measurement when core already generates ahead on another thread.
That branch requires no production change or core change.

### Throughput method

The existing ignored measurement test now accepts `CU4_THROUGHPUT=serial` or `core-prefetch`.
It remains behind `long-tests` and reuses the real coordinator in `tests/common/lease.rs`.
Both modes launch the unchanged release miner and use the same 50 canonical lease salts.
One untimed single-salt lease warms the sampler before the timer starts.
Every timed salt uses 64 reads and 256 sweeps.

The serial control sends 50 single-salt leases. It waits for each completion and refund before sending the next lease.
The existing core mode sends one 50-salt lease, allowing the expander to generate ahead while the sampler runs.
The timer includes initial generation, sampling, result transport, lease completion, and refunds.
It excludes process startup, warmup, independent energy checks, and shutdown.
The serial control also includes 49 extra lease exchanges and repeated lease admission.
The throughput difference is not all generation because the comparison includes those extra exchanges and repeated admissions.
The original 4,800-node serial control spends 1.878946 seconds outside the sampler, or 38.35 ms per extra exchange.
That scale exceeds the measured 13.45 ms generation cost per salt.
The division does not isolate exchange latency because the residual also includes generation and other protocol work.
Both modes use the same production code.

The measurement checks salt and nonce order, 64 reads per salt, reported sweep counts, and exact energies against regenerated instances.
It also checks rejection and fatal-error absence, each expected refund, and a successful shutdown.
Sampler timing sums the 50 reported `device_access_time_us` values.
This sum includes CU-1, upload, device work, download, and final sampler scoring.
The remaining wall time includes initial generation, protocol work, and any idle gaps.
`Session::until` at `tests/common/lease.rs:202–225` awaits incoming protocol frames through `inbound.message()`.
The helper tests completion after each frame without a fixed polling interval or log-poll quantization.
The timer ends after the final completion and refund arrive, so network and scheduling delays remain in the wall time.

Raw session records: [2026-11-11-cu4-overlap-a4000.jsonl](2026-11-11-cu4-overlap-a4000.jsonl).

### Throughput results

The dataset contains all 27 recorded sessions: seven original runs plus all 20 new repeats.
Rows use their one-based JSONL line number. Each session covers 50 jobs on physical GPU 1.
The 4,800-node sessions use `12,4,62,1200`, and the 2,176-node sessions use `8,4,62,544`.
Rows 1–7 use commit `19287e0`. Rows 8–27 use `f403237` without production or benchmark changes.
N/A means the original record lacks summed sampler timing, so its outside-sampler time is also unknown.
The wall-time and throughput statistics include every run.

| Row | Nodes | Mode | Wall s | Sampler s | Outside sampler s | Jobs/s | Start→end C | End SM MHz |
|---:|---:|---|---:|---:|---:|---:|---|---:|
| 1 | 4,800 | `serial` | 10.428149 | N/A | N/A | 4.7947 | 62→66 | 1560 |
| 2 | 4,800 | `core-prefetch` | 6.141930 | N/A | N/A | 8.1408 | 65→75 | 1890 |
| 3 | 2,176 | `core-prefetch` | 2.996929 | N/A | N/A | 16.6837 | 64→71 | 1890 |
| 4 | 2,176 | `serial` | 3.659842 | 2.689768 | 0.970074 | 13.6618 | 65→70 | 1890 |
| 5 | 2,176 | `core-prefetch` | 2.994327 | 2.965065 | 0.029262 | 16.6982 | 65→72 | 1890 |
| 6 | 4,800 | `serial` | 6.912209 | 5.033263 | 1.878946 | 7.2336 | 66→75 | 1875 |
| 7 | 4,800 | `core-prefetch` | 5.437914 | 5.376243 | 0.061671 | 9.1947 | 68→78 | 1860 |
| 8 | 4,800 | `serial` | 8.576757 | 5.832188 | 2.744569 | 5.8297 | 65→73 | 1710 |
| 9 | 4,800 | `core-prefetch` | 4.478435 | 4.449387 | 0.029048 | 11.1646 | 66→76 | 1845 |
| 10 | 2,176 | `serial` | 3.651597 | 2.740495 | 0.911102 | 13.6926 | 65→71 | 1815 |
| 11 | 2,176 | `core-prefetch` | 2.166600 | 2.150621 | 0.015979 | 23.0776 | 65→73 | 1890 |
| 12 | 4,800 | `core-prefetch` | 4.452764 | 4.420198 | 0.032566 | 11.2290 | 66→77 | 1860 |
| 13 | 4,800 | `serial` | 6.213241 | 4.582754 | 1.630487 | 8.0473 | 67→76 | 1860 |
| 14 | 2,176 | `core-prefetch` | 2.102655 | 2.085974 | 0.016681 | 23.7795 | 66→74 | 1890 |
| 15 | 2,176 | `serial` | 2.888623 | 2.167726 | 0.720897 | 17.3093 | 66→74 | 1890 |
| 16 | 4,800 | `serial` | 6.129598 | 4.402014 | 1.727584 | 8.1571 | 67→76 | 1860 |
| 17 | 4,800 | `core-prefetch` | 4.337379 | 4.311111 | 0.026268 | 11.5277 | 67→77 | 1860 |
| 18 | 2,176 | `serial` | 2.941636 | 2.232171 | 0.709465 | 16.9973 | 66→73 | 1890 |
| 19 | 2,176 | `core-prefetch` | 2.021869 | 2.007104 | 0.014765 | 24.7296 | 64→73 | 1890 |
| 20 | 4,800 | `core-prefetch` | 4.970818 | 4.936391 | 0.034427 | 10.0587 | 66→75 | 1875 |
| 21 | 4,800 | `serial` | 6.095595 | 4.379600 | 1.715995 | 8.2026 | 67→77 | 1860 |
| 22 | 2,176 | `core-prefetch` | 2.347470 | 2.323941 | 0.023529 | 21.2995 | 66→74 | 1860 |
| 23 | 2,176 | `serial` | 2.778961 | 2.113491 | 0.665470 | 17.9923 | 66→73 | 1890 |
| 24 | 4,800 | `serial` | 6.253384 | 4.355608 | 1.897776 | 7.9957 | 67→76 | 1845 |
| 25 | 4,800 | `core-prefetch` | 4.335366 | 4.305017 | 0.030349 | 11.5331 | 66→77 | 1860 |
| 26 | 2,176 | `serial` | 2.853481 | 2.133413 | 0.720068 | 17.5225 | 66→73 | 1890 |
| 27 | 2,176 | `core-prefetch` | 2.199783 | 2.185566 | 0.014217 | 22.7295 | 65→74 | 1890 |

The following aggregates include every recorded run for each size and mode.
These session ratios are indicative because the measurements leave clocks and host scheduling under normal system control.

| Nodes | Mode | Runs | Median wall s | Full wall range s | Median jobs/s | Full jobs/s range |
|---|---|---:|---:|---|---:|---|
| 4,800 | `serial` | 7 | 6.253384 | 6.095595–10.428149 | 7.9957 | 4.7947–8.2026 |
| 4,800 | `core-prefetch` | 7 | 4.478435 | 4.335366–6.141930 | 11.1646 | 8.1408–11.5331 |
| 2,176 | `serial` | 6 | 2.915130 | 2.778961–3.659842 | 17.1533 | 13.6618–17.9923 |
| 2,176 | `core-prefetch` | 7 | 2.199783 | 2.021869–2.996929 | 22.7295 | 16.6837–24.7296 |

Median rate ratios, `core-prefetch / serial`: 4,800 nodes: 1.3963× (39.63% higher session throughput). 2,176 nodes: 1.3251× (32.51% higher session throughput).

The five new pairs per size use order `serial, core-prefetch` in rounds 1, 3, and 5.
Rounds 2 and 4 reverse the order. Each round measures 4,800 nodes before 2,176 nodes.
The new repeats have the following medians and full ranges, as a separate cohort alongside the all-run aggregates.

| Nodes | Mode | Runs | Median wall s | Full wall range s | Median jobs/s | Full jobs/s range |
|---|---|---:|---:|---|---:|---|
| 4,800 | `serial` | 5 | 6.213241 | 6.095595–8.576757 | 8.0473 | 5.8297–8.2026 |
| 4,800 | `core-prefetch` | 5 | 4.452764 | 4.335366–4.970818 | 11.2290 | 10.0587–11.5331 |
| 2,176 | `serial` | 5 | 2.888623 | 2.778961–3.651597 | 17.3093 | 13.6926–17.9923 |
| 2,176 | `core-prefetch` | 5 | 2.166600 | 2.021869–2.347470 | 23.0776 | 21.2995–24.7296 |

Median rate ratios, `core-prefetch / serial`: 4,800 nodes: 1.3954× (39.54% higher session throughput). 2,176 nodes: 1.3333× (33.33% higher session throughput).

Sampler timing is available for six runs per size and mode, including all five new repeats.
The next table uses every available sampler timing and leaves the three missing values unknown.

| Nodes | Mode | Timed runs | Median sampler s | Full sampler range s | Median outside s | Full outside range s |
|---|---|---:|---:|---|---:|---|
| 4,800 | `serial` | 6 | 4.492384 | 4.355608–5.832188 | 1.803265 | 1.630487–2.744569 |
| 4,800 | `core-prefetch` | 6 | 4.434793 | 4.305017–5.376243 | 0.031457 | 0.026268–0.061671 |
| 2,176 | `serial` | 6 | 2.199949 | 2.113491–2.740495 | 0.720482 | 0.665470–0.970074 |
| 2,176 | `core-prefetch` | 6 | 2.168094 | 2.007104–2.965065 | 0.016330 | 0.014217–0.029262 |

Sampler time also differs between modes, so a throughput ratio cannot represent generation alone.
The original timed pairs show 6.8% more sampler time with prefetch at 4,800 nodes and 10.2% more at 2,176 nodes.
Rows 6–7 and 4–5 show those differences, but the measurements do not isolate their cause.
The sampler table shows the full range across all instrumented runs.

Rows 7 and 5 leave only 62 ms and 29 ms outside the sampler.
These small gaps and the source trace provide evidence that generation already overlaps sampling.
The session throughput percentages also include the control's extra 49 lease exchanges.
The ratios measure existing session behavior without isolating savings from generation or CU-1.
CU-1 preparation stays serial immediately before device work.

The initial 4,800-node core-prefetch run takes 6.141930 seconds, compared with 5.437914 seconds in row 7.
Final SM readings of 1,890 and 1,860 MHz do not support excluding the first run for clocks.
All initial results remain in both the raw data and the all-run aggregates.

The repeat supervisor waits at least 45 seconds between processes and checks GPU 1 before each warmup.
It requires at most 68 C and zero reported GPU activity. After the interruption, it also requires no GPU compute process.

The initial supervisor stops after row 13 when its 44-second process deadline expires.
That row already contains a valid 6.213241-second timed result, and the test later reports success after 59.66 seconds.
The initial deadline kills only Cargo, allowing the test child to finish. All recorded session data remains included.

A separate GPU 1 process then appears at 100% GPU activity and 95 C, so the repeat sequence pauses until the card cools.
After resuming, the supervisor runs the same prebuilt ignored test directly and applies a 44-second deadline to the whole process group.

Possible contention remains unknown because the measurements do not record the start of the other workload.
All timed sessions stay below 45 seconds, and the statistics include every row despite possible contention.
The table's start and end readings bracket timed work after warmup.

### Rerun the session comparison

Build the miner once. Run one shape and mode per process.
Use `8,4,62,544` for the smaller case and `serial` for the control.
Run only one GPU command at a time.

```sh
cargo build --release --bin quip-cuda-otsuka
mkdir -p .tmp
CUDA_VISIBLE_DEVICES=1 \
TMPDIR="$PWD/.tmp" \
CU4_SHAPE=12,4,62,1200 \
CU4_THROUGHPUT=core-prefetch \
CU4_BINARY="$PWD/target/release/quip-cuda-otsuka" \
CU4_OUTPUT=docs/perf/cu4-overlap-rerun.jsonl \
cargo test --release --features long-tests --lib measure_generation_cost -- \
  --ignored --nocapture --test-threads=1
```

The existing ignored benchmark supports the 50-job measurement.
No new test enters the default suite or the cluster session timing gate.

### Session verification

The formatting check and both requested clippy checks pass for this record correction.
The 20 new ignored benchmark sessions check all 50 timed jobs in each session on physical GPU 1.
The benchmark checks salt and nonce order, reads, sweeps, energies, refunds, and shutdown.
The record correction changes neither production code nor benchmark code.

The earlier session verification records 438 passed and 264 ignored release tests, including the six ignored library tests.
It also records 45 passing cluster tests with ignored tests enabled and 24 passing cluster session timing tests.
These historical checks are distinct from the new throughput repeats and the three rerun host checks.

Vale reports zero errors and four Readability warnings at `--minAlertLevel=warning`.
The permitted metrics are ColemanLiau, FleschReadingEase, LIX, and SMOG.
