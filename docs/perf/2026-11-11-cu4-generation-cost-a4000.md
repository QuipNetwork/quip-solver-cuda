# Family generation costs

The CU-4 session measurements below supersede the original device-generation recommendation.
Core 0.0.8 already generates lease instances ahead of the sampler.
The maintainer selected host overlap instead of device generation.
Production behavior remains unchanged under the task's measurement-only stop condition.

The original spike recommended device generation under its stated decision rule.
At all three registered 4,800-node recipes, generation plus coefficient transfer is 13.47–13.73% of a low-budget mining job.
Generation alone is 13.39–13.65%, so the conclusion does not depend on counting packed neighbor bits as coefficient transfer.
The 10% condition passes at mining-relevant sizes. No admitted measured case exceeds the alternative 50 ms saving condition.
This is an upper-bound opportunity. No device generator exists in this measurement. Its added kernel time would reduce the saving.
The canonical host generator remains the implemented default. This work adds no generation kernel.

## Measurement environment

Physical GPU 1 is the only GPU used. Every GPU process has `CUDA_VISIBLE_DEVICES=1`.
Visible CUDA ordinal zero selects physical GPU 1. Commands run serially.
The GPU is NVIDIA RTX A4000, UUID `GPU-83b5eb66-e645-f1bb-23d6-95aa0ea03e09`, PCI `00000000:06:00.0`.
The driver is 615.71.10. The host CPU is AMD Ryzen 9 5950X. Rust and Cargo are 1.97.1.
The UTC measurement date is 2026-11-11, which is 2026-11-10 locally.

The registered low-budget recipes start at 65–68 C and finish at 69–73 C.
Their GPU phases last 0.88–0.89 seconds. Their reported SM clocks are 1,860–1,905 MHz.
The largest low-budget GPU phase lasts 3.054 seconds.
The high-budget sequence reaches 95 C and lower clocks. Treat all high-budget rows as thermally affected sensitivity data.
The high-budget 10,656-node case takes 52.53 seconds, including a 50.810-second GPU phase: it exceeds the 45-second limit.
Its SM clock drops from 1,590 to 1,485 MHz. It does not support the recommendation.
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

These rows are thermally affected. They do not support the recommendation.

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

All required checks pass with zero warnings:

- `cargo fmt --all --check`
- `cargo clippy --release --all-targets --all-features -- -D warnings`
- `cargo clippy --release --all-targets -- -D warnings`
- `cargo test --release`: 438 passed. The run ignores 258 tests and reports zero failures.

The test run uses `RUST_TEST_THREADS=1` and `CUDA_VISIBLE_DEVICES=1`.

## Existing host generation overlap

Core 0.0.8 generates family instances on a Tokio runtime worker, separate from the `quip-sampler` thread.
The expander task can migrate between runtime workers at await points.
It calls canonical generation synchronously within each task poll.
It does not generate instances on `quip-sampler`.

The source chain in `quip-solver-core-0.0.8/src/` is:

- `lib.rs:185`: `generates_locally` defaults to false. The Otsuka sampler keeps this default.
- `session.rs:1999`: the miner creates the multithread Tokio runtime.
- `session.rs:1155`: prefetch is twice the live stream width. `lease_depth` uses that bound.
- `session.rs:1184`: the `quip-sampler` thread consumes the job receiver.
- `session.rs:1197`: a shared semaphore bounds unfinished lease salts across expanders.
- `session.rs:1507`: the session spawns `Expander::run` as a Tokio task.
- `lease.rs:689`: the expander acquires a slot before drawing each salt.
- `lease.rs:708`: `salt_graph` generates the next instance before sending its job.
- `lease.rs:636`: `family_graph` calls `FamilyLease::generate` and converts the instance into a graph and family context.
- `lease.rs:133`: `FamilyLease::generate` admits the saved descriptor and calls `generate_instance` at line 143.
- `lease.rs:750`: the expander sends the generated job to the sampler channel.

Within this crate, `src/cluster/sampler.rs:442` receives each generated `WarmStreamJob`.
Line 449 builds `PreparedGraph` on `quip-sampler`, immediately before the synchronous device work at line 450.
`src/cluster/prepare.rs:290` delegates family preparation to `from_instance`.
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
The control includes 49 extra lease exchanges and repeated lease admission.
The throughput difference compares whole sessions and does not isolate generation savings.
Both modes use the same production code.

The measurement checks salt and nonce order, 64 reads per salt, reported sweep counts, and exact energies against regenerated instances.
It also checks rejection and fatal-error absence, each expected refund, and a successful shutdown.
Sampler timing sums the 50 reported `device_access_time_us` values.
This sum includes CU-1, upload, device work, download, and final sampler scoring.
The remaining wall time includes initial generation, protocol work, and any idle gaps.

Raw session records: [2026-11-11-cu4-overlap-a4000.jsonl](2026-11-11-cu4-overlap-a4000.jsonl).

### Throughput results

These rows use the final timing instrumentation. Each row covers 50 jobs on physical GPU 1.

| Nodes | Parameters m,t,weak,pairs | Mode | Wall seconds | Sampler seconds | Outside sampler seconds | Jobs/s |
|---|---|---|---:|---:|---:|---:|
| 4,800 | `12,4,62,1200` | Serial single-salt leases | 6.912209 | 5.033263 | 1.878946 | 7.2336 |
| 4,800 | `12,4,62,1200` | Existing core prefetch | 5.437914 | 5.376243 | 0.061671 | 9.1947 |
| 2,176 | `8,4,62,544` | Serial single-salt leases | 3.659842 | 2.689768 | 0.970074 | 13.6618 |
| 2,176 | `8,4,62,544` | Existing core prefetch | 2.994327 | 2.965065 | 0.029262 | 16.6982 |

The prefetched runs spend 98.9% and 99.0% of wall time inside the serial sampler.
Generation for 50 salts cannot lie serially outside sampling in gaps of only 62 ms and 29 ms.
The source trace and these gaps show that core already hides generation behind sampling.
The task's stop condition applies. This measurement adds no production worker or prepared successor.

Session throughput rises by 27.1% and 22.2% over the controls.
These percentages also include fewer lease exchanges and admission checks.
They do not estimate a new code change or isolate CU-1 savings.
The sampler remains serial for family jobs and still prepares each graph immediately before running it.

The final runs start at 65–68 C and finish at 70–78 C.
Their final SM clocks range from 1,860 to 1,890 MHz.
Each timed run lasts under seven seconds. The longest whole measurement test lasts 15.38 seconds.
That duration includes startup and host checks.
The seven raw records also keep three initial runs without summed sampler timing.
The first 4,800-node control ran at 1,560 MHz and took 10.428 seconds.
The main table uses the later run with summed sampler timing rather than that clock-affected initial control.
The initial prefetched runs measured 8.1408 jobs/s and 16.6837 jobs/s.

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

The final formatting check and both requested clippy checks pass with zero compiler or lint warnings.
The release suite passes 438 top-level tests and ignores 264 tests.
All 45 cluster host, device, and session tests pass with ignored tests enabled on GPU 1.
The CI timing command passes all 24 cluster session tests. Its longest test takes 4.346 seconds.

The first release suite run failed during the existing headless routing test's shutdown.
`tests/common/lease.rs:218` reported an HTTP/2 body-read error.
The focused reproduction, CI timing run, and fresh full suite pass without changes to that test or transport code.
The shutdown failure's cause remains unresolved. This work changes no production code.

Vale reports no prose errors. Its four remaining readability warnings reflect technical terms and source references in this record.
The record keeps those terms for precision.
