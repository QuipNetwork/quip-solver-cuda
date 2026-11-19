//! Headless cluster streaming checks and opt-in real-device conformance.

use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamOutcome,
};
use quip_solver_cuda::capacity::DeviceLimits;
use quip_solver_cuda::cluster::device::{ClusterDeviceError, ClusterRun};
use quip_solver_cuda::cluster::sampler::{
    cuda_otsuka_identity, ClusterBackend, ClusterWork, OtsukaSampler, HEADLESS_LIMITS,
};
use quip_solver_cuda::cluster::{Model, Motifs, PreparedGraph, Weights};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Longest wait for one session phase. It detects a hang, not a slow run:
/// CI's `--ensure-time` limit of 30 seconds bounds each test's total time,
/// and a shared runner can take several seconds longer than an idle card.
const PHASE_LIMIT: std::time::Duration = std::time::Duration::from_secs(15);

#[derive(Clone, Default)]
#[allow(clippy::struct_excessive_bools)] // Independent fault and shape switches.
struct FakeDevice {
    launches: Arc<Mutex<usize>>,
    cancel: Option<CancelToken>,
    fail: bool,
    short: bool,
    cancel_after: usize,
    memory_shortfall: bool,
    yield_after: Option<Arc<AtomicBool>>,
    /// Local build: reads of different energies in no energy order (read
    /// `r` flips a prefix of the alternating state whose length varies with
    /// `r`), for the row-order test.
    varied: bool,
    /// Local build: wall time per launch, so lease work queues up.
    sleep_ms: u64,
    /// Local build (all7): with the deep lane's probe, the lineage best the
    /// fake reports at every launch (`CU3_TEST_PROBE_E`). `None` never
    /// probes, as a backend without checkpoints.
    probe_energy: Option<i64>,
}

impl FakeDevice {
    /// The fake's reads for one run (energies are rescored by the sampler).
    fn reads(&self, graph: &PreparedGraph, work: &ClusterWork) -> ClusterRun {
        let count = if self.short { 1 } else { work.reads };
        let n = graph.original_graph().num_nodes();
        let varied = self.varied;
        ClusterRun {
            spins: (0..count)
                .map(|read| {
                    let flips = if varied {
                        (read * 7919 + 13) % n.max(1)
                    } else {
                        0
                    };
                    (0..n)
                        .map(|node| {
                            let s: i8 = if (read + node) % 2 == 0 { 1 } else { -1 };
                            if node < flips {
                                -s
                            } else {
                                s
                            }
                        })
                        .collect()
                })
                .collect(),
            energies: vec![123; count], // The sampler must score original coefficients.
        }
    }
}

impl ClusterBackend for FakeDevice {
    fn run(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        if self.memory_shortfall {
            return Err(quip_solver_cuda::cluster::PrepareError::DeviceMemory {
                requested: 1024,
                budget: 512,
            }
            .into());
        }
        work.for_each_launch(stopped, |_, _, _| {
            if self.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.sleep_ms));
            }
            *self.launches.lock().unwrap() += 1;
            if let Some(token) = &self.cancel {
                if *self.launches.lock().unwrap() >= self.cancel_after {
                    token.cancel_through(1);
                }
            }
            if self.fail {
                return Err(ClusterDeviceError::Parameter("scripted device error"));
            }
            Ok(())
        })?;
        if let Some(yielding) = &self.yield_after {
            yielding.store(true, Ordering::SeqCst);
        }
        if cancelled() {
            return Err(ClusterDeviceError::Cancelled);
        }
        Ok(self.reads(&graph, work))
    }

    /// Local build (all7): the probe after every launch; a hit skips the
    /// remaining launches (no sleep, not counted) and returns the reads.
    fn run_probed(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: &dyn Fn(i64) -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        let Some(energy) = self.probe_energy else {
            return self.run(graph, work, stopped, cancelled);
        };
        let hit = AtomicBool::new(false);
        work.for_each_launch(stopped, |_, _, _| {
            if hit.load(Ordering::Relaxed) {
                return Ok(());
            }
            if self.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.sleep_ms));
            }
            *self.launches.lock().unwrap() += 1;
            if probe(energy) {
                hit.store(true, Ordering::Relaxed);
            }
            Ok(())
        })?;
        if cancelled() {
            return Err(ClusterDeviceError::Cancelled);
        }
        Ok(self.reads(&graph, work))
    }
}

#[derive(Clone, Default)]
struct FakeSa {
    calls: Arc<Mutex<usize>>,
    yielding: bool,
    late_yield: Arc<AtomicBool>,
    streams: Arc<Mutex<usize>>,
    batch_size: usize,
}

impl Sampler for FakeSa {
    fn sample(
        &self,
        _graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        *self.calls.lock().unwrap() += 1;
        Ok((0..params.num_reads)
            .map(|_| SamplerResult {
                spins: vec![1, 1],
                energy_milli: -1000,
            })
            .collect())
    }

    fn sample_stream(
        &self,
        mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<quip_solver_core::StreamResult>,
        _cancel: CancelToken,
    ) {
        *self.streams.lock().unwrap() += 1;
        while let Some(first) = jobs.blocking_recv() {
            let mut batch = vec![first];
            for _ in 1..self.batch_size {
                batch.push(
                    jobs.blocking_recv()
                        .expect("SA stream must keep its input open"),
                );
            }
            for job in batch {
                let result = self.sample(&job.graph, &job.params);
                if out
                    .blocking_send(quip_solver_core::StreamResult {
                        job_id: job.job_id,
                        outcome: StreamOutcome::Completed(result),
                        device_access_time_us: 1,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }
    }

    fn stream_width(&self) -> usize {
        4
    }
    fn declared_stream_width() -> u32 {
        0
    }

    fn should_throttle(&self) -> bool {
        self.yielding || self.late_yield.load(Ordering::SeqCst)
    }
}

fn graph() -> IsingGraph {
    IsingGraph::new(vec![0.0; 2], vec![-1.0], vec![(0, 1)])
}

fn prepared(limits: &DeviceLimits) -> PreparedGraph {
    PreparedGraph::new(
        &Model {
            h_milli: &[0, 0],
            edges: &[(0, 1)],
            j_milli: &[-1000],
        },
        &Motifs {
            tiles: &[],
            gauges: &[],
            pairs: &[(0, 1)],
        },
        Weights::FULL,
        limits,
        3,
    )
    .unwrap()
}

fn job() -> StreamJob {
    StreamJob {
        job_id: b"cluster".to_vec(),
        graph: graph(),
        params: SampleParams {
            num_reads: 3,
            num_sweeps: 7,
            ..Default::default()
        },
        watermark: Some(1),
    }
}

#[test]
fn every_read_is_returned_in_order_and_rescored() {
    let device = FakeDevice::default();
    let sampler = OtsukaSampler::with_backends(device, FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
        panic!("expected complete reads")
    };
    assert_eq!(reads.len(), 3);
    assert_eq!(
        reads.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
        [vec![1, -1], vec![-1, 1], vec![1, -1]]
    );
    assert!(reads.iter().all(|r| r.energy_milli == 1000));
}

#[test]
fn cancellation_between_chunks_returns_cancelled_outcome() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device.clone(), FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(&job(), prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert_eq!(*device.launches.lock().unwrap(), 1);
}

#[test]
fn a_device_error_is_a_job_error() {
    let device = FakeDevice {
        fail: true,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device, FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    let StreamOutcome::Completed(Err(error)) = result.outcome else {
        panic!("expected job error")
    };
    assert!(error.to_string().contains("scripted device error"));
}

#[test]
fn incomplete_device_output_is_rejected() {
    let device = FakeDevice {
        short: true,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device, FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Err(error)) if error.is_fatal()));
}

#[test]
fn plain_jobs_use_the_sa_stream() {
    let device = FakeDevice::default();
    let sa = FakeSa::default();
    let sampler = OtsukaSampler::with_backends(device.clone(), sa.clone(), HEADLESS_LIMITS);
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (out, mut results) = tokio::sync::mpsc::channel(1);
    tx.blocking_send(job()).unwrap();
    drop(tx);
    sampler.sample_stream(rx, out, CancelToken::default());
    assert!(matches!(
        results.blocking_recv().unwrap().outcome,
        StreamOutcome::Completed(Ok(_))
    ));
    assert_eq!(*sa.calls.lock().unwrap(), 1);
    assert_eq!(*device.launches.lock().unwrap(), 0);
    assert_eq!(sampler.stream_width(), 4);
}

#[test]
fn governor_yield_is_busy_and_never_opens_the_cluster_device() {
    let device = FakeDevice::default();
    let sa = FakeSa {
        yielding: true,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device.clone(), sa, HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(
        result.outcome,
        StreamOutcome::Completed(Err(SampleError::DeviceBusy))
    ));
    assert_eq!(*device.launches.lock().unwrap(), 0);
}

#[test]
fn job_parameters_are_refusals_and_never_launch() {
    let device = FakeDevice::default();
    let sampler = OtsukaSampler::with_backends(device.clone(), FakeSa::default(), HEADLESS_LIMITS);
    let mut bad_params = vec![];
    for (sweeps, per_beta) in [(3_000_000_000, 1), (7, 3_000_000_000)] {
        bad_params.push(SampleParams {
            num_sweeps: sweeps,
            sweeps_per_beta: per_beta,
            ..job().params
        });
    }
    for range in [
        (f64::NAN, 1.0),
        (0.0, 1.0),
        (-1.0, 2.0),
        (0.1, f64::INFINITY),
        (0.1, f64::MAX),
        (f64::MIN_POSITIVE, 1.0),
    ] {
        bad_params.push(SampleParams {
            beta_range: Some(range),
            ..job().params
        });
    }
    for reads in [0, 257] {
        bad_params.push(SampleParams {
            num_reads: reads,
            ..job().params
        });
    }
    for params in bad_params {
        let mut request = job();
        request.params = params;
        let result = sampler.sample_prepared_for_test(
            &request,
            prepared(&HEADLESS_LIMITS),
            &CancelToken::default(),
        );
        assert!(matches!(
            result.outcome,
            StreamOutcome::Completed(Err(SampleError::Capacity))
        ));
    }
    assert_eq!(*device.launches.lock().unwrap(), 0);
}

#[test]
fn live_memory_shortfall_is_transient() {
    let device = FakeDevice {
        memory_shortfall: true,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device, FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(
        result.outcome,
        StreamOutcome::Completed(Err(SampleError::DeviceBusy))
    ));
}

#[test]
fn governor_yield_after_device_completion_keeps_finished_reads() {
    let late_yield = Arc::new(AtomicBool::new(false));
    let device = FakeDevice {
        yield_after: Some(late_yield.clone()),
        ..Default::default()
    };
    let sa = FakeSa {
        late_yield,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device, sa, HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(
        &job(),
        prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
}

#[test]
fn capabilities_are_headless_and_bounded() {
    assert!(!<OtsukaSampler as Sampler>::generates_locally());
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-otsuka"))
        .args(["--capabilities", "--device", "2147483647"])
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["solverId"], "quip-cuda-otsuka/1");
    assert_eq!(caps["backend"], "cuda");
    assert_eq!(caps["algorithm"], "otsuka");
    assert_eq!(caps["streamWidth"], 0);
    let id = cuda_otsuka_identity(&HEADLESS_LIMITS, usize::MAX);
    assert_eq!(caps["maxNodes"], id.max_nodes);
    assert_eq!(caps["maxEdges"], id.max_edges);
    assert_eq!(caps["maxDegree"], id.limits.max_degree.unwrap());
    for list in ["generatedFamilies", "acceptedFamilies"] {
        assert_eq!(caps[list][0]["familyId"], "quip/twisted-ice-dimer");
        assert_eq!(caps[list][0]["revisions"], serde_json::json!([1]));
        assert!(caps[list][0].get("paramsDigests").is_none());
    }
}

#[test]
fn cancellation_during_final_descent_stops_before_the_next_pass() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        cancel_after: 8,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device.clone(), FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(&job(), prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert_eq!(*device.launches.lock().unwrap(), 8);
}

#[test]
fn a_device_fault_remains_visible_when_cancellation_races_it() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        fail: true,
        ..Default::default()
    };
    let sampler = OtsukaSampler::with_backends(device, FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(&job(), prepared(&HEADLESS_LIMITS), &cancel);
    let StreamOutcome::Completed(Err(error)) = result.outcome else {
        panic!("device fault was suppressed")
    };
    assert!(error.is_fatal());
}

#[test]
fn an_abandoned_job_never_reaches_the_device() {
    let cancel = CancelToken::default();
    cancel.cancel_through(1);
    let device = FakeDevice::default();
    let sampler = OtsukaSampler::with_backends(device.clone(), FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(&job(), prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert_eq!(*device.launches.lock().unwrap(), 0);
}

#[test]
fn a_job_without_a_watermark_keeps_all_its_reads() {
    let cancel = CancelToken::default();
    cancel.cancel_through(u64::MAX);
    let mut request = job();
    request.watermark = None;
    let sampler =
        OtsukaSampler::with_backends(FakeDevice::default(), FakeSa::default(), HEADLESS_LIMITS);
    let result = sampler.sample_prepared_for_test(&request, prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
}

#[test]
fn linear_ladder_and_sweep_offsets_reach_each_launch() {
    let params = SampleParams {
        num_sweeps: 7,
        sweeps_per_beta: 2,
        beta_range: Some((0.5, 1.5)),
        schedule: quip_solver_core::BetaSchedule::Linear,
        ..Default::default()
    };
    let work = ClusterWork::new(&graph(), &params).unwrap();
    let mut sweeps = Vec::new();
    let mut descent = 0;
    work.for_each_launch(&|| false, |offset, beta, greedy| {
        if greedy {
            descent += 1;
        } else {
            sweeps.push((offset, beta));
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(
        sweeps,
        [(0, 0.5), (1, 0.5), (2, 1.0), (3, 1.0), (4, 1.5), (5, 1.5)]
    );
    assert_eq!(descent, 256);
}

#[test]
fn a_beta_rung_does_not_exceed_a_positive_job_sweep_budget() {
    let params = SampleParams {
        num_sweeps: 3,
        sweeps_per_beta: 100,
        ..Default::default()
    };
    let work = ClusterWork::new(&graph(), &params).unwrap();
    let mut sweeps = 0;
    work.for_each_launch(&|| false, |_, _, greedy| {
        if !greedy {
            sweeps += 1;
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(sweeps, 3);
}

#[test]
fn declared_shapes_fit_cu1_and_device_budgets() {
    use quip_solver_cuda::capacity::{cluster_footprint, ClusterShape};
    for shared in [48 << 10, 64 << 10, 100 << 10, 228 << 10] {
        let limits = DeviceLimits {
            shared_bytes_per_block_optin: shared,
            ..HEADLESS_LIMITS
        };
        let id = cuda_otsuka_identity(&limits, usize::MAX);
        let n = id.max_nodes as usize;
        let degree = id.limits.max_degree.unwrap() as usize;
        assert!(n * 9 + 20 <= shared);
        assert!(u64::from(id.max_edges) * 999 <= i32::MAX.unsigned_abs().into());
        let shape = ClusterShape {
            nodes: n,
            tiles: n / 2,
            links: n / 2,
            weak_width: degree,
            all_width: degree,
            colors: n,
        };
        assert!(cluster_footprint(shape, 256).total() <= limits.usable_bytes());
        assert!(cuda_otsuka_identity(&limits, 17).max_nodes <= 17);
    }
    // A complete weak model at the degree boundary must pass actual CU-1 admission.
    let n = 129;
    let edges: Vec<_> = (0..n)
        .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
        .collect();
    PreparedGraph::new(
        &Model {
            h_milli: &vec![0; n as usize],
            edges: &edges,
            j_milli: &vec![999; edges.len()],
        },
        &Motifs {
            tiles: &[],
            gauges: &[],
            pairs: &[],
        },
        Weights::FULL,
        &HEADLESS_LIMITS,
        256,
    )
    .unwrap();
}

#[allow(dead_code)] // The family tests use only part of the shared coordinator.
#[path = "common/lease.rs"]
mod family_session;

// The SDK constructs WarmStreamJob and FamilyContext only inside its session.
// Use the test executable as a headless miner to exercise that exact path.
#[test]
#[allow(clippy::too_many_lines)] // One knob per local-build feature.
fn headless_miner_entry() {
    let Ok(socket) = std::env::var("CU3_TEST_COORDINATOR") else {
        return;
    };
    // Local build: CU3_TEST_VARIED_READS gives the fake lanes reads of
    // different energies in no energy order.
    // CU3_TEST_LANE_SLEEP_MS slows every fake launch.
    let device = FakeDevice {
        varied: std::env::var("CU3_TEST_VARIED_READS").is_ok(),
        probe_energy: std::env::var("CU3_TEST_PROBE_E")
            .ok()
            .and_then(|v| v.parse().ok()),
        sleep_ms: std::env::var("CU3_TEST_LANE_SLEEP_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        ..FakeDevice::default()
    };
    let sa = FakeSa {
        batch_size: 2,
        ..Default::default()
    };
    // Local build: CU3_TEST_EXTRA_LANES runs the QUIP_OTSUKA_CONCURRENCY loop.
    let extra = std::env::var("CU3_TEST_EXTRA_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let sampler = if extra > 0 {
        OtsukaSampler::with_lanes(
            device.clone(),
            (0..extra).map(|_| device.clone()).collect(),
            sa.clone(),
            HEADLESS_LIMITS,
        )
    } else {
        OtsukaSampler::with_backends(device.clone(), sa.clone(), HEADLESS_LIMITS)
    };
    // Local build: CU3_TEST_STAGE0_KEEP turns on the stage-0 screen.
    let sampler = match std::env::var("CU3_TEST_STAGE0_KEEP")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(keep) => sampler.with_stage0_keep(keep),
        None => sampler,
    };
    // Local build: CU3_TEST_STAGE0_MODEL picks the stage-0 model.
    let sampler = match std::env::var("CU3_TEST_STAGE0_MODEL")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
    {
        Some(model) => sampler.with_stage0_model(model),
        None => sampler,
    };
    // Local build: CU3_TEST_STAGE0_PRE sets model 3's prefilter fraction.
    let pre = std::env::var("CU3_TEST_STAGE0_PRE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let sampler = if let Some(pre) = pre {
        sampler.with_stage0_pre(pre)
    } else {
        sampler
    };
    // Local build: CU3_TEST_STAGE1="frac,keep[,reads]" turns on stage 1.
    let sampler = match std::env::var("CU3_TEST_STAGE1").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    }) {
        Some(v) => sampler.with_stage1(quip_solver_cuda::cluster::stage1::Stage1Config {
            frac: v[0],
            keep: v[1],
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            reads: v.get(2).map_or(0, |r| *r as usize),
            submit_q: 0.5,
        }),
        None => sampler,
    };
    // Local build: CU3_TEST_POLISH="level,threads,topk,sweeps" turns on the CPU polish.
    let sampler = match std::env::var("CU3_TEST_POLISH").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<usize>().unwrap())
            .collect::<Vec<_>>()
    }) {
        Some(v) => sampler.with_polish(quip_solver_cuda::cluster::polish::PolishConfig {
            level: u8::try_from(v[0]).unwrap(),
            threads: v[1],
            topk: v[2],
            sweeps: v[3],
            queue: 64,
            ..Default::default()
        }),
        None => sampler,
    };
    // Local build (all5): CU3_TEST_STAGE2="q,k,max_active" turns on stage 2.
    let sampler = match std::env::var("CU3_TEST_STAGE2").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    }) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(v) => sampler.with_stage2(quip_solver_cuda::cluster::stage2::Stage2Config {
            q: v[0],
            k: v[1] as usize,
            guard_milli: 1250,
            max_active: v[2] as usize,
            release_milli: 500,
        }),
        None => sampler,
    };
    // Local build (all6): CU3_TEST_POOL="capacity[,max_age_ms]" turns on the
    // best-first pool; CU3_TEST_FAST_GEN screens with the cached generator.
    let sampler = match std::env::var("CU3_TEST_POOL").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<u64>().unwrap())
            .collect::<Vec<_>>()
    }) {
        Some(v) => sampler.with_pool(quip_solver_cuda::cluster::pool::PoolConfig {
            capacity: usize::try_from(v[0]).unwrap(),
            max_age: Duration::from_millis(v.get(1).copied().unwrap_or(120_000)),
        }),
        None => sampler,
    };
    let sampler = if std::env::var("CU3_TEST_FAST_GEN").is_ok() {
        sampler.with_fast_gen()
    } else {
        sampler
    };
    // Local build (all7): CU3_TEST_DEEP="lanes,mult,q,start_s,record[,after_stage2]"
    // turns on the deep lane (guard 1250, release 500).
    let sampler = match std::env::var("CU3_TEST_DEEP").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    }) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(v) => sampler.with_deep(quip_solver_cuda::cluster::deep::DeepConfig {
            lanes: v[0] as usize,
            mult: v[1] as usize,
            q: v[2],
            start: Duration::from_secs_f64(v[3]),
            record: v[4] != 0.0,
            guard_milli: 1250,
            release_milli: 500,
            after_stage2: v.get(5).is_some_and(|&a| a != 0.0),
            slack_milli: 0,
            log_drops: false,
            // all9: CU3_TEST_DEEP_ADAPTIVE="ramp0_s,ramp1_s,late_s,horizon_s,
            // mult_min,mult_max,gap_far_milli,step_s" (lanes is LMAX).
            adaptive: std::env::var("CU3_TEST_DEEP_ADAPTIVE").ok().map(|a| {
                let a: Vec<f64> = a.split(',').map(|x| x.trim().parse().unwrap()).collect();
                quip_solver_cuda::cluster::deep::Adaptive {
                    ramp0: Duration::from_secs_f64(a[0]),
                    ramp1: Duration::from_secs_f64(a[1]),
                    late: Duration::from_secs_f64(a[2]),
                    horizon: Duration::from_secs_f64(a[3]),
                    mult_min: a[4] as usize,
                    mult_max: a[5] as usize,
                    gap_far_milli: a[6] as i64,
                    step: Duration::from_secs_f64(a[7]),
                }
            }),
        }),
        None => sampler,
    };
    // Local build (all12): CU3_TEST_EASE="t0_s,gap_milli,lanes,end_s,post_s,
    // release_milli,safe_s,mult_max,max_runs" turns on the ease booster
    // (with CU3_TEST_DEEP).
    let sampler = match std::env::var("CU3_TEST_EASE").ok().map(|v| {
        v.split(',')
            .map(|x| x.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    }) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(v) => sampler.with_ease(quip_solver_cuda::cluster::ease::EaseConfig {
            t0: Duration::from_secs_f64(v[0]),
            gap_milli: v[1] as i64,
            lanes: v[2] as usize,
            end: Duration::from_secs_f64(v[3]),
            post: Duration::from_secs_f64(v[4]),
            release_milli: v[5] as i64,
            safe: Duration::from_secs_f64(v[6]),
            mult_max: v[7] as usize,
            max_runs: v[8] as u64,
        }),
        None => sampler,
    };
    let lease_only = std::env::var("CU3_TEST_LEASE_ONLY").is_ok();
    let args = quip_solver_core::CommonArgs {
        quip_coordinator: Some(socket),
        miner_id: Some("headless-cluster".into()),
        capabilities: false,
        solve: false,
        check: false,
        log_level: "error".into(),
        sweeps_per_beta: None,
    };
    let exit = quip_solver_core::run_code(
        cuda_otsuka_identity(&HEADLESS_LIMITS, usize::MAX),
        &args,
        || Ok(sampler),
    );
    assert_eq!(exit as u8, 0);
    // Local build: the parent reads the launch count from stderr.
    eprintln!("fake-launches {}", *device.launches.lock().unwrap());
    if lease_only {
        return;
    }
    assert_eq!(
        *sa.streams.lock().unwrap(),
        1,
        "plain jobs must share one persistent stream"
    );
    assert_eq!(*sa.calls.lock().unwrap(), 2);
    assert!(
        *device.launches.lock().unwrap() > 0,
        "validated family must reach the cluster backend"
    );
}

#[tokio::test]
async fn warm_routing_keeps_sa_width_and_refuses_only_the_bad_family_job() {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, ising_problem, CoefficientEncoding, EdgeList, IsingProblem, Job, JobKind,
    };
    use quip_solver_core::quip_protocol::wire::encode_i32_le;
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CUDA_VISIBLE_DEVICES", "");
        command
    })
    .await;
    // The initial credits resolve the live width and permit both jobs before either completes.
    assert!(session.log.credits[0] >= 4);
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
        .await;
    for id in [b"plain-1", b"plain-2"] {
        session
            .send(coord_msg::Msg::Job(Job {
                job_id: id.to_vec(),
                kind: JobKind::IsingSample as i32,
                generation: 1,
                ising: Some(IsingProblem {
                    graph: Some(ising_problem::Graph::Edges(EdgeList {
                        u: vec![0],
                        v: vec![1],
                    })),
                    encoding: CoefficientEncoding::I32 as i32,
                    scale: 1000,
                    h: encode_i32_le(&[0, 0]),
                    j: encode_i32_le(&[-1000]),
                    num_reads: 3,
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .await;
    }
    session
        .until("both plain jobs", PHASE_LIMIT, |log| {
            log.results.len() == 2 && log.refunds() == 2
        })
        .await;
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, [7; 32]).unwrap();

    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"bad-family",
            3_000_000_000,
            3,
        )))
        .await;
    session
        .until("bad job refusal and refund", PHASE_LIMIT, |log| {
            log.refunds() == 3 || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.log.rejects[0].job_id, b"bad-family");
    assert_eq!(
        session.log.rejects[0].reason,
        quip_solver_core::quip_proto::v1::RejectReason::TooLarge as i32
    );
    assert_eq!(session.log.results.len(), 2);
    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"good-family",
            4,
            3,
        )))
        .await;
    session
        .until(
            "next family completes in same session",
            PHASE_LIMIT,
            |log| log.results.len() == 3 && log.refunds() == 4,
        )
        .await;
    assert_eq!(session.log.results[2].job_id, b"good-family");
    assert!(session.log.fatal.is_none());
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.shutdown(2000).await, 0);
}

/// Local build: the concurrent family loop (three lanes) keeps the plain SA
/// stream, refuses only the bad family job, and completes every good one.
#[tokio::test]
async fn concurrent_lanes_complete_every_family_job() {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, ising_problem, CoefficientEncoding, EdgeList, IsingProblem, Job, JobKind,
    };
    use quip_solver_core::quip_protocol::wire::encode_i32_le;
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CUDA_VISIBLE_DEVICES", "");
        command
    })
    .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
        .await;
    for id in [b"plain-1", b"plain-2"] {
        session
            .send(coord_msg::Msg::Job(Job {
                job_id: id.to_vec(),
                kind: JobKind::IsingSample as i32,
                generation: 1,
                ising: Some(IsingProblem {
                    graph: Some(ising_problem::Graph::Edges(EdgeList {
                        u: vec![0],
                        v: vec![1],
                    })),
                    encoding: CoefficientEncoding::I32 as i32,
                    scale: 1000,
                    h: encode_i32_le(&[0, 0]),
                    j: encode_i32_le(&[-1000]),
                    num_reads: 3,
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .await;
    }
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, [7; 32]).unwrap();
    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"bad-family",
            3_000_000_000,
            3,
        )))
        .await;
    for id in [b"fam-1", b"fam-2", b"fam-3", b"fam-4"] {
        session
            .send(coord_msg::Msg::Job(canonical::job(
                &instance, &params, id, 4, 3,
            )))
            .await;
    }
    session
        .until("every job settles", PHASE_LIMIT, |log| {
            log.results.len() + log.rejects.len() == 7 || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.log.rejects[0].job_id, b"bad-family");
    let mut done: Vec<Vec<u8>> = session
        .log
        .results
        .iter()
        .map(|r| r.job_id.clone())
        .collect();
    done.sort();
    assert_eq!(
        done,
        [
            &b"fam-1"[..],
            b"fam-2",
            b"fam-3",
            b"fam-4",
            b"plain-1",
            b"plain-2"
        ]
        .map(<[u8]>::to_vec)
        .to_vec()
    );
    assert_eq!(session.shutdown(2000).await, 0);
}

/// Local build: run one family lease of `salts` salts through the session
/// (core's default expander, the path a dimer lease takes) with three fake
/// lanes and the given stage-0 keep. Returns (salts done, results).
async fn family_lease(keep: Option<&str>, salts: u64) -> (u64, usize) {
    let env: Vec<(&str, &str)> = keep
        .map(|k| ("CU3_TEST_STAGE0_KEEP", k))
        .into_iter()
        .collect();
    family_lease_env(&env, salts).await
}

/// [`family_lease`] with an explicit stage-0 model (`CU3_TEST_STAGE0_MODEL`).
async fn family_lease_model(keep: Option<&str>, model: Option<&str>, salts: u64) -> (u64, usize) {
    let mut env: Vec<(&str, &str)> = Vec::new();
    if let Some(keep) = keep {
        env.push(("CU3_TEST_STAGE0_KEEP", keep));
    }
    if let Some(model) = model {
        env.push(("CU3_TEST_STAGE0_MODEL", model));
    }
    family_lease_env(&env, salts).await
}

/// Local build: [`family_lease`] with arbitrary `CU3_TEST_*` knobs.
async fn family_lease_env(env: &[(&str, &str)], salts: u64) -> (u64, usize) {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind, Topology,
    };
    let (base, params) = canonical::fixture();
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CU3_TEST_LEASE_ONLY", "1")
            .env("CUDA_VISIBLE_DEVICES", "");
        command.envs(env.iter().copied());
        command
    })
    .await;
    let hash = [0x5e; 32];
    let nodes = base.node_ids().to_vec();
    let edges: Vec<(u32, u32)> = base
        .edges()
        .iter()
        .map(|&(u, v)| (nodes[u as usize], nodes[v as usize]))
        .collect();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: nodes.clone(),
            edges: Some(EdgeList {
                u: edges.iter().map(|e| e.0).collect(),
                v: edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:fam".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(&params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    session
        .until("lease done", Duration::from_mins(2), |log| {
            !log.lease_done.is_empty() || !log.rejects.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert!(session.log.rejects.is_empty(), "{:?}", session.log.rejects);
    let done = session.log.lease_done[0].salts_done;
    let results = session.log.results.len();
    assert_eq!(session.shutdown(2000).await, 0);
    (done, results)
}

/// Local build: [`family_lease_env`] that also returns, per result, the
/// salt nonce and every reported solution energy after checking each one
/// exactly against the canonical instance of that nonce.
async fn family_lease_checked(
    env: &[(&str, &str)],
    salts: u64,
) -> (u64, Vec<([u8; 32], Vec<i64>)>) {
    let (done, results, _) = family_lease_targeted(env, salts, i64::MAX).await;
    (done, results)
}

/// Local build (all5): [`family_lease_checked`] under a session target of
/// `max_energy_milli`, also returning the fake lanes' launch count.
#[allow(clippy::too_many_lines)] // One scripted session, start to shutdown.
async fn family_lease_targeted(
    env: &[(&str, &str)],
    salts: u64,
    max_energy_milli: i64,
) -> (u64, Vec<([u8; 32], Vec<i64>)>, usize) {
    let (done, out, launches, _) = family_lease_logged(env, salts, max_energy_milli).await;
    (done, out, launches)
}

/// Local build (all6): [`family_lease_targeted`] that also returns the
/// miner's stderr.
#[allow(clippy::too_many_lines)] // One scripted session, start to shutdown.
async fn family_lease_logged(
    env: &[(&str, &str)],
    salts: u64,
    max_energy_milli: i64,
) -> (u64, Vec<([u8; 32], Vec<i64>)>, usize, String) {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind, Topology,
    };
    use quip_solver_core::quip_protocol::wire::decode_spins_packed;
    let (base, params) = canonical::fixture();
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CU3_TEST_LEASE_ONLY", "1")
            .env("CUDA_VISIBLE_DEVICES", "");
        command.envs(env.iter().copied());
        command
    })
    .await;
    let hash = [0x5e; 32];
    let nodes = base.node_ids().to_vec();
    let edges: Vec<(u32, u32)> = base
        .edges()
        .iter()
        .map(|&(u, v)| (nodes[u as usize], nodes[v as usize]))
        .collect();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: nodes.clone(),
            edges: Some(EdgeList {
                u: edges.iter().map(|e| e.0).collect(),
                v: edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(max_energy_milli, 4)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:fam".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(&params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    session
        .until("lease done", Duration::from_mins(2), |log| {
            !log.lease_done.is_empty() || !log.rejects.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert!(session.log.rejects.is_empty(), "{:?}", session.log.rejects);
    let done = session.log.lease_done[0].salts_done;
    let mut out = Vec::new();
    for result in &session.log.results {
        let nonce: [u8; 32] = result.nonce.as_slice().try_into().unwrap();
        let instance = canonical::instance(&base, &params, nonce).expect("canonical instance");
        let n = instance.node_ids().len();
        let mut energies = Vec::new();
        for solution in &result.solutions {
            let spins = decode_spins_packed(&solution.spins, n).unwrap();
            let mut e = instance.offset_milli();
            for (k, &(u, v)) in instance.edges().iter().enumerate() {
                e += i64::from(instance.j()[k])
                    * i64::from(spins[u as usize])
                    * i64::from(spins[v as usize]);
            }
            for (i, &h) in instance.h().iter().enumerate() {
                e += i64::from(h) * i64::from(spins[i]);
            }
            assert_eq!(
                e, solution.energy_milli,
                "reported energy is the exact energy of the spins"
            );
            energies.push(e);
        }
        out.push((nonce, energies));
    }
    assert_eq!(session.shutdown(2000).await, 0);
    let launches: usize = session
        .stderr()
        .lines()
        .find_map(|l| l.strip_prefix("fake-launches "))
        .and_then(|n| n.trim().parse().ok())
        .expect("the miner reports its launch count");
    out.sort_by_key(|r| r.0);
    (done, out, launches, session.stderr())
}

/// Local build: with the CPU polish on, every lease salt still counts as done,
/// the same salts report, every reported energy is the exact energy of its
/// spins, and no salt's best gets worse (the fake device's alternating states
/// are far from a minimum, so polish lowers them). Descent only (level 1) and
/// the cluster continuation (level 2), each through the pool.
/// CPU only (fake lanes, `CUDA_VISIBLE_DEVICES` empty).
#[tokio::test]
async fn polish_keeps_lease_accounting_and_reports_exact_lower_energies() {
    let salts = 24;
    let (done_off, off) = family_lease_checked(&[], salts).await;
    let (done_l1, l1) = family_lease_checked(&[("CU3_TEST_POLISH", "1,2,4,0")], salts).await;
    let (done_l2, l2) = family_lease_checked(&[("CU3_TEST_POLISH", "2,2,2,50")], salts).await;
    for done in [done_off, done_l1, done_l2] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    assert!(!off.is_empty());
    let best = |r: &[i64]| r.iter().copied().min().unwrap();
    for polished in [&l1, &l2] {
        assert_eq!(polished.len(), off.len(), "the same salts report");
        let mut lower = 0;
        for (a, b) in off.iter().zip(polished.iter()) {
            assert_eq!(a.0, b.0);
            assert!(best(&b.1) <= best(&a.1), "polish never raises the best");
            lower += usize::from(best(&b.1) < best(&a.1));
        }
        assert!(
            lower * 2 >= off.len(),
            "polish lowered {lower} of {} salts",
            off.len()
        );
    }
    #[allow(clippy::cast_possible_wrap)]
    let mean =
        |r: &[([u8; 32], Vec<i64>)]| r.iter().map(|x| best(&x.1)).sum::<i64>() / r.len() as i64;
    eprintln!(
        "polish lease: off {} l1 {} l2 {}",
        mean(&off),
        mean(&l1),
        mean(&l2)
    );
    assert!(mean(&l2) <= mean(&l1));
}

/// Local build (`budfox/otsuka-all3`): stage 1 and the CPU polish compose on
/// lease salts. Every salt counts as done, every reported energy is the exact
/// energy of its spins (checked by [`family_lease_checked`]), stage 1 alone
/// reports only reads the unscreened run reports for that salt (the fake
/// lane's short and full reads are equal, so none is swapped in, and a
/// screened salt keeps a subset), and the polish, which sees the reads stage 1
/// returns, never raises a reported salt's best.
/// CPU only (fake lanes, `CUDA_VISIBLE_DEVICES` empty).
#[tokio::test]
async fn stage1_then_polish_report_exact_energies_and_count_every_salt() {
    let salts = 48;
    let (done_off, off) = family_lease_checked(&[], salts).await;
    let (done_s1, s1) = family_lease_checked(&[("CU3_TEST_STAGE1", "0.5,0.5")], salts).await;
    let (done_both, both) = family_lease_checked(
        &[
            ("CU3_TEST_STAGE1", "0.5,0.5"),
            ("CU3_TEST_POLISH", "2,2,2,50"),
        ],
        salts,
    )
    .await;
    for done in [done_off, done_s1, done_both] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    let off: std::collections::HashMap<[u8; 32], Vec<i64>> = off.into_iter().collect();
    assert_eq!(off.len(), usize::try_from(salts).unwrap());
    for (label, run) in [("stage 1", &s1), ("stage 1 + polish", &both)] {
        assert!(
            !run.is_empty() && run.len() < off.len(),
            "{label}: {} of {} salts reported",
            run.len(),
            off.len()
        );
    }
    for (nonce, energies) in &s1 {
        let mut stock = off[nonce].clone();
        for e in energies {
            let at = stock
                .iter()
                .position(|x| x == e)
                .expect("a read the stock run reports");
            stock.swap_remove(at);
        }
    }
    for (nonce, energies) in &both {
        let best = energies.iter().copied().min().unwrap();
        assert!(
            best <= off[nonce].iter().copied().min().unwrap(),
            "polish never raises the best"
        );
    }
}

/// Local build: every family lease result leaves the sampler with its rows in
/// ascending energy, so row 0 (what the coordinator submits at
/// `min_solutions = 1`) is the lowest. Fake lanes return reads of different
/// energies in no energy order; checked with the polish off, polish 2, and
/// stage 1 on (its merge and kept short reads). Energies stay exact
/// ([`family_lease_checked`]). CPU only.
#[tokio::test]
async fn result_rows_leave_in_ascending_energy() {
    let salts = 24;
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let runs = [
        ("polish off", vec![varied]),
        ("polish 2", vec![varied, ("CU3_TEST_POLISH", "2,2,2,50")]),
        ("stage 1", vec![varied, ("CU3_TEST_STAGE1", "0.5,0.5")]),
        (
            "stage 1 + polish 2",
            vec![
                varied,
                ("CU3_TEST_STAGE1", "0.5,0.5"),
                ("CU3_TEST_POLISH", "2,2,2,50"),
            ],
        ),
    ];
    for (label, env) in runs {
        let (done, results) = family_lease_checked(&env, salts).await;
        assert_eq!(done, salts, "{label}: every salt counts as done");
        assert!(!results.is_empty(), "{label}: some salt reports");
        let mut distinct = 0;
        for (_, energies) in &results {
            assert!(
                energies.windows(2).all(|w| w[0] <= w[1]),
                "{label}: rows not ascending: {energies:?}"
            );
            assert_eq!(
                energies[0],
                *energies.iter().min().unwrap(),
                "{label}: row 0 is the best"
            );
            distinct += usize::from(energies.first() != energies.last());
        }
        assert!(
            distinct > 0,
            "{label}: the fake reads must differ for the check to bite"
        );
    }
}

/// Local build: `QUIP_OTSUKA_SCREEN_THREADS=1` is core's own lease path:
/// the sampler does not generate locally, and a screened lease reports the
/// same salts with the same energies as with the knob unset. CPU only.
#[tokio::test]
async fn screen_threads_one_is_the_session_path() {
    assert!(
        !<OtsukaSampler as Sampler>::generates_locally(),
        "unset: core's path"
    );
    let salts = 64;
    let keep = ("CU3_TEST_STAGE0_KEEP", "0.25");
    let (done_unset, unset) = family_lease_checked(&[keep], salts).await;
    let (done_one, one) =
        family_lease_checked(&[keep, ("QUIP_OTSUKA_SCREEN_THREADS", "1")], salts).await;
    assert_eq!((done_unset, done_one), (salts, salts));
    assert!(!unset.is_empty());
    assert_eq!(unset, one, "THREADS=1 reports the same salts and energies");
}

/// Local build: `QUIP_OTSUKA_SCREEN_THREADS=4` generates and screens lease
/// salts on four workers. Every salt counts as done exactly once (core's
/// sink refuses a repeat, and `salts_done` equals the lease size), no salt
/// reports twice, every energy is exact for its nonce (offset included), and
/// the screen keeps about the serial path's fraction. CPU only.
#[tokio::test]
async fn screen_threads_four_counts_every_salt_once_and_keeps_the_fraction() {
    let salts = 96;
    let keep = ("CU3_TEST_STAGE0_KEEP", "0.25");
    let (done_serial, serial) = family_lease_checked(&[keep], salts).await;
    let (done_pool, pool) =
        family_lease_checked(&[keep, ("QUIP_OTSUKA_SCREEN_THREADS", "4")], salts).await;
    let (done_all, all) = family_lease_checked(&[("QUIP_OTSUKA_SCREEN_THREADS", "4")], salts).await;
    eprintln!(
        "screen pool lease: serial {done_serial}/{}, pool {done_pool}/{}, pool keep-all {done_all}/{}",
        serial.len(),
        pool.len(),
        all.len()
    );
    for done in [done_serial, done_pool, done_all] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    let unique: std::collections::HashSet<[u8; 32]> = pool.iter().map(|r| r.0).collect();
    assert_eq!(unique.len(), pool.len(), "no salt reports twice");
    assert_eq!(
        all.len(),
        usize::try_from(salts).unwrap(),
        "unscreened: every salt reports once"
    );
    let (a, b) = (serial.len(), pool.len());
    assert!(
        a.abs_diff(b) <= (a / 3).max(6) && b * 2 <= all.len(),
        "pool kept {b}, serial {a} of {salts}"
    );
}

/// Local build: a round change (`Cancel`) stops a local lease promptly.
/// Fake lanes take ~1 ms per launch (about 257 launches per salt at this
/// session's sweeps) and the screen keeps everything, so nearly all of 400
/// salts are still unsolved (queued or unclaimed) when the first result
/// arrives and the round is cancelled. The lease reports a few salts done,
/// and the lanes launch for only the salts already running: queued salts of
/// the cancelled round never reach them.
/// CPU only.
#[tokio::test]
async fn screen_threads_cancel_drops_queued_work() {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, Cancel, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind,
        Topology,
    };
    let salts: u64 = 400;
    let (base, params) = canonical::fixture();
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CU3_TEST_LEASE_ONLY", "1")
            .env("CU3_TEST_LANE_SLEEP_MS", "1")
            .env("QUIP_OTSUKA_SCREEN_THREADS", "4")
            .env("CUDA_VISIBLE_DEVICES", "");
        command
    })
    .await;
    let hash = [0x5e; 32];
    let nodes = base.node_ids().to_vec();
    let edges: Vec<(u32, u32)> = base
        .edges()
        .iter()
        .map(|&(u, v)| (nodes[u as usize], nodes[v as usize]))
        .collect();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: nodes.clone(),
            edges: Some(EdgeList {
                u: edges.iter().map(|e| e.0).collect(),
                v: edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:fam".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(&params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    let started = std::time::Instant::now();
    session
        .until("first result", Duration::from_mins(2), |log| {
            !log.results.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    eprintln!("first result after {:?}", started.elapsed());
    session
        .send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
        .await;
    session
        .until("lease done", Duration::from_mins(2), |log| {
            !log.lease_done.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    let done = session.log.lease_done[0].salts_done;
    assert_eq!(session.shutdown(2000).await, 0);
    let launches: usize = session
        .stderr()
        .lines()
        .find_map(|l| l.strip_prefix("fake-launches "))
        .and_then(|n| n.trim().parse().ok())
        .expect("the miner reports its launch count");
    // Launches one solved salt takes: the lane's ladder plus one descent.
    let per_salt = 257;
    let full = usize::try_from(salts).unwrap() * per_salt;
    eprintln!("cancelled local lease: {done} of {salts} salts done, {launches} of {full} launches");
    assert!(
        done < salts / 10,
        "the lease stopped early: {done} of {salts}"
    );
    // Three lanes, at most a few salts each between the first result and
    // the stop; the queue (two salts per lane) is dropped unsolved.
    assert!(
        launches < 12 * per_salt,
        "queued salts reached the lanes: {launches} of {full}"
    );
}

/// Local build (all5): stage 2 on the local path. Under an unreachable
/// target it holds deep first runs and re-anneals them (more launches), yet
/// every salt still counts as done once and reports exactly once, with
/// exact energies and ascending rows, and no salt's best is worse than
/// without stage 2. Under a target that accepts everything it never holds
/// (the guard), so the launches match stage 2 off. With stage 1 on as well
/// (reruns skip stage 1), no salt reports twice. CPU only.
#[tokio::test]
async fn stage2_reruns_count_every_salt_once_and_never_hold_a_winner() {
    let salts = 96;
    let threads = ("QUIP_OTSUKA_SCREEN_THREADS", "4");
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let on = ("CU3_TEST_STAGE2", "0.5,2,3");
    let low = i64::MIN / 2;
    let (done_off, off, launches_off) = family_lease_targeted(&[threads, varied], salts, low).await;
    let (done_on, held, launches_on) =
        family_lease_targeted(&[threads, varied, on], salts, low).await;
    let (done_win, open, launches_win) =
        family_lease_targeted(&[threads, varied, on], salts, i64::MAX).await;
    let (done_s1, s1, _) = family_lease_targeted(
        &[threads, varied, on, ("CU3_TEST_STAGE1", "0.5,0.5")],
        salts,
        low,
    )
    .await;
    eprintln!(
        "stage 2 lease: launches off {launches_off}, held {launches_on}, open target {launches_win}"
    );
    for done in [done_off, done_on, done_win, done_s1] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    let n = usize::try_from(salts).unwrap();
    for (label, run) in [("off", &off), ("held", &held), ("open", &open)] {
        assert_eq!(run.len(), n, "{label}: every salt reports exactly once");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), n, "{label}: no salt reports twice");
    }
    let unique: std::collections::HashSet<[u8; 32]> = s1.iter().map(|r| r.0).collect();
    assert_eq!(unique.len(), s1.len(), "stage 1 + 2: no salt reports twice");
    assert!(launches_on > launches_off, "stage 2 re-annealed some salts");
    assert_eq!(
        launches_win, launches_off,
        "a reachable target holds nothing"
    );
    for ((nonce, a), (nonce_b, b)) in off.iter().zip(&held) {
        assert_eq!(nonce, nonce_b);
        assert!(b.windows(2).all(|w| w[0] <= w[1]), "rows ascending");
        assert!(b[0] <= a[0], "stage 2 never raises a salt's best");
    }
}

/// Local build (all7): the deep lane on the local path. Under an unreachable
/// target it holds first runs in the trigger and re-anneals them long on the
/// deep lane (more launches), yet every salt counts as done once and
/// reports exactly once, with exact energies, ascending rows, and no salt's
/// best worse than with the deep lane off. Under a target that accepts
/// everything it never holds (the guard): launches match off. With the
/// round record on, with stage 1 and stage 2 on as well, and with the deep
/// lane after stage 2, no salt reports twice. CPU only.
#[tokio::test]
async fn deep_lane_counts_every_salt_once_and_never_holds_a_winner() {
    let salts = 96;
    let threads = ("QUIP_OTSUKA_SCREEN_THREADS", "4");
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let deep = ("CU3_TEST_DEEP", "1,4,0.5,0,0");
    let low = i64::MIN / 2;
    let (done_off, off, launches_off) = family_lease_targeted(&[threads, varied], salts, low).await;
    let (done_on, held, launches_on) =
        family_lease_targeted(&[threads, varied, deep], salts, low).await;
    let (done_win, open, launches_win) =
        family_lease_targeted(&[threads, varied, deep], salts, i64::MAX).await;
    let (done_rec, rec, launches_rec) = family_lease_targeted(
        &[threads, varied, ("CU3_TEST_DEEP", "1,4,0.5,0,1")],
        salts,
        low,
    )
    .await;
    let (done_all, all, _) = family_lease_targeted(
        &[
            threads,
            varied,
            deep,
            ("CU3_TEST_STAGE1", "0.5,0.5"),
            ("CU3_TEST_STAGE2", "0.5,2,3"),
        ],
        salts,
        low,
    )
    .await;
    eprintln!(
        "deep lane lease: launches off {launches_off}, held {launches_on}, record {launches_rec}, open target {launches_win}"
    );
    for done in [done_off, done_on, done_win, done_rec, done_all] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    let n = usize::try_from(salts).unwrap();
    for (label, run) in [
        ("off", &off),
        ("held", &held),
        ("open", &open),
        ("record", &rec),
    ] {
        assert_eq!(run.len(), n, "{label}: every salt reports exactly once");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), n, "{label}: no salt reports twice");
    }
    let unique: std::collections::HashSet<[u8; 32]> = all.iter().map(|r| r.0).collect();
    assert_eq!(
        unique.len(),
        all.len(),
        "deep + stage 1 + stage 2: no salt reports twice"
    );
    let (done_after, after, launches_after) = family_lease_targeted(
        &[
            threads,
            varied,
            ("CU3_TEST_DEEP", "1,4,0.5,0,0,1"),
            ("CU3_TEST_STAGE2", "0.5,2,3"),
        ],
        salts,
        low,
    )
    .await;
    assert_eq!(
        done_after, salts,
        "deep after stage 2: every salt counts as done"
    );
    assert_eq!(
        after.len(),
        n,
        "deep after stage 2: every salt reports exactly once"
    );
    let unique: std::collections::HashSet<[u8; 32]> = after.iter().map(|r| r.0).collect();
    assert_eq!(unique.len(), n, "deep after stage 2: no salt reports twice");
    assert!(
        launches_after > launches_off,
        "deep after stage 2: held and re-annealed"
    );
    assert!(
        launches_on > launches_off,
        "the deep lane re-annealed some salts"
    );
    // Whether a later first run beats the warm-up's record is up to the
    // salts (the unit tests pin the record rule); here only the accounting.
    assert!(
        launches_rec >= launches_off,
        "the record trigger never skips work"
    );
    assert_eq!(
        launches_win, launches_off,
        "a reachable target holds nothing"
    );
    for ((nonce, a), (nonce_b, b)) in off.iter().zip(&held) {
        assert_eq!(nonce, nonce_b);
        assert!(b.windows(2).all(|w| w[0] <= w[1]), "rows ascending");
        assert!(b[0] <= a[0], "the deep lane never raises a salt's best");
    }
}

/// Local build (all12): the ease booster on the local path, on the two deep
/// lanes of three, with the deep lane's own holds off (allocation 0). With
/// `T0` beyond the session it holds nothing: launches and reads match the
/// booster off. Holding every round best (open gap, static target, no
/// release) with up to 4 boost runs per instance: more launches, yet every
/// salt counts as done once and reports exactly once, rows ascending, and
/// no salt's best is worse than off. A probe that releases at once, and the
/// booster next to holding deep lanes, stage 1 and stage 2, account every
/// salt once too. CPU only.
#[tokio::test]
async fn ease_boost_counts_every_salt_once_and_late_t0_holds_nothing() {
    let salts = 96;
    let threads = ("QUIP_OTSUKA_SCREEN_THREADS", "4");
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let deep = ("CU3_TEST_DEEP", "2,4,0.5,0,0");
    let no_holds = ("CU3_TEST_DEEP_ADAPTIVE", "3600,3600,3600,3600,3,4,0,0");
    let low = i64::MIN / 2;
    let (done_off, off, launches_off) =
        family_lease_targeted(&[threads, varied, deep, no_holds], salts, low).await;
    let late = ("CU3_TEST_EASE", "3600,9e18,2,3600,90,0,3600,2,4");
    let (done_late, late_run, launches_late) =
        family_lease_targeted(&[threads, varied, deep, no_holds, late], salts, low).await;
    let on = ("CU3_TEST_EASE", "0,9e18,2,3600,90,0,3600,2,4");
    let (done_on, held, launches_on) =
        family_lease_targeted(&[threads, varied, deep, no_holds, on], salts, low).await;
    let crossing = ("CU3_TEST_PROBE_E", "-4700000000000000000");
    let (done_cross, cross, launches_cross) =
        family_lease_targeted(&[threads, varied, deep, no_holds, on, crossing], salts, low).await;
    let (done_all, all, _) = family_lease_targeted(
        &[
            threads,
            varied,
            ("CU3_TEST_DEEP", "1,4,0.5,0,0"),
            on,
            ("CU3_TEST_STAGE1", "0.5,0.5"),
            ("CU3_TEST_STAGE2", "0.5,2,3"),
        ],
        salts,
        low,
    )
    .await;
    eprintln!(
        "ease boost lease: launches off {launches_off}, late {launches_late}, on {launches_on}, crossing {launches_cross}"
    );
    let n = usize::try_from(salts).unwrap();
    for (label, done, run) in [
        ("off", done_off, &off),
        ("late", done_late, &late_run),
        ("on", done_on, &held),
        ("crossing", done_cross, &cross),
    ] {
        assert_eq!(done, salts, "{label}: every salt counts as done");
        assert_eq!(run.len(), n, "{label}: every salt reports exactly once");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), n, "{label}: no salt reports twice");
    }
    // Stage 1 sends promoted reads only: count, and no repeat.
    assert_eq!(done_all, salts, "all: every salt counts as done");
    let unique: std::collections::HashSet<[u8; 32]> = all.iter().map(|r| r.0).collect();
    assert_eq!(
        unique.len(),
        all.len(),
        "ease + deep + stage 1 + stage 2: no salt reports twice"
    );
    assert_eq!(
        launches_late, launches_off,
        "a booster that never holds adds nothing"
    );
    assert_eq!(late_run, off, "a booster that never holds changes no read");
    assert!(
        launches_on > launches_off,
        "the booster re-annealed round bests"
    );
    for ((nonce, a), (nonce_b, b)) in off.iter().zip(&held) {
        assert_eq!(nonce, nonce_b);
        assert!(b.windows(2).all(|w| w[0] <= w[1]), "rows ascending");
        assert!(b[0] <= a[0], "the booster never raises a salt's best");
    }
}

/// Local build (all7): a long run whose probed lineage best reaches the
/// live target is stopped at that probe and delivered at once (far fewer
/// launches than running every long run to its end), with the same
/// accounting: every salt done once, reported once, exact energies. CPU only.
#[tokio::test]
async fn deep_lane_delivers_at_once_when_the_long_run_crosses() {
    let salts = 96;
    let threads = ("QUIP_OTSUKA_SCREEN_THREADS", "4");
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let deep = ("CU3_TEST_DEEP", "1,4,0.5,0,0");
    let low = i64::MIN / 2;
    let (_, _, launches_off) = family_lease_targeted(&[threads, varied], salts, low).await;
    let (done_full, full, launches_full) =
        family_lease_targeted(&[threads, varied, deep], salts, low).await;
    // Far below the target plus the release margin: every probe crosses.
    let crossing = ("CU3_TEST_PROBE_E", "-4700000000000000000");
    let (done_cross, cross, launches_cross) =
        family_lease_targeted(&[threads, varied, deep, crossing], salts, low).await;
    // A probe that never crosses runs every long run to its end.
    let never = ("CU3_TEST_PROBE_E", "4700000000000000000");
    let (done_never, never_run, launches_never) =
        family_lease_targeted(&[threads, varied, deep, never], salts, low).await;
    eprintln!(
        "deep lane probe: launches off {launches_off}, full {launches_full}, crossing {launches_cross}, never {launches_never}"
    );
    let n = usize::try_from(salts).unwrap();
    for (label, done, run) in [
        ("full", done_full, &full),
        ("crossing", done_cross, &cross),
        ("never", done_never, &never_run),
    ] {
        assert_eq!(done, salts, "{label}: every salt counts as done");
        assert_eq!(run.len(), n, "{label}: every salt reports exactly once");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), n, "{label}: no salt reports twice");
    }
    assert!(launches_full > launches_off);
    assert!(launches_never > launches_off);
    assert!(
        launches_cross < launches_full && launches_cross < launches_never,
        "a crossing long run stops at its first probe"
    );
}

/// Local build (all9): the adaptive deep lane on the local path, three
/// lanes of which two may run long runs. An allocation that stays 0 (RAMP0
/// beyond the session) holds nothing: launches match off. Holding from the
/// start, with the shortest multiple (horizon 0) or the longest, and with
/// stage 1, stage 2 and the pool on as well: every salt counts as done
/// once and reports exactly once, rows ascending, no salt's best worse than
/// off, and the lease still completes (the one ordinary lane, stage 2 and
/// the pool are not starved). CPU only.
#[tokio::test]
async fn deep_adaptive_counts_every_salt_once_and_zero_allocation_holds_nothing() {
    let salts = 96;
    let threads = ("QUIP_OTSUKA_SCREEN_THREADS", "4");
    let varied = ("CU3_TEST_VARIED_READS", "1");
    let deep = ("CU3_TEST_DEEP", "2,4,0.5,0,0");
    let low = i64::MIN / 2;
    let (done_off, off, launches_off) = family_lease_targeted(&[threads, varied], salts, low).await;
    let never = ("CU3_TEST_DEEP_ADAPTIVE", "3600,3600,3600,3600,3,4,0,0");
    let (done_never, never_run, launches_never) =
        family_lease_targeted(&[threads, varied, deep, never], salts, low).await;
    let short = ("CU3_TEST_DEEP_ADAPTIVE", "0,0,3600,0,2,4,0,0");
    let (done_short, short_run, launches_short) =
        family_lease_targeted(&[threads, varied, deep, short], salts, low).await;
    let long = ("CU3_TEST_DEEP_ADAPTIVE", "0,0,3600,3600,2,4,0,0");
    let (done_long, long_run, launches_long) =
        family_lease_targeted(&[threads, varied, deep, long], salts, low).await;
    let (done_all, all, _) = family_lease_targeted(
        &[
            threads,
            varied,
            deep,
            long,
            ("CU3_TEST_STAGE1", "0.5,0.5"),
            ("CU3_TEST_STAGE2", "0.5,2,3"),
            ("CU3_TEST_STAGE0_KEEP", "0.5"),
            ("CU3_TEST_POOL", "3"),
            ("CU3_TEST_LANE_SLEEP_MS", "1"),
        ],
        salts,
        low,
    )
    .await;
    eprintln!(
        "deep adaptive lease: launches off {launches_off}, never {launches_never}, short {launches_short}, long {launches_long}"
    );
    let n = usize::try_from(salts).unwrap();
    for (label, done, run) in [
        ("off", done_off, &off),
        ("never", done_never, &never_run),
        ("short", done_short, &short_run),
        ("long", done_long, &long_run),
    ] {
        assert_eq!(done, salts, "{label}: every salt counts as done");
        assert_eq!(run.len(), n, "{label}: every salt reports exactly once");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), n, "{label}: no salt reports twice");
    }
    assert_eq!(
        done_all, salts,
        "with stage 1, stage 2 and the pool: every salt done"
    );
    let unique: std::collections::HashSet<[u8; 32]> = all.iter().map(|r| r.0).collect();
    assert_eq!(
        unique.len(),
        all.len(),
        "with stage 1, stage 2 and the pool: no salt reports twice"
    );
    assert_eq!(launches_never, launches_off, "allocation 0 holds nothing");
    assert!(
        launches_short > launches_off,
        "the adaptive lane re-annealed some salts"
    );
    assert!(
        launches_long > launches_off,
        "the adaptive lane re-annealed some salts"
    );
    for run in [&never_run, &short_run, &long_run] {
        for ((nonce, a), (nonce_b, b)) in off.iter().zip(run) {
            assert_eq!(nonce, nonce_b);
            assert!(b.windows(2).all(|w| w[0] <= w[1]), "rows ascending");
            assert!(b[0] <= a[0], "the deep lane never raises a salt's best");
        }
    }
    for ((nonce, a), (nonce_b, b)) in off.iter().zip(&never_run) {
        assert_eq!(
            (nonce, a),
            (nonce_b, b),
            "allocation 0: the same reads as off"
        );
    }
}

/// Local build (all7): a round change while the deep lane holds salts and
/// runs a long anneal. Slow fake lanes; the lease is cancelled once some
/// results are out (past the trigger's warm-up). The lease ends promptly
/// with no fatal error (no `stopped lease workers did not retire`), and
/// the held and queued work never runs to its end. CPU only.
#[tokio::test]
async fn deep_lane_cancel_drops_held_and_running_work() {
    deep_cancel_case(&[("CU3_TEST_DEEP", "1,16,0.9,0,0")]).await;
}

/// Local build (all9): [`deep_lane_cancel_drops_held_and_running_work`]
/// with the adaptive allocation, two deep-capable lanes of three, holding
/// from the start (MULT 16 to a far horizon). CPU only.
#[tokio::test]
async fn deep_adaptive_cancel_drops_held_and_running_work() {
    deep_cancel_case(&[
        ("CU3_TEST_DEEP", "2,16,0.9,0,0"),
        ("CU3_TEST_DEEP_ADAPTIVE", "0,0,3600,3600,8,16,0,0"),
    ])
    .await;
}

/// Local build (all12): a round change while the ease booster holds round
/// bests and runs long boost runs (MULT 16) on two deep lanes: the lease
/// ends at once, no fatal error, no `did not retire`, no double report.
#[tokio::test]
async fn ease_boost_cancel_drops_held_and_running_work() {
    deep_cancel_case(&[
        ("CU3_TEST_DEEP", "2,4,0.5,0,0"),
        ("CU3_TEST_DEEP_ADAPTIVE", "3600,3600,3600,3600,3,4,0,0"),
        ("CU3_TEST_EASE", "0,9e18,2,36000,90,0,36000,16,48"),
    ])
    .await;
}

/// The cancel scenario of the deep-lane tests, with the deep lane's `env`.
#[allow(clippy::too_many_lines)] // One scripted session, start to shutdown.
async fn deep_cancel_case(env: &[(&str, &str)]) {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, Cancel, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind,
        Topology,
    };
    let salts: u64 = 400;
    let (base, params) = canonical::fixture();
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CU3_TEST_LEASE_ONLY", "1")
            .env("CU3_TEST_LANE_SLEEP_MS", "1")
            .env("CU3_TEST_VARIED_READS", "1")
            .env("QUIP_OTSUKA_SCREEN_THREADS", "4")
            .env("CUDA_VISIBLE_DEVICES", "");
        command.envs(env.iter().copied());
        command
    })
    .await;
    let hash = [0x5e; 32];
    let nodes = base.node_ids().to_vec();
    let edges: Vec<(u32, u32)> = base
        .edges()
        .iter()
        .map(|&(u, v)| (nodes[u as usize], nodes[v as usize]))
        .collect();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: nodes.clone(),
            edges: Some(EdgeList {
                u: edges.iter().map(|e| e.0).collect(),
                v: edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MIN / 2, 4)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:fam".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(&params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    session
        .until("results past the warm-up", Duration::from_mins(2), |log| {
            log.results.len() >= 40 || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    let cancelled_at = std::time::Instant::now();
    session
        .send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
        .await;
    session
        .until("lease done", Duration::from_mins(2), |log| {
            !log.lease_done.is_empty() || log.fatal.is_some()
        })
        .await;
    let ended = cancelled_at.elapsed();
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    let done = session.log.lease_done[0].salts_done;
    let reported = session.log.results.len();
    assert_eq!(session.shutdown(2000).await, 0);
    let stderr = session.stderr();
    assert!(!stderr.contains("did not retire"), "{stderr}");
    eprintln!("deep cancel: lease done {ended:?} after the cancel, {done} of {salts} salts done, {reported} reported");
    assert!(done < salts, "the lease stopped early: {done} of {salts}");
    assert!(
        ended < Duration::from_secs(5),
        "the lease ended promptly: {ended:?}"
    );
    let unique: std::collections::HashSet<&[u8]> = session
        .log
        .results
        .iter()
        .map(|r| r.nonce.as_slice())
        .collect();
    assert_eq!(unique.len(), reported, "no salt reports twice");
}

/// Local build (all6): the model-2 stage-0 score of a fixture salt, as the
/// screen computes it (lower is predicted deeper).
fn fixture_stage0(nonce: [u8; 32]) -> f32 {
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, nonce).expect("canonical instance");
    let (tiles, _) =
        quip_solver_core::ising_generators::ice::local_gauges(&instance).expect("tiles");
    let graph = IsingGraph {
        h: instance
            .h()
            .iter()
            .map(|&m| f64::from(m) / 1000.0)
            .collect(),
        j: instance
            .j()
            .iter()
            .map(|&m| f64::from(m) / 1000.0)
            .collect(),
        edges: instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    };
    quip_solver_cuda::family_screen::stage0_score_model(2, &graph, &tiles)
}

/// Local build (all6): the best-first pool. With slow lanes the screen runs
/// ahead of them; every salt still counts as done exactly once and reports
/// at most once with exact energies, fewer salts are annealed than all5's
/// FIFO queue anneals at the same keep, and the annealed salts have a
/// better (lower) mean stage-0 score. With the pool and the cached
/// generator together, nothing disagrees and the accounting holds. CPU only.
#[tokio::test]
async fn pool_counts_every_salt_once_and_anneals_the_best_scores() {
    let salts = 160;
    let base = [
        ("QUIP_OTSUKA_SCREEN_THREADS", "4"),
        ("CU3_TEST_VARIED_READS", "1"),
        ("CU3_TEST_STAGE0_KEEP", "0.5"),
        ("CU3_TEST_STAGE0_MODEL", "2"),
        ("CU3_TEST_LANE_SLEEP_MS", "1"),
    ];
    let mut pooled = base.to_vec();
    pooled.push(("CU3_TEST_POOL", "3"));
    let mut fast = pooled.clone();
    fast.push(("CU3_TEST_FAST_GEN", "1"));
    let (done_fifo, fifo, _, _) = family_lease_logged(&base, salts, i64::MAX).await;
    let (done_pool, pool, _, _) = family_lease_logged(&pooled, salts, i64::MAX).await;
    let (done_fast, quick, _, log) = family_lease_logged(&fast, salts, i64::MAX).await;
    for (label, done, run) in [
        ("fifo", done_fifo, &fifo),
        ("pool", done_pool, &pool),
        ("pool+fast", done_fast, &quick),
    ] {
        assert_eq!(done, salts, "{label}: every salt counts as done");
        let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
        assert_eq!(unique.len(), run.len(), "{label}: no salt reports twice");
        assert!(!run.is_empty(), "{label}: the lanes annealed something");
        for (_, energies) in run {
            assert!(
                energies.windows(2).all(|w| w[0] <= w[1]),
                "{label}: rows ascending"
            );
        }
    }
    assert!(
        !log.contains("disagrees"),
        "cached generator disagreed: {log}"
    );
    #[allow(clippy::cast_precision_loss)]
    let mean = |run: &[([u8; 32], Vec<i64>)]| {
        run.iter()
            .map(|r| f64::from(fixture_stage0(r.0)))
            .sum::<f64>()
            / run.len() as f64
    };
    let (m_fifo, m_pool, m_fast) = (mean(&fifo), mean(&pool), mean(&quick));
    eprintln!(
        "pool lease: annealed fifo {} (mean score {m_fifo:.4}), pool {} ({m_pool:.4}), pool+fast {} ({m_fast:.4})",
        fifo.len(),
        pool.len(),
        quick.len()
    );
    assert!(
        pool.len() < fifo.len(),
        "the pool anneals fewer salts than the FIFO"
    );
    assert!(
        m_pool < m_fifo,
        "the pool anneals better scores: {m_pool} vs {m_fifo}"
    );
    assert!(
        m_fast < m_fifo,
        "and so with the cached generator: {m_fast} vs {m_fifo}"
    );
}

/// Local build (all6): the pool with stage 1, stage 2 and the polish: every
/// salt done once, no salt reports twice, exact energies, rows ascending.
/// The pool's age limit at its minimum releases entries while the lanes are
/// busy. CPU only.
#[tokio::test]
async fn pool_composes_with_stage1_stage2_and_polish() {
    let salts = 96;
    let knobs = [
        ("QUIP_OTSUKA_SCREEN_THREADS", "4"),
        ("CU3_TEST_VARIED_READS", "1"),
        ("CU3_TEST_STAGE0_KEEP", "0.5"),
        ("CU3_TEST_STAGE0_MODEL", "3"),
        ("CU3_TEST_STAGE1", "0.5,0.5"),
        ("CU3_TEST_STAGE2", "0.5,2,3"),
        ("CU3_TEST_POLISH", "1,2,4,64"),
        ("CU3_TEST_POOL", "4,1000"),
        ("CU3_TEST_FAST_GEN", "1"),
        ("CU3_TEST_LANE_SLEEP_MS", "1"),
    ];
    let (done, run, _, log) = family_lease_logged(&knobs, salts, i64::MIN / 2).await;
    assert_eq!(done, salts, "every salt counts as done");
    let unique: std::collections::HashSet<[u8; 32]> = run.iter().map(|r| r.0).collect();
    assert_eq!(unique.len(), run.len(), "no salt reports twice");
    assert!(!run.is_empty());
    for (_, energies) in &run {
        assert!(energies.windows(2).all(|w| w[0] <= w[1]), "rows ascending");
    }
    assert!(!log.contains("disagrees"), "{log}");
    assert!(!log.contains("panicked"), "{log}");
}

/// Local build (all6): a round cancel with the pool full and the lanes busy
/// ends the lease promptly: the pooled salts of the stopped lease are
/// dropped, never annealed, and nothing hangs or faults. CPU only.
#[tokio::test]
async fn pool_cancel_drops_pooled_work() {
    use family_session::{target, Session};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, Cancel, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind,
        Topology,
    };
    let salts: u64 = 400;
    let (base, params) = canonical::fixture();
    let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .env("CU3_TEST_COORDINATOR", socket)
            .env("CU3_TEST_EXTRA_LANES", "2")
            .env("CU3_TEST_LEASE_ONLY", "1")
            .env("CU3_TEST_LANE_SLEEP_MS", "1")
            .env("CU3_TEST_STAGE0_KEEP", "0.5")
            .env("CU3_TEST_STAGE0_MODEL", "2")
            .env("CU3_TEST_POOL", "8")
            .env("CU3_TEST_FAST_GEN", "1")
            .env("QUIP_OTSUKA_SCREEN_THREADS", "4")
            .env("CUDA_VISIBLE_DEVICES", "");
        command
    })
    .await;
    let hash = [0x5e; 32];
    let nodes = base.node_ids().to_vec();
    let edges: Vec<(u32, u32)> = base
        .edges()
        .iter()
        .map(|&(u, v)| (nodes[u as usize], nodes[v as usize]))
        .collect();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: nodes.clone(),
            edges: Some(EdgeList {
                u: edges.iter().map(|e| e.0).collect(),
                v: edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:fam".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(&params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    session
        .until("first result", Duration::from_mins(2), |log| {
            !log.results.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    let cancelled = std::time::Instant::now();
    session
        .send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
        .await;
    session
        .until("lease done", Duration::from_mins(2), |log| {
            !log.lease_done.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    let ended = cancelled.elapsed();
    let done = session.log.lease_done[0].salts_done;
    assert_eq!(session.shutdown(2000).await, 0);
    let launches: usize = session
        .stderr()
        .lines()
        .find_map(|l| l.strip_prefix("fake-launches "))
        .and_then(|n| n.trim().parse().ok())
        .expect("the miner reports its launch count");
    let per_salt = 257;
    eprintln!("cancelled pooled lease: {done} of {salts} done, {launches} launches, ended {ended:?} after the cancel");
    assert!(
        ended < Duration::from_secs(5),
        "the lease ended promptly: {ended:?}"
    );
    // Three lanes, at most a few salts each before the stop; the pooled
    // salts (up to eight) never reach a lane.
    assert!(
        launches < 12 * per_salt,
        "pooled salts reached the lanes: {launches}"
    );
}

/// Local build: stage-0 model 3, stage 1 and the CPU polish compose on the
/// screen pool and on the serial path: every salt done, exact energies,
/// rows ascending, fewer salts reported than screened. CPU only.
#[tokio::test]
async fn model3_stage1_polish_compose_on_both_screen_paths() {
    let salts = 96;
    let knobs = [
        ("CU3_TEST_VARIED_READS", "1"),
        ("CU3_TEST_STAGE0_KEEP", "0.1"),
        ("CU3_TEST_STAGE0_MODEL", "3"),
        ("CU3_TEST_STAGE1", "0.5,0.5"),
        ("CU3_TEST_POLISH", "2,2,2,50"),
    ];
    let mut pool_knobs = knobs.to_vec();
    pool_knobs.push(("QUIP_OTSUKA_SCREEN_THREADS", "4"));
    for (label, env) in [("serial", knobs.to_vec()), ("pool", pool_knobs)] {
        let (done, results) = family_lease_checked(&env, salts).await;
        eprintln!(
            "model 3 + stage 1 + polish, {label}: {done} done, {} reported",
            results.len()
        );
        assert_eq!(done, salts, "{label}: every salt counts as done");
        assert!(
            !results.is_empty() && results.len() * 4 <= usize::try_from(salts).unwrap(),
            "{label}: reported {}",
            results.len()
        );
        for (_, energies) in &results {
            assert!(
                energies.windows(2).all(|w| w[0] <= w[1]),
                "{label}: rows ascending"
            );
        }
    }
}

/// Local build: with `QUIP_OTSUKA_STAGE0_KEEP` a screened salt still counts as
/// done for its lease (the coordinator's throughput) and sends nothing, and
/// only about the requested fraction reaches a lane.
#[tokio::test]
async fn stage0_screen_counts_every_lease_salt_and_solves_a_fraction() {
    let salts = 96;
    let (done_off, results_off) = family_lease(None, salts).await;
    let (done_on, results_on) = family_lease(Some("0.25"), salts).await;
    eprintln!("stage0 lease: off {done_off} done / {results_off} results, on {done_on} done / {results_on} results");
    assert_eq!(done_off, salts, "every salt is done with the screen off");
    assert_eq!(done_on, salts, "screened salts count as done");
    assert!(results_off > 0);
    assert!(
        results_on >= 1 && results_on * 2 <= results_off,
        "screen kept {results_on} of {results_off}"
    );
}

/// Local build: the same lease contract with stage-0 model 2
/// (`QUIP_OTSUKA_STAGE0_MODEL=2`): every salt done, about the kept fraction
/// solved. CPU only (fake lanes, `CUDA_VISIBLE_DEVICES` empty).
#[tokio::test]
async fn stage0_model2_screen_counts_every_lease_salt_and_solves_a_fraction() {
    let salts = 96;
    let (done_off, results_off) = family_lease(None, salts).await;
    let (done_on, results_on) = family_lease_model(Some("0.25"), Some("2"), salts).await;
    eprintln!("stage0 model 2 lease: off {done_off} done / {results_off} results, on {done_on} done / {results_on} results");
    assert_eq!(done_off, salts, "every salt is done with the screen off");
    assert_eq!(done_on, salts, "screened salts count as done");
    assert!(results_off > 0);
    assert!(
        results_on >= 1 && results_on * 2 <= results_off,
        "screen kept {results_on} of {results_off}"
    );
}

/// Local build (agent P): the same lease contract with stage-0 model 3
/// (`QUIP_OTSUKA_STAGE0_MODEL=3`): model 1 passes half the salts, model 2 half
/// of those. Every salt is done, about a quarter solved, and never more than
/// the prefilter alone passes. CPU only (fake lanes, `CUDA_VISIBLE_DEVICES`
/// empty).
#[tokio::test]
async fn stage0_model3_cascade_counts_every_lease_salt_and_solves_a_fraction() {
    let salts = 96;
    let (done_off, results_off) = family_lease(None, salts).await;
    let (done_on, results_on) = family_lease_env(
        &[
            ("CU3_TEST_STAGE0_KEEP", "0.25"),
            ("CU3_TEST_STAGE0_MODEL", "3"),
            ("CU3_TEST_STAGE0_PRE", "0.5"),
        ],
        salts,
    )
    .await;
    let (done_pre, results_pre) = family_lease_env(
        &[
            ("CU3_TEST_STAGE0_KEEP", "0.5"),
            ("CU3_TEST_STAGE0_MODEL", "3"),
            ("CU3_TEST_STAGE0_PRE", "0.5"),
        ],
        salts,
    )
    .await;
    eprintln!(
        "stage0 model 3 lease: off {done_off}/{results_off}, cascade {done_on}/{results_on}, prefilter only {done_pre}/{results_pre}"
    );
    for done in [done_off, done_on, done_pre] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    assert!(results_off > 0);
    assert!(
        results_on >= 1 && results_on * 2 <= results_off,
        "cascade kept {results_on} of {results_off}"
    );
    assert!(
        results_pre >= 1 && results_pre < results_off && results_on <= results_pre,
        "prefilter alone kept {results_pre}, cascade {results_on}, of {results_off}"
    );
}

/// Local build: with stage 1 on, a salt screened out after its short run
/// still counts as done and sends nothing; promoted salts send their reads.
/// Alone, and behind the stage-0 screen.
#[tokio::test]
async fn stage1_screen_counts_every_lease_salt_and_sends_only_promoted_reads() {
    let salts = 96;
    let (done_off, results_off) = family_lease_env(&[], salts).await;
    let (done_s1, results_s1) = family_lease_env(&[("CU3_TEST_STAGE1", "0.5,0.25")], salts).await;
    let (done_both, results_both) = family_lease_env(
        &[
            ("CU3_TEST_STAGE0_KEEP", "0.5"),
            ("CU3_TEST_STAGE1", "0.25,0.25,2"),
        ],
        salts,
    )
    .await;
    let (done_all, results_all) = family_lease_env(&[("CU3_TEST_STAGE1", "0.5,1")], salts).await;
    eprintln!(
        "stage1 lease: off {done_off}/{results_off}, s1 {done_s1}/{results_s1},          s0+s1 {done_both}/{results_both}, keep-1 {done_all}/{results_all}"
    );
    for done in [done_off, done_s1, done_both, done_all] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    assert_eq!(
        results_all, results_off,
        "keep 1 promotes and reports every salt"
    );
    assert!(
        results_s1 >= 1 && results_s1 * 2 <= results_off,
        "stage 1 kept {results_s1} of {results_off}"
    );
    assert!(
        results_both >= 1 && results_both <= results_s1,
        "stage 0 + 1 kept {results_both} of {results_off}"
    );
}

/// Local build (integration of `budfox/otsuka-screen2` and
/// `budfox/otsuka-2stage`): stage 0 with model 2 (contracted cycles) gates
/// first, then stage 1's short run gates the salts it passes. Every salt still
/// counts as done, and the two screens together send no more than stage 1
/// alone behind the stage-0 model-1 screen at the same keeps would allow.
/// CPU only (fake lanes, `CUDA_VISIBLE_DEVICES` empty).
#[tokio::test]
async fn stage0_model2_then_stage1_compose_on_lease_salts() {
    let salts = 96;
    let (done_off, results_off) = family_lease_env(&[], salts).await;
    let (done_m2, results_m2) = family_lease_env(
        &[
            ("CU3_TEST_STAGE0_KEEP", "0.5"),
            ("CU3_TEST_STAGE0_MODEL", "2"),
        ],
        salts,
    )
    .await;
    let (done_both, results_both) = family_lease_env(
        &[
            ("CU3_TEST_STAGE0_KEEP", "0.5"),
            ("CU3_TEST_STAGE0_MODEL", "2"),
            ("CU3_TEST_STAGE1", "0.25,0.25,2"),
        ],
        salts,
    )
    .await;
    eprintln!(
        "stage0 model 2 + stage1 lease: off {done_off}/{results_off}, \
         m2 {done_m2}/{results_m2}, m2+s1 {done_both}/{results_both}"
    );
    for done in [done_off, done_m2, done_both] {
        assert_eq!(done, salts, "every salt counts as done");
    }
    assert!(results_off > 0);
    assert!(
        results_m2 >= 1 && results_m2 < results_off,
        "stage 0 model 2 kept {results_m2} of {results_off}"
    );
    assert!(
        results_both >= 1 && results_both <= results_m2,
        "stage 0 model 2 + stage 1 kept {results_both} of {results_m2}"
    );
}

/// Local build: GPU smoke test of the stage-0 screen on the live topology,
/// through the real binary and core's default lease path. Ignored: run it in
/// a throwaway container with `OTSUKA_SMOKE_BIN=/path/quip-cuda-otsuka` and
/// the knob environment (`QUIP_OTSUKA_*`, inherited by the miner). Every
/// returned energy is checked exactly against the canonical instance.
/// `OTSUKA_SMOKE_SALTS` (48), `OTSUKA_SMOKE_SWEEPS` (2048).
#[tokio::test]
#[ignore = "needs a CUDA device and OTSUKA_SMOKE_BIN"]
#[allow(clippy::too_many_lines)]
async fn gpu_stage0_family_lease_smoke() {
    use family_session::{target, Session};
    use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
    use quip_solver_core::ising_generators::family::{generate_instance, BaseTopology};
    use quip_solver_core::ising_generators::ice::{RECIPES, TWISTED_DIMER_V1};
    use quip_solver_core::quip_proto::v1::{
        coord_msg, EdgeList, GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind, Topology,
    };
    use quip_solver_core::quip_protocol::wire::decode_spins_packed;
    let Ok(binary) = std::env::var("OTSUKA_SMOKE_BIN") else {
        return;
    };
    let var = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let salts = var("OTSUKA_SMOKE_SALTS", 48);
    let sweeps = u32::try_from(var("OTSUKA_SMOKE_SWEEPS", 2048)).unwrap();
    let blob = include_bytes!("fixtures/aglais-z12t4-spec120.bin.gz");
    let topo = ising_generators::blob::decode(blob)
        .unwrap()
        .topology()
        .unwrap();
    let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
    let params = RECIPES[0].params;
    let digest = params_digest(params);
    let contracts = [TWISTED_DIMER_V1];
    let admitted = admit(
        &contracts,
        TWISTED_DIMER_V1.family_id.as_bytes(),
        1,
        params,
        &digest,
    )
    .unwrap();
    let bounds = admitted.size_bounds(base.stats()).unwrap();
    let started = std::time::Instant::now();
    let mut session = Session::start(&binary, 4, "").await;
    let hash = [0x5e; 32];
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: topo.nodes.clone(),
            edges: Some(EdgeList {
                u: topo.edges.iter().map(|e| e.0).collect(),
                v: topo.edges.iter().map(|e| e.1).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(target(i64::MAX, sweeps)))
        .await;
    session
        .send(coord_msg::Msg::Job(Job {
            job_id: b"lease:smoke".to_vec(),
            kind: JobKind::IsingGenerate as i32,
            generation: 1,
            deadline_ms: 0,
            ising: None,
            provenance: None,
            generator: Some(IsingProblemGenerator {
                algorithm: GeneratorAlgorithm::FamilyChacha8V1 as i32,
                topology_hash: hash.to_vec(),
                last_proof_block_hash: vec![0x11; 32],
                miner_account: vec![0x22; 32],
                base_salt: vec![0; 32],
                salt_start: 1,
                salt_count: salts,
                family: Some(canonical::descriptor(params)),
                topology_order_digest: base.order_digest().to_vec(),
            }),
        }))
        .await;
    session
        .until("lease done", Duration::from_mins(4), |log| {
            !log.lease_done.is_empty() || !log.rejects.is_empty() || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert!(session.log.rejects.is_empty(), "{:?}", session.log.rejects);
    let done = &session.log.lease_done[0];
    let mut best = i64::MAX;
    let mut checked = 0;
    for result in &session.log.results {
        let nonce: [u8; 32] = result.nonce.as_slice().try_into().unwrap();
        let instance = generate_instance(&admitted, &base, &bounds, nonce).unwrap();
        let n = instance.node_ids().len();
        for solution in &result.solutions {
            let spins = decode_spins_packed(&solution.spins, n).unwrap();
            let mut e = instance.offset_milli();
            for (k, &(u, v)) in instance.edges().iter().enumerate() {
                e += i64::from(instance.j()[k])
                    * i64::from(spins[u as usize])
                    * i64::from(spins[v as usize]);
            }
            for (i, &h) in instance.h().iter().enumerate() {
                e += i64::from(h) * i64::from(spins[i]);
            }
            assert_eq!(e, solution.energy_milli, "reported energy is exact");
            best = best.min(e);
            checked += 1;
        }
    }
    let log = session.stderr();
    let pairs = log.matches("otsuka stage0 pair").count();
    // Local build: stage-1 decisions (QUIP_OTSUKA_STAGE1_* in the environment).
    let s1 = log.matches("otsuka stage1 pair").count();
    let s1_promoted = log
        .lines()
        .filter(|l| l.contains("otsuka stage1 pair") && l.contains("Promoted"))
        .count();
    eprintln!(
        "smoke: {salts} salts, salts_done {}, results {}, solutions checked {checked}, best {best}, \
         stage0 pair lines {pairs}, stage1 pairs {s1} promoted {s1_promoted}, {:.1} s",
        done.salts_done,
        session.log.results.len(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(done.salts_done, salts);
    assert!(!session.log.results.is_empty());
    assert_eq!(session.shutdown(2000).await, 0);
}

mod canonical {
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::{
        descriptor::params_digest, family::BaseTopology, ice::TWISTED_DIMER_V1, instance::Instance,
    };
    use quip_solver_core::quip_proto::v1::{
        ising_problem, CoefficientEncoding, EdgeList, FamilyBinding, FamilyDescriptor,
        IsingProblem, Job, JobKind,
    };
    use quip_solver_core::quip_protocol::wire::encode_i32_le;
    pub(super) fn fixture() -> (BaseTopology, Vec<u8>) {
        let reference = reference_revision(TWISTED_DIMER_V1.family_id, 1).unwrap();
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let base = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let params = reference
            .params
            .iter()
            .find(|p| {
                export_instances(reference.contract, p, &base, &[[7; 32]])
                    .is_ok_and(|instances| instances.into_iter().any(|i| i.is_ok()))
            })
            .unwrap()
            .to_vec();
        (base, params)
    }

    pub(super) fn job(
        instance: &Instance,
        params: &[u8],
        id: &[u8],
        sweeps: u32,
        reads: u32,
    ) -> Job {
        Job {
            job_id: id.to_vec(),
            kind: JobKind::IsingSample as i32,
            generation: 1,
            ising: Some(IsingProblem {
                graph: Some(ising_problem::Graph::Edges(EdgeList {
                    u: instance.edges().iter().map(|&(u, _)| u).collect(),
                    v: instance.edges().iter().map(|&(_, v)| v).collect(),
                })),
                encoding: CoefficientEncoding::I32 as i32,
                scale: 1000,
                h: encode_i32_le(instance.h()),
                j: encode_i32_le(instance.j()),
                num_reads: reads,
                num_sweeps: sweeps,
                family: Some(FamilyBinding {
                    family: Some(descriptor(params)),
                    nonce: vec![7; 32],
                    instance_digest: instance.digest().to_vec(),
                    metadata: instance.metadata().to_vec(),
                    topology_order_digest: instance.topology_order_digest().to_vec(),
                    node_ids: instance.node_ids().to_vec(),
                    offset_milli: instance.offset_milli(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub(super) fn descriptor(params: &[u8]) -> FamilyDescriptor {
        FamilyDescriptor {
            family_id: TWISTED_DIMER_V1.family_id.into(),
            revision: 1,
            params: params.to_vec(),
            params_digest: params_digest(params).to_vec(),
        }
    }

    pub(super) fn instance(
        base: &BaseTopology,
        params: &[u8],
        nonce: [u8; 32],
    ) -> Option<Instance> {
        export_instances(&TWISTED_DIMER_V1, params, base, &[nonce])
            .unwrap()
            .pop()
            .unwrap()
            .ok()
    }
}

mod gpu {
    use super::canonical::{descriptor, fixture, instance};
    use super::family_session::{lease, target, Session};
    use super::PHASE_LIMIT;
    use quip_solver_conformance::families::drive_miner_family;
    use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::quip_proto::v1::{coord_msg, EdgeList, GeneratorAlgorithm, Topology};
    use quip_solver_core::quip_protocol::lease::LeaseSpec;
    use quip_solver_core::quip_protocol::scoring::energy_from_milli;
    use quip_solver_core::quip_protocol::wire::decode_spins_packed;
    use serial_test::serial;

    const BIN: &str = env!("CARGO_BIN_EXE_quip-cuda-otsuka");
    const HASH: [u8; 32] = [0x73; 32];

    fn hardware() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let (major, minor) = ctx.compute_capability().unwrap();
        eprintln!(
            "CU-3 hardware: device=0 name={} compute={major}.{minor}",
            ctx.name().unwrap()
        );
    }

    fn assert_scores(result: &quip_solver_core::quip_proto::v1::Result, instance: &Instance) {
        let edges: Vec<_> = instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        assert!(!result.solutions.is_empty());
        for solution in &result.solutions {
            let spins = decode_spins_packed(&solution.spins, instance.node_ids().len()).unwrap();
            assert_eq!(
                solution.energy_milli,
                energy_from_milli(&spins, instance.h(), instance.j(), &edges)
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial]
    fn synchronized_sweeps_and_descent_match_an_unchunked_run() {
        use quip_solver_cuda::cluster::{
            device::{ClusterDevice, ClusterSchedule},
            sampler::ClusterWork,
            PreparedGraph, Weights,
        };
        hardware();
        let (base, family_params) = fixture();
        let instance = instance(&base, &family_params, [7; 32]).unwrap();
        let graph = quip_solver_core::IsingGraph::new(
            instance
                .h()
                .iter()
                .map(|&h| f64::from(h) / 1000.0)
                .collect(),
            instance
                .j()
                .iter()
                .map(|&j| f64::from(j) / 1000.0)
                .collect(),
            instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect(),
        );
        let params = quip_solver_core::SampleParams {
            num_reads: 64,
            num_sweeps: 24,
            sweeps_per_beta: 6,
            beta_range: Some((0.1, 1.7)),
            schedule: quip_solver_core::BetaSchedule::Geometric,
            seed: 73,
        };
        let prepared = PreparedGraph::from_instance(
            &instance,
            &TWISTED_DIMER_V1,
            Weights::FULL,
            &super::HEADLESS_LIMITS,
            64,
        )
        .unwrap();
        assert!(!prepared.layout().tiles.is_empty());
        assert!(!prepared.layout().pairs.is_empty());
        assert!(prepared.layout().weak_deg.iter().any(|&d| d > 0));
        let mut device = ClusterDevice::open(0, prepared).unwrap();
        let work = ClusterWork::new(&graph, &params).unwrap();
        let actual = device.run_work(&work, &|| false, &|| false).unwrap();
        assert_eq!(actual.spins.len(), 64);
        assert_eq!(actual.energies.len(), 64);
        for chunk in [2, 4, 6] {
            // 6 runs a whole rung in one launch. 4 also exercises a short tail.
            // Both paths use the same geometric ladder, with six sweeps per
            // rung. Chunk boundaries fall both inside and at the end of rungs.
            let expected = device.run_work_with_chunk_for_test(&work, chunk).unwrap();
            assert_eq!(actual.spins, expected.spins, "chunk={chunk}");
            assert_eq!(actual.energies, expected.energies, "chunk={chunk}");
        }
        // The host ladder sets a constant beta per launch. Also exercise
        // beta_at with a varying device schedule and nonzero sweep offsets.
        // Both paths evaluate the same betas on the device, so powf rounding
        // cannot differ between host and device or weaken exact comparison.
        let schedule = ClusterSchedule {
            sweeps: 24,
            chunk: 24,
            beta_min: 0.1,
            beta_max: 1.7,
            ..Default::default()
        };
        let (expected, expected_rng) = device
            .run_before_descent_for_test(&schedule, params.num_reads, params.seed)
            .unwrap();
        let expected_final = device
            .run(&schedule, params.num_reads, params.seed)
            .unwrap();
        for chunk in [1, 2, 4, 6, 7] {
            // Seven does not divide the schedule and leaves a short tail.
            let chunked = ClusterSchedule {
                chunk,
                ..schedule.clone()
            };
            let (actual, actual_rng) = device
                .run_before_descent_for_test(&chunked, params.num_reads, params.seed)
                .unwrap();
            assert_eq!(actual.spins, expected.spins, "varying beta chunk={chunk}");
            assert_eq!(
                actual.energies, expected.energies,
                "varying beta chunk={chunk}"
            );
            assert_eq!(actual_rng, expected_rng, "varying beta chunk={chunk}");
            let actual_final = device.run(&chunked, params.num_reads, params.seed).unwrap();
            assert_eq!(
                actual_final.spins, expected_final.spins,
                "descent chunk={chunk}"
            );
            assert_eq!(
                actual_final.energies, expected_final.energies,
                "descent chunk={chunk}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires a CUDA GPU"]
    #[serial]
    async fn family_progresses_while_sa_is_active_then_cancels_cleanly() {
        use quip_solver_core::quip_proto::v1::Cancel;
        hardware();
        let (base, params) = fixture();
        let instance = instance(&base, &params, [7; 32]).unwrap();
        let mut session = Session::start_with(4, "num_sweeps = 4", |socket| {
            let mut command = tokio::process::Command::new(BIN);
            command.args(["--quip-coordinator", socket, "--sweeps-per-beta", "1024"]);
            command
        })
        .await;
        session
            .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
            .await;
        let mut plain = super::canonical::job(&instance, &params, b"active-sa", 10_000_000, 64);
        plain.ising.as_mut().unwrap().family = None;
        session.send(coord_msg::Msg::Job(plain)).await;
        session
            .until_stderr("self-feeding session launching", PHASE_LIMIT)
            .await;
        session
            .send(coord_msg::Msg::Job(super::canonical::job(
                &instance,
                &params,
                b"concurrent-family",
                4,
                64,
            )))
            .await;
        session
            .until("family while SA remains active", PHASE_LIMIT, |log| {
                !log.results_for(b"concurrent-family").is_empty()
            })
            .await;
        assert!(session.log.results_for(b"active-sa").is_empty());
        assert!(session.log.rejects.is_empty());
        assert_scores(session.log.results_for(b"concurrent-family")[0], &instance);
        session
            .send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
            .await;
        session
            .until("family and cancelled SA credits", PHASE_LIMIT, |log| {
                log.refunds() == 2
            })
            .await;
        let mut next = super::canonical::job(&instance, &params, b"next-sa", 4, 64);
        next.generation = 2;
        next.ising.as_mut().unwrap().family = None;
        session.send(coord_msg::Msg::Job(next)).await;
        session
            .until("next SA job", PHASE_LIMIT, |log| {
                !log.results_for(b"next-sa").is_empty() && log.refunds() == 3
            })
            .await;
        assert!(session.log.fatal.is_none());
        assert_eq!(session.shutdown(2000).await, 0);
    }

    #[tokio::test]
    #[ignore = "requires a CUDA GPU"]
    #[serial]
    async fn explicit_twisted_dimer_job_scores_on_original_model() {
        hardware();
        let (base, params) = fixture();
        let instance = instance(&base, &params, [7; 32]).unwrap();
        let mut session = Session::start(BIN, 1, "num_sweeps = 4").await;
        session
            .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
            .await;
        session
            .send(coord_msg::Msg::Job(super::canonical::job(
                &instance,
                &params,
                b"explicit-cluster",
                4,
                64,
            )))
            .await;
        session
            .until("explicit result and credit", PHASE_LIMIT, |log| {
                !log.results.is_empty() && log.refunds() == 1
            })
            .await;
        assert!(session.log.rejects.is_empty());
        assert_scores(&session.log.results[0], &instance);
        assert_eq!(session.shutdown(2000).await, 0);
        assert!(session.stderr().contains("cuda kernel cluster:"));
    }

    #[tokio::test]
    #[ignore = "requires a CUDA GPU"]
    #[serial]
    async fn canonical_host_family_lease_reaches_cluster_and_refunds_credit() {
        hardware();
        let (base, params) = fixture();
        let mut request = lease(b"host-cluster", 1, HASH, 0, 2);
        let generator = request.generator.as_mut().unwrap();
        generator.algorithm = GeneratorAlgorithm::FamilyChacha8V1 as i32;
        generator.family = Some(descriptor(&params));
        generator.topology_order_digest = base.order_digest().to_vec();
        let spec = LeaseSpec::from_proto(generator).unwrap();
        let mut session = Session::start(BIN, 1, "num_sweeps = 4").await;
        assert!(!session
            .log
            .hello
            .as_ref()
            .unwrap()
            .capabilities
            .as_ref()
            .unwrap()
            .features
            .iter()
            .any(|f| f == "local-generation"));
        session
            .send(coord_msg::Msg::Topology(Topology {
                hash: HASH.to_vec(),
                nodes: base.node_ids().to_vec(),
                edges: Some(EdgeList {
                    u: base.edges().iter().map(|&(u, _)| u).collect(),
                    v: base.edges().iter().map(|&(_, v)| v).collect(),
                }),
                allowed_h_milli: vec![0],
                allowed_j_milli: vec![-1000, 1000],
            }))
            .await;
        session
            .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
            .await;
        session.send(coord_msg::Msg::Job(request)).await;
        session
            .until("family lease and credit", PHASE_LIMIT, |log| {
                log.done_count(b"host-cluster") == 1 && log.refunds() == 1
            })
            .await;
        assert!(session.log.rejects.is_empty());
        let mut expected_results = 0;
        for index in 0..2 {
            let nonce = spec.nonce(index).unwrap();
            if let Some(instance) = instance(&base, &params, nonce) {
                expected_results += 1;
                let result = session
                    .log
                    .results
                    .iter()
                    .find(|r| r.nonce == nonce)
                    .unwrap();
                assert_scores(result, &instance);
            }
        }
        assert!(expected_results > 0);
        assert_eq!(session.log.results.len(), expected_results);
        assert_eq!(session.shutdown(2000).await, 0);
        assert!(session.stderr().contains("cuda kernel cluster:"));
    }

    #[tokio::test]
    #[ignore = "requires a CUDA GPU"]
    #[serial]
    async fn sc7_family_conformance_records_hardware() {
        hardware();
        let directory = tempfile::tempdir().unwrap();
        let socket = format!("unix://{}", directory.path().join("sc7.sock").display());
        let report = drive_miner_family(BIN, &[], &socket).await;
        eprintln!("{}", report.summary());
        assert!(
            report.stderr.contains("cuda kernel cluster:"),
            "{}",
            report.stderr
        );
        assert!(
            report.is_conformant(),
            "{}\n{}",
            report.summary(),
            report.stderr
        );
        for name in [
            "supported-revision",
            "explicit-supported",
            "credit-ledger-balanced",
        ] {
            assert!(
                report.cases.iter().any(|case| case.name == name
                    && case.outcome == quip_solver_conformance::solve::CaseOutcome::Passed),
                "{name}: {}",
                report.summary()
            );
        }
    }
}
