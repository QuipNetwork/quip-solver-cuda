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
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct FakeDevice {
    launches: Arc<Mutex<usize>>,
    cancel: Option<CancelToken>,
    fail: bool,
    short: bool,
    cancel_after: usize,
}

impl ClusterBackend for FakeDevice {
    fn run(
        &self,
        _graph: PreparedGraph,
        work: &ClusterWork,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        work.for_each_launch(cancelled, |_, _, _| {
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
        let count = if self.short { 1 } else { work.reads };
        Ok(ClusterRun {
            spins: (0..count)
                .map(|read| {
                    if read % 2 == 0 {
                        vec![1, -1]
                    } else {
                        vec![-1, 1]
                    }
                })
                .collect(),
            energies: vec![123; count], // The sampler must score original coefficients.
        })
    }
}

#[derive(Clone, Default)]
struct FakeSa {
    calls: Arc<Mutex<usize>>,
    yielding: bool,
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

    fn should_throttle(&self) -> bool {
        self.yielding
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
    assert!(matches!(result.outcome, StreamOutcome::Completed(Err(_))));
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
    assert_eq!(sampler.stream_width(), 1);
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
    assert_eq!(caps["streamWidth"], 1);
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

#[cfg(feature = "long-tests")]
#[allow(dead_code)] // The family tests use only part of the shared coordinator.
#[path = "common/lease.rs"]
mod family_session;

#[cfg(feature = "long-tests")]
mod gpu {
    use super::family_session::{lease, target, Session};
    use quip_solver_conformance::families::{
        drive_miner_family, export_instances, reference_revision,
    };
    use quip_solver_core::ising_generators::descriptor::params_digest;
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::quip_proto::v1::{
        coord_msg, ising_problem, CoefficientEncoding, EdgeList, FamilyBinding, FamilyDescriptor,
        GeneratorAlgorithm, IsingProblem, Job, JobKind, Topology,
    };
    use quip_solver_core::quip_protocol::lease::LeaseSpec;
    use quip_solver_core::quip_protocol::scoring::energy_from_milli;
    use quip_solver_core::quip_protocol::wire::{decode_spins_packed, encode_i32_le};
    use serial_test::serial;
    use std::time::Duration;

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

    fn fixture() -> (BaseTopology, Vec<u8>) {
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

    fn descriptor(params: &[u8]) -> FamilyDescriptor {
        FamilyDescriptor {
            family_id: TWISTED_DIMER_V1.family_id.into(),
            revision: 1,
            params: params.to_vec(),
            params_digest: params_digest(params).to_vec(),
        }
    }

    fn instance(base: &BaseTopology, params: &[u8], nonce: [u8; 32]) -> Option<Instance> {
        export_instances(&TWISTED_DIMER_V1, params, base, &[nonce])
            .unwrap()
            .pop()
            .unwrap()
            .ok()
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
        use quip_solver_cuda::cluster::device::{ClusterDevice, ClusterSchedule};
        use quip_solver_cuda::cluster::sampler::ClusterWork;
        hardware();
        let params = quip_solver_core::SampleParams {
            num_reads: 3,
            num_sweeps: 7,
            beta_range: Some((0.7, 0.7)),
            seed: 73,
            ..Default::default()
        };
        let mut device = ClusterDevice::open(0, super::prepared(&super::HEADLESS_LIMITS)).unwrap();
        let expected = device
            .run(
                &ClusterSchedule {
                    sweeps: 7,
                    chunk: 7,
                    beta_min: 0.7,
                    beta_max: 0.7,
                    ..Default::default()
                },
                3,
                73,
            )
            .unwrap();
        let work = ClusterWork::new(&super::graph(), &params).unwrap();
        let actual = device.run_work(&work, &|| false).unwrap();
        assert_eq!(actual.spins, expected.spins);
        assert_eq!(actual.energies, expected.energies);
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
            .send(coord_msg::Msg::Job(Job {
                job_id: b"explicit-cluster".to_vec(),
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
                    family: Some(FamilyBinding {
                        family: Some(descriptor(&params)),
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
            }))
            .await;
        session
            .until(
                "explicit result and credit",
                Duration::from_secs(3),
                |log| !log.results.is_empty() && log.refunds() == 1,
            )
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
            .until("family lease and credit", Duration::from_secs(3), |log| {
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
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
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
