//! Obtain session-validated contexts without exposing a production test constructor.

use quip_solver_core::families::{Declaration, FamilyContext};
use quip_solver_core::ising_generators::descriptor::params_digest;
use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, miner_msg,
    miner_service_server::{MinerService, MinerServiceServer},
    CoefficientEncoding, Configure, CoordMsg, EdgeList, FamilyBinding, FamilyDescriptor,
    IsingProblem, Job, JobKind, MinerMsg, Shutdown, Welcome,
};
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use quip_solver_core::{
    CommonArgs, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, WarmStart,
};
use quip_solver_cuda::families::flow_loops::{FlowLoopsSampler, CUDA_FLOW_LOOPS_IDENTITY};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};

type Contexts = Vec<(FamilyContext, WarmStart)>;

struct Capture(Arc<Mutex<Contexts>>);

impl Sampler for Capture {
    fn sample(&self, _: &IsingGraph, _: &SampleParams) -> Result<Vec<SamplerResult>, SampleError> {
        panic!("family job must reach sample_family")
    }
    fn accepts_family_metadata() -> bool {
        true
    }
    fn accepts_warm_start() -> bool {
        true
    }
    fn declaration() -> Declaration {
        FlowLoopsSampler::declaration()
    }
    fn sample_family(
        &self,
        _: &IsingGraph,
        _: &SampleParams,
        warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.0
            .lock()
            .unwrap()
            .push((family.clone(), warm.unwrap().clone()));
        Err(SampleError::Declined("context captured".into()))
    }
}

struct Coordinator(Vec<Job>);

#[tonic::async_trait]
impl MinerService for Coordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;
    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let (tx, rx) = mpsc::channel(16);
        let jobs = self.0.clone();
        let mut inbound = request.into_inner();
        tokio::spawn(async move {
            let send = |msg| tx.send(Ok(CoordMsg { msg: Some(msg) }));
            assert!(matches!(
                inbound.message().await.unwrap().unwrap().msg,
                Some(miner_msg::Msg::Hello(_))
            ));
            send(coord_msg::Msg::Welcome(Welcome {
                protocol_version: 2,
            }))
            .await
            .unwrap();
            send(coord_msg::Msg::Configure(Configure {
                queue_depth: 2,
                idle_timeout_s: 60,
                heartbeat_s: 15,
                reconnect_window_s: 0,
                backend_toml: String::new(),
            }))
            .await
            .unwrap();
            let mut dispatched = false;
            let mut rejects = 0;
            while let Some(frame) = inbound.message().await.unwrap() {
                match frame.msg {
                    Some(miner_msg::Msg::JobRequest(_)) if !dispatched => {
                        for job in &jobs {
                            send(coord_msg::Msg::Job(job.clone())).await.unwrap();
                        }
                        dispatched = true;
                    }
                    Some(miner_msg::Msg::Reject(_)) => {
                        rejects += 1;
                        if rejects == jobs.len() {
                            send(coord_msg::Msg::Shutdown(Shutdown { grace_ms: 1000 }))
                                .await
                                .unwrap();
                        }
                    }
                    _ => {}
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

pub(super) fn contexts() -> &'static Contexts {
    static CONTEXTS: OnceLock<Contexts> = OnceLock::new();
    CONTEXTS.get_or_init(|| {
        let jobs = [super::FLOW_V1, super::POLARITON_V1]
            .iter()
            .enumerate()
            .map(|(k, contract)| {
                let (params, i) =
                    super::instance(contract, "../topologies/routing-small.bin.gz", 1);
                Job {
                    job_id: vec![u8::try_from(k).unwrap()],
                    kind: JobKind::IsingSample as i32,
                    deadline_ms: u64::MAX,
                    ising: Some(IsingProblem {
                        graph: Some(ising_problem::Graph::Edges(EdgeList {
                            u: i.edges().iter().map(|&(u, _)| u).collect(),
                            v: i.edges().iter().map(|&(_, v)| v).collect(),
                        })),
                        encoding: CoefficientEncoding::I32 as i32,
                        scale: 1000,
                        h: encode_i32_le(i.h()),
                        j: encode_i32_le(i.j()),
                        num_reads: 5,
                        num_sweeps: 10,
                        initial_spins: vec![vec![0; i.h().len().div_ceil(8)]],
                        family: Some(FamilyBinding {
                            family: Some(FamilyDescriptor {
                                family_id: contract.family_id.into(),
                                revision: contract.revision,
                                params_digest: params_digest(&params).to_vec(),
                                params,
                            }),
                            nonce: i.nonce().unwrap().to_vec(),
                            instance_digest: i.digest().to_vec(),
                            metadata: i.metadata().to_vec(),
                            topology_order_digest: i.topology_order_digest().to_vec(),
                            node_ids: i.node_ids().to_vec(),
                            offset_milli: i.offset_milli(),
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            })
            .collect();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let root = std::env::current_dir().unwrap().join(".tmp");
            std::fs::create_dir_all(&root).unwrap();
            let directory = tempfile::tempdir_in(root).unwrap();
            let path = directory.path().join("session.sock");
            let incoming = UnixListenerStream::new(tokio::net::UnixListener::bind(&path).unwrap());
            let server = tokio::spawn(
                tonic::transport::Server::builder()
                    .add_service(MinerServiceServer::new(Coordinator(jobs)))
                    .serve_with_incoming(incoming),
            );
            let sampler = Capture(Arc::clone(&captured));
            // The local coordinator uses no account credentials.
            assert!(
                std::env::var_os("QUIP_SESSION_TOKEN").is_none(),
                "run this fixture without an account token"
            );
            std::env::set_var("QUIP_SESSION_TOKEN", "local-test-token");
            let common = CommonArgs {
                quip_coordinator: Some(format!("unix://{}", path.display())),
                miner_id: Some("flow-test".into()),
                capabilities: false,
                solve: false,
                check: false,
                log_level: "error".into(),
                sweeps_per_beta: None,
            };
            let run = tokio::task::spawn_blocking(move || {
                quip_solver_core::run_code(CUDA_FLOW_LOOPS_IDENTITY, &common, || Ok(sampler))
            });
            let exit = tokio::time::timeout(Duration::from_secs(2), run)
                .await
                .unwrap()
                .unwrap();
            std::env::remove_var("QUIP_SESSION_TOKEN");
            assert_eq!(exit, quip_solver_core::ExitCode::Clean);
            server.abort();
        });
        let values = Arc::try_unwrap(captured).unwrap().into_inner().unwrap();
        assert_eq!(values.len(), 2);
        values
    })
}
