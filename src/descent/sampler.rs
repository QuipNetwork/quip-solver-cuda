//! The `quip-cuda-msa-descent` session adapter: the MSA kernel stream, then
//! the host descent of each completed job on a post-process thread.
//!
//! The pattern follows `OtsukaSampler` (src/cluster/sampler.rs): the intake
//! keeps the graph of each job and forwards the plain job to the inner
//! stream. The post-process thread receives each kernel result, descends
//! its reads, rescores them exactly, and forwards the result. The GPU keeps
//! running the next jobs while the host descends. Lease salts go to the
//! inner sampler without descent.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::{CHARGED_ICE_V1, ICE_CHARGE_RECIPE};
use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamOutcome, StreamResult, WarmStreamJob,
};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use super::greedy::DescentError;
use super::tiles::instance_tiles;
use super::{descend, Plan, POLICY_ID};

/// The stable id of this binary (FAM-60).
pub const MSA_DESCENT_SOLVER_ID: &str = "quip-cuda-msa-descent/1";

/// The binary accepts explicit `quip/charged-ice@1` jobs of the `ice-charge`
/// recipe. It generates no family work.
const ACCEPTED: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[CHARGED_ICE_V1],
    only_params: &[ICE_CHARGE_RECIPE.params],
}];

/// The inner MSA sampler with the `msa-nat-loop/1` descent after each job.
#[derive(Debug)]
pub struct MsaDescentSampler<S> {
    inner: S,
}

impl<S> MsaDescentSampler<S> {
    /// Wrap the kernel sampler.
    #[must_use]
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

/// The metadata tiles of a selected family job: a `quip/charged-ice@1` job
/// with the `ice-charge` recipe parameters. `None` for every other job,
/// whose tiles are then recovered from the couplings.
///
/// # Errors
///
/// [`SampleError::Declined`] when a selected instance does not match the job
/// graph or its tile section is malformed.
pub fn family_tiles(
    graph: &IsingGraph,
    family: Option<&FamilyContext>,
) -> Result<Option<Vec<[u32; 4]>>, SampleError> {
    let Some(family) = family else {
        return Ok(None);
    };
    let selected = family.family_id() == CHARGED_ICE_V1.family_id
        && family.revision() == CHARGED_ICE_V1.revision
        && family.params() == ICE_CHARGE_RECIPE.params;
    if !selected {
        return Ok(None);
    }
    instance_tiles(graph, family.instance())
        .map(Some)
        .map_err(SampleError::Declined)
}

/// What the post-process thread needs to descend one job.
struct Pending {
    graph: IsingGraph,
    seed: u64,
    tiles: Option<Vec<[u32; 4]>>,
    watermark: Option<u64>,
}

/// Why a descent did not return states.
enum Failure {
    Cancelled,
    Sample(SampleError),
}

/// Descend the raw kernel reads of one job and rescore every state.
fn finish(
    pending: &Pending,
    raw: Vec<SamplerResult>,
    kernel_s: f64,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let started = Instant::now();
    let graph = &pending.graph;
    let plan = match Plan::new(graph, pending.tiles.clone()) {
        Ok(plan) => plan,
        Err(why) => {
            tracing::warn!(
                policy = POLICY_ID,
                why,
                "descent skipped; raw reads returned"
            );
            return Ok(raw);
        }
    };
    let spins: Vec<Vec<i8>> = raw.into_iter().map(|r| r.spins).collect();
    let threads = crate::streaming::score_threads();
    let out = descend(&plan, &spins, pending.seed, threads, cancel).map_err(|e| match e {
        DescentError::Cancelled => Failure::Cancelled,
        e => Failure::Sample(SampleError::DeviceFault(format!("descent: {e}"))),
    })?;
    let mut reads = Vec::with_capacity(out.reads.len());
    for spins in out.reads {
        let energy = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
        if energy != plan.ising().energy(&spins) {
            return Err(Failure::Sample(SampleError::DeviceFault(
                "descent rescore differs from the exact energy".into(),
            )));
        }
        reads.push(SamplerResult {
            spins,
            energy_milli: energy,
        });
    }
    let report = out.report;
    let best = reads.iter().map(|r| r.energy_milli).min();
    if best != Some(report.final_best) || report.final_best > report.raw_best {
        return Err(Failure::Sample(SampleError::DeviceFault(
            "descended best is above the raw best or differs from the report".into(),
        )));
    }
    let counts = plan.tile_counts();
    tracing::info!(
        policy = POLICY_ID,
        tiles_from = plan.source().name(),
        tiles = counts.tiles,
        tiled_spins = plan.tiles().tiled().len(),
        moves = plan.greedy().num_moves(),
        reads = reads.len(),
        kernel_s,
        descent_s = started.elapsed().as_secs_f64(),
        raw_best = report.raw_best,
        greedy_best = report.greedy_best,
        loop_best = ?report.loop_best,
        loop_from = ?report.loop_from,
        loop_rounds = report.loop_rounds,
        final_best = report.final_best,
        rescore_ok = true,
        "msa descent"
    );
    Ok(reads)
}

impl<S: Sampler> MsaDescentSampler<S> {
    /// Sample one job on the inner sampler, then descend it.
    fn sample_descended(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        tiles: Option<Vec<[u32; 4]>>,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let started = Instant::now();
        let raw = self.inner.sample(graph, params)?;
        let pending = Pending {
            graph: graph.clone(),
            seed: params.seed,
            tiles,
            watermark: None,
        };
        finish(&pending, raw, started.elapsed().as_secs_f64(), None).map_err(|f| match f {
            Failure::Sample(e) => e,
            Failure::Cancelled => SampleError::DeviceBusy,
        })
    }

    /// The streaming pipeline behind both stream entry points.
    fn pipeline(
        &self,
        mut next: impl FnMut() -> Option<(StreamJob, Option<FamilyContext>)>,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
    ) {
        let depth = self.inner.stream_width().saturating_mul(2).max(1);
        let (inner_tx, inner_rx) = channel::<StreamJob>(depth);
        let (done_tx, done_rx) = channel::<StreamResult>(depth);
        let pending: Mutex<HashMap<Vec<u8>, VecDeque<Pending>>> = Mutex::default();
        let waiting = &pending;
        std::thread::scope(|scope| {
            let inner_cancel = cancel.clone();
            scope.spawn(move || self.inner.sample_stream(inner_rx, done_tx, inner_cancel));
            scope.spawn(move || post_process(done_rx, out, cancel, waiting));
            while let Some((job, family)) = next() {
                if out.is_closed() {
                    break;
                }
                let tiles = match family_tiles(&job.graph, family.as_ref()) {
                    Ok(tiles) => tiles,
                    Err(e) => {
                        let declined = StreamResult {
                            job_id: job.job_id,
                            outcome: StreamOutcome::Completed(Err(e)),
                            device_access_time_us: 0,
                        };
                        if out.blocking_send(declined).is_err() {
                            break;
                        }
                        continue;
                    }
                };
                // Record the job before the kernel can finish it.
                waiting
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(job.job_id.clone())
                    .or_default()
                    .push_back(Pending {
                        graph: job.graph.clone(),
                        seed: job.params.seed,
                        tiles,
                        watermark: job.watermark,
                    });
                if inner_tx.blocking_send(job).is_err() {
                    break;
                }
            }
            drop(inner_tx);
        });
    }
}

/// Descend each kernel result and forward it, in completion order.
fn post_process(
    mut done: Receiver<StreamResult>,
    out: &Sender<StreamResult>,
    cancel: &CancelToken,
    pending: &Mutex<HashMap<Vec<u8>, VecDeque<Pending>>>,
) {
    while let Some(result) = done.blocking_recv() {
        let job = {
            let mut map = pending.lock().unwrap_or_else(PoisonError::into_inner);
            let job = map.get_mut(&result.job_id).and_then(VecDeque::pop_front);
            if map.get(&result.job_id).is_some_and(VecDeque::is_empty) {
                map.remove(&result.job_id);
            }
            job
        };
        let outcome = match (result.outcome, job) {
            (StreamOutcome::Completed(Ok(raw)), Some(job)) => {
                // Microseconds to seconds; a job runs far below 2^52 us.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "kernel times are far below 2^52 microseconds"
                )]
                let kernel_s = result.device_access_time_us as f64 / 1e6;
                match finish(&job, raw, kernel_s, Some((cancel, job.watermark))) {
                    Ok(reads) => StreamOutcome::Completed(Ok(reads)),
                    Err(Failure::Cancelled) => StreamOutcome::Cancelled,
                    Err(Failure::Sample(e)) => StreamOutcome::Completed(Err(e)),
                }
            }
            (StreamOutcome::Completed(Ok(_)), None) => StreamOutcome::Completed(Err(
                SampleError::DeviceFault("a kernel result has no pending job".into()),
            )),
            (other, _) => other,
        };
        let forwarded = StreamResult {
            job_id: result.job_id,
            outcome,
            device_access_time_us: result.device_access_time_us,
        };
        if out.blocking_send(forwarded).is_err() {
            break;
        }
    }
}

impl<S: Sampler> Sampler for MsaDescentSampler<S> {
    fn generates_locally() -> bool {
        S::generates_locally()
    }

    /// Lease salts run on the inner sampler without descent.
    fn sample_lease(
        &self,
        lease: &quip_solver_core::Lease,
        topology: &quip_solver_core::TopologyView,
        params: &SampleParams,
        sink: &quip_solver_core::LeaseSink,
    ) -> Result<(), SampleError> {
        self.inner.sample_lease(lease, topology, params, sink)
    }

    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.sample_descended(graph, params, None)
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: MSA_DESCENT_SOLVER_ID,
            generated: &[],
            accepted: ACCEPTED,
        }
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&quip_solver_core::WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let tiles = family_tiles(graph, Some(family))?;
        self.sample_descended(graph, params, tiles)
    }

    fn sample_stream(
        &self,
        mut jobs: Receiver<StreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.pipeline(
            || jobs.blocking_recv().map(|job| (job, None)),
            &out,
            &cancel,
        );
    }

    fn sample_stream_warm(
        &self,
        mut jobs: Receiver<WarmStreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.pipeline(
            || jobs.blocking_recv().map(|item| (item.job, item.family)),
            &out,
            &cancel,
        );
    }

    fn stream_width(&self) -> usize {
        self.inner.stream_width()
    }

    fn declared_stream_width() -> u32 {
        S::declared_stream_width()
    }

    fn native_topology(&self) -> Option<quip_solver_core::NativeTopology> {
        self.inner.native_topology()
    }

    fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    fn should_throttle(&self) -> bool {
        self.inner.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        self.inner.max_reads()
    }

    fn apply_config(&self, backend_toml: &str) {
        self.inner.apply_config(backend_toml);
    }
}
