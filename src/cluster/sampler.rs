//! Otsuka on validated twisted-ice-dimer jobs, production CUDA SA otherwise.

use super::deep::{deep_seed, take_best, Deep, DeepConfig, DeepGroup, PROBE_EVERY};
use super::device::{ClusterDevice, ClusterDeviceError, ClusterKernel, ClusterRun};
use super::ease::{ease_seed, Ease, EaseConfig, EaseGroup};
use super::polish::{polish_reads, PolishConfig, PolishGraph, PolishOutcome};
use super::pool::{Pool, PoolConfig};
use super::stage1::{Stage1, Stage1Config, Stage1Outcome, Stage1Trace};
use super::stage2::{best_of, rerun_seed, Group, Payload, Stage2, Stage2Config};
use super::tuning::OtsukaTuning;
use super::{PrepareError, PreparedGraph, MAX_PACKED_NODES};
use crate::capacity::{
    cluster_footprint, ClusterShape, DeviceLimits, CLUSTER_DYNAMIC_PADDING,
    CLUSTER_SHARED_BYTES_PER_NODE, CLUSTER_STATIC_SHARED_BYTES,
};
use crate::cuda_device::CudaDevice;
use crate::family_screen::{cascade_keeps, instance_stage0_model, raw_stage0_model, Stage0Gate};
use crate::nvml_gov::UtilGovernor;
use crate::streaming::{build_beta_schedule, StallConfig};
use crate::{cuda_sa_identity, CudaSampler, KernelKind};
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext, FamilySource};
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::quip_proto::v1::Algorithm;
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{
    BackendIdentity, BetaSchedule, CancelToken, IsingGraph, SampleError, SampleParams, Sampler,
    SamplerResult, StreamJob, StreamOutcome, StreamResult, WarmStart, WarmStreamJob,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{channel, Receiver, Sender};

mod local;
use local::{fast_gen_enabled, screen_threads, Dest, FastGen, ScreenStats, ScreenTokens};

/// Stable binary identity.
pub const OTSUKA_SOLVER_ID: &str = "quip-cuda-otsuka/1";

/// Conservative headless budgets. Live sessions use the selected device facts.
pub const HEADLESS_LIMITS: DeviceLimits = DeviceLimits {
    shared_bytes_per_block: 48 << 10,
    shared_bytes_per_block_optin: 48 << 10,
    free_bytes: 64 << 20,
    sm_count: 1,
    threads_per_sm: 2048,
};

// A topology bound, enforced by the session for both family and plain jobs.
const MAX_DEGREE: usize = 128;
const MAX_READS: usize = 256;
const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// Conservative capacity for every admitted family shape, using CU-1 budgets.
/// All valid revision-1 parameters are supported within these topology limits.
#[must_use]
pub fn cuda_otsuka_identity(limits: &DeviceLimits, max_nodes: usize) -> BackendIdentity {
    let shared = limits
        .shared_bytes_per_block_optin
        .saturating_sub(CLUSTER_DYNAMIC_PADDING + CLUSTER_STATIC_SHARED_BYTES)
        / CLUSTER_SHARED_BYTES_PER_NODE;
    let mut nodes = shared.min(MAX_PACKED_NODES).min(max_nodes);
    while nodes > 0 {
        let shape = ClusterShape {
            nodes,
            tiles: nodes / 2, // Valid metadata permits at most two tiles per node.
            links: nodes / 2,
            weak_width: MAX_DEGREE.min(nodes - 1),
            all_width: MAX_DEGREE.min(nodes - 1),
            colors: nodes,
        };
        if cluster_footprint(shape, MAX_READS).total() <= limits.usable_bytes() {
            break;
        }
        nodes -= 1;
    }
    // Every residual is at most 999 milli with full motif weights.
    let edges = (nodes.saturating_mul(MAX_DEGREE.min(nodes.saturating_sub(1))) / 2)
        .min(i32::MAX.unsigned_abs() as usize / 999);
    let mut id = cuda_sa_identity(nodes, edges);
    id.algorithm = Algorithm::Otsuka;
    id.limits.max_degree = Some(u32::try_from(MAX_DEGREE).unwrap_or(u32::MAX));
    id
}

/// Host schedule shared by real and fake launch drivers.
#[doc(hidden)]
pub struct ClusterWork {
    /// Independent reads, in request order.
    pub reads: usize,
    pub(crate) seed: u64,
    pub(crate) betas: Vec<f32>,
    pub(crate) sweeps_per_beta: usize,
    /// Local build: the ladder is geometric, so chunked launches may replace
    /// it with the kernel's own per-sweep geometric ladder between its ends.
    pub(crate) geometric: bool,
}

impl ClusterWork {
    /// Build the shared CUDA ladder with the CPU Otsuka per-rung sweep cap.
    ///
    /// # Errors
    /// Returns a parameter error for unsupported counts or beta values.
    #[doc(hidden)]
    pub fn new(graph: &IsingGraph, params: &SampleParams) -> Result<Self, ClusterDeviceError> {
        if params.num_sweeps > i32::MAX.unsigned_abs() as usize
            || params.sweeps_per_beta > i32::MAX.unsigned_abs() as usize
        {
            return Err(ClusterDeviceError::JobParameter("sweep count exceeds i32"));
        }
        if params.num_reads == 0
            || params.num_reads > MAX_READS
            || params.beta_range.is_some_and(|(hot, cold)| {
                !hot.is_finite() || !cold.is_finite() || hot <= 0.0 || cold <= 0.0
            })
        {
            return Err(ClusterDeviceError::JobParameter(
                "invalid read count or beta range",
            ));
        }
        let per_beta = if params.num_sweeps == 0 {
            params.sweeps_per_beta.max(1)
        } else {
            params.sweeps_per_beta.max(1).min(params.num_sweeps)
        };
        let (betas, sweeps_per_beta) = build_beta_schedule(
            graph,
            params.num_sweeps,
            per_beta,
            params.beta_range,
            params.schedule,
            1.0,
        );
        let total = betas.len().checked_mul(sweeps_per_beta);
        if total.is_none_or(|n| n == 0 || n > i32::MAX.unsigned_abs() as usize)
            || betas.iter().any(|b| !b.is_finite() || *b <= 0.0)
        {
            return Err(ClusterDeviceError::JobParameter(
                "invalid read count or beta ladder",
            ));
        }
        Ok(Self {
            reads: params.num_reads,
            seed: params.seed,
            betas,
            sweeps_per_beta,
            geometric: params.schedule == BetaSchedule::Geometric,
        })
    }

    /// One synchronized sweep per callback, then 256 single greedy passes.
    /// A converged greedy state stays unchanged, including its RNG state.
    /// During launches, cancellation waits at most one sweep and its synchronization.
    ///
    /// # Errors
    /// Returns cancellation or the first launch failure, without partial reads.
    pub fn for_each_launch(
        &self,
        cancelled: &dyn Fn() -> bool,
        mut launch: impl FnMut(usize, f32, bool) -> Result<(), ClusterDeviceError>,
    ) -> Result<(), ClusterDeviceError> {
        self.for_each_chunk(cancelled, 1, |start, _, beta, descent| {
            launch(start, beta, descent)
        })
    }

    pub(crate) fn for_each_chunk(
        &self,
        stopped: &dyn Fn() -> bool,
        chunk: usize,
        mut launch: impl FnMut(usize, usize, f32, bool) -> Result<(), ClusterDeviceError>,
    ) -> Result<(), ClusterDeviceError> {
        if chunk == 0 {
            return Err(ClusterDeviceError::Parameter("chunk must be positive"));
        }
        let mut sweep = 0;
        for &beta in &self.betas {
            let mut remaining = self.sweeps_per_beta;
            while remaining > 0 {
                checkpoint(stopped)?;
                let count = chunk.min(remaining);
                launch(sweep, count, beta, false)?;
                sweep += count;
                remaining -= count;
            }
        }
        // Production yields between every greedy pass. The test oracle runs
        // the original kernel's single 256-pass descent instead.
        let passes = if chunk == 1 { 1 } else { 256 };
        for _ in 0..256 / passes {
            checkpoint(stopped)?;
            launch(0, passes, 1.0, true)?;
        }
        Ok(())
    }
}

fn checkpoint(cancelled: &dyn Fn() -> bool) -> Result<(), ClusterDeviceError> {
    if cancelled() {
        Err(ClusterDeviceError::Cancelled)
    } else {
        Ok(())
    }
}

/// Minimal device boundary for the headless session tests.
#[doc(hidden)]
pub trait ClusterBackend: Send + Sync + 'static {
    /// Execute a prepared job without dropping or sorting reads.
    ///
    /// # Errors
    /// Returns admission, cancellation, or device errors.
    fn run(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError>;

    /// Local build (all7, deep lane): [`Self::run`] with a probe of the
    /// lowest lineage best (lane energy, milli) about every second; `true`
    /// ends the anneal there and returns its reads as a completed run. The
    /// default never probes (a backend without checkpoints runs to the end).
    ///
    /// # Errors
    /// As [`Self::run`].
    fn run_probed(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        _probe: &dyn Fn(i64) -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run(graph, work, stopped, cancelled)
    }
}

/// Preloaded cluster function on the selected CUDA device.
pub struct GpuCluster {
    kernel: ClusterKernel,
    tuning: OtsukaTuning,
}

impl GpuCluster {
    /// Local build: load one cluster lane (its own CUDA stream) with explicit
    /// knobs, for benchmarks outside the session.
    ///
    /// # Errors
    /// Returns cluster compilation or driver errors.
    #[doc(hidden)]
    pub fn load(device_index: usize, tuning: OtsukaTuning) -> Result<Self, ClusterDeviceError> {
        Ok(Self {
            kernel: ClusterKernel::load_variant(
                device_index,
                tuning.fast,
                tuning.sweep_v2,
                tuning.device_pa,
                tuning.sweep_v3,
                tuning.sweep_v4,
                tuning.sweep_v5,
                tuning.sweep_v6,
                tuning.union_sv,
            )?,
            tuning,
        })
    }
}

impl ClusterBackend for GpuCluster {
    fn run(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        checkpoint(stopped)?;
        ClusterDevice::open_with_kernel(graph, &self.kernel)?.run_work_tuned(
            work,
            &self.tuning,
            stopped,
            cancelled,
        )
    }

    fn run_probed(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: &dyn Fn(i64) -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_probed_every(graph, work, stopped, cancelled, probe, PROBE_EVERY)
    }
}

impl GpuCluster {
    /// Local build (all7): [`ClusterBackend::run_probed`] probing at the
    /// first checkpoint after each `every`, for GPU tests (`Duration::ZERO`
    /// probes at every checkpoint).
    ///
    /// # Errors
    /// As [`ClusterBackend::run`].
    #[doc(hidden)]
    pub fn run_probed_every(
        &self,
        graph: PreparedGraph,
        work: &ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: &dyn Fn(i64) -> bool,
        every: Duration,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        checkpoint(stopped)?;
        ClusterDevice::open_with_kernel(graph, &self.kernel)?.run_work_probed(
            work,
            &self.tuning,
            stopped,
            cancelled,
            Some(probe),
            every,
        )
    }
}

/// Serial family jobs alongside one persistent production SA stream.
pub struct OtsukaSampler<D = GpuCluster, S = CudaSampler> {
    cluster: D,
    sa: S,
    limits: DeviceLimits,
    /// Local build: further cluster lanes, one CUDA stream each, for
    /// `QUIP_OTSUKA_CONCURRENCY`. Empty keeps the stock serial loop.
    extra: Vec<D>,
    tuning: OtsukaTuning,
    /// Local build: the stage-0 screen (`QUIP_OTSUKA_STAGE0_KEEP` below 1).
    stage0: Option<Mutex<Stage0Gate>>,
    /// Local build: model 3's model-1 prefilter gate, in front of `stage0`.
    stage0_pre: Option<Mutex<Stage0Gate>>,
    /// Local build: the stage-1 short-anneal screen (`QUIP_OTSUKA_STAGE1_FRAC`).
    stage1: Option<Stage1>,
    /// Local build: the host CPU polish of lease reads (`QUIP_OTSUKA_CPU_POLISH`).
    polish: Option<PolishConfig>,
    /// Local build: polish counters for the periodic summary line.
    polish_stats: PolishStats,
    /// Local build: the lanes' work queue while a stream session runs, for
    /// locally generated lease salts (`QUIP_OTSUKA_SCREEN_THREADS` above 1).
    lane_queue: Mutex<Option<SyncSender<LaneItem>>>,
    /// Local build: screen and lane counters for the periodic screen line.
    screen_stats: ScreenStats,
    /// Local build: CPU slots shared by every lease's screen workers.
    screen_tokens: ScreenTokens,
    /// Local build (all5): stage 2 (`QUIP_OTSUKA_STAGE2_Q`, local path only).
    stage2: Option<Stage2>,
    /// Local build (all5): stage-2 reruns, taken by a lane before any fresh
    /// screened salt.
    stage2_reruns: Mutex<VecDeque<LaneItem>>,
    /// Local build (all5): the held stage-2 groups, for the cap and the
    /// release watcher.
    stage2_groups: Mutex<Vec<Weak<S2Group>>>,
    /// Local build (all6): the best-first pool of screened local lease
    /// salts (`QUIP_OTSUKA_POOL`, local path with stage 0 on only).
    pool: Option<Pool<LaneItem>>,
    /// Local build (all6): the cached dimer generator for the local screen
    /// (`QUIP_OTSUKA_FAST_GEN`), built per topology and family parameters.
    fast_gen: Option<Mutex<Option<Arc<FastGen>>>>,
    /// Local build (all7): the late-round deep lane (`QUIP_OTSUKA_DEEP_LANES`,
    /// local path only).
    deep: Option<Deep>,
    /// Local build (all7): long runs waiting for a deep lane.
    deep_queue: Mutex<VecDeque<LaneItem>>,
    /// Local build (all7): the held deep groups, for the cap and the watcher.
    deep_groups: Mutex<Vec<Weak<DGroup>>>,
    /// Local build (all12): the ease-race booster (`QUIP_OTSUKA_EASE_BOOST`,
    /// with the deep lane only).
    ease: Option<Ease>,
    /// Local build (all12): boost runs waiting for a deep lane (taken before
    /// the deep lane's own long runs).
    ease_queue: Mutex<VecDeque<LaneItem>>,
    /// Local build (all12): the held ease groups, for the watcher.
    ease_groups: Mutex<Vec<Weak<EGroup>>>,
}

/// Local build (all7): a deep-lane group over the sampler's own types.
type DGroup = DeepGroup<StreamJob, FamilyRef, Dest>;

/// Local build (all12): an ease group over the sampler's own types.
type EGroup = EaseGroup<StreamJob, FamilyRef, Dest>;

/// Local build (all5): a stage-2 group over the sampler's own job, family
/// and destination types.
type S2Group = Group<StreamJob, FamilyRef, Dest>;

/// Local build (all5): how often the stage-2 watcher checks held groups
/// against the live target.
const STAGE2_WATCH: Duration = Duration::from_millis(50);

/// Local build (all5): how long an idle lane waits on the fresh queue before
/// it looks at the stage-2 reruns again (stage 2 on only).
const STAGE2_LANE_POLL: Duration = Duration::from_millis(20);

/// Local build: the family facts a lane needs. Core's [`FamilyContext`] on
/// the session path. On the local path (`QUIP_OTSUKA_SCREEN_THREADS` above
/// one) the instance this sampler generated, since only core can build a
/// `FamilyContext`.
enum FamilyRef {
    Core(FamilyContext),
    Local {
        instance: Arc<Instance>,
        contract: &'static Contract,
    },
}

impl FamilyRef {
    fn instance(&self) -> &Instance {
        match self {
            Self::Core(family) => family.instance(),
            Self::Local { instance, .. } => instance,
        }
    }

    fn contract(&self) -> &'static Contract {
        match self {
            Self::Core(family) => family.contract(),
            Self::Local { contract, .. } => contract,
        }
    }

    /// A lease salt (interchangeable, so screened, staged and polished).
    fn is_lease(&self) -> bool {
        match self {
            Self::Core(family) => matches!(family.source(), FamilySource::Lease { .. }),
            Self::Local { .. } => true,
        }
    }
}

/// Local build: one validated family job for a lane, and where its result goes.
struct LaneItem {
    job: StreamJob,
    family: FamilyRef,
    stage0: Option<f32>,
    dest: Dest,
    /// Local build (all5): a stage-2 rerun (its group and 1-based index).
    /// Its `dest` is never used: the group delivers the first run's.
    rerun: Option<(Arc<S2Group>, u64)>,
    /// Local build (all7): a deep lane's long run (its group). Its `dest` is
    /// never used either.
    deep: Option<Arc<DGroup>>,
    /// Local build (all12): an ease boost run (its group). Its `dest` is
    /// never used either.
    ease: Option<Arc<EGroup>>,
}

/// Local build: one completed lease job waiting for the CPU polish.
struct PolishJob {
    job: StreamJob,
    family: FamilyRef,
    result: StreamResult,
    dest: Dest,
}

/// Local build: polish counters, summarized every [`POLISH_SUMMARY_EVERY`] jobs.
#[derive(Debug, Default)]
struct PolishStats {
    jobs: AtomicU64,
    improved: AtomicU64,
    gain_milli: AtomicU64,
    micros: AtomicU64,
    saturated: AtomicU64,
    cancelled: AtomicU64,
    mismatched: AtomicU64,
}

/// Local build: log a polish summary line every this many polished jobs.
const POLISH_SUMMARY_EVERY: u64 = 200;

/// Local build: `XORed` into the job seed for the polish draws.
const POLISH_SEED_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;

/// Local build: the stage-0 gates' verdict on one instance.
enum Gate0 {
    /// Solve it, with its score when the screen scored it.
    Pass(Option<f32>),
    /// Screened out.
    Reject,
}

/// Local build: the stage-0 decision for one validated family job.
enum Stage0 {
    /// Solve it. Carries its score when the screen is on, for the debug log.
    Pass(Option<f32>),
    /// Screened out: completed with no reads, so its lease counts the salt
    /// as done and nothing is sent.
    Reject(StreamResult),
}

fn stage0_gate(tuning: &OtsukaTuning) -> Option<Mutex<Stage0Gate>> {
    if tuning.stage0_keep >= 1.0 {
        return None;
    }
    if tuning.stage0_model == 3 {
        // Model 3: model 2 keeps its share of the prefilter's survivors.
        let (_, main) = cascade_keeps(tuning.stage0_keep, tuning.stage0_pre);
        return (main < 1.0).then(|| Mutex::new(Stage0Gate::new(main)));
    }
    Some(Mutex::new(Stage0Gate::new(tuning.stage0_keep)))
}

/// Local build: model 3's model-1 prefilter gate (screen on, model 3 only).
fn stage0_pre_gate(tuning: &OtsukaTuning) -> Option<Mutex<Stage0Gate>> {
    if tuning.stage0_keep >= 1.0 || tuning.stage0_model != 3 {
        return None;
    }
    let (pre, _) = cascade_keeps(tuning.stage0_keep, tuning.stage0_pre);
    (pre < 1.0).then(|| Mutex::new(Stage0Gate::new(pre)))
}

/// Local build (all6): the pool from the environment. It needs the local
/// lease path and a stage-0 score to rank by, so without either it stays off
/// (said once at startup).
fn pool_from_env(tuning: &OtsukaTuning) -> Option<Pool<LaneItem>> {
    let config = PoolConfig::from_env()?;
    if screen_threads() <= 1 || tuning.stage0_keep >= 1.0 {
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_POOL needs QUIP_OTSUKA_SCREEN_THREADS above 1 and QUIP_OTSUKA_STAGE0_KEEP below 1; pool off"
        );
        return None;
    }
    eprintln!(
        "quip-otsuka: pool on: capacity={} max_age_ms={}",
        config.capacity,
        config.max_age.as_millis()
    );
    Some(Pool::new(config))
}

/// Local build (all7): the deep lane from the environment. It needs the
/// local lease path and at least one lane besides the deep ones; otherwise
/// it stays off (said once at startup).
fn deep_from_env(tuning: &OtsukaTuning) -> Option<Deep> {
    let config = DeepConfig::from_env()?;
    if screen_threads() <= 1 || config.lanes >= tuning.concurrency.max(1) {
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_DEEP_LANES needs QUIP_OTSUKA_SCREEN_THREADS above 1 and fewer deep lanes than QUIP_OTSUKA_CONCURRENCY; deep lane off"
        );
        return None;
    }
    eprintln!(
        "quip-otsuka: deep lane on: lanes={} mult={} start_s={} q={} record={} guard_milli={} release_milli={} after_stage2={}",
        config.lanes,
        config.mult,
        config.start.as_secs_f64(),
        config.q,
        config.record,
        config.guard_milli,
        config.release_milli,
        config.after_stage2
    );
    // all9: the v2 knobs, said only when one is not the all8 behaviour.
    if config.slack_milli != 0 || config.log_drops || config.adaptive.is_some() {
        eprintln!(
            "quip-otsuka: deep lane v2: slack_milli={} log_drops={} adaptive={}",
            config.slack_milli,
            config.log_drops,
            config.adaptive.map_or_else(
                || "off".to_owned(),
                |a| format!(
                    "lmax={} ramp0_s={} ramp1_s={} late_s={} horizon_s={} mult_min={} mult_max={} gap_far_milli={} step_s={}",
                    config.lanes,
                    a.ramp0.as_secs_f64(),
                    a.ramp1.as_secs_f64(),
                    a.late.as_secs_f64(),
                    a.horizon.as_secs_f64(),
                    a.mult_min,
                    a.mult_max,
                    a.gap_far_milli,
                    a.step.as_secs_f64()
                )
            )
        );
    }
    Some(Deep::new(config, Instant::now()))
}

/// Local build (all12): the ease booster from the environment. It needs the
/// deep lane (its lanes, probe and run-time estimate); otherwise it stays
/// off (said once at startup).
fn ease_from_env(deep: Option<&Deep>) -> Option<Ease> {
    let config = EaseConfig::from_env(deep.map_or(0, |d| d.config().lanes))?;
    eprintln!(
        "quip-otsuka: ease boost on: t0_s={} gap_milli={} lanes={} end_s={} post_s={} release_milli={} safe_s={} mult_max={} max_runs={}",
        config.t0.as_secs_f64(),
        config.gap_milli,
        config.lanes,
        config.end.as_secs_f64(),
        config.post.as_secs_f64(),
        config.release_milli,
        config.safe.as_secs_f64(),
        config.mult_max,
        config.max_runs
    );
    Some(Ease::new(config))
}

/// Local build (all5): stage 2 from the environment. It needs the local
/// lease path, so with `QUIP_OTSUKA_SCREEN_THREADS` at 1 it stays off (said
/// once at startup).
fn stage2_from_env() -> Option<Stage2> {
    let config = Stage2Config::from_env()?;
    if screen_threads() <= 1 {
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_STAGE2_Q needs QUIP_OTSUKA_SCREEN_THREADS above 1; stage 2 off"
        );
        return None;
    }
    eprintln!(
        "quip-otsuka: stage 2 on: q={} k={} guard_milli={} max_active={} release_milli={}",
        config.q, config.k, config.guard_milli, config.max_active, config.release_milli
    );
    Some(Stage2::new(config))
}

impl OtsukaSampler {
    /// Bind the selected device, governor, and stall policy.
    ///
    /// # Errors
    /// Returns cluster compilation or driver errors before streaming starts.
    pub fn new(
        device: CudaDevice,
        gov: UtilGovernor,
        stall: StallConfig,
    ) -> Result<Self, ClusterDeviceError> {
        let tuning = OtsukaTuning::from_env();
        let lane = || -> Result<GpuCluster, ClusterDeviceError> {
            Ok(GpuCluster {
                kernel: ClusterKernel::load_variant(
                    device.device_index,
                    tuning.fast,
                    tuning.sweep_v2,
                    tuning.device_pa,
                    tuning.sweep_v3,
                    tuning.sweep_v4,
                    tuning.sweep_v5,
                    tuning.sweep_v6,
                    tuning.union_sv,
                )?,
                tuning,
            })
        };
        let cluster = lane()?;
        let extra = (1..tuning.concurrency)
            .map(|_| lane())
            .collect::<Result<Vec<_>, _>>()?;
        let limits = device.limits;
        let mut sampler = Self {
            cluster,
            sa: CudaSampler::new(device, gov, KernelKind::Sa, stall),
            limits,
            extra,
            tuning,
            stage0: stage0_gate(&tuning),
            stage0_pre: stage0_pre_gate(&tuning),
            stage1: Stage1Config::from_env().map(Stage1::new),
            polish: PolishConfig::from_env(),
            polish_stats: PolishStats::default(),
            lane_queue: Mutex::new(None),
            screen_stats: ScreenStats::default(),
            screen_tokens: ScreenTokens::new(screen_threads()),
            stage2: stage2_from_env(),
            stage2_reruns: Mutex::new(VecDeque::new()),
            stage2_groups: Mutex::new(Vec::new()),
            pool: pool_from_env(&tuning),
            fast_gen: fast_gen_enabled().then(|| Mutex::new(None)),
            deep: deep_from_env(&tuning),
            deep_queue: Mutex::new(VecDeque::new()),
            deep_groups: Mutex::new(Vec::new()),
            ease: None,
            ease_queue: Mutex::new(VecDeque::new()),
            ease_groups: Mutex::new(Vec::new()),
        };
        // all12: after the deep lane (its startup lines come first).
        sampler.ease = ease_from_env(sampler.deep.as_ref());
        Ok(sampler)
    }
}

impl<D: ClusterBackend, S: Sampler> OtsukaSampler<D, S> {
    /// Inject a fake device and fallback without opening CUDA.
    #[doc(hidden)]
    #[must_use]
    pub fn with_backends(cluster: D, sa: S, limits: DeviceLimits) -> Self {
        Self {
            cluster,
            sa,
            limits,
            extra: Vec::new(),
            tuning: OtsukaTuning::default(),
            stage0: None,
            stage0_pre: None,
            stage1: None,
            polish: None,
            polish_stats: PolishStats::default(),
            lane_queue: Mutex::new(None),
            screen_stats: ScreenStats::default(),
            screen_tokens: ScreenTokens::new(screen_threads()),
            stage2: None,
            stage2_reruns: Mutex::new(VecDeque::new()),
            stage2_groups: Mutex::new(Vec::new()),
            pool: None,
            fast_gen: None,
            deep: None,
            deep_queue: Mutex::new(VecDeque::new()),
            deep_groups: Mutex::new(Vec::new()),
            ease: None,
            ease_queue: Mutex::new(VecDeque::new()),
            ease_groups: Mutex::new(Vec::new()),
        }
    }

    /// Local build: fake lanes for the concurrent family loop in tests.
    #[doc(hidden)]
    #[must_use]
    pub fn with_lanes(cluster: D, extra: Vec<D>, sa: S, limits: DeviceLimits) -> Self {
        Self {
            cluster,
            sa,
            limits,
            extra,
            tuning: OtsukaTuning::default(),
            stage0: None,
            stage0_pre: None,
            stage1: None,
            polish: None,
            polish_stats: PolishStats::default(),
            lane_queue: Mutex::new(None),
            screen_stats: ScreenStats::default(),
            screen_tokens: ScreenTokens::new(screen_threads()),
            stage2: None,
            stage2_reruns: Mutex::new(VecDeque::new()),
            stage2_groups: Mutex::new(Vec::new()),
            pool: None,
            fast_gen: None,
            deep: None,
            deep_queue: Mutex::new(VecDeque::new()),
            deep_groups: Mutex::new(Vec::new()),
            ease: None,
            ease_queue: Mutex::new(VecDeque::new()),
            ease_groups: Mutex::new(Vec::new()),
        }
    }

    /// Local build: turn on the stage-0 screen for tests (`keep` below 1).
    #[doc(hidden)]
    #[must_use]
    pub fn with_stage0_keep(mut self, keep: f64) -> Self {
        self.tuning.stage0_keep = keep;
        self.stage0 = stage0_gate(&self.tuning);
        self.stage0_pre = stage0_pre_gate(&self.tuning);
        self
    }

    /// Local build: pick the stage-0 model for tests (1 lean, 2 contracted
    /// cycles, 3 model-1 prefilter then model 2), as
    /// `QUIP_OTSUKA_STAGE0_MODEL` does.
    #[doc(hidden)]
    #[must_use]
    pub fn with_stage0_model(mut self, model: u8) -> Self {
        self.tuning.stage0_model = model;
        self.stage0 = stage0_gate(&self.tuning);
        self.stage0_pre = stage0_pre_gate(&self.tuning);
        self
    }

    /// Local build: model 3's prefilter fraction for tests, as
    /// `QUIP_OTSUKA_STAGE0_PRE` does.
    #[doc(hidden)]
    #[must_use]
    pub fn with_stage0_pre(mut self, pre: f64) -> Self {
        self.tuning.stage0_pre = pre;
        self.stage0 = stage0_gate(&self.tuning);
        self.stage0_pre = stage0_pre_gate(&self.tuning);
        self
    }

    /// Local build: turn on the stage-1 screen for tests.
    #[doc(hidden)]
    #[must_use]
    pub fn with_stage1(mut self, config: Stage1Config) -> Self {
        self.stage1 = Some(Stage1::new(config));
        self
    }

    /// Local build (all5): turn on stage 2 for tests (it acts on the local
    /// lease path only).
    #[doc(hidden)]
    #[must_use]
    pub fn with_stage2(mut self, config: Stage2Config) -> Self {
        self.stage2 = Some(Stage2::new(config));
        self
    }

    /// Local build (all6): turn on the best-first pool for tests (it acts on
    /// the local lease path with stage 0 on only).
    #[doc(hidden)]
    #[must_use]
    pub fn with_pool(mut self, config: PoolConfig) -> Self {
        self.pool = Some(Pool::new(config));
        self
    }

    /// Local build (all7): turn on the deep lane for tests (local lease path
    /// only; `config.lanes` must be below the lane count).
    #[doc(hidden)]
    #[must_use]
    pub fn with_deep(mut self, config: DeepConfig) -> Self {
        self.deep = Some(Deep::new(config, Instant::now()));
        self
    }

    /// Local build (all12): turn on the ease booster for tests (with the
    /// deep lane on).
    #[doc(hidden)]
    #[must_use]
    pub fn with_ease(mut self, config: EaseConfig) -> Self {
        self.ease = Some(Ease::new(config));
        self
    }

    /// Local build (all6): screen local lease salts with the cached dimer
    /// generator, for tests (`QUIP_OTSUKA_FAST_GEN`).
    #[doc(hidden)]
    #[must_use]
    pub fn with_fast_gen(mut self) -> Self {
        self.fast_gen = Some(Mutex::new(None));
        self
    }

    /// Local build: turn on the CPU polish for tests.
    #[doc(hidden)]
    #[must_use]
    pub fn with_polish(mut self, config: PolishConfig) -> Self {
        self.polish = Some(config);
        self
    }

    /// Local build: start the polish pool in `scope`, when polish is on.
    /// Each pool thread takes a queued lease result, polishes it, and sends
    /// it to `out`. The threads stop when every sender is dropped and the
    /// queue is drained.
    fn start_polish<'scope, 'env>(
        &'env self,
        scope: &'scope std::thread::Scope<'scope, 'env>,
        out: &'env Sender<StreamResult>,
        cancel: &'env CancelToken,
    ) -> Option<SyncSender<PolishJob>> {
        let config = self.polish?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<PolishJob>(config.queue.max(1));
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..config.threads.max(1) {
            let rx = Arc::clone(&rx);
            scope.spawn(move || loop {
                let next = rx.lock().unwrap_or_else(PoisonError::into_inner).recv();
                let Ok(item) = next else { break };
                let PolishJob {
                    job,
                    family,
                    result,
                    dest,
                } = self.polish_job(item, &config, cancel);
                if !self.send_out(job, family, result, dest, out) {
                    break;
                }
            });
        }
        Some(tx)
    }

    /// Local build: send a family result, through the polish pool when it
    /// applies (a completed lease result with reads, at or below the cutoff,
    /// and room in the queue). Returns false when `out` is closed.
    fn deliver(
        &self,
        pool: Option<&SyncSender<PolishJob>>,
        job: StreamJob,
        family: FamilyRef,
        result: StreamResult,
        dest: Dest,
        out: &Sender<StreamResult>,
    ) -> bool {
        let wanted = match (pool, self.polish.as_ref(), &result.outcome) {
            (Some(_), Some(config), StreamOutcome::Completed(Ok(reads))) => {
                family.is_lease()
                    && reads
                        .iter()
                        .map(|r| r.energy_milli)
                        .min()
                        .is_some_and(|best| config.wants(best))
            }
            _ => false,
        };
        if let (true, Some(tx)) = (wanted, pool) {
            match tx.try_send(PolishJob {
                job,
                family,
                result,
                dest,
            }) {
                Ok(()) => return true,
                Err(TrySendError::Full(item) | TrySendError::Disconnected(item)) => {
                    self.polish_stats.saturated.fetch_add(1, Ordering::Relaxed);
                    return self.send_out(item.job, item.family, item.result, item.dest, out);
                }
            }
        }
        self.send_out(job, family, result, dest, out)
    }

    /// Local build: the last step of every family result: sort its reads
    /// and send it. all12: with the ease booster on, a local lease result
    /// may be held for a boost instead ([`Self::maybe_ease`]).
    fn send_out(
        &self,
        job: StreamJob,
        family: FamilyRef,
        mut result: StreamResult,
        dest: Dest,
        out: &Sender<StreamResult>,
    ) -> bool {
        sort_reads(&mut result);
        match self.maybe_ease(job, family, result, dest) {
            Some((_, _, result, dest)) => dest.send(result, out),
            None => true, // held in an ease group
        }
    }

    /// Local build (all12): hold a local lease result that is the round's
    /// best and just short of the live target, and queue boost runs of its
    /// instance for the deep lanes (`None`); else hand it back. See
    /// [`super::ease`].
    fn maybe_ease(
        &self,
        job: StreamJob,
        family: FamilyRef,
        result: StreamResult,
        dest: Dest,
    ) -> Option<(StreamJob, FamilyRef, StreamResult, Dest)> {
        let (Some(ease), Some(deep)) = (
            self.ease.as_ref().filter(|e| e.active()),
            self.deep.as_ref(),
        ) else {
            return Some((job, family, result, dest));
        };
        let (Dest::Local(reply), FamilyRef::Local { instance, contract }) = (&dest, &family) else {
            return Some((job, family, result, dest));
        };
        if reply.stop_flag().load(Ordering::Relaxed) {
            return Some((job, family, result, dest));
        }
        let Some(best) = best_of(&result) else {
            return Some((job, family, result, dest));
        };
        let offset = reply.offset();
        let reported = best.saturating_add(offset);
        let cap = (i32::MAX.unsigned_abs() as usize)
            .checked_div(job.params.num_sweeps)
            .unwrap_or(0);
        let now = Instant::now();
        let grant = match ease.consider(deep, reported, job.params.seed, cap, now) {
            Ok(grant) => grant,
            Err(why) => {
                tracing::debug!(best = reported, ?why, "otsuka ease: sent");
                return Some((job, family, result, dest));
            }
        };
        let plan = grant.plan;
        let (instance, contract, stop) = (Arc::clone(instance), *contract, reply.stop_flag());
        let runs: Vec<StreamJob> = (0..plan.runs as u64)
            .map(|k| StreamJob {
                job_id: job.job_id.clone(),
                graph: job.graph.clone(),
                params: SampleParams {
                    seed: ease_seed(job.params.seed, grant.first + k),
                    num_sweeps: job.params.num_sweeps * plan.mult,
                    ..job.params.clone()
                },
                watermark: job.watermark,
            })
            .collect();
        let epoch = deep.round_now(now).1;
        let group = Arc::new(EGroup::new(
            reported,
            offset,
            grant.elapsed,
            epoch,
            plan.mult,
            plan.runs,
            stop,
            grant.retired,
            Payload {
                job,
                family,
                dest,
                result,
            },
        ));
        {
            let mut groups = self
                .ease_groups
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
            groups.push(Arc::downgrade(&group));
        }
        tracing::info!(
            round_s = grant.elapsed.as_secs_f64(),
            best = reported,
            target = grant.target,
            gap_milli = plan.gap,
            mult = plan.mult,
            lanes = plan.runs,
            run_s = deep.run_estimate().as_secs_f64(),
            eta_s = plan.eta.as_secs_f64(),
            seed_index = grant.first,
            retarget = grant.retarget,
            "otsuka ease boost"
        );
        let mut queue = self
            .ease_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for run in runs {
            queue.push_back(LaneItem {
                job: run,
                family: FamilyRef::Local {
                    instance: Arc::clone(&instance),
                    contract,
                },
                stage0: None,
                dest: Dest::Session, // unused: the group delivers
                rerun: None,
                deep: None,
                ease: Some(Arc::clone(&group)),
            });
        }
        None
    }

    /// Local build (all12): run one boost anneal on a deep lane, probing its
    /// lineage best about every second, and deliver its group when this
    /// run ends it. A skipped run (retired, harvested, delivered or stopped
    /// group) never launches. False when `out` is closed.
    #[allow(clippy::too_many_arguments)]
    fn run_ease(
        &self,
        lane: &D,
        ease: &Ease,
        deep: &Deep,
        item: &LaneItem,
        group: &EGroup,
        pool: Option<&SyncSender<PolishJob>>,
        out: &Sender<StreamResult>,
    ) -> bool {
        let run = if group.begin() {
            let began = Instant::now();
            let cancelled = || group.stopped();
            let probe = |device_best: i64| group.probe(ease, deep, device_best);
            let result = Self::finish_with(&item.job, &cancelled, || {
                let prepared = self.prepared(
                    &item.job.graph,
                    &item.job.params,
                    item.family.instance(),
                    item.family.contract(),
                )?;
                self.reads_on_probed(
                    lane,
                    &item.job.graph,
                    &item.job.params,
                    prepared,
                    &cancelled,
                    Some(&probe),
                )
                .map_err(sample_error)
            });
            Some((result, began.elapsed()))
        } else {
            None
        };
        match group.finish_run(ease, deep, run) {
            Some(p) => self.deliver(pool, p.job, p.family, p.result, p.dest, out),
            None => true,
        }
    }

    /// Local build (all12): the held ease groups the watcher should deliver
    /// now. Groups of a stopped lease are dropped on the way.
    fn ease_due(&self, ease: &Ease, deep: &Deep) -> Vec<Payload<StreamJob, FamilyRef, Dest>> {
        let held: Vec<Arc<EGroup>> = {
            let mut groups = self
                .ease_groups
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
            groups.iter().filter_map(Weak::upgrade).collect()
        };
        let now = Instant::now();
        ease.tick(deep, now);
        held.iter()
            .filter_map(|g| g.release_due(ease, deep, now))
            .collect()
    }

    /// Local build: polish one queued lease result on a pool thread. A
    /// watermark cancelled before or during the polish turns the outcome
    /// into `Cancelled`, as the lane's `finish` does for a job cancelled
    /// while it ran. Any other failure sends the reads unchanged.
    fn polish_job(
        &self,
        item: PolishJob,
        config: &PolishConfig,
        cancel: &CancelToken,
    ) -> PolishJob {
        let PolishJob {
            job,
            family,
            mut result,
            dest,
        } = item;
        let cancelled = || dest.stopped(cancel, job.watermark);
        if cancelled() {
            self.polish_stats.cancelled.fetch_add(1, Ordering::Relaxed);
            result.outcome = StreamOutcome::Cancelled;
            return PolishJob {
                job,
                family,
                result,
                dest,
            };
        }
        let StreamOutcome::Completed(Ok(reads)) = &mut result.outcome else {
            return PolishJob {
                job,
                family,
                result,
                dest,
            };
        };
        let graph = match PolishGraph::from_instance(
            family.instance(),
            family.contract(),
            self.tuning.weights(),
        ) {
            Ok(graph) => graph,
            Err(e) => {
                tracing::warn!(error = %e, "otsuka polish: host graph refused; reads sent unpolished");
                return PolishJob {
                    job,
                    family,
                    result,
                    dest,
                };
            }
        };
        let graph_ref = &job.graph;
        let score = |s: &[i8]| energy_milli(s, &graph_ref.h, &graph_ref.j, &graph_ref.edges);
        let started = Instant::now();
        let outcome = polish_reads(
            &graph,
            reads,
            config,
            job.params.seed ^ POLISH_SEED_SALT,
            &score,
            &cancelled,
        );
        let micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        if cancelled() || outcome.is_none() {
            self.polish_stats.cancelled.fetch_add(1, Ordering::Relaxed);
            result.outcome = StreamOutcome::Cancelled;
            return PolishJob {
                job,
                family,
                result,
                dest,
            };
        }
        if let Some(outcome) = outcome {
            self.record_polish(&job, &outcome, micros);
        }
        PolishJob {
            job,
            family,
            result,
            dest,
        }
    }

    /// Local build: count one polished job; debug line per job, info summary
    /// every [`POLISH_SUMMARY_EVERY`] jobs.
    fn record_polish(&self, job: &StreamJob, outcome: &PolishOutcome, micros: u64) {
        let st = &self.polish_stats;
        let gain = u64::try_from(outcome.before.saturating_sub(outcome.after)).unwrap_or(0);
        st.improved
            .fetch_add(u64::from(gain > 0), Ordering::Relaxed);
        st.gain_milli.fetch_add(gain, Ordering::Relaxed);
        st.micros.fetch_add(micros, Ordering::Relaxed);
        st.mismatched
            .fetch_add(outcome.mismatched as u64, Ordering::Relaxed);
        let jobs = st.jobs.fetch_add(1, Ordering::Relaxed) + 1;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let id: String = job.job_id.iter().fold(String::new(), |mut s, b| {
                use std::fmt::Write;
                write!(s, "{b:02x}").expect("writing to a String cannot fail");
                s
            });
            tracing::debug!(
                job = %id,
                before = outcome.before,
                after = outcome.after,
                improved_reads = outcome.improved,
                ms = micros / 1000,
                "otsuka polish"
            );
        }
        if jobs.is_multiple_of(POLISH_SUMMARY_EVERY) {
            #[allow(clippy::cast_precision_loss)]
            let per = |x: u64| x as f64 / jobs as f64;
            tracing::info!(
                jobs,
                improved = st.improved.load(Ordering::Relaxed),
                mean_gain_milli = per(st.gain_milli.load(Ordering::Relaxed)),
                mean_ms = per(st.micros.load(Ordering::Relaxed)) / 1000.0,
                saturated = st.saturated.load(Ordering::Relaxed),
                cancelled = st.cancelled.load(Ordering::Relaxed),
                mismatched = st.mismatched.load(Ordering::Relaxed),
                "otsuka polish summary"
            );
        }
    }

    /// Local build: anneal one validated family job on `lane`. Without stage
    /// 1, and for every explicit family job, this is the stock single run.
    /// With it, a lease salt takes the short run first and the full run only
    /// when promoted ([`Stage1::solve`]). A stage-2 rerun (`rerun`) is one
    /// full anneal on the job's parameters (its seed already replaced): it
    /// never passes stage 1 again and never feeds stage 1's windows.
    /// `full_ran` tells whether a full anneal completed (stage 2's trigger
    /// looks at those results only).
    fn solve_family(
        &self,
        lane: &D,
        job: &StreamJob,
        family: &FamilyRef,
        cancelled: &dyn Fn() -> bool,
        rerun: bool,
        full_ran: &mut bool,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let anneal = |params: &SampleParams| -> Result<Vec<SamplerResult>, SampleError> {
            let prepared =
                self.prepared(&job.graph, params, family.instance(), family.contract())?;
            self.reads_on(lane, &job.graph, params, prepared, cancelled)
                .map_err(sample_error)
        };
        let stage1 = self.stage1.as_ref().filter(|_| family.is_lease() && !rerun);
        let Some(stage1) = stage1 else {
            let reads = anneal(&job.params)?;
            *full_ran = true;
            return Ok(reads);
        };
        // A yielded full run keeps the short reads; a fault or a round change
        // propagates (finish() turns the latter into Cancelled).
        let shape = Some((job.graph.h.len(), job.graph.edges.len()));
        let (reads, trace) = stage1.solve_shaped(shape, &job.params, anneal, |e| {
            !e.is_fatal() && !cancelled()
        })?;
        *full_ran = trace.outcome == Stage1Outcome::Promoted;
        log_stage1(job, &trace);
        Ok(reads)
    }

    /// Local build (all5): after the FIRST full run of a local lease salt,
    /// either hand the result back for delivery, or hold it in a stage-2
    /// group and queue `K` reruns of the same instance ahead of fresh salts
    /// (`None`). See [`super::stage2`].
    fn maybe_deepen(
        &self,
        job: StreamJob,
        family: FamilyRef,
        result: StreamResult,
        dest: Dest,
    ) -> Option<(StreamJob, FamilyRef, StreamResult, Dest)> {
        let Some(stage2) = self.stage2.as_ref() else {
            return Some((job, family, result, dest));
        };
        let (Dest::Local(reply), FamilyRef::Local { instance, contract }) = (&dest, &family) else {
            return Some((job, family, result, dest));
        };
        let Some(best) = best_of(&result) else {
            return Some((job, family, result, dest));
        };
        let offset = reply.offset();
        let reported = best.saturating_add(offset);
        let shape = (job.graph.h.len(), job.graph.edges.len());
        if !stage2.should_deepen(shape, reported) {
            return Some((job, family, result, dest));
        }
        let (instance, contract, stop) = (Arc::clone(instance), *contract, reply.stop_flag());
        // The cap, checked and filled under one lock.
        let mut groups = self
            .stage2_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
        if groups.len() >= stage2.config().max_active {
            return Some((job, family, result, dest));
        }
        let k = stage2.config().k;
        let reruns: Vec<StreamJob> = (1..=k as u64)
            .map(|j| StreamJob {
                job_id: job.job_id.clone(),
                graph: job.graph.clone(),
                params: SampleParams {
                    seed: rerun_seed(job.params.seed, j),
                    ..job.params.clone()
                },
                watermark: job.watermark,
            })
            .collect();
        let group = Arc::new(S2Group::new(
            reported,
            offset,
            k,
            stop,
            Payload {
                job,
                family,
                dest,
                result,
            },
        ));
        groups.push(Arc::downgrade(&group));
        drop(groups);
        let mut queue = self
            .stage2_reruns
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for (j, rerun) in (1..).zip(reruns) {
            queue.push_back(LaneItem {
                job: rerun,
                family: FamilyRef::Local {
                    instance: Arc::clone(&instance),
                    contract,
                },
                stage0: None,
                dest: Dest::Session, // unused: the group delivers
                rerun: Some((Arc::clone(&group), j)),
                deep: None,
                ease: None,
            });
        }
        None
    }

    /// Local build (all7): after the FIRST full run of a local lease salt,
    /// either hand the result back (stage 2 or delivery), or hold it in a
    /// deep group and queue one long run of the same instance for the deep
    /// lanes (`None`). See [`super::deep`].
    fn maybe_deep(
        &self,
        job: StreamJob,
        family: FamilyRef,
        result: StreamResult,
        dest: Dest,
    ) -> Option<(StreamJob, FamilyRef, StreamResult, Dest)> {
        let Some(deep) = self.deep.as_ref() else {
            return Some((job, family, result, dest));
        };
        let (Dest::Local(reply), FamilyRef::Local { instance, contract }) = (&dest, &family) else {
            return Some((job, family, result, dest));
        };
        let Some(best) = best_of(&result) else {
            return Some((job, family, result, dest));
        };
        let offset = reply.offset();
        let reported = best.saturating_add(offset);
        let shape = (job.graph.h.len(), job.graph.edges.len());
        let now = Instant::now();
        // The cap, checked and filled under one lock.
        let mut groups = self
            .deep_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
        if !deep.should_deep(shape, reported, now, groups.len()) {
            return Some((job, family, result, dest));
        }
        // all9: MULT, or the adaptive length for this time into the round.
        let mult = deep.mult_for(deep.round_elapsed(now));
        let Some(num_sweeps) = job
            .params
            .num_sweeps
            .checked_mul(mult)
            .filter(|&n| n <= i32::MAX.unsigned_abs() as usize)
        else {
            return Some((job, family, result, dest));
        };
        let (instance, contract, stop) = (Arc::clone(instance), *contract, reply.stop_flag());
        let long = StreamJob {
            job_id: job.job_id.clone(),
            graph: job.graph.clone(),
            params: SampleParams {
                seed: deep_seed(job.params.seed),
                num_sweeps,
                ..job.params.clone()
            },
            watermark: job.watermark,
        };
        let group = Arc::new(
            DGroup::new(
                reported,
                offset,
                deep.round_elapsed(now),
                stop,
                Payload {
                    job,
                    family,
                    dest,
                    result,
                },
            )
            .with_mult(mult),
        );
        groups.push(Arc::downgrade(&group));
        drop(groups);
        tracing::debug!(
            first = reported,
            round_s = deep.round_elapsed(now).as_secs_f64(),
            num_sweeps,
            "otsuka deep: holding a first run for a long run"
        );
        self.deep_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(LaneItem {
                job: long,
                family: FamilyRef::Local { instance, contract },
                stage0: None,
                dest: Dest::Session, // unused: the group delivers
                rerun: None,
                deep: Some(group),
                ease: None,
            });
        None
    }

    /// Local build (all7): run one long anneal on a deep lane, probing its
    /// lineage best about every second, and deliver its group. False when
    /// `out` is closed. A long run of a stopped lease or of a delivered
    /// group never launches, or stops at its next chunk.
    fn run_deep(
        &self,
        lane: &D,
        deep: &Deep,
        item: &LaneItem,
        group: &DGroup,
        pool: Option<&SyncSender<PolishJob>>,
        out: &Sender<StreamResult>,
    ) -> bool {
        if !group.start() {
            return true;
        }
        let cancelled = || group.stopped();
        let probe = |device_best: i64| group.probe(deep, device_best);
        let result = Self::finish_with(&item.job, &cancelled, || {
            let prepared = self.prepared(
                &item.job.graph,
                &item.job.params,
                item.family.instance(),
                item.family.contract(),
            )?;
            self.reads_on_probed(
                lane,
                &item.job.graph,
                &item.job.params,
                prepared,
                &cancelled,
                Some(&probe),
            )
            .map_err(sample_error)
        });
        match group.complete(deep, result) {
            Some(p) => self.deliver(pool, p.job, p.family, p.result, p.dest, out),
            None => true,
        }
    }

    /// Local build (all7): the held deep groups the watcher should deliver
    /// now. Groups of a stopped lease are dropped on the way.
    fn deep_due(&self, deep: &Deep) -> Vec<Payload<StreamJob, FamilyRef, Dest>> {
        let held: Vec<Arc<DGroup>> = {
            let mut groups = self
                .deep_groups
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
            groups.iter().filter_map(Weak::upgrade).collect()
        };
        let now = Instant::now();
        // all9: keep the adaptive allocation (and its log) current.
        if deep.config().adaptive.is_some() {
            deep.allocation(now);
        }
        held.iter()
            .filter_map(|g| g.release_due(deep, now))
            .collect()
    }

    /// Local build (all5): the held groups the watcher should deliver now
    /// (the target eased to within the release margin). Groups of a stopped
    /// lease are dropped on the way.
    fn stage2_due(&self, stage2: &Stage2) -> Vec<Payload<StreamJob, FamilyRef, Dest>> {
        let held: Vec<Arc<S2Group>> = {
            let mut groups = self
                .stage2_groups
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            groups.retain(|g| g.upgrade().is_some_and(|g| g.pending()));
            groups.iter().filter_map(Weak::upgrade).collect()
        };
        held.iter().filter_map(|g| g.release_due(stage2)).collect()
    }

    /// Local build (all5): the stage-2 watcher, a thread of the stream
    /// session's scope while stage 2 runs on the local path. Every
    /// [`STAGE2_WATCH`] it delivers held groups the live target has eased
    /// to, so a hold never sits on a result the coordinator could submit.
    ///
    /// all7: it also watches the deep lane's held groups (`deep`).
    fn stage2_watch(
        &self,
        stage2: Option<&Stage2>,
        deep: Option<&Deep>,
        over: &AtomicBool,
        pool: Option<&SyncSender<PolishJob>>,
        out: &Sender<StreamResult>,
    ) {
        while !over.load(Ordering::Acquire) {
            std::thread::sleep(STAGE2_WATCH);
            let mut due = stage2.map(|s| self.stage2_due(s)).unwrap_or_default();
            if let Some(deep) = deep {
                due.extend(self.deep_due(deep));
                // all12: and the ease booster's held groups.
                if let Some(ease) = self.ease.as_ref().filter(|e| e.active()) {
                    due.extend(self.ease_due(ease, deep));
                }
            }
            for p in due {
                if !self.deliver(pool, p.job, p.family, p.result, p.dest, out) {
                    return;
                }
            }
        }
    }

    /// Local build (all5): the next lane item with stage 2 or the pool on: a
    /// rerun first, else a fresh salt. Polls, so a lane parked on an empty
    /// fresh queue still sees a new rerun within [`STAGE2_LANE_POLL`]. `Err`
    /// once the fresh queue is closed.
    ///
    /// all6: with the pool (`pool`), a fresh salt is a session-path item
    /// from the queue if one is waiting, else the pool's best screened local
    /// salt; a lane with nothing to do sleeps on the pool until an offer.
    ///
    /// all7: a deep lane (`deep_lane`) takes a queued long run before
    /// anything else.
    fn next_lane_item(
        &self,
        fresh: &Mutex<std::sync::mpsc::Receiver<LaneItem>>,
        reruns: bool,
        pool: Option<&Pool<LaneItem>>,
        deep_lane: bool,
    ) -> Result<LaneItem, ()> {
        loop {
            // all12: a deep lane takes a boost run before the deep lane's
            // own long runs (a skippable one first, to retire it at once).
            if deep_lane && self.ease.is_some() {
                let boost = take_best(
                    &mut self
                        .ease_queue
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner),
                    |item| {
                        item.ease.as_ref().map_or(i64::MIN, |g| {
                            if g.skippable() {
                                i64::MIN
                            } else {
                                0
                            }
                        })
                    },
                );
                if let Some(item) = boost {
                    return Ok(item);
                }
            }
            if deep_lane {
                let mut queue = self
                    .deep_queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                // all9: adaptive takes the best first run's long run (a
                // stopped group's first, to skip it at once).
                let adaptive = self
                    .deep
                    .as_ref()
                    .is_some_and(|d| d.config().adaptive.is_some());
                let long = if adaptive {
                    take_best(&mut queue, |item| {
                        item.deep.as_ref().map_or(i64::MIN, |g| {
                            if g.stopped() {
                                i64::MIN
                            } else {
                                g.first_best()
                            }
                        })
                    })
                } else {
                    queue.pop_front()
                };
                drop(queue);
                if let Some(item) = long {
                    return Ok(item);
                }
            }
            let rerun = if reruns {
                self.stage2_reruns
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pop_front()
            } else {
                None
            };
            if let Some(item) = rerun {
                return Ok(item);
            }
            if let Some(pool) = pool {
                let queued = fresh
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .try_recv();
                match queued {
                    Ok(item) => {
                        self.screen_stats.dequeued();
                        return Ok(item);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => return Err(()),
                }
                if let Some(item) = pool.take(&local::stale_item) {
                    return Ok(item);
                }
                pool.wait(STAGE2_LANE_POLL);
                continue;
            }
            let next = fresh
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .recv_timeout(STAGE2_LANE_POLL);
            match next {
                Ok(item) => {
                    self.screen_stats.dequeued();
                    return Ok(item);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(()),
            }
        }
    }

    /// Local build: solve one lane item on `lane` and deliver it (stage 2
    /// may hold a local salt's first full result, or the item is a stage-2
    /// rerun). False when `out` is closed.
    #[allow(clippy::too_many_arguments)]
    fn lane_item(
        &self,
        lane: &D,
        stage2: Option<&Stage2>,
        deep_on: bool,
        mut item: LaneItem,
        pool: Option<&SyncSender<PolishJob>>,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
    ) -> bool {
        if let Some(rerun) = item.rerun.take() {
            return match stage2 {
                Some(stage2) => self.run_rerun(lane, stage2, &item, (&rerun.0, rerun.1), pool, out),
                None => true, // unreachable: reruns exist only with stage 2
            };
        }
        if let Some(group) = item.deep.take() {
            return match self.deep.as_ref() {
                Some(deep) => self.run_deep(lane, deep, &item, &group, pool, out),
                None => true, // unreachable: deep runs exist only with the deep lane
            };
        }
        if let Some(group) = item.ease.take() {
            return match (self.ease.as_ref(), self.deep.as_ref()) {
                (Some(ease), Some(deep)) => {
                    self.run_ease(lane, ease, deep, &item, &group, pool, out)
                }
                _ => true, // unreachable: boost runs exist only with the booster
            };
        }
        let LaneItem {
            job,
            family,
            stage0,
            dest,
            rerun: _,
            deep: _,
            ease: _,
        } = item;
        let cancelled = || dest.stopped(cancel, job.watermark);
        let mut full_ran = false;
        let began = Instant::now();
        let result = Self::finish_with(&job, &cancelled, || {
            self.solve_family(lane, &job, &family, &cancelled, false, &mut full_ran)
        });
        log_stage0_pair(&job, stage0, &result);
        if !full_ran {
            return self.deliver(pool, job, family, result, dest, out);
        }
        // all9: time the job length for the adaptive long-run multiple
        // (a no-op without it).
        if let Some(deep) = self.deep.as_ref().filter(|_| deep_on) {
            if best_of(&result).is_some() {
                deep.note_run(began.elapsed(), 1);
            }
        }
        // all7: the deep lane and stage 2 each may hold a first full run, in
        // the order `QUIP_OTSUKA_DEEP_AFTER_STAGE2` sets (deep first by
        // default). When one holds it, the other still records it, so both
        // windows and the round record see every first run.
        let deep = self.deep.as_ref().filter(|_| deep_on);
        let facts = if deep.is_some() && stage2.is_some() {
            first_facts(&job, &result, &dest)
        } else {
            None
        };
        let mut first = Some((job, family, result, dest));
        let deep_first = !deep.is_some_and(|d| d.config().after_stage2);
        for deep_turn in [deep_first, !deep_first] {
            let Some((job, family, result, dest)) = first.take() else {
                break;
            };
            first = match (deep_turn, deep, stage2) {
                (true, Some(_), _) => {
                    let kept = self.maybe_deep(job, family, result, dest);
                    if let (None, Some(stage2), Some((shape, best))) = (&kept, stage2, facts) {
                        stage2.observe(shape, best);
                    }
                    kept
                }
                (false, _, Some(_)) => {
                    let kept = self.maybe_deepen(job, family, result, dest);
                    if let (None, Some(deep), Some((shape, best))) = (&kept, deep, facts) {
                        deep.observe(shape, best, Instant::now());
                    }
                    kept
                }
                _ => Some((job, family, result, dest)),
            };
        }
        match first {
            Some((job, family, result, dest)) => self.deliver(pool, job, family, result, dest, out),
            None => true, // held in a deep or stage-2 group
        }
    }

    /// Local build (all5): run one stage-2 rerun on `lane` and merge it into
    /// its group, delivering the group when that ends it. False when `out`
    /// is closed. A rerun of a stopped lease or of a delivered group stops
    /// at its next chunk, or never launches.
    fn run_rerun(
        &self,
        lane: &D,
        stage2: &Stage2,
        item: &LaneItem,
        (group, index): (&S2Group, u64),
        pool: Option<&SyncSender<PolishJob>>,
        out: &Sender<StreamResult>,
    ) -> bool {
        let cancelled = || group.stopped();
        let mut full_ran = false;
        let result = Self::finish_with(&item.job, &cancelled, || {
            self.solve_family(
                lane,
                &item.job,
                &item.family,
                &cancelled,
                true,
                &mut full_ran,
            )
        });
        match group.complete(stage2, result, index) {
            Some(p) => self.deliver(pool, p.job, p.family, p.result, p.dest, out),
            None => true,
        }
    }

    /// Local build: score a validated family lease salt and decide. With the
    /// screen off this is always `Pass(None)` and costs nothing. Explicit
    /// family jobs, and a salt whose score cannot be computed, pass.
    fn stage0(&self, job: &StreamJob, family: &FamilyContext, cancel: &CancelToken) -> Stage0 {
        let lease = matches!(family.source(), FamilySource::Lease { .. });
        match self.stage0_decide(family.instance(), &job.graph.edges, lease) {
            Gate0::Pass(score) => Stage0::Pass(score),
            Gate0::Reject => Stage0::Reject(Self::finish(job, cancel, || Ok(Vec::new()))),
        }
    }

    /// Local build: the stage-0 gates on one instance whose edges are
    /// `edges` (the job graph's). Shared by the session path's dispatcher
    /// and the local screen workers; each gate decides under its own lock,
    /// so concurrent workers see the serial rolling-quantile rule, in
    /// arrival order.
    fn stage0_decide(
        &self,
        instance: &Instance,
        edges: &Arc<[(usize, usize)]>,
        lease: bool,
    ) -> Gate0 {
        // Only lease salts are interchangeable. An explicit job is always solved.
        if !lease {
            return Gate0::Pass(None);
        }
        self.screen_stats.screened.fetch_add(1, Ordering::Relaxed);
        let verdict = self.stage0_gates(instance, edges);
        if matches!(verdict, Gate0::Pass(_)) {
            self.screen_stats.kept.fetch_add(1, Ordering::Relaxed);
        }
        verdict
    }

    /// Local build (all6): [`Self::stage0_decide`] for a lease salt given as
    /// raw parts (`n` nodes, the job graph's `edges`, couplings `j` aligned
    /// with them, and the selected tiles), as the cached generator yields
    /// them. Same gates, same counters, same decisions as on the instance.
    fn stage0_decide_raw(
        &self,
        n: usize,
        edges: &Arc<[(usize, usize)]>,
        j: &[i32],
        tiles: &[[u32; 4]],
    ) -> Gate0 {
        self.screen_stats.screened.fetch_add(1, Ordering::Relaxed);
        let verdict = self.stage0_gates_with(|model| raw_stage0_model(model, n, edges, j, tiles));
        if matches!(verdict, Gate0::Pass(_)) {
            self.screen_stats.kept.fetch_add(1, Ordering::Relaxed);
        }
        verdict
    }

    fn stage0_gates(&self, instance: &Instance, edges: &Arc<[(usize, usize)]>) -> Gate0 {
        self.stage0_gates_with(|model| instance_stage0_model(model, instance, edges))
    }

    /// The stage-0 gates with `score(model)` giving one model's score.
    fn stage0_gates_with(&self, score_of: impl Fn(u8) -> Option<f32>) -> Gate0 {
        if self.stage0.is_none() && self.stage0_pre.is_none() {
            return Gate0::Pass(None);
        }
        // Model 3: the lean model-1 gate first, so model 2 scores only its
        // survivors.
        if let Some(pre) = self.stage0_pre.as_ref() {
            let Some(lean) = score_of(1) else {
                tracing::warn!("stage-0 prefilter score unavailable; solving the job");
                return Gate0::Pass(None);
            };
            if !pre
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .admit(lean)
            {
                return Gate0::Reject;
            }
        }
        let Some(gate) = self.stage0.as_ref() else {
            // Model 3 with keep >= pre: model 1 alone decided.
            return Gate0::Pass(None);
        };
        let Some(score) = score_of(self.tuning.stage0_model) else {
            tracing::warn!("stage-0 score unavailable; solving the job");
            return Gate0::Pass(None);
        };
        if gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .admit(score)
        {
            Gate0::Pass(Some(score))
        } else {
            Gate0::Reject
        }
    }

    fn reads(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        prepared: PreparedGraph,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<SamplerResult>, ClusterDeviceError> {
        self.reads_on(&self.cluster, graph, params, prepared, cancelled)
    }

    fn reads_on(
        &self,
        cluster: &D,
        graph: &IsingGraph,
        params: &SampleParams,
        prepared: PreparedGraph,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<SamplerResult>, ClusterDeviceError> {
        self.reads_on_probed(cluster, graph, params, prepared, cancelled, None)
    }

    /// [`Self::reads_on`], with the deep lane's probe (all7) when given.
    fn reads_on_probed(
        &self,
        cluster: &D,
        graph: &IsingGraph,
        params: &SampleParams,
        prepared: PreparedGraph,
        cancelled: &dyn Fn() -> bool,
        probe: Option<&dyn Fn(i64) -> bool>,
    ) -> Result<Vec<SamplerResult>, ClusterDeviceError> {
        // A governor yield refuses this job as busy. Only the session token
        // makes finish() report Cancelled and suppress the upstream result.
        let stopped = || cancelled() || self.sa.should_throttle();
        checkpoint(&stopped)?;
        let work = ClusterWork::new(graph, params)?;
        let run = match probe {
            Some(probe) => cluster.run_probed(prepared, &work, &stopped, cancelled, probe)?,
            None => cluster.run(prepared, &work, &stopped, cancelled)?,
        };
        if run.spins.len() != params.num_reads
            || run.energies.len() != params.num_reads
            || run
                .spins
                .iter()
                .any(|s| s.len() != graph.h.len() || s.iter().any(|v| ![-1, 1].contains(v)))
        {
            return Err(ClusterDeviceError::Parameter(
                "incomplete or invalid cluster output",
            ));
        }
        let mut reads = Vec::with_capacity(params.num_reads);
        for spins in run.spins {
            checkpoint(cancelled)?;
            reads.push(SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            });
        }
        Ok(reads)
    }

    fn prepared(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        instance: &Instance,
        contract: &'static Contract,
    ) -> Result<PreparedGraph, SampleError> {
        if instance.h().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
            return Err(SampleError::Capacity);
        }
        let weights = self.tuning.weights();
        // `PreparedGraph::from_family` is exactly this call on a `FamilyContext`.
        PreparedGraph::from_instance_scaled(
            instance,
            contract,
            weights,
            &self.limits,
            params.num_reads,
            self.tuning.weak_scale,
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "cluster admission refused");
            SampleError::Capacity
        })
    }

    fn finish(
        job: &StreamJob,
        cancel: &CancelToken,
        run: impl FnOnce() -> Result<Vec<SamplerResult>, SampleError>,
    ) -> StreamResult {
        Self::finish_with(job, &|| cancel.is_cancelled(job.watermark), run)
    }

    /// [`Self::finish`] with an explicit stop rule: the job's watermark on the
    /// session path, the lease's stop flag on the local path.
    fn finish_with(
        job: &StreamJob,
        cancelled: &dyn Fn() -> bool,
        run: impl FnOnce() -> Result<Vec<SamplerResult>, SampleError>,
    ) -> StreamResult {
        let start = Instant::now();
        let result = if cancelled() { Ok(Vec::new()) } else { run() };
        let fatal = result.as_ref().is_err_and(SampleError::is_fatal);
        let outcome = if cancelled() && !fatal {
            StreamOutcome::Cancelled
        } else {
            StreamOutcome::Completed(result)
        };
        StreamResult {
            job_id: job.job_id.clone(),
            outcome,
            device_access_time_us: u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
        }
    }

    /// Exercise the production result and cancellation path with a prepared fixture.
    #[doc(hidden)]
    #[must_use]
    pub fn sample_prepared_for_test(
        &self,
        job: &StreamJob,
        prepared: PreparedGraph,
        cancel: &CancelToken,
    ) -> StreamResult {
        Self::finish(job, cancel, || {
            self.reads(&job.graph, &job.params, prepared, &|| {
                cancel.is_cancelled(job.watermark)
            })
            .map_err(sample_error)
        })
    }
}

impl<D: ClusterBackend, S: Sampler> OtsukaSampler<D, S> {
    /// Local build: `QUIP_OTSUKA_CONCURRENCY` lanes pull family jobs from one
    /// bounded queue; plain jobs still go to the resident SA stream. With
    /// local generation (`QUIP_OTSUKA_SCREEN_THREADS` above 1) the queue is
    /// also published for [`Sampler::sample_lease`], whose screen workers
    /// feed it lease salts.
    #[allow(clippy::too_many_lines)] // One session's lanes, watcher and feed.
    fn concurrent_family_loop<'scope, 'env>(
        &'env self,
        scope: &'scope std::thread::Scope<'scope, 'env>,
        jobs: &mut Receiver<WarmStreamJob>,
        out: &'env Sender<StreamResult>,
        cancel: &'env CancelToken,
        sa_tx: &Sender<StreamJob>,
    ) {
        let lanes: Vec<&D> = std::iter::once(&self.cluster).chain(&self.extra).collect();
        let lane_count = lanes.len();
        let polish = self.start_polish(scope, out, cancel);
        let local = screen_threads() > 1;
        // Stage 2 acts on local lease salts only (the sampler owns their
        // delivery). Stale reruns and groups of an earlier session go.
        let stage2 = self.stage2.as_ref().filter(|_| local);
        self.stage2_reruns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        self.stage2_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        // all6: the pool works local lease salts only; an earlier session's
        // entries belong to leases that are gone.
        let pool = self.pool.as_ref().filter(|_| local);
        if let Some(pool) = self.pool.as_ref() {
            drop(pool.clear());
        }
        // all7: the deep lane works local lease salts only, on the first
        // `lanes` lanes, and always leaves one lane to the rest. An earlier
        // session's held groups and long runs are gone.
        let deep = self
            .deep
            .as_ref()
            .filter(|d| local && d.config().lanes < lane_count);
        if self.deep.is_some() && deep.is_none() {
            tracing::warn!(
                lanes = lane_count,
                "deep lane needs the local path and a spare lane; off"
            );
        }
        self.deep_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        self.deep_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        if let Some(deep) = deep {
            deep.new_round(Instant::now());
        }
        // all12: the ease booster runs with the deep lane only; an earlier
        // session's held groups and boost runs are gone.
        self.ease_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        self.ease_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        if let Some(ease) = self.ease.as_ref() {
            ease.set_active(deep.is_some());
        }
        let deep_lanes = deep.map_or(0, |d| d.config().lanes);
        let stage2_over = Arc::new(AtomicBool::new(false));
        if stage2.is_some() || deep.is_some() {
            let (over, polish) = (Arc::clone(&stage2_over), polish.clone());
            scope.spawn(move || self.stage2_watch(stage2, deep, &over, polish.as_ref(), out));
        }
        // The local path keeps one more kept salt per lane ready.
        let depth = if local { 2 * lane_count } else { lane_count };
        let (fam_tx, fam_rx) = std::sync::mpsc::sync_channel::<LaneItem>(depth);
        let fam_rx = std::sync::Arc::new(Mutex::new(fam_rx));
        for (index, lane) in lanes.into_iter().enumerate() {
            let fam_rx = std::sync::Arc::clone(&fam_rx);
            let polish = polish.clone();
            let deep_lane = index < deep_lanes;
            scope.spawn(move || loop {
                let waited = Instant::now();
                let next = if stage2.is_some() || pool.is_some() || deep.is_some() {
                    self.next_lane_item(&fam_rx, stage2.is_some(), pool, deep_lane)
                } else {
                    let next = fam_rx.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    if next.is_ok() {
                        self.screen_stats.dequeued();
                    }
                    next.map_err(|_| ())
                };
                self.screen_stats.lane_waited(waited.elapsed());
                let Ok(item) = next else {
                    break;
                };
                let delivered = self.lane_item(
                    lane,
                    stage2,
                    deep.is_some(),
                    item,
                    polish.as_ref(),
                    out,
                    cancel,
                );
                self.screen_stats.maybe_log(lane_count);
                if let Some(pool) = pool {
                    pool.maybe_log(self.screen_stats.screened.load(Ordering::Relaxed));
                }
                if !delivered {
                    break;
                }
            });
        }
        if local {
            *self
                .lane_queue
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(fam_tx.clone());
        }
        while let Some(item) = jobs.blocking_recv() {
            if out.is_closed() {
                break;
            }
            let job = item.job;
            match item.family.filter(supported) {
                Some(family) => match self.stage0(&job, &family, cancel) {
                    Stage0::Pass(stage0) => {
                        let item = LaneItem {
                            job,
                            family: FamilyRef::Core(family),
                            stage0,
                            dest: Dest::Session,
                            rerun: None,
                            deep: None,
                            ease: None,
                        };
                        if fam_tx.send(item).is_err() {
                            break;
                        }
                        self.screen_stats.enqueued();
                    }
                    Stage0::Reject(result) => {
                        if out.blocking_send(result).is_err() {
                            break;
                        }
                    }
                },
                None => {
                    if sa_tx.blocking_send(job).is_err() {
                        break;
                    }
                }
            }
            self.screen_stats.maybe_log(lane_count);
        }
        // The session is over: no new local lease salts. Lanes finish what
        // is queued (a stopped lease's salts are dropped unsolved) and exit
        // when every sender is gone. The stage-2 watcher exits too (it holds
        // a polish-pool sender).
        *self
            .lane_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
        stage2_over.store(true, Ordering::Release);
    }
}

/// Local build (all7): a local first run's problem shape and best reported
/// energy (lane best plus the instance offset), as the deep lane's and
/// stage 2's triggers see it.
fn first_facts(
    job: &StreamJob,
    result: &StreamResult,
    dest: &Dest,
) -> Option<((usize, usize), i64)> {
    let Dest::Local(reply) = dest else {
        return None;
    };
    let best = best_of(result)?.saturating_add(reply.offset());
    Some(((job.graph.h.len(), job.graph.edges.len()), best))
}

/// Local build: the stage-0 score next to the solved energy, for calibration
/// (`RUST_LOG=quip_solver_cuda::cluster::sampler=debug`).
fn log_stage0_pair(job: &StreamJob, score: Option<f32>, result: &StreamResult) {
    if let (Some(stage0), StreamOutcome::Completed(Ok(reads))) = (score, &result.outcome) {
        if let Some(full) = reads.iter().map(|r| r.energy_milli).min() {
            let id: String = job.job_id.iter().fold(String::new(), |mut s, b| {
                use std::fmt::Write;
                write!(s, "{b:02x}").expect("writing to a String cannot fail");
                s
            });
            tracing::debug!(job = %id, stage0, full, "otsuka stage0 pair");
        }
    }
}

/// Local build: the short and full bests of one stage-1 job, for
/// calibration (`RUST_LOG=quip_solver_cuda::cluster::sampler=debug`).
fn log_stage1(job: &StreamJob, trace: &Stage1Trace) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let id: String = job.job_id.iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write;
            write!(s, "{b:02x}").expect("writing to a String cannot fail");
            s
        });
        tracing::debug!(
            job = %id,
            short = trace.short_best,
            full = trace.full_best,
            short_kept = trace.short_kept,
            outcome = ?trace.outcome,
            "otsuka stage1 pair"
        );
    }
}

/// Local build: order a completed family result's reads by energy, lowest
/// first, as the last step before it leaves the sampler (after the stage-1
/// merge and the CPU polish, which both change energies in place).
///
/// The coordinator keeps the solver's row order: its energy filter and Z2
/// dedup keep the first occurrence, and with `min_solutions = 1`
/// `quip_protocol::diversity::select_diverse(rows, 1)` is `[0]`, so it
/// submits row 0, not the lowest row. Core forwards rows in order too. The
/// sort is stable and moves each read whole (spins with their energy).
fn sort_reads(result: &mut StreamResult) {
    if let StreamOutcome::Completed(Ok(reads)) = &mut result.outcome {
        reads.sort_by_key(|r| r.energy_milli);
    }
}

fn supported(family: &FamilyContext) -> bool {
    family.family_id() == TWISTED_DIMER_V1.family_id
        && family.revision() == TWISTED_DIMER_V1.revision
}

fn sample_error(error: ClusterDeviceError) -> SampleError {
    match error {
        ClusterDeviceError::Cancelled
        | ClusterDeviceError::Prepare(PrepareError::DeviceMemory { .. }) => SampleError::DeviceBusy,
        ClusterDeviceError::JobParameter(_) => SampleError::Capacity,
        ClusterDeviceError::Prepare(e) => {
            tracing::warn!(error = %e, "cluster admission refused");
            SampleError::Capacity
        }
        ClusterDeviceError::Driver(e) => crate::sampler::SampleError::from(e).into(),
        ClusterDeviceError::Cuda(e) => SampleError::DeviceFault(e.to_string()),
        ClusterDeviceError::Parameter(e) => SampleError::DeviceFault(e.into()),
        e @ ClusterDeviceError::MemoryPoolsUnsupported => SampleError::DeviceFault(e.to_string()),
        ClusterDeviceError::Spin(e) => SampleError::DeviceFault(e.to_string()),
    }
}

impl<D: ClusterBackend, S: Sampler> Sampler for OtsukaSampler<D, S> {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.sa.sample(graph, params)
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    /// Local build: `QUIP_OTSUKA_SCREEN_THREADS` above 1 generates and
    /// screens lease salts here, on a worker pool, instead of in core's
    /// serial expander ([`Self::sample_lease`]). 1 (the default) keeps
    /// core's path exactly.
    fn generates_locally() -> bool {
        screen_threads() > 1
    }

    fn sample_lease(
        &self,
        lease: &quip_solver_core::Lease,
        topology: &quip_solver_core::TopologyView,
        params: &SampleParams,
        sink: &quip_solver_core::LeaseSink,
    ) -> Result<(), SampleError> {
        self.sample_lease_local(lease, topology, params, sink)
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: OTSUKA_SOLVER_ID,
            generated: TWISTED_DIMER,
            accepted: TWISTED_DIMER,
        }
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        if !supported(family) {
            return self.sample(graph, params);
        }
        let prepared = self.prepared(graph, params, family.instance(), family.contract())?;
        self.reads(graph, params, prepared, &|| false)
            .map_err(sample_error)
    }

    fn sample_stream(
        &self,
        jobs: Receiver<StreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.sa.sample_stream(jobs, out, cancel);
    }

    fn sample_stream_warm(
        &self,
        mut jobs: Receiver<WarmStreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        std::thread::scope(|scope| {
            // Keep the production self-feeding session alive across plain jobs.
            let (sa_tx, sa_rx) = channel(self.sa.stream_width().saturating_mul(2).max(1));
            let sa_out = out.clone();
            let sa_cancel = cancel.clone();
            scope.spawn(|| self.sa.sample_stream(sa_rx, sa_out, sa_cancel));
            // Local lease generation needs the lane queue, even with one lane.
            if !self.extra.is_empty() || screen_threads() > 1 {
                self.concurrent_family_loop(scope, &mut jobs, &out, &cancel, &sa_tx);
                drop(sa_tx);
                return;
            }
            let pool = self.start_polish(scope, &out, &cancel);
            while let Some(item) = jobs.blocking_recv() {
                if out.is_closed() {
                    break;
                }
                let job = item.job;
                if let Some(family) = item.family.filter(supported) {
                    let sent = match self.stage0(&job, &family, &cancel) {
                        Stage0::Reject(result) => out.blocking_send(result).is_ok(),
                        Stage0::Pass(stage0) => {
                            let cancelled = || cancel.is_cancelled(job.watermark);
                            let family = FamilyRef::Core(family);
                            let mut full_ran = false;
                            let result = Self::finish(&job, &cancel, || {
                                self.solve_family(
                                    &self.cluster,
                                    &job,
                                    &family,
                                    &cancelled,
                                    false,
                                    &mut full_ran,
                                )
                            });
                            log_stage0_pair(&job, stage0, &result);
                            self.deliver(pool.as_ref(), job, family, result, Dest::Session, &out)
                        }
                    };
                    if !sent {
                        break;
                    }
                } else if sa_tx.blocking_send(job).is_err() {
                    break;
                }
            }
            drop(sa_tx);
        });
    }

    fn stream_width(&self) -> usize {
        // Local build (all5, agent X): on the local path the session sizes
        // its lease credit window from this (prefetch = 2 * width, window =
        // prefetch + queue depth). The SA stream width (~170 on a 5090) held
        // ~343 leases of 4500 salts at once: none ever completed, the
        // coordinator's salts/s stayed 0, and every lease spawned THREADS+1
        // threads (~2400). The cluster lanes are the real width. With
        // THREADS = 1 (core's path) this is the SA width, as in all4.
        if local::screen_threads() > 1 {
            1 + self.extra.len()
        } else {
            self.sa.stream_width()
        }
    }
    fn declared_stream_width() -> u32 {
        S::declared_stream_width()
    }

    fn max_reads(&self) -> u32 {
        u32::try_from(MAX_READS).unwrap_or(u32::MAX)
    }
    fn utilization(&self) -> f64 {
        self.sa.utilization()
    }
    fn should_throttle(&self) -> bool {
        self.sa.should_throttle()
    }
    fn apply_config(&self, backend_toml: &str) {
        self.sa.apply_config(backend_toml);
    }
}
