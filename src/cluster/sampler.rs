//! Otsuka on validated twisted-ice-dimer jobs, production CUDA SA otherwise.

use super::device::{ClusterDevice, ClusterDeviceError, ClusterKernel, ClusterRun};
use super::{PrepareError, PreparedGraph, Weights, MAX_PACKED_NODES};
use crate::capacity::{
    cluster_footprint, ClusterShape, DeviceLimits, CLUSTER_DYNAMIC_PADDING,
    CLUSTER_SHARED_BYTES_PER_NODE, CLUSTER_STATIC_SHARED_BYTES,
};
use crate::cuda_device::CudaDevice;
use crate::nvml_gov::UtilGovernor;
use crate::streaming::{build_beta_schedule, StallConfig};
use crate::{cuda_sa_identity, CudaSampler, KernelKind};
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::quip_proto::v1::Algorithm;
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, Lease, LeaseSink, SampleError, SampleParams, Sampler,
    SamplerResult, StreamJob, StreamOutcome, StreamResult, TopologyView, WarmStart, WarmStreamJob,
};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub(crate) mod lease_run;

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
const MAX_READS: usize = 8192;
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
        // The coordinator sends one sweep per rung. Eight per rung halves the launch count and
        // leaves the energy unchanged; every rung boundary is still a cancellation checkpoint.
        let requested = if params.sweeps_per_beta <= 1 {
            std::env::var("QC_SPB").ok().and_then(|v| v.parse().ok()).unwrap_or(32)
        } else {
            params.sweeps_per_beta
        };
        let per_beta = if params.num_sweeps == 0 {
            requested.max(1)
        } else {
            requested.max(1).min(params.num_sweeps)
        };
        let (betas, sweeps_per_beta) = build_beta_schedule(
            graph,
            params.num_sweeps,
            per_beta,
            params.beta_range.or(Some((1.2, 8.0))),
            params.schedule,
            1.0,
        );
        // Two-piece schedule (QC_KNEE_FRAC > 0): geometric lo -> QC_KNEE_BETA over the first fraction of the betas, then
        // QC_KNEE_BETA -> QC_BETA_MAX. The weak layer (|J| = 0.045) is still hot at beta = 8; the tail freezes it.
        let betas = {
            let kf: f64 = std::env::var("QC_KNEE_FRAC").ok().map_or(KNEE_FRAC_DEFAULT, |v| v.parse().unwrap());
            if kf > 0.0 && kf < 1.0 && betas.len() > 2 {
                let kb: f64 = std::env::var("QC_KNEE_BETA").ok().map_or(KNEE_BETA_DEFAULT, |v| v.parse().unwrap());
                let hi: f64 = std::env::var("QC_BETA_MAX").ok().map_or(BETA_MAX_DEFAULT, |v| v.parse().unwrap());
                let (lo, _) = params.beta_range.unwrap_or((1.2, 8.0));
                let n = betas.len();
                if std::env::var("QC_KNEE_DEBUG").is_ok() { eprintln!("KNEE active: n={n} lo={lo} kb={kb} hi={hi} kf={kf}"); }
                (0..n)
                    .map(|i| {
                        let f = i as f64 / (n - 1) as f64;
                        let b = if f < kf { lo * (kb / lo).powf(f / kf) } else { kb * (hi / kb).powf((f - kf) / (1.0 - kf)) };
                        b as f32
                    })
                    .collect::<Vec<f32>>()
            } else {
                betas
            }
        };
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
}

/// Preloaded cluster function on the selected CUDA device.
pub struct GpuCluster {
    /// One kernel handle (and CUDA stream) per worker, so concurrent jobs overlap on the device.
    kernels: Vec<ClusterKernel>,
}

thread_local! {
    /// Index of the worker thread running the current job.
    static WORKER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
        let kernel = &self.kernels[WORKER.with(std::cell::Cell::get) % self.kernels.len()];
        ClusterDevice::open_with_kernel(graph, kernel)?.run_work(work, stopped, cancelled)
    }
}

/// Serial family jobs alongside one persistent production SA stream.
pub struct OtsukaSampler<D = GpuCluster, S = CudaSampler> {
    cluster: D,
    sa: S,
    limits: DeviceLimits,
    /// Best screened energy (milli) of recent jobs; ranks a new job among them.
    screen: Mutex<Vec<VecDeque<i64>>>,
}

/// Combined-score ranking from stage B on (QS_COMB=0 restores the per-stage energy rank).
fn comb_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QS_COMB").map_or(false, |v| v == "1"))
}

/// Weighted sum of the z-scores of the stage energies `e[1..]` (stage 0 is the cheap, almost uninformative probe), in
/// thousandths; more negative is better. Per-stage `(mean, sd, weight)` come from live measurements and can be replaced
/// with QS_ZSTAT="m;s;w|m;s;w|..." (energy units); the means only shift the score, the sd and weights decide the mix.
fn combined_score_milli(stage_e: &[i64]) -> i64 {
    static CFG: std::sync::OnceLock<Vec<(f64, f64, f64)>> = std::sync::OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let spec = std::env::var("QS_ZSTAT").unwrap_or_else(|_| "-5050.6;4.1;1.15|-5064.2;3.7;0.9|-5071.4;3.4;1.5".into());
        spec.split('|')
            .filter_map(|t| {
                let v: Vec<f64> = t.split(';').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 3 && v[1] > 0.0).then(|| (v[0], v[1], v[2]))
            })
            .collect()
    });
    let mut sum = 0.0;
    for (j, &e) in stage_e.iter().enumerate().skip(1) {
        let (m, sd, w) = cfg.get(j - 1).copied().unwrap_or((cfg.last().map_or(0.0, |c| c.0), 3.5, 1.0));
        sum += w * (e as f64 * 0.001 - m) / sd;
    }
    (sum * 1000.0).round() as i64
}

const KNEE_FRAC_DEFAULT: f64 = 0.0;
const KNEE_BETA_DEFAULT: f64 = 8.0;
const BETA_MAX_DEFAULT: f64 = 16.0;

/// Reference energy (milli) a job must beat to pass stage `k`, from live measurements. `QS_PRIOR` overrides
/// it as `a;b;c` in energy units.
fn prior_milli(k: usize) -> i64 {
    let spec = std::env::var("QS_PRIOR").unwrap_or_else(|_| if lease_run::msa_mode() { "-5054.5;-5069".into() } else { "-4795;-5052.5;-5064;-5068".into() });
    let v: Vec<f64> = spec.split(';').filter_map(|x| x.trim().parse().ok()).collect();
    let u = v.get(k).copied().unwrap_or(f64::NEG_INFINITY);
    if u.is_finite() { (u * 1000.0).round() as i64 } else { i64::MIN / 2 }
}

/// The instance and contract a job is built from, without a session context.
pub(crate) struct Fam<'a> {
    pub(crate) instance: &'a quip_solver_core::ising_generators::instance::Instance,
    pub(crate) contract: &'static quip_solver_core::ising_generators::descriptor::Contract,
}

/// Screen settings: short runs on every job, a deep run only on the best jobs.
struct ScreenCfg {
    on: bool,
    /// `(reads, sweeps, q)`: a job passes a stage when it ranks in the best `1/q` of that stage's window.
    stages: Vec<(usize, usize, usize)>,
    min: usize,
    win: usize,
}

impl ScreenCfg {
    fn from_env() -> Self {
        let n = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let spec = std::env::var("QS_STAGES").unwrap_or_else(|_| if lease_run::msa_mode() { "256:8192:4;512:32768:4".into() } else { "256:64:32;256:8192:16;512:16384:2;512:32768:4".into() });
        let stages = spec
            .split(';')
            .filter_map(|st| {
                let v: Vec<usize> = st.split(':').filter_map(|x| x.parse().ok()).collect();
                (v.len() == 3 && v[0] > 0 && v[1] > 0 && v[2] > 0).then(|| (v[0], v[1], v[2]))
            })
            .collect();
        Self { on: n("QS_ON", 1) != 0, stages, min: n("QS_MIN", 24), win: n("QS_WIN", 256).max(8) }
    }
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
        let workers = std::env::var("QS_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).max(1);
        let mut kernels = Vec::with_capacity(workers);
        for _ in 0..workers {
            kernels.push(ClusterKernel::load(device.device_index)?);
        }
        let cluster = GpuCluster { kernels };
        let limits = device.limits;
        Ok(Self {
            cluster,
            sa: CudaSampler::new(device, gov, KernelKind::Sa, stall),
            limits,
            screen: Mutex::new(Vec::new()),
        })
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
            screen: Mutex::new(Vec::new()),
        }
    }

    /// Short screen first; the full budget only when the job ranks among the best recent ones.
    /// A rejected job returns its screen reads, which are ordinary valid samples.
    fn screened_reads(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        family: &Fam<'_>,
        cancelled: &dyn Fn() -> bool,
        deep_out: &std::sync::atomic::AtomicBool,
        target: Option<i64>,
        mut pre: Option<PreparedGraph>,
        stage_e: &mut Vec<i64>,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let cfg = ScreenCfg::from_env();
        if !cfg.on || params.num_reads < 2 || cfg.stages.is_empty() {
            let prepared = self.prepared(graph, params, family)?;
            return self.reads(graph, params, prepared, cancelled).map_err(sample_error);
        }
        let mut last_reads: Option<Vec<SamplerResult>> = None;
        for (k, &(reads_k, sweeps_k, q)) in cfg.stages.iter().enumerate() {
            if params.num_sweeps <= sweeps_k {
                break;
            }
            let mut sp = params.clone();
            sp.num_reads = reads_k.min(params.num_reads);
            sp.num_sweeps = sweeps_k;
                        // The first stage may arrive prepared by a generator thread, so the device thread only launches kernels.
            let prepared = match pre.take() {
                Some(p) if k == 0 => p,
                _ => self.prepared(graph, &sp, family)?,
            };
            super::device::set_keep_override(1);
            let reads = self.reads(graph, &sp, prepared, cancelled).map_err(sample_error)?;
            super::device::set_keep_override(0);
            let best = reads.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX);
            stage_e.push(best);
            // From stage B on, a job is ranked by the weighted sum of its z-scores over the stages it has run (B, C, D...): each
            // stage's energy is a noisy view of the same instance quality, and the sum predicts the deep result better than the
            // newest stage alone (live data, 229 deep jobs: top quartile by B+C gives 19% deep <= -5083, by C alone 12%).
            let value = if k >= 1 && comb_on() { combined_score_milli(&stage_e) } else { best };
            let pass = {
                let mut ws = self.screen.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                while ws.len() <= k {
                    // A fresh window starts as copies of the stage's reference threshold, so screening works at once.
                    let prior = if ws.len() >= 1 && comb_on() { 0 } else { prior_milli(ws.len()) };
                    ws.push(std::iter::repeat(prior).take(cfg.min).collect());
                }
                let w = &mut ws[k];
                let n = w.len();
                let rank = w.iter().filter(|&&e| e < value).count();
                w.push_back(value);
                while w.len() > cfg.win {
                    w.pop_front();
                }
                n >= cfg.min && rank * q < n
            };
            if !pass {
                return Ok(reads);
            }
            last_reads = Some(reads);
        }
        let prepared = self.prepared(graph, params, family)?;
        let mut all = match self.reads(graph, params, prepared, cancelled) {
            Ok(r) => {
                deep_out.store(true, std::sync::atomic::Ordering::Relaxed);
                r
            }
            // A lineage dropped at the racing checkpoint is not a cancellation: report the last screening stage.
            Err(_) if !cancelled() => return Ok(last_reads.unwrap_or_default()).and_then(|r: Vec<SamplerResult>| if r.is_empty() { Err(SampleError::DeviceBusy) } else { Ok(r) }),
            Err(e) => return Err(sample_error(e)),
        };
        stage_e.push(all.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX));
        // Variable-length attempt: while the best nonce is promising but short of the target, run more independent
        // lineages on the same instance until it reaches the target, the round turns over, or the limit is hit.
        let envf = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
        let exit = match target {
            // Stop as soon as the round's live target is met with a small margin (target is in milli).
            Some(t) => t.saturating_sub((envf("QS_MARGIN", 0.3) * 1000.0).round() as i64),
            None => (envf("QS_EXIT", -5081.0) * 1000.0).round() as i64,
        };
        let promise = (envf("QS_PROMISE", -5081.0) * 1000.0).round() as i64;
        let k_max = envf("QS_FOLLOW", if lease_run::msa_mode() { 6.0 } else { 4.0 }) as u64;
        let mut best = all.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX);
        let mut k = 1u64;
        while best > exit && best <= promise && k <= k_max && !cancelled() {
            let mut p2 = params.clone();
            p2.seed = params.seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let Ok(prepared) = self.prepared(graph, &p2, family) else { break };
            let more = match self.reads(graph, &p2, prepared, cancelled) {
                Ok(m) => m,
                Err(_) if !cancelled() => {
                    k += 1;
                    continue;
                }
                Err(_) => break,
            };
            all.extend(more);
            best = all.iter().map(|r| r.energy_milli).min().unwrap_or(best);
            k += 1;
        }
        if all.len() > 16 {
            all.sort_by_key(|r| r.energy_milli);
            all.truncate(16);
        }
        Ok(all)
    }

    fn reads(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        prepared: PreparedGraph,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<SamplerResult>, ClusterDeviceError> {
        // A governor yield refuses this job as busy. Only the session token
        // makes finish() report Cancelled and suppress the upstream result.
        let stopped = || cancelled() || self.sa.should_throttle();
        checkpoint(&stopped)?;
        let t_new = Instant::now();
        let work = ClusterWork::new(graph, params)?;
        super::device::PROF_US[3].fetch_add(u64::try_from(t_new.elapsed().as_micros()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
        let t_run = Instant::now();
        let run = self.cluster.run(prepared, &work, &stopped, cancelled)?;
        super::device::PROF_US[4].fetch_add(u64::try_from(t_run.elapsed().as_micros()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
        let t_score = Instant::now();
        if run.spins.is_empty()
            || run.spins.len() > params.num_reads
            || run.energies.len() != run.spins.len()
            || run
                .spins
                .iter()
                .any(|s| s.len() != graph.h.len() || s.iter().any(|v| ![-1, 1].contains(v)))
        {
            return Err(ClusterDeviceError::Parameter(
                "incomplete or invalid cluster output",
            ));
        }
        let mut reads = Vec::with_capacity(run.spins.len());
        for spins in run.spins {
            checkpoint(cancelled)?;
            reads.push(SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            });
        }
        super::device::PROF_US[5].fetch_add(u64::try_from(t_score.elapsed().as_micros()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
        Ok(reads)
    }

    fn prepared(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        family: &Fam<'_>,
    ) -> Result<PreparedGraph, SampleError> {
        if family.instance.h().len() != graph.h.len()
            || family.instance.edges().len() != graph.edges.len()
        {
            return Err(SampleError::Capacity);
        }
        PreparedGraph::from_instance(family.instance, family.contract, Weights::from_env(), &self.limits, params.num_reads).map_err(
            |e| {
                tracing::warn!(error = %e, "cluster admission refused");
                SampleError::Capacity
            },
        )
    }

    fn finish(
        job: &StreamJob,
        cancel: &CancelToken,
        run: impl FnOnce() -> Result<Vec<SamplerResult>, SampleError>,
    ) -> StreamResult {
        let start = Instant::now();
        let result = if cancel.is_cancelled(job.watermark) {
            Ok(Vec::new())
        } else {
            run()
        };
        let fatal = result.as_ref().is_err_and(SampleError::is_fatal);
        let outcome = if cancel.is_cancelled(job.watermark) && !fatal {
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

    fn generates_locally() -> bool
    where
        Self: Sized,
    {
        true
    }

    fn sample_lease(
        &self,
        lease: &Lease,
        _topology: &TopologyView,
        params: &SampleParams,
        out: &LeaseSink,
    ) -> Result<(), SampleError> {
        self.run_lease(lease, params, out)
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
        let fam = Fam { instance: family.instance(), contract: family.contract() };
        let prepared = self.prepared(graph, params, &fam)?;
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
            let workers = std::env::var("QS_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).max(1);
            let (wtx, wrx) = std::sync::mpsc::sync_channel::<WarmStreamJob>(workers);
            let wrx = std::sync::Arc::new(Mutex::new(wrx));
            for w in 0..workers {
                let wrx = std::sync::Arc::clone(&wrx);
                let out = out.clone();
                let cancel = cancel.clone();
                scope.spawn(move || {
                    WORKER.with(|c| c.set(w));
                    loop {
                        let item = {
                            let guard = wrx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                            guard.recv()
                        };
                        let Ok(item) = item else { break };
                        let job = item.job;
                        let Some(family) = item.family.as_ref().filter(|f| supported(f)) else { continue };
                        let result = Self::finish(&job, &cancel, || {
                            let fam = Fam { instance: family.instance(), contract: family.contract() };
                            let deep = std::sync::atomic::AtomicBool::new(false);
                            self.screened_reads(&job.graph, &job.params, &fam, &|| {
                                cancel.is_cancelled(job.watermark)
                            }, &deep, None, None, &mut Vec::new())
                        });
                        if out.blocking_send(result).is_err() {
                            break;
                        }
                    }
                });
            }
            while let Some(item) = jobs.blocking_recv() {
                if out.is_closed() {
                    break;
                }
                if item.family.as_ref().is_some_and(|f| supported(f)) {
                    if wtx.send(item).is_err() {
                        break;
                    }
                } else if sa_tx.blocking_send(item.job).is_err() {
                    break;
                }
            }
            drop(wtx);
            drop(sa_tx);
        });
    }

    fn stream_width(&self) -> usize {
        self.sa.stream_width()
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
