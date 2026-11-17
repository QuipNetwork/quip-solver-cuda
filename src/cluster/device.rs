//! On-demand Otsuka cluster optimizer. Component updates are a heuristic.

use super::{PrepareError, PreparedGraph, SpinError};
use crate::capacity::{CLUSTER_STATIC_SHARED_BYTES, CLUSTER_THREADS};
use crate::cuda_device::{compile_for_arch, device_arch, driver_version, CudaError};
use crate::jit_cache;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const SOURCE: &str = include_str!("../../kernels/cluster.cu");

#[cfg(all(test, feature = "long-tests"))]
#[path = "generation_cost.rs"]
mod generation_cost;

/// Research schedule and policy defaults from `run_cluster.py:39-46`.
/// Weights belong to the prepared graph. Research defaults use 250 milli.
#[derive(Clone, Debug)]
pub struct ClusterSchedule {
    /// Total cluster sweeps, in `1..=i32::MAX`.
    pub sweeps: usize,
    /// Period of colored single-spin sweeps. Zero disables these sweeps.
    pub single_every: usize,
    /// Random groups for parallel component decisions. Must be positive.
    pub rounds: usize,
    /// Maximum sweeps per launch. Must be positive.
    pub chunk: usize,
    /// Positive starting beta.
    pub beta_min: f32,
    /// Optional intermediate beta. Zero disables the knee.
    pub beta_knee: f32,
    /// Fraction of the schedule at the knee, strictly between zero and one.
    /// Zero disables the knee.
    pub knee_frac: f32,
    /// Positive final beta.
    pub beta_max: f32,
}

impl Default for ClusterSchedule {
    fn default() -> Self {
        Self {
            sweeps: 65536,
            single_every: 1,
            rounds: 2,
            chunk: 2048,
            beta_min: 1.0,
            beta_knee: 0.0,
            knee_frac: 0.0,
            beta_max: 12.0,
        }
    }
}

/// Admission, compilation, allocation, launch, or result failure.
#[derive(Debug, thiserror::Error)]
pub enum ClusterDeviceError {
    /// Cluster jobs require allocations that do not synchronize resident SA.
    #[error("cluster requires CUDA memory pools for asynchronous allocation; use quip-cuda-sa on this device")]
    MemoryPoolsUnsupported,
    /// The caller stopped this job between synchronized launches.
    #[error("cluster job cancelled")]
    Cancelled,
    /// The job supplied an unsupported schedule or read count.
    #[error("invalid cluster job parameter: {0}")]
    JobParameter(&'static str),
    /// A schedule or launch dimension is invalid.
    #[error("invalid cluster parameter: {0}")]
    Parameter(&'static str),
    /// CU-1 capacity admission rejected the launch on this device.
    #[error(transparent)]
    Prepare(#[from] PrepareError),
    /// Kernel compilation or module loading failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// A CUDA driver operation failed.
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    /// Device output contains an invalid spin state.
    #[error(transparent)]
    Spin(#[from] SpinError),
}

// Params in gpucluster.cu:63-66, passed by value. No pointer or Rust padding fields.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    sweeps: i32,
    single_every: i32,
    rounds: i32,
    chunk: i32,
    beta_min: f32,
    beta_knee: f32,
    knee_frac: f32,
    beta_max: f32,
    tile_w: f32,
    dimer_w: f32,
}
// SAFETY: repr(C), ten initialized 32-bit scalars, matching CUDA Params exactly.
unsafe impl DeviceRepr for Params {}

impl ClusterSchedule {
    /// Check finite betas and kernel integer widths without opening CUDA.
    ///
    /// # Errors
    /// Returns a parameter error for an invalid count or beta.
    pub fn validate(&self) -> Result<(), ClusterDeviceError> {
        self.params().map(|_| ())
    }

    fn params(&self) -> Result<Params, ClusterDeviceError> {
        let int =
            |v| i32::try_from(v).map_err(|_| ClusterDeviceError::Parameter("count exceeds i32"));
        if self.sweeps == 0 || self.chunk == 0 || self.rounds == 0 {
            return Err(ClusterDeviceError::Parameter(
                "sweeps, chunk, and rounds must be positive",
            ));
        }
        if ![self.beta_min, self.beta_knee, self.knee_frac, self.beta_max]
            .iter()
            .all(|v| v.is_finite())
            || self.beta_min <= 0.0
            || self.beta_max <= 0.0
            || self.beta_knee < 0.0
            || !(0.0..1.0).contains(&self.knee_frac)
        {
            return Err(ClusterDeviceError::Parameter(
                "betas must be finite and positive; knee fraction must be in [0, 1)",
            ));
        }
        Ok(Params {
            sweeps: int(self.sweeps)?,
            single_every: int(self.single_every)?,
            rounds: int(self.rounds)?,
            chunk: int(self.chunk)?,
            beta_min: self.beta_min,
            beta_knee: self.beta_knee,
            knee_frac: self.knee_frac,
            beta_max: self.beta_max,
            tile_w: 250.0,
            dimer_w: 250.0,
        })
    }
}

/// Every anneal's final state and original-model energy.
#[derive(Debug)]
pub struct ClusterRun {
    /// Spins in original node order, one vector per anneal.
    pub spins: Vec<Vec<i8>>,
    /// Host-rescored energy in milli, one entry per anneal.
    pub energies: Vec<i64>,
}

/// One prepared model and loaded function on one CUDA device.
/// Opening this solver compiles only the cluster kernel.
pub struct ClusterDevice {
    graph: PreparedGraph,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    tiles: CudaSlice<i32>,
    gauges: CudaSlice<i8>,
    pairs: CudaSlice<i32>,
    weak_deg: CudaSlice<i32>,
    weak_ell: CudaSlice<i32>,
    all_deg: CudaSlice<i32>,
    all_ell: CudaSlice<i32>,
    color_off: CudaSlice<i32>,
    h: CudaSlice<i32>,
    tile_w: CudaSlice<i32>,
    link_w: CudaSlice<i32>,
}

pub(crate) struct ClusterKernel {
    stream: Arc<CudaStream>,
    function: CudaFunction,
}

impl ClusterKernel {
    pub(crate) fn load(device_index: usize) -> Result<Self, ClusterDeviceError> {
        let ctx = CudaContext::new(device_index)?;
        // Synchronous allocation can wait for the persistent SA kernel.
        // Refuse before JIT or stream creation, while SA is still idle.
        check_async_alloc(ctx.has_async_alloc())?;
        let arch = device_arch(&ctx);
        let module = jit_cache::load_or_compile(
            &ctx,
            "cluster",
            SOURCE,
            &[],
            &format!("sm_{arch}"),
            driver_version()?,
            0,
            || compile_for_arch(SOURCE, 0, arch, &[]),
        )?;
        // Resolve the function before SA launches. Lazy module loading can
        // synchronize the primary context, even with nonblocking streams.
        let function = module.load_function("quip_cluster_anneal")?;
        let static_shared = usize::try_from(function.shared_size_bytes()?)
            .map_err(|_| ClusterDeviceError::Parameter("negative static shared memory"))?;
        if static_shared > CLUSTER_STATIC_SHARED_BYTES {
            return Err(ClusterDeviceError::Parameter(
                "static shared memory exceeds CU-1 reservation",
            ));
        }
        // cudarc synchronizes the context when creating its first stream.
        // Retain this stream too, so no per-job creation waits on resident SA.
        let stream = ctx.new_stream()?;
        Ok(Self { stream, function })
    }
}

fn envn(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Otsuka population-anneal settings. Each comes from the `otsuka_*` key of the miner's config
/// section, else the `QUIP_OTSUKA_*` environment variable, else the default:
///
/// - `beta`: the family beta range `(hot, cold)` for jobs that carry none. Default `(1, 8)`, near the
///   CPU sampler's `FAMILY_BETA_RANGE`; the generic SA range (about 0.04 to 105) wastes most of the
///   ladder on twisted-ice instances. `"stock"` restores the job's own range.
/// - `chunk`: sweeps per kernel launch. Default 16.
/// - `resamples`: population steps per anneal. Default 256; 0 turns the population step off.
/// - `houdayer`: Houdayer swap rounds per population step from beta 3. Default 1.
/// - `reads`, `sweeps`: pin the job budget. Default 0 (the job's own budget).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Knobs {
    pub beta: Option<(f64, f64)>,
    pub chunk: usize,
    pub resamples: usize,
    pub houdayer: usize,
    pub reads: usize,
    pub sweeps: usize,
}

static KNOBS: std::sync::RwLock<Option<Knobs>> = std::sync::RwLock::new(None);

/// The default family beta range `(hot, cold)`.
const DEFAULT_BETA: (f64, f64) = (1.0, 8.0);

/// A usable beta range: both bounds finite and positive, hot below cold.
fn beta_pair(hot: f64, cold: f64) -> Option<(f64, f64)> {
    (hot.is_finite() && cold.is_finite() && hot > 0.0 && hot < cold).then_some((hot, cold))
}

/// `"stock"` is `Ok(None)` (the job's own range), `"hot,cold"` is `Ok(Some(..))`, anything else is
/// an error so the caller keeps its current range.
fn parse_beta(v: &str) -> Result<Option<(f64, f64)>, ()> {
    if v.trim() == "stock" {
        return Ok(None);
    }
    let parts: Vec<f64> = v
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| ())?;
    match parts[..] {
        [hot, cold] => beta_pair(hot, cold).map(Some).ok_or(()),
        _ => Err(()),
    }
}

/// [`parse_beta`] for a setting named `source`, warning and keeping `current` on a bad value.
fn beta_or(source: &str, v: &str, current: Option<(f64, f64)>) -> Option<(f64, f64)> {
    parse_beta(v).unwrap_or_else(|()| {
        tracing::warn!(source, value = v, "ignoring an invalid otsuka beta range");
        current
    })
}

pub(crate) fn knobs() -> Knobs {
    if let Some(k) = *KNOBS
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return k;
    }
    let k = Knobs {
        beta: match std::env::var("QUIP_OTSUKA_BETA") {
            Ok(v) => beta_or("QUIP_OTSUKA_BETA", &v, Some(DEFAULT_BETA)),
            Err(_) => Some(DEFAULT_BETA),
        },
        chunk: envn("QUIP_OTSUKA_CHUNK", 16).max(1),
        resamples: envn("QUIP_OTSUKA_RESAMPLES", 256),
        houdayer: envn("QUIP_OTSUKA_HOUDAYER", 1),
        reads: envn("QUIP_OTSUKA_READS", 0),
        sweeps: envn("QUIP_OTSUKA_SWEEPS", 0),
    };
    *KNOBS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(k);
    k
}

/// The `otsuka_*` keys of the miner's config section override env and defaults.
#[expect(
    clippy::cast_precision_loss,
    reason = "an integer beta bound in the config is a small number"
)]
pub(crate) fn knobs_from_toml(backend_toml: &str) {
    let Ok(t) = backend_toml.parse::<toml::Table>() else {
        return;
    };
    let mut k = knobs();
    let int = |key: &str| {
        t.get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|v| usize::try_from(v).ok())
    };
    if let Some(v) = int("otsuka_chunk") {
        k.chunk = v.max(1);
    }
    if let Some(v) = int("otsuka_resamples") {
        k.resamples = v;
    }
    if let Some(v) = int("otsuka_houdayer") {
        k.houdayer = v;
    }
    if let Some(v) = int("otsuka_reads") {
        k.reads = v;
    }
    if let Some(v) = int("otsuka_sweeps") {
        k.sweeps = v;
    }
    match t.get("otsuka_beta") {
        Some(toml::Value::String(v)) => k.beta = beta_or("otsuka_beta", v, k.beta),
        Some(toml::Value::Array(a)) => {
            let f = |x: &toml::Value| x.as_float().or_else(|| x.as_integer().map(|i| i as f64));
            match (a.len(), a.first().and_then(f), a.get(1).and_then(f)) {
                (2, Some(hot), Some(cold)) if beta_pair(hot, cold).is_some() => {
                    k.beta = Some((hot, cold));
                }
                _ => tracing::warn!(value = ?a, "ignoring an invalid otsuka_beta"),
            }
        }
        Some(other) => tracing::warn!(value = ?other, "ignoring an invalid otsuka_beta"),
        None => {}
    }
    *KNOBS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(k);
    tracing::info!(?k, "otsuka settings");
}

/// Host state of the population anneal (resampling and Houdayer moves between launches).
struct Pop {
    every: usize,
    since: usize,
    last_beta: f32,
    houdayer: usize,
    rng: u64,
    best: Option<(Vec<i8>, i64)>,
    adj: Vec<Vec<u32>>,
    steps: usize,
    host: std::time::Duration,
}

impl Pop {
    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits scaled to [0, 1) are exact in f64"
    )]
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The beta from which the population step pairs replicas for Houdayer moves, as in the CPU
/// sampler. Below it the replicas differ almost everywhere and a move only swaps a pair.
const HOUDAYER_BETA: f32 = 3.0;

/// Systematic resampling: the replica each slot copies, drawn in proportion to
/// `exp(-step * (E - E_min))` (energies in milli) with the single uniform offset `unit` in `[0, 1)`.
/// The result keeps the population size and lists replicas in ascending order.
#[expect(
    clippy::cast_precision_loss,
    reason = "milli energies and replica counts are far below 2^52"
)]
fn systematic_resample(energy: &[i64], step: f64, unit: f64) -> Vec<usize> {
    let r = energy.len();
    if r == 0 {
        return Vec::new();
    }
    let low = energy.iter().copied().min().unwrap_or(0);
    let weights: Vec<f64> = energy
        .iter()
        .map(|&e| (-step * (e - low) as f64 / 1000.0).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let stride = total / r as f64;
    let mut mark = unit * stride;
    let mut picks = Vec::with_capacity(r);
    let mut reach = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        reach += w;
        while mark < reach && picks.len() < r {
            picks.push(i);
            mark += stride;
        }
    }
    // Rounding in the running sum can leave the last slot unfilled.
    picks.resize(r, r - 1);
    picks
}

/// `rounds` rounds of Houdayer moves over random pairs of `states`. Each move picks a node where the
/// pair differs, grows the connected cluster of differing nodes over `adj`, and swaps it between
/// the two. Every edge that leaves the cluster ends on a node where the pair agrees, so a move
/// keeps the summed energy of the pair.
#[expect(
    clippy::cast_possible_truncation,
    reason = "node and replica indices stay below MAX_PACKED_NODES and MAX_READS"
)]
fn houdayer_rounds(
    states: &mut [Vec<i8>],
    adj: &[Vec<u32>],
    rounds: usize,
    next: &mut dyn FnMut() -> u64,
) {
    let replicas = states.len();
    let nodes = adj.len();
    let mut order: Vec<usize> = (0..replicas).collect();
    let mut seen = vec![false; nodes];
    let mut differ: Vec<u32> = Vec::new();
    let mut cluster: Vec<u32> = Vec::new();
    for _ in 0..rounds {
        for i in (1..replicas).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        for pair in order.chunks_exact(2) {
            let (lo, hi) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
            let (left, right) = states.split_at_mut(hi);
            let (sa, sb) = (&mut left[lo], &mut right[0]);
            differ.clear();
            differ.extend((0..nodes as u32).filter(|&u| sa[u as usize] != sb[u as usize]));
            if differ.is_empty() {
                continue;
            }
            let root = differ[(next() % differ.len() as u64) as usize];
            cluster.clear();
            cluster.push(root);
            seen[root as usize] = true;
            let mut at = 0;
            while at < cluster.len() {
                let node = cluster[at] as usize;
                at += 1;
                for &v in &adj[node] {
                    let v = v as usize;
                    if !seen[v] && sa[v] != sb[v] {
                        seen[v] = true;
                        cluster.push(v as u32);
                    }
                }
            }
            for &node in &cluster {
                let node = node as usize;
                seen[node] = false;
                sa[node] = -sa[node];
                sb[node] = -sb[node];
            }
        }
    }
}

struct WorkControl<'a> {
    work: &'a super::sampler::ClusterWork,
    stopped: &'a dyn Fn() -> bool,
    cancelled: &'a dyn Fn() -> bool,
    chunk: usize,
    /// Run the host population step between launches (the miner's job path).
    population: bool,
}

impl ClusterDevice {
    /// Open a device, check its real budgets, and upload the CU-1 layout.
    /// Retains the device primary context with normal event tracking.
    ///
    /// # Errors
    /// Returns capacity, compilation, allocation, or driver errors.
    pub fn open(device_index: usize, graph: PreparedGraph) -> Result<Self, ClusterDeviceError> {
        let kernel = ClusterKernel::load(device_index)?;
        Self::open_with_kernel(graph, &kernel)
    }

    pub(crate) fn open_with_kernel(
        graph: PreparedGraph,
        kernel: &ClusterKernel,
    ) -> Result<Self, ClusterDeviceError> {
        let ctx = kernel.stream.context();
        check_shared(
            &graph,
            attribute(
                ctx,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?,
        )?;
        check_memory(&graph, graph.footprint().instances, ctx.mem_get_info()?.0)?;
        let function = kernel.function.clone();
        function.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(graph.dynamic_shared_bytes())
                .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds i32"))?,
        )?;
        let stream = Arc::clone(&kernel.stream);
        let d = graph.layout();
        Ok(Self {
            tiles: upload(&stream, &d.tiles)?,
            gauges: upload(&stream, &d.gauges)?,
            pairs: upload(&stream, &d.pairs)?,
            weak_deg: upload(&stream, &d.weak_deg)?,
            weak_ell: upload(&stream, &d.weak_ell)?,
            all_deg: upload(&stream, &d.all_deg)?,
            all_ell: upload(&stream, &d.all_ell)?,
            color_off: upload(&stream, &d.color_off)?,
            h: upload(&stream, &d.h)?,
            tile_w: upload(&stream, &d.tile_w)?,
            link_w: upload(&stream, &d.link_w)?,
            graph,
            stream,
            function,
        })
    }

    /// Run a bounded sequence of cluster launches followed by final descent.
    /// All calls start from the supplied seed, without continuing a prior run.
    /// State and RNG storage follow `gpucluster.cu:309-321`.
    ///
    /// # Errors
    /// Rejects invalid schedules, overflowing dimensions, and device budgets
    /// before allocation. Returns all CUDA failures and invalid output states.
    pub fn run(
        &mut self,
        schedule: &ClusterSchedule,
        anneals: usize,
        seed: u64,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_inner(schedule, anneals, seed, true, None)
            .map(|(run, _)| run)
    }

    /// Test access to the state before final descent, with all per-thread RNG
    /// words in anneal-major order. Spins retain the original node order.
    /// This skips only final descent and exists for integration-test oracles.
    ///
    /// # Errors
    /// Returns the same validation and CUDA errors as `run`.
    #[doc(hidden)]
    pub fn run_before_descent_for_test(
        &mut self,
        schedule: &ClusterSchedule,
        anneals: usize,
        seed: u64,
    ) -> Result<(ClusterRun, Vec<u64>), ClusterDeviceError> {
        self.run_inner(schedule, anneals, seed, false, None)
    }

    /// Run a host-built ladder with one synchronized sweep per launch.
    ///
    /// # Errors
    /// Returns admission, device, or cancellation errors without partial output.
    #[doc(hidden)]
    pub fn run_work(
        &mut self,
        work: &super::sampler::ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_work_chunked(work, stopped, cancelled, 1, false)
    }

    /// Run a host-built ladder as the miner does: `otsuka_chunk` sweeps per launch and the host
    /// population step between launches (see [`Knobs`]).
    ///
    /// Unlike [`Self::run_work`], the result depends on the launch chunk, because the population
    /// step runs at launch boundaries.
    ///
    /// # Errors
    /// Returns the same validation and device errors as `run_work`.
    #[doc(hidden)]
    pub fn run_work_population(
        &mut self,
        work: &super::sampler::ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_work_chunked(work, stopped, cancelled, knobs().chunk, true)
    }

    /// Run the identical host ladder with larger launch chunks as a GPU oracle.
    /// Each chunk stays within one beta rung. Final descent uses one launch.
    ///
    /// # Errors
    /// Returns the same validation and device errors as `run_work`.
    #[doc(hidden)]
    pub fn run_work_with_chunk_for_test(
        &mut self,
        work: &super::sampler::ClusterWork,
        chunk: usize,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_work_chunked(work, &|| false, &|| false, chunk, false)
    }

    fn run_work_chunked(
        &mut self,
        work: &super::sampler::ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        chunk: usize,
        population: bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        let schedule = ClusterSchedule {
            sweeps: work.betas.len() * work.sweeps_per_beta,
            chunk: 1,
            ..Default::default()
        };
        self.run_inner(
            &schedule,
            work.reads,
            work.seed,
            true,
            Some(&WorkControl {
                work,
                stopped,
                cancelled,
                chunk,
                population,
            }),
        )
        .map(|(run, _)| run)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the population step hooks into the launch loop it shares with stock runs"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "node indices stay below MAX_PACKED_NODES, far below u32::MAX"
    )]
    fn run_inner(
        &mut self,
        schedule: &ClusterSchedule,
        anneals: usize,
        seed: u64,
        final_descent: bool,
        control: Option<&WorkControl<'_>>,
    ) -> Result<(ClusterRun, Vec<u64>), ClusterDeviceError> {
        let params = schedule.params()?;
        let ctx = self.stream.context();
        check_shared(
            &self.graph,
            attribute(
                ctx,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?,
        )?;
        // Model buffers already occupy memory. Add them back for the CU-1 total estimate.
        let model_bytes = self.tiles.num_bytes()
            + self.gauges.num_bytes()
            + self.pairs.num_bytes()
            + self.weak_deg.num_bytes()
            + self.weak_ell.num_bytes()
            + self.all_deg.num_bytes()
            + self.all_ell.num_bytes()
            + self.color_off.num_bytes()
            + self.h.num_bytes()
            + self.tile_w.num_bytes()
            + self.link_w.num_bytes();
        let free = ctx.mem_get_info()?.0.saturating_add(model_bytes);
        let (spin_count, rng_count, blocks) = launch_shape(&self.graph, anneals)?;
        check_memory(&self.graph, anneals, free)?;
        let (spins, rng) = initial_state(spin_count, rng_count, seed);
        let mut spins = self.stream.clone_htod(&spins)?;
        let mut rng = self.stream.clone_htod(&rng)?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: u32::try_from(self.graph.dynamic_shared_bytes())
                .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds u32"))?,
        };
        let total_sweeps = usize::try_from(params.sweeps).unwrap_or(0);
        let resamples = knobs().resamples;
        let population = control.is_some_and(|c| c.population);
        let mut pop = (population && resamples > 0 && anneals > 1).then(|| Pop {
            every: (total_sweeps / resamples).max(1),
            since: 0,
            last_beta: 0.0,
            houdayer: knobs().houdayer,
            rng: seed ^ 0x9e37_79b9_7f4a_7c15 | 1,
            best: None,
            adj: {
                let g = self.graph.original_graph();
                let mut adj = vec![Vec::new(); g.num_nodes()];
                for &(u, v, _) in &g.edges {
                    adj[u].push(v as u32);
                    adj[v].push(u as u32);
                }
                adj
            },
            steps: 0,
            host: std::time::Duration::ZERO,
        });
        if let Some(control) = control {
            control.work.for_each_chunk(
                control.stopped,
                control.chunk,
                |start, count, beta, descent| {
                    let constant_beta = Params {
                        beta_min: beta,
                        beta_max: beta,
                        ..params
                    };
                    self.launch(
                        constant_beta,
                        i32::try_from(start)
                            .map_err(|_| ClusterDeviceError::Parameter("sweep exceeds i32"))?,
                        if descent {
                            0
                        } else {
                            i32::try_from(count)
                                .map_err(|_| ClusterDeviceError::Parameter("chunk exceeds i32"))?
                        },
                        if !descent {
                            0
                        } else if count == 1 {
                            2
                        } else {
                            1
                        },
                        config,
                        &mut spins,
                        &mut rng,
                    )?;
                    self.stream.synchronize()?;
                    if let Some(pop) = pop.as_mut() {
                        if !descent {
                            if pop.steps == 0 && pop.since == 0 {
                                pop.last_beta = beta;
                            }
                            pop.since += count;
                            if pop.since >= pop.every {
                                pop.since = 0;
                                self.population_step(pop, beta, &mut spins)?;
                            }
                        }
                    }
                    Ok(())
                },
            )?;
        } else {
            let mut start = 0;
            while start < params.sweeps {
                let count = params.chunk.min(params.sweeps - start);
                self.launch(params, start, count, 0, config, &mut spins, &mut rng)?;
                start += count;
            }
            if final_descent {
                self.launch(params, 0, 0, 1, config, &mut spins, &mut rng)?;
            }
        }
        let states = self.stream.clone_dtoh(&spins)?;
        let rng_state = if final_descent {
            Vec::new()
        } else {
            self.stream.clone_dtoh(&rng)?
        };
        self.stream.synchronize()?;
        let mut result = ClusterRun {
            spins: Vec::with_capacity(anneals),
            energies: Vec::with_capacity(anneals),
        };
        for s in states.chunks_exact(self.graph.original_graph().num_nodes()) {
            if control.is_some_and(|c| (c.cancelled)()) {
                return Err(ClusterDeviceError::Cancelled);
            }
            let original = self.graph.unpermute(s)?;
            result.energies.push(self.graph.energy(&original)?);
            result.spins.push(original);
        }
        if let Some(pop) = pop {
            tracing::debug!(
                steps = pop.steps,
                host_ms = u64::try_from(pop.host.as_millis()).unwrap_or(u64::MAX),
                "otsuka population steps"
            );
            if let Some((state, e)) = pop.best {
                let low = result.energies.iter().copied().min().unwrap_or(i64::MAX);
                if e < low {
                    if let Some(worst) =
                        (0..result.energies.len()).max_by_key(|&i| result.energies[i])
                    {
                        result.energies[worst] = e;
                        result.spins[worst] = state;
                    }
                }
            }
        }
        Ok((result, rng_state))
    }

    /// One population step on the host. Downloads the replicas, resamples them in proportion to
    /// `exp(-(beta - last_beta) (E - E_min))` with systematic resampling (true model energies), takes
    /// Houdayer cluster swaps between random pairs from beta 3, and uploads the result. This is the
    /// GPU counterpart of `anneal_population` in quip-solver-cpu's Otsuka sampler.
    fn population_step(
        &self,
        pop: &mut Pop,
        beta: f32,
        spins: &mut CudaSlice<i8>,
    ) -> Result<(), ClusterDeviceError> {
        let t0 = std::time::Instant::now();
        let g = self.graph.original_graph();
        let n = g.num_nodes();
        let o2d = self.graph.original_to_device();
        let d2o = self.graph.device_to_original();
        let dev = self.stream.clone_dtoh(spins)?;
        self.stream.synchronize()?;
        let mut states: Vec<Vec<i8>> = dev
            .chunks_exact(n)
            .map(|s| o2d.iter().map(|&d| s[d]).collect())
            .collect();
        let r = states.len();
        let mut energy = vec![0i64; r];
        let per = r.div_ceil(8).max(1);
        std::thread::scope(|sc| {
            for (es, ss) in energy.chunks_mut(per).zip(states.chunks(per)) {
                sc.spawn(move || {
                    for (e, s) in es.iter_mut().zip(ss) {
                        *e = g.energy(s);
                    }
                });
            }
        });
        let (low_at, low) = energy
            .iter()
            .copied()
            .enumerate()
            .min_by_key(|&(_, e)| e)
            .unwrap_or((0, 0));
        if pop.best.as_ref().is_none_or(|b| low < b.1) {
            pop.best = Some((states[low_at].clone(), low));
        }
        let step = f64::from(beta - pop.last_beta);
        if step > 0.0 {
            let picks = systematic_resample(&energy, step, pop.unit());
            states = picks.iter().map(|&i| states[i].clone()).collect();
        }
        if pop.houdayer > 0 && beta >= HOUDAYER_BETA {
            let adj = std::mem::take(&mut pop.adj);
            houdayer_rounds(&mut states, &adj, pop.houdayer, &mut || pop.next());
            pop.adj = adj;
        }
        let mut up = vec![0i8; r * n];
        for (block, state) in up.chunks_exact_mut(n).zip(&states) {
            for (d, slot) in block.iter_mut().enumerate() {
                *slot = state[d2o[d]];
            }
        }
        self.stream.memcpy_htod(&up, spins)?;
        self.stream.synchronize()?;
        pop.last_beta = beta;
        pop.steps += 1;
        pop.host += t0.elapsed();
        Ok(())
    }

    // The flat argument order matches quip_cluster_anneal in kernels/cluster.cu.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        params: Params,
        start: i32,
        count: i32,
        final_pass: i32,
        config: LaunchConfig,
        spins: &mut CudaSlice<i8>,
        rng: &mut CudaSlice<u64>,
    ) -> Result<(), ClusterDeviceError> {
        let h = self.graph.layout().header;
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(&h[0])
            .arg(&h[1])
            .arg(&h[2])
            .arg(&h[5])
            .arg(&self.tiles)
            .arg(&self.gauges)
            .arg(&self.pairs)
            .arg(&self.weak_deg)
            .arg(&self.weak_ell)
            .arg(&self.all_deg)
            .arg(&self.all_ell)
            .arg(&self.color_off)
            .arg(&self.h)
            .arg(&self.tile_w)
            .arg(&self.link_w)
            .arg(&params)
            .arg(&start)
            .arg(&count)
            .arg(spins)
            .arg(rng)
            .arg(&final_pass);
        // SAFETY: CU-1 validates indices, ELL products, and shared layout.
        // launch_shape checks per-anneal allocations. Arguments match the CUDA ABI.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

fn upload<T: DeviceRepr + Default + Clone>(
    stream: &Arc<CudaStream>,
    values: &[T],
) -> Result<CudaSlice<T>, ClusterDeviceError> {
    if values.is_empty() {
        Ok(stream.clone_htod(&[T::default()])?)
    } else {
        Ok(stream.clone_htod(values)?)
    }
}

fn attribute(ctx: &CudaContext, attr: CUdevice_attribute) -> Result<usize, ClusterDeviceError> {
    usize::try_from(ctx.attribute(attr)?)
        .map_err(|_| ClusterDeviceError::Parameter("negative CUDA attribute"))
}

fn check_async_alloc(has_async_alloc: bool) -> Result<(), ClusterDeviceError> {
    if has_async_alloc {
        Ok(())
    } else {
        Err(ClusterDeviceError::MemoryPoolsUnsupported)
    }
}

fn check_shared(graph: &PreparedGraph, budget: usize) -> Result<(), ClusterDeviceError> {
    let requested = graph
        .dynamic_shared_bytes()
        .checked_add(CLUSTER_STATIC_SHARED_BYTES)
        .ok_or(ClusterDeviceError::Parameter("shared memory size overflow"))?;
    if requested > budget {
        return Err(PrepareError::SharedMemory { requested, budget }.into());
    }
    Ok(())
}

fn check_memory(
    graph: &PreparedGraph,
    anneals: usize,
    free: usize,
) -> Result<(), ClusterDeviceError> {
    let mut footprint = graph.footprint();
    footprint.instances = anneals;
    let requested = footprint.total();
    let budget = free / crate::capacity::MEMORY_HEADROOM_DEN * crate::capacity::MEMORY_HEADROOM_NUM;
    if requested > budget {
        return Err(PrepareError::DeviceMemory { requested, budget }.into());
    }
    Ok(())
}

fn launch_shape(
    graph: &PreparedGraph,
    anneals: usize,
) -> Result<(usize, usize, u32), ClusterDeviceError> {
    if anneals == 0 {
        return Err(PrepareError::EmptyLaunch.into());
    }
    // CUDA grid x is signed 31-bit on every supported architecture.
    let blocks = i32::try_from(anneals)
        .ok()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(ClusterDeviceError::Parameter("anneals exceed grid width"))?;
    let count = |width: usize| {
        anneals
            .checked_mul(width)
            .filter(|&n| n <= isize::MAX.unsigned_abs())
            .ok_or(ClusterDeviceError::Parameter("allocation size overflow"))
    };
    let spins = count(graph.original_graph().num_nodes())?;
    let rng = count(CLUSTER_THREADS)?;
    count(CLUSTER_THREADS * std::mem::size_of::<u64>())?;
    Ok((spins, rng, blocks))
}

// gpucluster.cu:311-319. Wrapping arithmetic preserves unsigned C++ operations.
fn initial_state(spins: usize, threads: usize, seed: u64) -> (Vec<i8>, Vec<u64>) {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x0123_4567);
    let rng = (0..threads)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) | 1
        })
        .collect();
    let mut y = seed ^ 0xD1B5_4A32_D192_ED03;
    let spins = (0..spins)
        .map(|_| {
            y ^= y << 13;
            y ^= y >> 7;
            y ^= y << 17;
            if (y >> 33) & 1 == 1 {
                1
            } else {
                -1
            }
        })
        .collect();
    (spins, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::DeviceLimits;
    use crate::cluster::{Model, Motifs, Weights};

    #[test]
    fn beta_settings_accept_ranges_and_stock_and_reject_the_rest() {
        assert_eq!(parse_beta("stock"), Ok(None));
        assert_eq!(parse_beta(" 0.5 , 12 "), Ok(Some((0.5, 12.0))));
        for bad in [
            "", "1", "1,8,9", "8,1", "0,8", "-1,8", "1,x", "1,inf", "stock,1",
        ] {
            assert_eq!(parse_beta(bad), Err(()), "{bad:?}");
        }
        assert_eq!(
            beta_or("test", "8,1", Some(DEFAULT_BETA)),
            Some(DEFAULT_BETA)
        );
        assert_eq!(beta_or("test", "stock", Some(DEFAULT_BETA)), None);
    }

    #[test]
    fn systematic_resampling_keeps_the_count_and_favours_low_energies() {
        // Zero step: every replica is kept once.
        assert_eq!(
            systematic_resample(&[5, -3, 0, 7], 0.0, 0.5),
            vec![0, 1, 2, 3]
        );
        // A large step puts every slot on the lowest replica.
        assert_eq!(
            systematic_resample(&[5_000, -3_000, 0, 7_000], 50.0, 0.3),
            vec![1; 4]
        );
        // Any offset in [0, 1) gives a full, ordered population.
        for unit in [0.0, 0.25, 0.999_999] {
            let picks = systematic_resample(&[0, 1_000, 2_000, 3_000, 0], 1.0, unit);
            assert_eq!(picks.len(), 5);
            assert!(picks.windows(2).all(|w| w[0] <= w[1]));
        }
        assert!(systematic_resample(&[], 1.0, 0.5).is_empty());
    }

    fn pair_energy(edges: &[(usize, usize, i64)], s: &[i8]) -> i64 {
        edges
            .iter()
            .map(|&(u, v, j)| j * i64::from(s[u]) * i64::from(s[v]))
            .sum()
    }

    #[test]
    fn houdayer_moves_keep_each_pairs_summed_energy() {
        let mut rng = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let n = 40;
        let mut edges = Vec::new();
        for u in 0..n {
            for v in u + 1..n {
                if next() % 5 == 0 {
                    edges.push((u, v, i64::try_from(next() % 2001).unwrap() - 1000));
                }
            }
        }
        let mut adj = vec![Vec::new(); n];
        for &(u, v, _) in &edges {
            adj[u].push(u32::try_from(v).unwrap());
            adj[v].push(u32::try_from(u).unwrap());
        }
        let mut states: Vec<Vec<i8>> = (0..2)
            .map(|_| {
                (0..n)
                    .map(|_| if next() % 2 == 0 { 1 } else { -1 })
                    .collect()
            })
            .collect();
        for _ in 0..50 {
            let before = pair_energy(&edges, &states[0]) + pair_energy(&edges, &states[1]);
            let spins_before: Vec<i32> = (0..n)
                .map(|u| i32::from(states[0][u]) + i32::from(states[1][u]))
                .collect();
            houdayer_rounds(&mut states, &adj, 1, &mut next);
            let after = pair_energy(&edges, &states[0]) + pair_energy(&edges, &states[1]);
            assert_eq!(before, after);
            // A swap moves spins between the pair; it never changes their per-node sum.
            let spins_after: Vec<i32> = (0..n)
                .map(|u| i32::from(states[0][u]) + i32::from(states[1][u]))
                .collect();
            assert_eq!(spins_before, spins_after);
        }
    }

    #[test]
    fn cluster_requires_memory_pools() {
        assert!(matches!(
            check_async_alloc(false),
            Err(ClusterDeviceError::MemoryPoolsUnsupported)
        ));
        assert!(check_async_alloc(true).is_ok());
    }

    #[test]
    fn params_match_cuda_value_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 40);
        assert_eq!(std::mem::align_of::<Params>(), 4);
        assert_eq!(std::mem::offset_of!(Params, sweeps), 0);
        assert_eq!(std::mem::offset_of!(Params, single_every), 4);
        assert_eq!(std::mem::offset_of!(Params, rounds), 8);
        assert_eq!(std::mem::offset_of!(Params, chunk), 12);
        assert_eq!(std::mem::offset_of!(Params, beta_min), 16);
        assert_eq!(std::mem::offset_of!(Params, beta_knee), 20);
        assert_eq!(std::mem::offset_of!(Params, knee_frac), 24);
        assert_eq!(std::mem::offset_of!(Params, beta_max), 28);
        assert_eq!(std::mem::offset_of!(Params, tile_w), 32);
        assert_eq!(std::mem::offset_of!(Params, dimer_w), 36);
    }

    #[test]
    fn initialization_matches_research_seed_one() {
        let (spins, rng) = initial_state(8, 4, 1);
        assert_eq!(spins, [-1, -1, -1, 1, 1, -1, 1, -1]);
        assert_eq!(
            rng,
            [
                0xc05a_6778_50dc_981b,
                0x9e32_cdf7_9483_70bd,
                0xa776_5f79_6f00_bbef,
                0xbbbb_23fe_6921_fe53
            ]
        );
    }

    #[test]
    fn launch_rechecks_actual_device_and_anneal_count() {
        let limits = DeviceLimits {
            shared_bytes_per_block: 48 << 10,
            shared_bytes_per_block_optin: 100 << 10,
            free_bytes: 1 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        };
        let graph = PreparedGraph::new(
            &Model {
                h_milli: &[0],
                edges: &[],
                j_milli: &[],
            },
            &Motifs {
                tiles: &[],
                gauges: &[],
                pairs: &[],
            },
            Weights::FULL,
            &limits,
            1,
        )
        .unwrap();
        assert!(check_shared(&graph, 29).is_ok());
        assert!(matches!(
            check_shared(&graph, 28),
            Err(ClusterDeviceError::Prepare(
                PrepareError::SharedMemory { .. }
            ))
        ));
        assert_eq!(launch_shape(&graph, 2).unwrap(), (2, 1024, 2));
        assert!(launch_shape(&graph, 0).is_err());
        assert!(launch_shape(&graph, usize::MAX).is_err());
        let free = graph.footprint().total().div_ceil(4) * 5;
        assert!(check_memory(&graph, 1, free).is_ok());
        assert!(check_memory(&graph, 2, free).is_err());
        assert!(check_memory(&graph, usize::MAX, free).is_err());
    }
}
