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
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamJob, StreamOutcome, StreamResult, WarmStart, WarmStreamJob,
};
use std::time::Instant;
use tokio::sync::mpsc::{channel, Receiver, Sender};

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
        let mut sweep = 0;
        for &beta in &self.betas {
            for _ in 0..self.sweeps_per_beta {
                checkpoint(cancelled)?;
                launch(sweep, beta, false)?;
                sweep += 1;
            }
        }
        for _ in 0..256 {
            checkpoint(cancelled)?;
            launch(0, 1.0, true)?;
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
    kernel: ClusterKernel,
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
        ClusterDevice::open_with_kernel(graph, &self.kernel)?.run_work(work, stopped, cancelled)
    }
}

/// Serial family jobs alongside one persistent production SA stream.
pub struct OtsukaSampler<D = GpuCluster, S = CudaSampler> {
    cluster: D,
    sa: S,
    limits: DeviceLimits,
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
        let cluster = GpuCluster {
            kernel: ClusterKernel::load(device.device_index)?,
        };
        let limits = device.limits;
        Ok(Self {
            cluster,
            sa: CudaSampler::new(device, gov, KernelKind::Sa, stall),
            limits,
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
        }
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
        let work = ClusterWork::new(graph, params)?;
        let run = self.cluster.run(prepared, &work, &stopped, cancelled)?;
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
        family: &FamilyContext,
    ) -> Result<PreparedGraph, SampleError> {
        if family.instance().h().len() != graph.h.len()
            || family.instance().edges().len() != graph.edges.len()
        {
            return Err(SampleError::Capacity);
        }
        PreparedGraph::from_family(family, Weights::FULL, &self.limits, params.num_reads).map_err(
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
        let prepared = self.prepared(graph, params, family)?;
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
            while let Some(item) = jobs.blocking_recv() {
                if out.is_closed() {
                    break;
                }
                let job = item.job;
                if let Some(family) = item.family.as_ref().filter(|f| supported(f)) {
                    let result = Self::finish(&job, &cancel, || {
                        let prepared = self.prepared(&job.graph, &job.params, family)?;
                        self.reads(&job.graph, &job.params, prepared, &|| {
                            cancel.is_cancelled(job.watermark)
                        })
                        .map_err(sample_error)
                    });
                    if out.blocking_send(result).is_err() {
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
