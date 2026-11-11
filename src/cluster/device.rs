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

struct WorkControl<'a> {
    work: &'a super::sampler::ClusterWork,
    stopped: &'a dyn Fn() -> bool,
    cancelled: &'a dyn Fn() -> bool,
    chunk: usize,
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
        self.run_work_chunked(work, stopped, cancelled, 1)
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
        self.run_work_chunked(work, &|| false, &|| false, chunk)
    }

    fn run_work_chunked(
        &mut self,
        work: &super::sampler::ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        chunk: usize,
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
            }),
        )
        .map(|(run, _)| run)
    }

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
        Ok((result, rng_state))
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
