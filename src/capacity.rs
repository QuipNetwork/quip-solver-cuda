//! Node-capacity policy for the self-feeding kernels.
//!
//! SA uses its fixed local frame through 5000 nodes and global state above it.
//! Gibbs and MSA keep hot spins in shared memory and spill the rest.
//! Footprints bound session allocations before device buffers are built.

use crate::coefficients::{CoefficientPath, CoefficientPlan};
use crate::kernel::KernelKind;
use thiserror::Error;

/// Shipped `unpacked_state` size in `kernels/sa.cu`. Also the floor: a
/// smaller array saves nothing measurable and only narrows what runs.
pub const SA_DEFAULT_NODES: usize = 5000;

/// Threads per block the SA kernel launches, and so the multiplier on its
/// per-thread `delta_energy` workspace. Mirrors `total_threads` in
/// `streaming::build_algo_state`.
pub const SA_THREADS_PER_NONCE: usize = 256;

/// Fraction of free device memory the capacity derivation will spend, as
/// numerator over denominator.
///
/// Headroom leaves room for driver allocations and other device users beyond
/// the session buffers and the explicit pool allowance.
pub const MEMORY_HEADROOM_NUM: usize = 4;
/// See [`MEMORY_HEADROOM_NUM`].
pub const MEMORY_HEADROOM_DEN: usize = 5;

/// Session allowance for CUDA pool growth and allocation overhead.
const POOL_ALLOWANCE_BYTES: usize = 32 << 20;

/// Device facts the capacity derivation needs, read once at open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`.
    pub shared_bytes_per_block: usize,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`. Bounds the prefix,
    /// whose block opts in to more than the default per-block limit.
    pub shared_bytes_per_block_optin: usize,
    /// Free device memory at open, from `cuMemGetInfo`. Bounds session buffers.
    pub free_bytes: usize,
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    pub sm_count: usize,
    /// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`. The driver
    /// reserves local memory for full occupancy, so this multiplies the SA
    /// frame whether or not that many threads ever run.
    pub threads_per_sm: usize,
}

impl DeviceLimits {
    /// Free memory less the headroom the model does not account for.
    #[must_use]
    pub fn usable_bytes(&self) -> usize {
        self.free_bytes / MEMORY_HEADROOM_DEN * MEMORY_HEADROOM_NUM
    }
}

/// Concurrent nonce groups at the kernel's fixed SM width.
#[must_use]
pub fn instances(kernel: KernelKind, sm_count: usize) -> usize {
    (sm_count / kernel.sms_per_instance()).max(1)
}

/// Coefficient storage and arithmetic for both fields and couplings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Unit couplings and fields in {-1, 0, 1}, stored as i8.
    Binary,
    /// Scaled i8 coefficients with i32 arithmetic.
    Int8,
    /// Exact coefficients wider than i8, stored as i32.
    Int32,
    /// Real coefficients stored and evaluated as f32.
    Float32,
}

impl Precision {
    /// Map the selected coefficient path and unit-model property to precision.
    #[must_use]
    pub fn from_plan(plan: CoefficientPlan, unit: bool) -> Self {
        match plan.path {
            CoefficientPath::Int8 if unit => Self::Binary,
            CoefficientPath::Int8 | CoefficientPath::WeightedInt8 => Self::Int8,
            CoefficientPath::Wide => Self::Int32,
            CoefficientPath::Float32 => Self::Float32,
        }
    }
}

/// Device representation of the coupling matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// Two directed CSR entries per undirected coupling.
    Sparse,
    /// Row-major dense matrix storage.
    Dense,
}

/// Topology sizes needed to bound session allocations without building CSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Number of spins and fields.
    pub nodes: usize,
    /// Number of undirected couplings.
    pub couplers: usize,
    /// Largest node degree, bounding the greedy colouring by degree plus one.
    pub max_degree: usize,
}

/// Device memory shared by the session and repeated for each nonce group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    /// Session-wide device allocations, including topology and lane tables.
    /// This is global memory, distinct from CUDA block shared memory.
    pub shared_bytes: usize,
    /// Device memory per concurrent nonce group.
    pub per_instance_bytes: usize,
    /// Concurrent nonce groups, derived from the device's SM count.
    pub instances: usize,
}

impl Footprint {
    /// Total device memory, saturating so oversized shapes cannot wrap to fit.
    #[must_use]
    pub fn total(&self) -> usize {
        self.shared_bytes
            .saturating_add(self.per_instance_bytes.saturating_mul(self.instances))
    }
}

/// Device bytes for independent constrained loop chains.
/// Includes signed incidence, endpoints, labels, costs, targets, short cycles,
/// spins, running charges, three BFS arrays, scores, and validation flags.
/// Products saturate so an oversized launch cannot wrap into admission.
#[must_use]
pub fn flow_loop_footprint(
    links: usize,
    vertices: usize,
    cycles: usize,
    replicas: usize,
) -> Footprint {
    Footprint {
        shared_bytes: POOL_ALLOWANCE_BYTES
            .saturating_add(links.saturating_mul(17))
            .saturating_add(vertices.saturating_mul(20))
            .saturating_add(cycles.saturating_mul(20).max(4)),
        per_instance_bytes: links
            .saturating_add(vertices.saturating_mul(16))
            .saturating_add(20),
        instances: replicas,
    }
}

/// Threads per cluster anneal, matching `gpucluster.cu:272`.
pub const CLUSTER_THREADS: usize = 512;

/// Shared bytes per node: two int arrays and one byte spin (`gpucluster.cu:139-143`).
pub const CLUSTER_SHARED_BYTES_PER_NODE: usize = 9;

/// Dynamic shared padding requested by `gpucluster.cu:322`.
pub const CLUSTER_DYNAMIC_PADDING: usize = 16;

/// The static `changed` int in `gpucluster.cu:144`.
pub const CLUSTER_STATIC_SHARED_BYTES: usize = 4;

/// Sizes of the prepared cluster arrays. ELL widths count padded slots per node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterShape {
    /// Number of nodes.
    pub nodes: usize,
    /// Number of four-node tiles.
    pub tiles: usize,
    /// Number of pair links.
    pub links: usize,
    /// Maximum nonzero residual degree.
    pub weak_width: usize,
    /// Maximum nonzero original degree.
    pub all_width: usize,
    /// Number of contiguous color ranges.
    pub colors: usize,
}

/// Cluster global memory, including explicit fields and both weight arrays.
///
/// Counts `gpucluster.cu:298-307` uploads and `309-321` spin and RNG state.
/// Each empty upload reserves one element, as in `gpucluster.cu:259-262`.
/// Permutations and the original graph stay on the host. Parent and component
/// accumulators occupy block shared memory, not global memory.
#[must_use]
pub fn cluster_footprint(shape: ClusterShape, anneals: usize) -> Footprint {
    let n = shape.nodes;
    let t = shape.tiles;
    let p = shape.links;
    let int_bytes = |count: usize| count.max(1).saturating_mul(4);
    let mut shared = POOL_ALLOWANCE_BYTES;
    for count in [
        t.saturating_mul(4),
        p.saturating_mul(2),
        n,
        n.saturating_mul(shape.weak_width),
        n,
        n.saturating_mul(shape.all_width),
        shape.colors.saturating_add(1),
        n,
        t,
        p,
    ] {
        shared = shared.saturating_add(int_bytes(count));
    }
    shared = shared.saturating_add(t.saturating_mul(4).max(1));
    Footprint {
        shared_bytes: shared,
        per_instance_bytes: n.saturating_add(CLUSTER_THREADS.saturating_mul(8)),
        instances: anneals,
    }
}

/// Device allocation estimate for the selected kernel at full width.
#[must_use]
pub fn footprint(
    kernel: KernelKind,
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Footprint {
    match kernel {
        KernelKind::Msa => msa_footprint(shape, precision, storage, reads, betas, limits),
        KernelKind::Sa => sa_footprint(shape, precision, storage, reads, betas, limits),
        KernelKind::Gibbs => gibbs_footprint(shape, precision, storage, reads, betas, limits),
    }
}

/// Storage selected for a model. MSA uses dense storage above degree 62.
/// SA and Gibbs choose the smaller footprint.
#[must_use]
pub fn route_storage(
    kernel: KernelKind,
    shape: ModelShape,
    precision: Precision,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Storage {
    match kernel {
        KernelKind::Msa if shape.max_degree > MSA_MAX_DEGREE => Storage::Dense,
        KernelKind::Sa | KernelKind::Gibbs => {
            let dense = footprint(
                kernel,
                shape,
                precision,
                Storage::Dense,
                reads,
                betas,
                limits,
            );
            let sparse = footprint(
                kernel,
                shape,
                precision,
                Storage::Sparse,
                reads,
                betas,
                limits,
            );
            if dense.total() <= sparse.total() {
                Storage::Dense
            } else {
                Storage::Sparse
            }
        }
        KernelKind::Msa => Storage::Sparse,
    }
}

/// Read cap for the selected kernel on this device.
#[must_use]
pub fn max_reads(kernel: KernelKind, limits: &DeviceLimits) -> usize {
    match kernel {
        KernelKind::Msa => msa_device_reads(limits),
        KernelKind::Sa | KernelKind::Gibbs => 256,
    }
}

/// Device allocation estimate for MSA sessions at full width.
///
/// CSR and colour tables, beta schedules and integer acceptance tables are
/// session-wide. Each instance owns three coefficient/sample/energy slots,
/// eight control words and its spin spill. Empty buffers get stand-ins like
/// `SelfFeedingSession::build`. Colour counts use the greedy degree-plus-one
/// bound. A 32 MiB allowance covers CUDA pool growth and allocation overhead
/// measured by the A4000 allocation test.
///
/// Schedule tables use `betas.max(1)` entries, matching the session builders.
/// This assumes the default shared-spin layout, without a bench shared-spin
/// cap. Dense storage counts exact buffer bytes, with three coefficient matrix
/// slots, both packed spin layouts, and integer acceptance tables when used.
#[must_use]
pub fn msa_footprint(
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Footprint {
    msa_footprint_with_shared_cap(shape, precision, storage, reads, betas, limits, None)
}

/// Footprint with the benchmark's optional shared-spin cap.
pub(crate) fn msa_footprint_with_shared_cap(
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
    shared_cap: Option<usize>,
) -> Footprint {
    let count = instances(KernelKind::Msa, limits.sm_count);
    if storage == Storage::Dense {
        let coefficient_bytes = match precision {
            Precision::Binary | Precision::Int8 => 1,
            Precision::Int32 | Precision::Float32 => 4,
        };
        let mut shared_bytes = betas
            .max(1)
            .saturating_mul(4)
            .saturating_add(8)
            .saturating_add(POOL_ALLOWANCE_BYTES);
        if precision == Precision::Int8 || precision == Precision::Int32 {
            shared_bytes = shared_bytes
                .saturating_add(betas.max(1).saturating_mul(12))
                .saturating_add(crate::accept_table::NEGLOG_LEN * 8);
        }
        let n = shape.nodes;
        let reads = reads.max(1);
        let words = reads.div_ceil(MSA_LANES);
        let per_instance = n
            .saturating_mul(n)
            .max(1)
            .saturating_add(n.max(1))
            .saturating_mul(coefficient_bytes)
            .saturating_mul(3)
            .saturating_add(reads.saturating_mul(n.div_ceil(8).max(1)).saturating_mul(3))
            .saturating_add(reads.saturating_mul(12))
            .saturating_add(32)
            .saturating_add(n.saturating_mul(words).max(1).saturating_mul(8))
            .saturating_add(
                n.div_ceil(32)
                    .saturating_mul(words)
                    .saturating_mul(64)
                    .max(1)
                    .saturating_mul(4),
            );
        return Footprint {
            shared_bytes,
            per_instance_bytes: per_instance,
            instances: count,
        };
    }
    let betas = betas.max(1);
    let n = shape.nodes;
    let nnz = shape.couplers.saturating_mul(2).max(1);
    let colours = n.min(shape.max_degree.saturating_add(1)).max(1);
    let mut shared = n
        .saturating_add(1) // d_row_ptr
        .saturating_add(nnz) // d_col_ind
        .saturating_add(colours.saturating_mul(2)) // starts and counts
        .saturating_add(n.max(1)) // colour nodes
        .saturating_mul(4)
        .saturating_add(betas.saturating_mul(4)) // d_beta
        .saturating_add(POOL_ALLOWANCE_BYTES);
    let coefficient_bytes = match precision {
        Precision::Binary | Precision::Int8 => 1,
        Precision::Int32 | Precision::Float32 => 4,
    };
    match precision {
        Precision::Int8 | Precision::Int32 => {
            shared = shared
                .saturating_add(betas.saturating_mul(8 + 4)) // beta_fixed and delta_cut
                .saturating_add(crate::accept_table::NEGLOG_LEN * 8);
        }
        Precision::Binary | Precision::Float32 => {}
    }
    let reads = reads.max(1);
    let (words, shared_spins) = msa_spin_layout(
        n,
        reads,
        msa_dynamic_shared_bytes(limits.shared_bytes_per_block_optin),
        shared_cap,
    );
    let spill = msa_spill_words(n, shared_spins, words, 1)
        .max(1)
        .saturating_mul(8);
    let per_instance = n
        .max(1)
        .saturating_add(nnz)
        .saturating_mul(coefficient_bytes)
        .saturating_mul(3) // d_h and d_j, three slots
        .saturating_add(reads.saturating_mul(n.div_ceil(8).max(1)).saturating_mul(3))
        .saturating_add(reads.saturating_mul(4).saturating_mul(3)) // energies
        .saturating_add(8 * 4) // d_ctrl
        .saturating_add(spill);
    Footprint {
        shared_bytes: shared,
        per_instance_bytes: per_instance,
        instances: count,
    }
}

/// SA session allocation bound at full launch width.
/// Output slots are bounded at 256 reads regardless of the requested count.
/// Global spins and deltas use one element per launched thread and node.
/// Sparse mining frames are reserved at resident-thread occupancy.
/// Dense sessions always use global state.
#[must_use]
pub fn sa_footprint(
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    _reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Footprint {
    let n = shape.nodes;
    let nnz = match storage {
        Storage::Sparse => shape.couplers.saturating_mul(2).max(1),
        Storage::Dense => n.saturating_mul(n).max(1),
    };
    let coef_bytes = match precision {
        Precision::Binary | Precision::Int8 => 1,
        Precision::Int32 | Precision::Float32 => 4,
    };
    let topology_bytes = match storage {
        Storage::Sparse => n.saturating_add(1).saturating_add(nnz).saturating_mul(4),
        Storage::Dense => 8, // Empty CSR arrays each retain one i32 stand-in.
    };
    let mut shared = topology_bytes
        .saturating_add(betas.max(1).saturating_mul(4))
        .saturating_add(POOL_ALLOWANCE_BYTES);
    // Unit fields can add one to the degree. Above 62, reserve the i32
    // delta path even when every coupling is binary.
    let wide_binary = precision == Precision::Binary && shape.max_degree > 62;
    let delta_bytes = if precision == Precision::Binary && !wide_binary {
        1
    } else {
        4
    };
    if precision == Precision::Int8 || precision == Precision::Int32 || wide_binary {
        shared = shared
            .saturating_add(betas.max(1).saturating_mul(12))
            .saturating_add(crate::accept_table::NEGLOG_LEN * 8);
    }
    if storage == Storage::Sparse && n <= SA_DEFAULT_NODES {
        shared = shared.saturating_add(
            (SA_DEFAULT_NODES + SA_DEFAULT_NODES.div_ceil(8))
                .saturating_mul(limits.threads_per_sm)
                .saturating_mul(limits.sm_count),
        );
    }
    let per_instance_bytes = n
        .max(1)
        .saturating_add(nnz)
        .saturating_mul(coef_bytes)
        .saturating_mul(3)
        .saturating_add(
            n.div_ceil(8)
                .max(1)
                .saturating_mul(SA_THREADS_PER_NONCE)
                .saturating_mul(3),
        )
        .saturating_add(SA_THREADS_PER_NONCE * 12 + 32)
        .saturating_add(
            n.max(1)
                .saturating_mul(SA_THREADS_PER_NONCE)
                .saturating_mul(delta_bytes + 1),
        );
    Footprint {
        shared_bytes: shared,
        per_instance_bytes,
        instances: instances(KernelKind::Sa, limits.sm_count),
    }
}

/// Usable device memory per MSA instance, before session-wide allocations.
#[must_use]
pub fn msa_per_instance_budget(limits: &DeviceLimits) -> usize {
    limits.usable_bytes() / instances(KernelKind::Msa, limits.sm_count)
}

/// Shipped static shared-state size in `kernels/gibbs.cu`.
pub const GIBBS_DEFAULT_NODES: usize = 4800;

/// Two shared ints, rounded to the driver's 16-byte static allocation boundary.
pub const GIBBS_STATIC_SHARED_BYTES: usize = 16;

/// Eight warp partials for the 256-thread dense row reduction.
pub(crate) const GIBBS_DENSE_REDUCTION_BYTES: usize = 32;

/// Spins held in the dynamic shared prefix of a Gibbs block.
#[must_use]
pub fn gibbs_shared_spins(nodes: usize, optin_bytes: usize, cap: Option<usize>) -> usize {
    nodes
        .min(optin_bytes.saturating_sub(GIBBS_STATIC_SHARED_BYTES))
        .min(cap.unwrap_or(usize::MAX))
}

/// Gibbs session allocation bound at full launch width.
/// Colour starts and counts are tiled per instance; nodes and CSR are shared.
/// Each instance owns three coefficient/output slots and four block spills.
#[must_use]
pub fn gibbs_footprint(
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Footprint {
    gibbs_footprint_with_shared_cap(shape, precision, storage, reads, betas, limits, None)
}

pub(crate) fn gibbs_footprint_with_shared_cap(
    shape: ModelShape,
    precision: Precision,
    storage: Storage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Footprint {
    let n = shape.nodes;
    let (nnz, colours, csr_bytes, reduction_bytes) = match storage {
        Storage::Sparse => {
            let nnz = shape.couplers.saturating_mul(2).max(1);
            (
                nnz,
                n.min(shape.max_degree.saturating_add(1)).max(1),
                n.saturating_add(1).saturating_add(nnz).saturating_mul(4),
                0,
            )
        }
        Storage::Dense => (
            n.saturating_mul(n).max(1),
            n.max(1),
            8,
            GIBBS_DENSE_REDUCTION_BYTES,
        ),
    };
    let coefficient_bytes = match precision {
        Precision::Binary | Precision::Int8 => 1,
        Precision::Int32 | Precision::Float32 => 4,
    };
    let shared_bytes = csr_bytes
        .saturating_add(n.max(1).saturating_mul(4))
        .saturating_add(betas.max(1).saturating_mul(4))
        .saturating_add(POOL_ALLOWANCE_BYTES);
    let reads = reads.max(1);
    let prefix = gibbs_shared_spins(
        n,
        limits
            .shared_bytes_per_block_optin
            .saturating_sub(reduction_bytes),
        cap,
    );
    let spill = n
        .saturating_sub(prefix)
        .saturating_mul(KernelKind::Gibbs.sms_per_instance());
    let per_instance_bytes = n
        .max(1)
        .saturating_add(nnz)
        .saturating_mul(coefficient_bytes)
        .saturating_mul(3)
        .saturating_add(reads.saturating_mul(n.div_ceil(8).max(1)).saturating_mul(3))
        .saturating_add(reads.saturating_mul(12))
        .saturating_add(32)
        .saturating_add(colours.saturating_mul(8))
        .saturating_add(spill.max(1));
    Footprint {
        shared_bytes,
        per_instance_bytes,
        instances: instances(KernelKind::Gibbs, limits.sm_count),
    }
}

/// Minimum requested MSA capacity, also used when opening to read limits.
/// The msa kernel has no compiled node array: its spin state uses dynamic
/// shared memory and a global spill buffer. The miner derives its default
/// maximum from device memory after opening.
pub const MSA_DEFAULT_NODES: usize = 5000;

/// Default mining schedule length: quip-solver-core 0.0.3 `job.rs` uses
/// `DEFAULT_NUM_SWEEPS = 64` and `sweeps_per_beta.unwrap_or(1)`.
/// `streaming::build_beta_schedule` divides sweeps by sweeps per beta.
pub const MSA_DEFAULT_BETAS: usize = 64;

/// Reads packed into one 64-bit word by the msa kernel (`kernels/msa.cu`).
pub const MSA_LANES: usize = 64;

/// Default reads from the shared-memory probe at the mining baseline size.
#[must_use]
pub fn msa_device_reads(limits: &DeviceLimits) -> usize {
    let (_, shared) = msa_spin_layout(
        MSA_DEFAULT_NODES,
        MSA_MAX_READS,
        msa_dynamic_shared_bytes(limits.shared_bytes_per_block_optin),
        None,
    );
    if shared == MSA_DEFAULT_NODES {
        MSA_MAX_READS
    } else {
        MSA_LANES
    }
}

/// Replica words per spin an msa session allocates at 128 reads. Shared memory
/// no longer bounds this: spins that do not fit spill to global memory.
pub const MSA_REPLICA_WORDS: usize = 2;

/// Read cap for the msa kernel at [`MSA_REPLICA_WORDS`].
pub const MSA_MAX_READS: usize = MSA_LANES * MSA_REPLICA_WORDS;

/// Threads per block the msa kernel launches. Measured faster than 512 and
/// 1024 on an RTX 5090 (MR !26). `streaming::algo_limits` carries it to the
/// launch as the block size.
pub const MSA_THREADS_PER_NONCE: u32 = 256;

/// Neighbours per spin the msa kernel counts (`MSA_MAX_DEG` in
/// `kernels/msa.cu`), 20 per pass. With the field bond the count reaches 63,
/// the most the kernel's six bit planes hold, and the same ceiling as
/// `quip-cpu-msa`. Denser topologies use dense storage.
pub const MSA_MAX_DEGREE: usize = 62;

/// Static `__shared__` members of the msa kernel: `s_active_slot` and
/// `s_abort`, one `int` each (8 bytes of variables). The compiled function
/// reports 16: measured opening `kernels/msa.cu` on an A4000 (`sm_86`,
/// driver 610.57), where the driver rounds the static shared allocation
/// up to a 16-byte boundary. They count against the opt-in cap, so the dynamic
/// budget is the opt-in less this. `CudaDevice::open_with_nodes` refuses a
/// loaded function that reports more, so the budget never over-advertises.
/// A function that reports less leaves the budget conservative by under
/// one spin.
pub const MSA_STATIC_SHARED_BYTES: usize = 16;

/// Dynamic shared memory the msa kernel uses outside the spin state: the
/// 8192-byte threshold row plus 64 `u64` acceptance cuts (`MSA_ROW` and
/// `MSA_MAX_FIELD + 1` in `kernels/msa.cu`).
pub const MSA_FIXED_SHARED_BYTES: usize = 8192 + 64 * 8;

/// Dynamic shared memory one msa block may ask for on a device with this
/// opt-in ceiling: the ceiling less the kernel's static members.
#[must_use]
pub fn msa_dynamic_shared_bytes(shared_bytes_per_block_optin: usize) -> usize {
    shared_bytes_per_block_optin.saturating_sub(MSA_STATIC_SHARED_BYTES)
}

/// Bytes of dynamic shared memory an msa session needs for `nodes` spins at
/// `words` replica words each.
#[must_use]
pub fn msa_shared_bytes(nodes: usize, words: usize) -> usize {
    nodes
        .saturating_mul(words)
        .saturating_mul(8)
        .saturating_add(MSA_FIXED_SHARED_BYTES)
}

/// Most spins whose msa state fits `dynamic_shared_bytes` at `words` replica
/// words per spin. Spins past this spill to global memory.
#[must_use]
pub fn msa_shared_spins(dynamic_shared_bytes: usize, words: usize) -> usize {
    let per_node = words.saturating_mul(8).max(1);
    dynamic_shared_bytes.saturating_sub(MSA_FIXED_SHARED_BYTES) / per_node
}

/// `u64` words of global spill buffer an msa session allocates: the spins
/// past `shared_spins`, at `words` replica words each, for every nonce.
#[must_use]
pub fn msa_spill_words(nodes: usize, shared_spins: usize, words: usize, nonces: usize) -> usize {
    nodes
        .saturating_sub(shared_spins)
        .saturating_mul(words)
        .saturating_mul(nonces)
}

/// Replica words and shared spins used by both allocation and capacity checks.
pub(crate) fn msa_spin_layout(
    nodes: usize,
    reads: usize,
    dynamic_shared_bytes: usize,
    cap: Option<usize>,
) -> (usize, usize) {
    let words = reads
        .max(1)
        .div_ceil(MSA_LANES)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    let shared = nodes
        .min(msa_shared_spins(dynamic_shared_bytes, words))
        .min(cap.unwrap_or(usize::MAX));
    (words, shared)
}

/// Connectivity used to derive a maximum MSA model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Fixed maximum degree, with `nodes * degree / 2` couplers.
    Degree(usize),
    /// Every distinct pair of nodes is coupled.
    Complete,
}

impl Connectivity {
    /// Shape at a given node count, saturating for oversized requests.
    #[must_use]
    pub fn shape(self, nodes: usize) -> ModelShape {
        let degree = match self {
            Self::Degree(d) => d,
            Self::Complete => nodes.saturating_sub(1),
        };
        ModelShape {
            nodes,
            couplers: nodes.saturating_mul(degree) / 2,
            max_degree: degree,
        }
    }
}

/// Largest model for the selected kernel at the default beta count.
#[must_use]
pub fn max_model(
    kernel: KernelKind,
    conn: Connectivity,
    precision: Precision,
    reads: usize,
    limits: &DeviceLimits,
) -> ModelShape {
    match kernel {
        KernelKind::Msa => msa_max_model(conn, precision, reads, limits),
        KernelKind::Sa | KernelKind::Gibbs => {
            let mut high = i32::MAX as usize - 256;
            if let Connectivity::Degree(d) = conn {
                high = high.min((i32::MAX as usize - 1) / d.max(1));
            } else {
                high = high.min(46_340);
            }
            let at = |n| {
                let shape = conn.shape(n);
                let storage =
                    route_storage(kernel, shape, precision, reads, MSA_DEFAULT_BETAS, limits);
                footprint(
                    kernel,
                    shape,
                    precision,
                    storage,
                    reads,
                    MSA_DEFAULT_BETAS,
                    limits,
                )
                .total()
            };
            // The local frame disappears above the mining threshold. Search
            // the global interval first, then the local interval if none fits.
            let first_global = SA_DEFAULT_NODES + 1;
            let global_fits = kernel == KernelKind::Sa
                && first_global <= high
                && at(first_global) <= limits.usable_bytes();
            let mut low = if global_fits { first_global } else { 0 };
            if kernel == KernelKind::Sa && !global_fits {
                high = high.min(SA_DEFAULT_NODES);
            }
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if at(mid) <= limits.usable_bytes() {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            if let Connectivity::Degree(d) = conn {
                if low <= d {
                    low = 0;
                }
            }
            conn.shape(low)
        }
    }
}

/// Largest model that fits the default beta schedule and kernel index widths.
#[must_use]
pub fn msa_max_model(
    conn: Connectivity,
    precision: Precision,
    reads: usize,
    limits: &DeviceLimits,
) -> ModelShape {
    // Select storage for the topology family before searching its node count.
    // Complete-model capacity keeps dense storage even at small memory budgets.
    let storage = route_storage(
        KernelKind::Msa,
        conn.shape(usize::MAX),
        precision,
        reads,
        MSA_DEFAULT_BETAS,
        limits,
    );
    let mut high = if storage == Storage::Dense {
        // Dense row-major matrix indices are signed 32-bit integers.
        46_340
    } else {
        let words = reads
            .max(1)
            .div_ceil(MSA_LANES)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX);
        let ceiling = ((i32::MAX as usize - MSA_THREADS_PER_NONCE as usize) / words)
            .min(i32::MAX as usize - 8191);
        match conn {
            Connectivity::Degree(0) => ceiling,
            Connectivity::Degree(d) => ceiling.min((i32::MAX as usize - 20) / d),
            Connectivity::Complete => unreachable!(),
        }
    };
    let mut low = 0;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if msa_footprint(
            conn.shape(mid),
            precision,
            storage,
            reads,
            MSA_DEFAULT_BETAS,
            limits,
        )
        .total()
            <= limits.usable_bytes()
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    if let Connectivity::Degree(d) = conn {
        if low <= d {
            low = 0;
        }
    }
    conn.shape(low)
}

/// Largest uncoupled Binary model that fits the device at `reads`.
/// The default mining beta count bounds the schedule allocation. Longer
/// schedules still pass through the per-model footprint check.
#[must_use]
pub fn msa_max_nodes(limits: &DeviceLimits, reads: usize) -> usize {
    // `msa.cu` uses int for N * words and the final block-sized loop
    // increment. Its var + MSA_ROW_MASK also needs 8191 nodes of headroom,
    // which covers N + 7, sample packing and the colour-loop increment.
    let words = reads.max(1).div_ceil(MSA_LANES).checked_next_power_of_two();
    let Some(words) = words else { return 0 };
    let ceiling = (i32::MAX as usize - MSA_THREADS_PER_NONCE as usize)
        .checked_div(words)
        .unwrap_or(0)
        .min(i32::MAX as usize - 8191);
    msa_max_fitting(limits, reads, ceiling, |nodes| ModelShape {
        nodes,
        couplers: 0,
        max_degree: 0,
    })
}

/// Largest Binary coupler count across sparse degree 62 and complete dense models.
/// Nodes use the smallest degree-bounded shape. Node and coupler maxima are
/// independent bounds, so a model near both can still exceed device memory.
#[must_use]
pub fn msa_max_couplers(limits: &DeviceLimits, reads: usize) -> usize {
    // Each coupling has two CSR entries. Leave room for the sparse kernel's
    // MSA_CHUNK (20) increment at the end of a row.
    let ceiling = (i32::MAX as usize - 20) / 2;
    let max_nodes = msa_max_nodes(limits, reads);
    let sparse = msa_max_fitting(
        limits,
        reads,
        ceiling.min(max_nodes.saturating_mul(MSA_MAX_DEGREE) / 2),
        |couplers| ModelShape {
            nodes: (couplers * 2)
                .div_ceil(MSA_MAX_DEGREE)
                .max(MSA_MAX_DEGREE + 1),
            couplers,
            max_degree: MSA_MAX_DEGREE,
        },
    );
    sparse.max(msa_max_model(Connectivity::Complete, Precision::Binary, reads, limits).couplers)
}

fn msa_max_fitting(
    limits: &DeviceLimits,
    reads: usize,
    ceiling: usize,
    shape: impl Fn(usize) -> ModelShape,
) -> usize {
    let mut low = 0;
    let mut high = ceiling;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let footprint = msa_footprint(
            shape(mid),
            Precision::Binary,
            Storage::Sparse,
            reads,
            MSA_DEFAULT_BETAS,
            limits,
        );
        if footprint.total() <= limits.usable_bytes() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// Which device limit bounds a kernel's capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResource {
    /// Session buffers are bounded by device memory.
    DeviceMemory,
}

impl std::fmt::Display for BudgetResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceMemory => write!(f, "memory"),
        }
    }
}

/// Why a capacity request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CapacityError {
    /// Above what this device can hold. `resource` names which limit bound
    /// it. Session buffers must fit the usable device memory.
    #[error("requested {requested} nodes exceeds the device {resource} budget of {budget}")]
    AboveDeviceBudget {
        /// Nodes asked for.
        requested: usize,
        /// Budget derived from the device.
        budget: usize,
        /// Which device limit bound this request.
        resource: BudgetResource,
    },
}

/// Capacity used when `--max-nodes` is absent.
#[must_use]
pub fn default_nodes(kernel: KernelKind) -> usize {
    match kernel {
        KernelKind::Sa => SA_DEFAULT_NODES,
        KernelKind::Msa => MSA_DEFAULT_NODES,
        KernelKind::Gibbs => GIBBS_DEFAULT_NODES,
    }
}

/// Capacity to advertise in `--capabilities`.
///
/// Kernels derive maxima from device memory and advertise zero without
/// device limits. A requested capacity is an optional lower cap.
#[must_use]
pub fn advertised_nodes(
    kernel: KernelKind,
    requested: usize,
    limits: Option<&DeviceLimits>,
) -> usize {
    let want = requested.max(default_nodes(kernel));
    match kernel {
        KernelKind::Sa | KernelKind::Gibbs => limits.map_or(0, |limits| {
            requested.min(
                max_model(
                    kernel,
                    Connectivity::Degree(0),
                    Precision::Binary,
                    256,
                    limits,
                )
                .nodes,
            )
        }),
        KernelKind::Msa => limits.map_or(0, |limits| {
            want.min(msa_max_nodes(limits, msa_device_reads(limits)))
        }),
    }
}

/// Binary coupler limit that an SA or Gibbs identity advertises: the complete
/// model, the fewest nodes for its coupler count, at the full read count.
/// Node and coupler limits are independent bounds, and each model still
/// passes admission at its own precision, reads, and schedule length.
#[must_use]
pub fn advertised_couplers(kernel: KernelKind, limits: &DeviceLimits) -> usize {
    match kernel {
        KernelKind::Sa | KernelKind::Gibbs => {
            max_model(
                kernel,
                Connectivity::Complete,
                Precision::Binary,
                256,
                limits,
            )
            .couplers
        }
        KernelKind::Msa => msa_max_couplers(limits, msa_device_reads(limits)),
    }
}

/// Resolve a requested capacity against the kernel and the device.
///
/// MSA raises requests below its default and rejects requests above its
/// device bound. SA and Gibbs preserve the requested cap and check memory
/// per model.
///
/// # Errors
///
/// [`CapacityError::AboveDeviceBudget`] when the request exceeds
/// [`msa_max_nodes`] for this device.
pub fn resolve(
    kernel: KernelKind,
    requested: usize,
    limits: &DeviceLimits,
) -> Result<usize, CapacityError> {
    let floor = default_nodes(kernel);
    let want = requested.max(floor);
    let (budget, resource) = match kernel {
        KernelKind::Sa | KernelKind::Gibbs => return Ok(requested),
        KernelKind::Msa => (
            msa_max_nodes(limits, msa_device_reads(limits)),
            BudgetResource::DeviceMemory,
        ),
    };
    if want > budget {
        return Err(CapacityError::AboveDeviceBudget {
            requested: want,
            budget,
            resource,
        });
    }
    Ok(want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficients::{CoefficientPath, CoefficientPlan};
    use crate::kernel::KernelKind;

    #[test]
    fn gibbs_instances_use_four_sms() {
        assert_eq!(instances(KernelKind::Gibbs, 48), 12);
    }

    #[test]
    fn gibbs_max_model_exceeds_old_shared_ceiling() {
        let limits = a4000_limits();
        let m = max_model(
            KernelKind::Gibbs,
            Connectivity::Degree(20),
            Precision::Binary,
            256,
            &limits,
        );
        assert!(m.nodes > GIBBS_DEFAULT_NODES);
    }

    #[test]
    fn gibbs_explicit_cap_below_the_shared_array_is_preserved() {
        let limits = a4000_limits();
        assert_eq!(resolve(KernelKind::Gibbs, 1000, &limits), Ok(1000));
        assert_eq!(
            advertised_nodes(KernelKind::Gibbs, 1000, Some(&limits)),
            1000
        );
        let memory_limit = max_model(
            KernelKind::Gibbs,
            Connectivity::Degree(0),
            Precision::Binary,
            256,
            &limits,
        )
        .nodes;
        assert_eq!(
            advertised_nodes(KernelKind::Gibbs, usize::MAX, Some(&limits)),
            memory_limit
        );
        assert_eq!(advertised_nodes(KernelKind::Gibbs, usize::MAX, None), 0);
    }

    fn a4000_limits() -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49_152,
            shared_bytes_per_block_optin: 101_376,
            free_bytes: 15 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        }
    }

    #[test]
    fn advertised_couplers_follow_the_complete_binary_footprint() {
        for kernel in [KernelKind::Sa, KernelKind::Gibbs] {
            for limits in [a4000_limits(), a4000(512 << 20)] {
                let bound = max_model(
                    kernel,
                    Connectivity::Complete,
                    Precision::Binary,
                    256,
                    &limits,
                )
                .couplers;
                assert_eq!(advertised_couplers(kernel, &limits), bound, "{kernel:?}");
            }
            // complete:1415 has 1,000,405 couplers, past the old fixed 1,000,000.
            assert!(advertised_couplers(kernel, &a4000_limits()) > 1_000_405);
        }
    }

    #[test]
    fn sa_instances_follow_sm_count() {
        assert_eq!(instances(KernelKind::Sa, 48), 48);
    }

    #[test]
    fn gibbs_dense_footprint_counts_matrices_colours_and_spill() {
        let limits = a4000_limits();
        let shape = Connectivity::Complete.shape(600);
        for (precision, per_instance) in [
            (Precision::Binary, 1_147_305),
            (Precision::Int8, 1_147_305),
            (Precision::Int32, 4_392_705),
            (Precision::Float32, 4_392_705),
        ] {
            let footprint = gibbs_footprint(shape, precision, Storage::Dense, 256, 64, &limits);
            assert_eq!(footprint.instances, 12);
            assert_eq!(footprint.shared_bytes, POOL_ALLOWANCE_BYTES + 2_664);
            assert_eq!(footprint.per_instance_bytes, per_instance);
            let spilled = gibbs_footprint_with_shared_cap(
                shape,
                precision,
                Storage::Dense,
                256,
                64,
                &limits,
                Some(0),
            );
            assert_eq!(spilled.per_instance_bytes, per_instance + 2_399);
            assert_eq!(
                route_storage(KernelKind::Gibbs, shape, precision, 256, 64, &limits),
                Storage::Dense
            );
            assert_eq!(
                route_storage(
                    KernelKind::Gibbs,
                    Connectivity::Degree(20).shape(600),
                    precision,
                    256,
                    64,
                    &limits
                ),
                Storage::Sparse
            );
        }
    }

    #[test]
    fn gibbs_small_complete_model_uses_the_smaller_sparse_footprint() {
        let limits = a4000_limits();
        let shape = Connectivity::Complete.shape(32);
        let dense = gibbs_footprint(shape, Precision::Float32, Storage::Dense, 256, 64, &limits);
        let sparse = gibbs_footprint(shape, Precision::Float32, Storage::Sparse, 256, 64, &limits);
        // Three extra diagonals per instance cost 4,608 bytes. Removing
        // shared CSR saves 4,092 bytes, so dense costs 516 bytes more.
        assert_eq!(dense.total() - sparse.total(), 516);
        assert_eq!(
            route_storage(
                KernelKind::Gibbs,
                shape,
                Precision::Float32,
                256,
                64,
                &limits
            ),
            Storage::Sparse
        );
    }

    #[test]
    fn sa_routes_complete_graphs_dense() {
        let limits = a4000_limits();
        let shape = Connectivity::Complete.shape(2_000);
        assert_eq!(
            route_storage(KernelKind::Sa, shape, Precision::Int8, 256, 64, &limits),
            Storage::Dense
        );
        let sparse = Connectivity::Degree(20).shape(2_000);
        assert_eq!(
            route_storage(KernelKind::Sa, sparse, Precision::Int8, 256, 64, &limits),
            Storage::Sparse
        );
    }

    #[test]
    fn sa_dense_binary_capacity_includes_wide_deltas() {
        let limits = a4000_limits();
        let shape = Connectivity::Complete.shape(2_000);
        assert_eq!(
            sa_footprint(shape, Precision::Binary, Storage::Dense, 32, 64, &limits),
            sa_footprint(shape, Precision::Int8, Storage::Dense, 32, 64, &limits),
        );
        let model = max_model(
            KernelKind::Sa,
            Connectivity::Complete,
            Precision::Binary,
            32,
            &limits,
        );
        let at = |n| {
            sa_footprint(
                Connectivity::Complete.shape(n),
                Precision::Binary,
                Storage::Dense,
                32,
                64,
                &limits,
            )
            .total()
        };
        assert!(at(model.nodes) <= limits.usable_bytes());
        assert!(at(model.nodes + 1) > limits.usable_bytes());
    }

    #[test]
    fn sa_footprint_counts_launched_threads_not_reads() {
        let limits = a4000_limits();
        let shape = Connectivity::Degree(20).shape(20_000);
        let at = |reads| {
            footprint(
                KernelKind::Sa,
                shape,
                Precision::Binary,
                Storage::Sparse,
                reads,
                64,
                &limits,
            )
        };
        assert_eq!(
            at(64),
            at(256),
            "SA launches 256 threads per nonce at every read count"
        );
    }

    #[test]
    fn sa_max_model_searches_past_the_local_frame_drop() {
        let mut limits = a4000_limits();
        limits.free_bytes = 512 << 20;
        let conn = Connectivity::Degree(20);
        assert!(
            sa_footprint(
                conn.shape(5001),
                Precision::Int32,
                Storage::Sparse,
                256,
                64,
                &limits
            )
            .total()
                <= limits.usable_bytes()
        );
        let model = max_model(KernelKind::Sa, conn, Precision::Int32, 256, &limits);
        assert!(model.nodes >= 5001);
        assert!(
            sa_footprint(
                conn.shape(model.nodes + 1),
                Precision::Int32,
                Storage::Sparse,
                256,
                64,
                &limits
            )
            .total()
                > limits.usable_bytes()
        );
    }

    #[test]
    fn sa_max_model_exceeds_old_local_ceiling() {
        let limits = a4000_limits();
        let m = max_model(
            KernelKind::Sa,
            Connectivity::Degree(20),
            Precision::Binary,
            256,
            &limits,
        );
        assert!(m.nodes > SA_DEFAULT_NODES);
        let fp = footprint(
            KernelKind::Sa,
            m,
            Precision::Binary,
            Storage::Sparse,
            256,
            64,
            &limits,
        );
        assert!(fp.total() <= limits.usable_bytes());
        let next = Connectivity::Degree(20).shape(m.nodes + 1);
        assert!(
            footprint(
                KernelKind::Sa,
                next,
                Precision::Binary,
                Storage::Sparse,
                256,
                64,
                &limits
            )
            .total()
                > limits.usable_bytes()
        );
    }

    #[test]
    fn small_complete_msa_model_keeps_dense_accounting() {
        let shape = Connectivity::Complete.shape(32);
        let mut limits = a4000_limits();
        let expected = msa_footprint(shape, Precision::Binary, Storage::Dense, 128, 64, &limits);
        limits.free_bytes = expected.total().div_ceil(4) * 5;
        let model = max_model(
            KernelKind::Msa,
            Connectivity::Complete,
            Precision::Binary,
            128,
            &limits,
        );
        assert_eq!(model.nodes, 32);
        let storage = route_storage(
            KernelKind::Msa,
            Connectivity::Complete.shape(usize::MAX),
            Precision::Binary,
            128,
            64,
            &limits,
        );
        assert_eq!(storage, Storage::Dense);
        let cells = crate::probe::probe(KernelKind::Msa, &limits, 128);
        let cell = cells
            .iter()
            .find(|cell| {
                cell.connectivity == Connectivity::Complete && cell.precision == Precision::Binary
            })
            .unwrap();
        assert_eq!(cell.nodes, 32);
        assert_eq!(cell.storage, Storage::Dense);
        assert_eq!(cell.per_instance_bytes, expected.per_instance_bytes);
        assert_eq!(
            footprint(
                KernelKind::Msa,
                model,
                Precision::Binary,
                storage,
                128,
                64,
                &limits
            ),
            expected
        );
    }

    #[test]
    fn msa_dispatch_matches_msa_footprint() {
        let limits = a4000_limits();
        let shape = Connectivity::Degree(20).shape(64_000);
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            assert_eq!(
                footprint(
                    KernelKind::Msa,
                    shape,
                    precision,
                    Storage::Sparse,
                    128,
                    MSA_DEFAULT_BETAS,
                    &limits
                ),
                msa_footprint(
                    shape,
                    precision,
                    Storage::Sparse,
                    128,
                    MSA_DEFAULT_BETAS,
                    &limits
                ),
            );
        }
    }

    #[test]
    fn msa_max_model_unchanged_by_dispatch() {
        let limits = a4000_limits();
        for conn in [
            Connectivity::Degree(20),
            Connectivity::Degree(60),
            Connectivity::Complete,
        ] {
            for precision in [
                Precision::Binary,
                Precision::Int8,
                Precision::Int32,
                Precision::Float32,
            ] {
                assert_eq!(
                    max_model(KernelKind::Msa, conn, precision, 128, &limits),
                    msa_max_model(conn, precision, 128, &limits),
                );
            }
        }
    }

    #[test]
    fn msa_routes_by_degree() {
        let limits = a4000_limits();
        let at = |d| {
            route_storage(
                KernelKind::Msa,
                Connectivity::Degree(d).shape(10_000),
                Precision::Binary,
                128,
                64,
                &limits,
            )
        };
        assert_eq!(at(62), Storage::Sparse);
        assert_eq!(at(63), Storage::Dense);
    }

    #[test]
    fn max_reads_per_kernel() {
        let limits = a4000_limits();
        assert_eq!(max_reads(KernelKind::Sa, &limits), 256);
        assert_eq!(max_reads(KernelKind::Gibbs, &limits), 256);
        assert_eq!(
            max_reads(KernelKind::Msa, &limits),
            msa_device_reads(&limits)
        );
    }

    #[test]
    fn msa_accepts_a_million_uncoupled_nodes_when_the_footprint_fits() {
        let limits = a4000_limits();
        let shape = ModelShape {
            nodes: 1_000_000,
            couplers: 0,
            max_degree: 0,
        };
        assert!(
            msa_footprint(shape, Precision::Binary, Storage::Sparse, 128, 64, &limits).total()
                <= limits.usable_bytes()
        );
        assert_eq!(
            resolve(KernelKind::Msa, shape.nodes, &limits),
            Ok(shape.nodes)
        );
    }

    #[test]
    fn instances_divide_sms_by_kernel_width() {
        assert_eq!(instances(KernelKind::Msa, 48), 48);
        assert_eq!(instances(KernelKind::Sa, 48), 48);
        assert_eq!(instances(KernelKind::Gibbs, 48), 12);
        assert_eq!(instances(KernelKind::Gibbs, 3), 1);
        assert_eq!(instances(KernelKind::Msa, 0), 1);
    }

    #[test]
    fn footprint_grows_with_couplers_and_lane_width() {
        let l = a4000_limits();
        let s = |c| ModelShape {
            nodes: 100_000,
            couplers: c,
            max_degree: 40,
        };
        let u = msa_footprint(s(2_000_000), Precision::Binary, Storage::Sparse, 128, 1, &l);
        let w = msa_footprint(s(2_000_000), Precision::Int32, Storage::Sparse, 128, 1, &l);
        assert!(w.total() > u.total());
        assert!(
            msa_footprint(s(4_000_000), Precision::Binary, Storage::Sparse, 128, 1, &l).total()
                > u.total()
        );
        assert_eq!(u.instances, 48);
    }

    #[test]
    fn precision_keeps_unit_models_binary_and_wide_models_int32() {
        for unit in [false, true] {
            assert_eq!(
                Precision::from_plan(
                    CoefficientPlan {
                        path: CoefficientPath::Wide,
                        scale_milli: 1
                    },
                    unit
                ),
                Precision::Int32
            );
        }
        for (path, expected) in [
            (CoefficientPath::WeightedInt8, Precision::Int8),
            (CoefficientPath::Float32, Precision::Float32),
        ] {
            for unit in [false, true] {
                assert_eq!(
                    Precision::from_plan(
                        CoefficientPlan {
                            path,
                            scale_milli: 1000
                        },
                        unit
                    ),
                    expected
                );
            }
        }
        let plan = CoefficientPlan {
            path: CoefficientPath::Int8,
            scale_milli: 1000,
        };
        assert_eq!(Precision::from_plan(plan, true), Precision::Binary);
        assert_eq!(Precision::from_plan(plan, false), Precision::Int8);
    }

    #[test]
    fn footprint_counts_three_csr_coefficient_slots_per_instance() {
        let l = a4000_limits();
        let shape = ModelShape {
            nodes: 100_000,
            couplers: 2_000_000,
            max_degree: 40,
        };
        let narrow = msa_footprint(shape, Precision::Int8, Storage::Sparse, 128, 1, &l);
        let wide = msa_footprint(shape, Precision::Int32, Storage::Sparse, 128, 1, &l);
        // Three slots, two directed halves per edge, three extra bytes per coefficient.
        assert_eq!(
            wide.per_instance_bytes - narrow.per_instance_bytes,
            36_900_000
        );
    }

    #[test]
    fn footprint_accounts_for_reads_and_spill_layout() {
        let l = a4000_limits();
        let shape = ModelShape {
            nodes: 100_000,
            couplers: 2_000_000,
            max_degree: 40,
        };
        let one = msa_footprint(shape, Precision::Binary, Storage::Sparse, 64, 1, &l);
        let two = msa_footprint(shape, Precision::Binary, Storage::Sparse, 128, 1, &l);
        // 64 more reads in three sample/energy slots, plus the larger spill.
        assert_eq!(two.per_instance_bytes - one.per_instance_bytes, 3_200_768);
        let mut no_shared = l;
        no_shared.shared_bytes_per_block_optin = 0;
        let spilled = msa_footprint(
            shape,
            Precision::Binary,
            Storage::Sparse,
            128,
            1,
            &no_shared,
        );
        assert_eq!(spilled.per_instance_bytes - two.per_instance_bytes, 92_656);
    }

    #[test]
    fn dense_binary_footprint_counts_session_allocations() {
        let footprint = msa_footprint(
            ModelShape {
                nodes: 2_000,
                couplers: 1_999_000,
                max_degree: 1_999,
            },
            Precision::Binary,
            Storage::Dense,
            128,
            64,
            &a4000_limits(),
        );
        // Pool allowance, two unused CSR stand-ins and 64 f32 beta values.
        assert_eq!(footprint.shared_bytes, 264 + (32 << 20));
        // Three i8 matrices and fields, three output slots, control,
        // two spin words per node, and 128 transposed u32 words per 32 nodes.
        assert_eq!(footprint.per_instance_bytes, 12_167_824);
        assert_eq!(footprint.instances, 48);
    }

    #[test]
    fn dense_capacity_boundaries_and_overflow() {
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let limits = a4000_limits();
            let size = |n| {
                msa_footprint(
                    ModelShape {
                        nodes: n,
                        couplers: 0,
                        max_degree: 0,
                    },
                    precision,
                    Storage::Dense,
                    128,
                    64,
                    &limits,
                )
                .total()
            };
            let mut low = 0usize;
            let mut high = 100_000;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if size(mid) <= limits.usable_bytes() {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            eprintln!(
                "dense {precision:?} A4000 maximum N={low} bytes={} next_bytes={}",
                size(low),
                size(low + 1)
            );
            assert!(low >= 4_000);
            assert!(size(low) <= limits.usable_bytes());
            assert!(size(low + 1) > limits.usable_bytes());
            assert_eq!(size(usize::MAX), usize::MAX);
        }
    }

    #[test]
    fn dense_weighted_footprint_counts_session_allocations() {
        for (precision, shared, per_instance) in [
            (Precision::Int8, 525_320, 12_167_824),
            (Precision::Int32, 525_320, 48_185_824),
            (Precision::Float32, 264, 48_185_824),
        ] {
            let footprint = msa_footprint(
                ModelShape {
                    nodes: 2_000,
                    couplers: 1_999_000,
                    max_degree: 1_999,
                },
                precision,
                Storage::Dense,
                128,
                64,
                &a4000_limits(),
            );
            assert_eq!(footprint.per_instance_bytes, per_instance);
            assert_eq!(footprint.shared_bytes, shared + (32 << 20));
        }
    }

    #[test]
    fn footprint_saturates_instead_of_wrapping_large_shapes() {
        let footprint = msa_footprint(
            ModelShape {
                nodes: usize::MAX,
                couplers: usize::MAX,
                max_degree: usize::MAX,
            },
            Precision::Int32,
            Storage::Sparse,
            usize::MAX,
            1,
            &a4000_limits(),
        );
        assert_eq!(footprint.total(), usize::MAX);
        assert_eq!(
            Footprint {
                shared_bytes: 1,
                per_instance_bytes: usize::MAX / 2 + 1,
                instances: 2
            }
            .total(),
            usize::MAX
        );
    }

    #[test]
    fn footprint_counts_every_beta() {
        let limits = a4000_limits();
        let shape = ModelShape {
            nodes: 1000,
            couplers: 1000,
            max_degree: 2,
        };
        // build allocates an f32 beta; the integer lane adds i64 beta_fixed
        // and i32 delta_cut. All three tables are shared across instances.
        for (precision, bytes_per_beta) in [(Precision::Binary, 4), (Precision::Int32, 16)] {
            for betas in [1, 65_536] {
                let extra = 17;
                let base = msa_footprint(shape, precision, Storage::Sparse, 128, betas, &limits);
                let longer = msa_footprint(
                    shape,
                    precision,
                    Storage::Sparse,
                    128,
                    betas + extra,
                    &limits,
                );
                assert_eq!(longer.total() - base.total(), extra * bytes_per_beta);
                assert_eq!(longer.per_instance_bytes, base.per_instance_bytes);
            }
        }
    }

    #[test]
    fn footprint_beta_counts_use_standins_and_saturate() {
        let limits = a4000_limits();
        let shape = ModelShape {
            nodes: 1000,
            couplers: 1000,
            max_degree: 2,
        };
        for precision in [Precision::Binary, Precision::Int32] {
            let footprint =
                |betas| msa_footprint(shape, precision, Storage::Sparse, 128, betas, &limits);
            assert_eq!(footprint(0), footprint(1));
            assert_eq!(footprint(usize::MAX).total(), usize::MAX);
        }
    }

    #[test]
    fn msa_budget_divides_usable_memory_between_instances() {
        assert_eq!(msa_per_instance_budget(&a4000_limits()), 268_435_456);
    }

    #[test]
    fn resolve_accepts_a_request_inside_both_bounds() {
        let limits = a4000(A4000_FREE);
        assert_eq!(resolve(KernelKind::Gibbs, 5640, &limits), Ok(5640));
        assert_eq!(resolve(KernelKind::Sa, 5640, &limits), Ok(5640));
    }

    #[test]
    fn sa_explicit_cap_below_the_local_frame_is_preserved() {
        let limits = a4000_limits();
        assert_eq!(resolve(KernelKind::Sa, 1000, &limits), Ok(1000));
        assert_eq!(advertised_nodes(KernelKind::Sa, 1000, Some(&limits)), 1000);
        let memory_limit = max_model(
            KernelKind::Sa,
            Connectivity::Degree(0),
            Precision::Binary,
            256,
            &limits,
        )
        .nodes;
        assert_eq!(
            advertised_nodes(KernelKind::Sa, usize::MAX, Some(&limits)),
            memory_limit
        );
    }

    /// MSA retains its default node floor.
    #[test]
    fn resolve_msa_raises_small_requests_to_the_default() {
        let limits = a4000(A4000_FREE);
        assert_eq!(resolve(KernelKind::Msa, 64, &limits), Ok(MSA_DEFAULT_NODES));
    }

    #[test]
    fn defaults_match_the_shipped_kernel_arrays() {
        assert_eq!(default_nodes(KernelKind::Sa), 5000);
        assert_eq!(default_nodes(KernelKind::Msa), 5000);
        assert_eq!(default_nodes(KernelKind::Gibbs), 4800);
    }

    /// An A4000 as this code sees it: 48 SMs, 1536 resident threads each,
    /// 48152 bytes of shared memory per block, and a nominally free 16 GiB.
    const A4000_SMS: usize = 48;
    const A4000_THREADS_PER_SM: usize = 1536;
    const A4000_FREE: usize = 16 * 1024 * 1024 * 1024;

    fn a4000(free_bytes: usize) -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49152,
            shared_bytes_per_block_optin: 101_376,
            free_bytes,
            sm_count: A4000_SMS,
            threads_per_sm: A4000_THREADS_PER_SM,
        }
    }

    /// A card with room for the default still opens on a plain invocation.
    #[test]
    fn an_ordinary_card_affords_the_default() {
        let limits = a4000(A4000_FREE);
        assert_eq!(
            resolve(KernelKind::Sa, SA_DEFAULT_NODES, &limits),
            Ok(SA_DEFAULT_NODES)
        );
    }

    #[test]
    fn msa_spill_words_covers_every_nonce_past_the_shared_spins() {
        assert_eq!(msa_spill_words(5000, 5000, 2, 48), 0);
        assert_eq!(msa_spill_words(5000, 6000, 2, 48), 0);
        assert_eq!(
            msa_spill_words(200_000, 5791, 2, 48),
            (200_000 - 5791) * 2 * 48
        );
        assert_eq!(msa_spill_words(10, 0, 1, 1), 10);
    }

    /// An A4000 opts in to 101376 bytes (99 KB) per block. Less the two
    /// static ints and the 8704-byte threshold structures, two 8-byte
    /// words per spin fit 5791 spins.
    #[test]
    fn msa_shared_spins_on_a_99_kb_optin() {
        assert_eq!(msa_dynamic_shared_bytes(101_376), 101_360);
        assert_eq!(msa_shared_spins(101_360, MSA_REPLICA_WORDS), 5791);
        assert!(msa_shared_bytes(5791, MSA_REPLICA_WORDS) <= 101_360);
        assert!(msa_shared_bytes(5792, MSA_REPLICA_WORDS) > 101_360);
    }

    /// The other opt-in sizes in the supported-arch table. Volta (96 KB),
    /// Ampere datacenter (163 KB) and Hopper (227 KB) hold the default at two
    /// words. Turing (64 KB) does not.
    #[test]
    fn msa_shared_spins_across_the_supported_optin_sizes() {
        let two = MSA_REPLICA_WORDS;
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(98_304), two),
            5599
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(166_912), two),
            9887
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(232_448), two),
            13_983
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(65_536), two),
            3551
        );
    }

    /// One word doubles the node budget, because the spin state is the only
    /// term that scales with the word count.
    #[test]
    fn msa_shared_spins_at_one_word_doubles_the_turing_ceiling() {
        assert_eq!(msa_shared_spins(msa_dynamic_shared_bytes(65_536), 1), 7102);
    }

    #[test]
    fn advertised_couplers_include_dense_complete_models() {
        assert!(msa_max_couplers(&a4000_limits(), 128) > 32_139_292);
    }

    #[test]
    fn a4000_advertised_limits_match_the_documented_example() {
        let limits = a4000_limits();
        assert_eq!(msa_max_nodes(&limits, 128), 3_987_504);
        assert_eq!(msa_max_couplers(&limits, 128), 44_486_028);
    }

    #[test]
    fn msa_default_beta_count_matches_default_sample_parameters() {
        let params = crate::SampleParams::default();
        assert_eq!(
            MSA_DEFAULT_BETAS,
            params.num_sweeps / params.sweeps_per_beta
        );
    }

    #[test]
    fn advertised_limits_fit_kernel_i32() {
        for limits in [a4000_limits(), a4000(usize::MAX)] {
            for reads in [64, 128] {
                let nodes = msa_max_nodes(&limits, reads);
                let couplers = msa_max_couplers(&limits, reads);
                let words = reads.div_ceil(MSA_LANES);
                let threads = MSA_THREADS_PER_NONCE as usize;
                let var = nodes - 1;
                let packed_size = nodes.div_ceil(8);
                // Worst-case intermediates from both builds of msa.cu,
                // including the final loop increments and padded sample byte.
                for (expression, value) in [
                    ("N", nodes),
                    ("N * words", nodes * words),
                    ("N + 7", nodes + 7),
                    ("var * words + w", var * words + words - 1),
                    ("i + blockDim.x", nodes * words - 1 + threads),
                    ("k + gstride", var + threads / words),
                    ("start + k", var),
                    ("var + 1", var + 1),
                    ("var + off", var + 8191),
                    ("++b", packed_size),
                    ("b * 8 + bit", (packed_size - 1) * 8 + 7),
                    ("nnz", couplers * 2),
                    ("base + MSA_CHUNK", couplers * 2 + 20),
                    ("base + q / pstart + q", couplers * 2 + 19),
                ] {
                    assert!(
                        i32::try_from(value).is_ok(),
                        "{expression} = {value} exceeds i32 at {nodes} nodes and {reads} reads"
                    );
                }
            }
        }
    }

    #[test]
    fn max_nodes_scales_with_free_memory() {
        let small = a4000(8 << 30);
        let large = a4000(16 << 30);
        assert!(msa_max_nodes(&large, 128) >= msa_max_nodes(&small, 128) * 2 - 1);
    }

    #[test]
    fn advertised_msa_axes_fit_the_default_schedule_footprint() {
        for limits in [a4000_limits(), a4000(64 << 20)] {
            for conn in [
                Connectivity::Degree(0),
                Connectivity::Degree(62),
                Connectivity::Complete,
            ] {
                let shape = msa_max_model(conn, Precision::Binary, 128, &limits);
                let footprint = |shape| {
                    msa_footprint(
                        shape,
                        Precision::Binary,
                        route_storage(
                            KernelKind::Msa,
                            shape,
                            Precision::Binary,
                            128,
                            MSA_DEFAULT_BETAS,
                            &limits,
                        ),
                        128,
                        MSA_DEFAULT_BETAS,
                        &limits,
                    )
                    .total()
                };
                assert!(shape.nodes > 0);
                assert!(footprint(shape) <= limits.usable_bytes());
                assert!(footprint(conn.shape(shape.nodes + 1)) > limits.usable_bytes());
                assert!(msa_max_couplers(&limits, 128) >= shape.couplers);
            }
            let nodes = msa_max_nodes(&limits, 128);
            assert_eq!(resolve(KernelKind::Msa, nodes, &limits), Ok(nodes));
            assert!(resolve(KernelKind::Msa, nodes + 1, &limits).is_err());
        }
    }

    #[test]
    fn msa_device_reads_keeps_turing_at_64() {
        assert_eq!(msa_device_reads(&a4000_limits()), 128);
        let limits = DeviceLimits {
            shared_bytes_per_block_optin: 65_536,
            ..a4000_limits()
        };
        assert_eq!(msa_device_reads(&limits), 64);
    }

    #[test]
    fn max_model_checks_every_precision_and_kernel_index_ceiling() {
        for limits in [a4000_limits(), a4000(64 << 20), a4000(usize::MAX)] {
            for conn in [
                Connectivity::Degree(0),
                Connectivity::Degree(20),
                Connectivity::Degree(60),
                Connectivity::Degree(63),
                Connectivity::Complete,
            ] {
                for precision in [
                    Precision::Binary,
                    Precision::Int8,
                    Precision::Int32,
                    Precision::Float32,
                ] {
                    let shape = msa_max_model(conn, precision, 128, &limits);
                    if shape.nodes == 0 {
                        continue;
                    }
                    let storage = route_storage(
                        KernelKind::Msa,
                        shape,
                        precision,
                        128,
                        MSA_DEFAULT_BETAS,
                        &limits,
                    );
                    assert!(
                        msa_footprint(shape, precision, storage, 128, MSA_DEFAULT_BETAS, &limits)
                            .total()
                            <= limits.usable_bytes()
                    );
                    if storage == Storage::Dense {
                        assert!(i32::try_from(shape.nodes * shape.nodes).is_ok());
                    } else {
                        assert!(i32::try_from(shape.couplers * 2 + 20).is_ok());
                    }
                    let next = conn.shape(shape.nodes + 1);
                    assert!(
                        msa_footprint(next, precision, storage, 128, MSA_DEFAULT_BETAS, &limits)
                            .total()
                            > limits.usable_bytes()
                            || (storage == Storage::Dense
                                && i32::try_from(next.nodes * next.nodes).is_err())
                            || (storage == Storage::Sparse
                                && (i32::try_from(next.couplers * 2 + 20).is_err()
                                    || i32::try_from(
                                        next.nodes * 2 + MSA_THREADS_PER_NONCE as usize
                                    )
                                    .is_err()
                                    || i32::try_from(next.nodes + 8191).is_err()))
                    );
                }
            }
        }
    }

    #[test]
    fn msa_limits_are_zero_without_session_memory() {
        for free_bytes in [0, 1024] {
            let limits = a4000(free_bytes);
            assert_eq!(msa_max_nodes(&limits, 128), 0);
            assert_eq!(msa_max_couplers(&limits, 128), 0);
        }
    }

    #[test]
    fn advertised_msa_uses_the_device_budget_and_optional_cap() {
        let limits = a4000_limits();
        assert_eq!(
            advertised_nodes(KernelKind::Msa, usize::MAX, Some(&limits)),
            msa_max_nodes(&limits, 128)
        );
        assert_eq!(advertised_nodes(KernelKind::Msa, 8192, Some(&limits)), 8192);
        assert_eq!(
            advertised_nodes(KernelKind::Msa, 64, Some(&limits)),
            MSA_DEFAULT_NODES
        );
        assert_eq!(advertised_nodes(KernelKind::Msa, usize::MAX, None), 0);
    }

    #[test]
    fn msa_read_cap_is_two_words_of_lanes() {
        assert_eq!(MSA_MAX_READS, 128);
    }
}
