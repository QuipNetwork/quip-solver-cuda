//! On-demand Otsuka cluster optimizer. Component updates are a heuristic.

use super::deep::PROBE_EVERY;
use super::tuning::OtsukaTuning;
use super::{PrepareError, PreparedGraph, SpinError};
use crate::capacity::{CLUSTER_STATIC_SHARED_BYTES, CLUSTER_THREADS};
use crate::cuda_device::{compile_for_arch, device_arch, driver_version, CudaError};
use crate::jit_cache;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;
use std::time::Duration;

const SOURCE: &str = include_str!("../../kernels/cluster.cu");
/// Local build: device population-annealing kernels, a separate module.
const PA_SOURCE: &str = include_str!("../../kernels/cluster_pa.cu");
/// Threads of `quip_pa_houdayer` (its `PA_HOUDAYER_THREADS`).
const PA_HOUDAYER_THREADS: u32 = 512;
/// Population segments queued ahead of the host in device mode.
const PA_QUEUE_DEPTH: usize = 2;

/// Local build: the four checkpoint kernels of `kernels/cluster_pa.cu`.
#[derive(Clone)]
pub(crate) struct PaKernels {
    energy_best: CudaFunction,
    plan: CudaFunction,
    gather: CudaFunction,
    houdayer: CudaFunction,
}

/// Dynamic shared bytes of `quip_pa_houdayer`: an int queue of n entries and
/// a seen bitmap of ceil(n / 32) words.
fn houdayer_shared(n: usize) -> usize {
    4 * (n + n.div_ceil(32))
}

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
    energy: CudaFunction,
    shared: usize,
    /// Local build: device PA kernels, when loaded and admissible for this graph.
    pa: Option<PaKernels>,
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
    /// Local build (`QUIP_OTSUKA_SWEEP_V3`): the full and weak ELL repacked
    /// into int4 chunks for the sweep launches (`pack_ell4`). Final passes and
    /// the PA kernels keep the plain ELL above.
    all_ell4: Option<CudaSlice<i32>>,
    weak_ell4: Option<CudaSlice<i32>>,
    /// Local build (`QUIP_OTSUKA_WEAK_SCALE`): plain ELLs with the weak
    /// couplings scaled, for anneal sweeps without the int4 packing (which is
    /// then built from these). `None` at scale 1. Final passes and the PA
    /// kernels keep the exact plain ELLs above.
    all_ell_anneal: Option<CudaSlice<i32>>,
    weak_ell_anneal: Option<CudaSlice<i32>>,
    /// Local build (`QUIP_OTSUKA_SWEEP_V4`): packed tile and pair records for
    /// the sweep draws (`pack_tiles_v4`, `pack_pairs_v4`).
    tiles_v4: Option<CudaSlice<i32>>,
    pairs_v4: Option<CudaSlice<i32>>,
}

#[allow(clippy::struct_excessive_bools)]
pub(crate) struct ClusterKernel {
    stream: Arc<CudaStream>,
    function: CudaFunction,
    energy: CudaFunction,
    /// Local build: compiled with `QUIP_OTSUKA_FAST` (16-bit parents).
    fast: bool,
    /// Local build: device PA kernels (`QUIP_OTSUKA_PA_DEVICE`).
    pa: Option<PaKernels>,
    /// Local build: compiled with `QUIP_OTSUKA_SWEEP_V3` (int4 ELL sweeps).
    sweep_v3: bool,
    /// Local build: compiled with `QUIP_OTSUKA_SWEEP_V4` (packed draw records).
    sweep_v4: bool,
    /// Local build: compiled with `QUIP_OTSUKA_SWEEP_V5` (16-bit ELL gathers).
    sweep_v5: bool,
    /// Local build: compiled with `QUIP_OTSUKA_SWEEP_V6` (class-split 16-bit
    /// gathers and the block's -s copy in shared memory).
    sweep_v6: bool,
}

/// Local build: the `QUIP_OTSUKA_FAST` NVRTC define.
const FAST_DEFINE: &str = "-DQUIP_OTSUKA_FAST=1";
/// Local build: the `QUIP_OTSUKA_SWEEP_V2` NVRTC define (restructured sweep).
const SWEEP_V2_DEFINE: &str = "-DQUIP_OTSUKA_SWEEP_V2=1";
/// Local build: the `QUIP_OTSUKA_SWEEP_V3` NVRTC define (V2 plus int4 ELL gathers).
const SWEEP_V3_DEFINE: &str = "-DQUIP_OTSUKA_SWEEP_V3=1";
/// Local build: the `QUIP_OTSUKA_SWEEP_V4` NVRTC define (V3 plus a balanced
/// union pass, batched compress and packed draw records).
const SWEEP_V4_DEFINE: &str = "-DQUIP_OTSUKA_SWEEP_V4=1";
/// Local build: the `QUIP_OTSUKA_SWEEP_V5` NVRTC define (V4 plus 16-bit ELL
/// entries for the Metropolis and accumulate gathers).
const SWEEP_V5_DEFINE: &str = "-DQUIP_OTSUKA_SWEEP_V5=1";
/// Local build: the `QUIP_OTSUKA_SWEEP_V6` NVRTC define (V5 plus ELL rows split
/// by coupling magnitude: an address, a shared load and an add per entry).
const SWEEP_V6_DEFINE: &str = "-DQUIP_OTSUKA_SWEEP_V6=1";
/// Local build: the `QUIP_OTSUKA_UNION_SV` NVRTC define (V4 with the
/// Shiloach-Vishkin-style union pass).
const UNION_SV_DEFINE: &str = "-DQUIP_OTSUKA_UNION_SV=1";

/// Local build (`QUIP_OTSUKA_SWEEP_V4`): one int4 per tile. `.x` = node 0 |
/// node 1 << 16, `.y` = node 2 | node 3 << 16, `.z` = gauge-positive bits 0-3
/// | gauge-negative bits 4-7, `.w` = the tile weight. The kernel's up mask
/// (`s * gauge > 0` per node) is `(sp & pos) | (!sp & neg)` with `sp` the
/// positive-spin bits, for any gauge value including zero.
fn pack_tiles_v4(
    tiles: &[i32],
    gauges: &[i8],
    tile_w: &[i32],
) -> Result<Vec<i32>, ClusterDeviceError> {
    let count = tiles.len() / 4;
    let node = |v: i32| -> Result<u32, ClusterDeviceError> {
        u16::try_from(v)
            .map(u32::from)
            .map_err(|_| ClusterDeviceError::Parameter("V4 packs tile nodes in 16 bits"))
    };
    let mut out = Vec::with_capacity(4 * count.max(1));
    for t in 0..count {
        let nd = &tiles[4 * t..4 * t + 4];
        let mut bits = 0u32;
        for k in 0..4 {
            let g = gauges.get(4 * t + k).copied().unwrap_or(0);
            if g > 0 {
                bits |= 1 << k;
            } else if g < 0 {
                bits |= 1 << (4 + k);
            }
        }
        out.extend([
            (node(nd[0])? | node(nd[1])? << 16).cast_signed(),
            (node(nd[2])? | node(nd[3])? << 16).cast_signed(),
            bits.cast_signed(),
            tile_w.get(t).copied().unwrap_or(0),
        ]);
    }
    Ok(out)
}

/// Local build (`QUIP_OTSUKA_SWEEP_V4`): one int2 per pair, `.x` = a | b << 16,
/// `.y` = the link weight.
fn pack_pairs_v4(pairs: &[i32], link_w: &[i32]) -> Result<Vec<i32>, ClusterDeviceError> {
    let count = pairs.len() / 2;
    let node = |v: i32| -> Result<u32, ClusterDeviceError> {
        u16::try_from(v)
            .map(u32::from)
            .map_err(|_| ClusterDeviceError::Parameter("V4 packs pair nodes in 16 bits"))
    };
    let mut out = Vec::with_capacity(2 * count.max(1));
    for p in 0..count {
        out.push((node(pairs[2 * p])? | node(pairs[2 * p + 1])? << 16).cast_signed());
        out.push(link_w.get(p).copied().unwrap_or(0));
    }
    Ok(out)
}

/// Local build: repack a column-major ELL (`ell[e * n + u]`, zero past each
/// row's degree) into int4 chunks: chunk `k` of node `u` is int4 `1 + k * n + u`,
/// holding entries `4k..4k+3`. Int4 0 is a header whose `.x` is the chunk count.
/// Zero padding words add `0 * s[0]`, so integer fields are unchanged.
fn pack_ell4(ell: &[i32], n: usize) -> Vec<i32> {
    if n == 0 {
        return vec![0; 4];
    }
    let width = ell.len() / n;
    let chunks = width.div_ceil(4);
    let mut out = vec![0; 4 * (1 + chunks * n)];
    out[0] = i32::try_from(chunks).unwrap_or(i32::MAX);
    for e in 0..width {
        for u in 0..n {
            out[4 * (1 + (e / 4) * n + u) + e % 4] = ell[e * n + u];
        }
    }
    out
}

/// Local build (`QUIP_OTSUKA_SWEEP_V5`): repack a column-major ELL into 16-bit
/// entries, `neighbor << 2 | code`, eight per int4: chunk `k` of node `u` is
/// int4 `1 + k * n + u`, entries `8k..8k+7`, entry `2q` in the low half of
/// word `q`. Int4 0 is the header `{chunks, 1, J0 | J1 << 16, J2 | J3 << 16}`
/// with `J[code]` the coupling. Padding entries are `u << 2` (the row's own
/// node, code 0): the accumulate's root test zeroes them, and Metropolis
/// subtracts their `J0 * s[u]` per row. `None` when the ELL has more
/// than four distinct couplings, a node index needs more than 14 bits, or a
/// coupling falls outside i16.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn pack_ell16(ell: &[i32], deg: &[i32], n: usize) -> Option<Vec<i32>> {
    if n == 0 || n >= 1 << 14 || deg.len() != n {
        return None;
    }
    let width = ell.len() / n;
    let mut codes: Vec<i16> = Vec::new();
    let chunks = width.div_ceil(8);
    let mut packed = vec![0u16; 8 * chunks * n];
    for u in 0..n {
        let own = u16::try_from(u << 2).ok()?;
        for e in 0..8 * chunks {
            packed[e * n + u] = own;
        }
        let d = usize::try_from(deg[u]).ok()?;
        if d > width {
            return None;
        }
        for e in 0..d {
            let w = ell[e * n + u].cast_unsigned();
            let (v, j) = (w >> 16, (w & 0xffff) as u16 as i16);
            if v as usize >= n {
                return None;
            }
            let c = match codes.iter().position(|&x| x == j) {
                Some(c) => c,
                None if codes.len() < 4 => {
                    codes.push(j);
                    codes.len() - 1
                }
                None => return None,
            };
            packed[e * n + u] = u16::try_from(v << 2 | c as u32).ok()?;
        }
    }
    let mut out = vec![0i32; 4 * (1 + chunks * n)];
    let half = |c: usize| u32::from(codes.get(c).copied().unwrap_or(0).cast_unsigned());
    out[0] = i32::try_from(chunks).ok()?;
    out[1] = 1;
    out[2] = (half(0) | half(1) << 16).cast_signed();
    out[3] = (half(2) | half(3) << 16).cast_signed();
    for e in 0..8 * chunks {
        for u in 0..n {
            let word = 4 * (1 + (e / 8) * n + u) + (e % 8) / 2;
            let shift = 16 * (e % 2);
            out[word] =
                (out[word].cast_unsigned() | u32::from(packed[e * n + u]) << shift).cast_signed();
        }
    }
    Some(out)
}

/// Local build (`QUIP_OTSUKA_SWEEP_V5`): the 16-bit layout when the graph
/// qualifies, else V4's int4 layout (header `.y` = 0), which the V5 kernel
/// reads with the V4 gathers.
fn pack_ell_v5(ell: &[i32], deg: &[i32], n: usize) -> Vec<i32> {
    pack_ell16(ell, deg, n).unwrap_or_else(|| {
        tracing::warn!(
            nodes = n,
            "QUIP_OTSUKA_SWEEP_V5: ELL not 16-bit packable; V4 gathers for this graph"
        );
        pack_ell4(ell, n)
    })
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): offset of the block's -s copy from
/// `s` in shared memory, the kernel's `V6_NEG_OFF(n)`: past `s`, the CU-1 flag
/// word and the V4 queue head.
const fn v6_neg_off(n: usize) -> usize {
    ((n + 3) & !3) + 8
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): extra dynamic shared bytes per block
/// for the -s copy and its zero byte, past the CU-1 layout (`9 n + 16`).
fn v6_shared_extra(n: usize) -> usize {
    // The copy ends at 8 n + V6_NEG_OFF(n) + n + 1 (the zero byte), rounded to a word.
    let end = (8 * n + v6_neg_off(n) + n + 1 + 3) & !3;
    end.saturating_sub(
        n * crate::capacity::CLUSTER_SHARED_BYTES_PER_NODE
            + crate::capacity::CLUSTER_DYNAMIC_PADDING,
    )
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): the rows of a column-major ELL split
/// into its two coupling magnitudes, `Some((js, jw, rows))` with `rows[u]` =
/// (strong entries, weak entries) as (neighbor, coupling sign) pairs in ELL
/// order. Strong is the larger magnitude unless a row holds more than eight
/// of it (then the smaller, when that fits; with a single magnitude every
/// entry is weak and `js` = 0). Zero couplings are dropped (they
/// add nothing). `None`: more than two magnitudes, more than eight strong
/// entries in some row either way, a neighbor out of range, or a node index
/// over 14 bits.
/// Local build (V6): per row, the (strong, weak) entries as (neighbor,
/// negative coupling) pairs.
type Rows6 = Vec<(Vec<(usize, bool)>, Vec<(usize, bool)>)>;

#[allow(clippy::cast_possible_wrap, clippy::needless_range_loop)] // i16 couplings; rows index deg and ell
fn ell6_rows(ell: &[i32], deg: &[i32], n: usize) -> Option<(i32, i32, Rows6)> {
    if n == 0 || n >= 1 << 14 || deg.len() != n {
        return None;
    }
    let width = ell.len() / n;
    let entry = |u: usize, e: usize| -> (usize, i32) {
        let w = ell[e * n + u].cast_unsigned();
        ((w >> 16) as usize, i32::from((w & 0xffff) as u16 as i16))
    };
    let mut mags: Vec<i32> = Vec::new();
    for u in 0..n {
        let d = usize::try_from(deg[u]).ok()?;
        if d > width {
            return None;
        }
        for e in 0..d {
            let (v, j) = entry(u, e);
            if v >= n {
                return None;
            }
            if j != 0 && !mags.contains(&j.abs()) {
                if mags.len() == 2 {
                    return None;
                }
                mags.push(j.abs());
            }
        }
    }
    mags.sort_unstable_by(|a, b| b.cmp(a));
    if mags.len() == 1 {
        mags.push(0);
    }
    let split = |js: i32| -> Option<Rows6> {
        let mut rows = Vec::with_capacity(n);
        for u in 0..n {
            let (mut strong, mut weak) = (Vec::new(), Vec::new());
            for e in 0..usize::try_from(deg[u]).ok()? {
                let (v, j) = entry(u, e);
                if j == 0 {
                    continue;
                }
                if j.abs() == js {
                    strong.push((v, j < 0))
                } else {
                    weak.push((v, j < 0))
                }
            }
            if strong.len() > 8 {
                return None;
            }
            rows.push((strong, weak));
        }
        Some(rows)
    };
    match mags.as_slice() {
        [] => Some((0, 0, vec![(Vec::new(), Vec::new()); n])),
        // With one magnitude, b = 0: the second try puts every entry in the weak class.
        &[a, b, ..] => split(a)
            .map(|r| (a, b, r))
            .or_else(|| split(b).map(|r| (b, a, r))),
        _ => None,
    }
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): the shared V6 layout around 16-bit
/// entries. Int4 0 is the header `{K, 2, js, jw}`; chunk `k` of row `u` is
/// int4 `1 + k * n + u` with eight entries, entry `2q + h` in half `h` of word
/// `q`, the strong entries in chunk 0 (padded to 8), the weak ones from chunk
/// 1 (padded to a whole chunk). Then one int per row: its chunk count `1 +
/// ceil(weak / 8)`. `half(u, v, neg)` encodes an entry, `pad(u, h)` the
/// padding of row `u` in word half `h`.
fn pack_ell6(
    js: i32,
    jw: i32,
    rows: &Rows6,
    n: usize,
    word: impl Fn(u32, u32) -> u32,
    half: impl Fn(usize, usize, bool) -> u32,
    pad: impl Fn(usize) -> u32,
) -> Option<Vec<i32>> {
    let kd: Vec<usize> = rows.iter().map(|(_, w)| 1 + w.len().div_ceil(8)).collect();
    let chunks = kd.iter().copied().max().unwrap_or(1).max(1);
    let mut out = vec![0i32; 4 * (1 + chunks * n) + n];
    out[0] = i32::try_from(chunks).ok()?;
    out[1] = 2;
    out[2] = js;
    out[3] = jw;
    for (u, (strong, weak)) in rows.iter().enumerate() {
        let mut slots = vec![pad(u); 8 * chunks];
        for (k, &(v, neg)) in strong.iter().enumerate() {
            slots[k] = half(u, v, neg);
        }
        for (k, &(v, neg)) in weak.iter().enumerate() {
            slots[8 + k] = half(u, v, neg);
        }
        for k in 0..kd[u] {
            for q in 0..4 {
                let (lo, hi) = (slots[8 * k + 2 * q], slots[8 * k + 2 * q + 1]);
                out[4 * (1 + k * n + u) + q] = word(lo, hi).cast_signed();
            }
        }
        out[4 * (1 + chunks * n) + u] = i32::try_from(kd[u]).ok()?;
    }
    Some(out)
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): the Metropolis (full) ELL. An entry
/// is a byte offset into the block's spin window: `v` for a positive
/// coupling, `V6_NEG_OFF(n) + v` (the -s copy) for a negative one, and
/// `V6_NEG_OFF(n) + n` (a zero byte) for padding; low entry in bits 0-15, high
/// entry in bits 16-31.
fn pack_ell6_all(ell: &[i32], deg: &[i32], n: usize) -> Option<Vec<i32>> {
    let (js, jw, rows) = ell6_rows(ell, deg, n)?;
    let off = v6_neg_off(n);
    if off + n > 0xffff {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    pack_ell6(
        js,
        jw,
        &rows,
        n,
        |lo, hi| lo | hi << 16,
        |_, v, neg| (if neg { off + v } else { v }) as u32,
        |_| (off + n) as u32,
    )
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): the accumulate (residual) ELL. A word
/// holds two entries: `4 * v_lo` in bits 0-15, `v_hi` in bits 16-29, the
/// coupling sign of the low entry in bit 30 and of the high one in bit 31.
/// Padding is the row's own node with a positive sign (its own root adds 0).
/// Entries are encoded as `v | neg << 14` and placed by `word`.
fn pack_ell6_weak(ell: &[i32], deg: &[i32], n: usize) -> Option<Vec<i32>> {
    let (js, jw, rows) = ell6_rows(ell, deg, n)?;
    #[allow(clippy::cast_possible_truncation)]
    pack_ell6(
        js,
        jw,
        &rows,
        n,
        |lo, hi| {
            ((lo & 0x3fff) << 2)
                | ((hi & 0x3fff) << 16)
                | ((lo >> 14) & 1) << 30
                | ((hi >> 14) & 1) << 31
        },
        |_, v, neg| v as u32 | u32::from(neg) << 14,
        |u| u as u32,
    )
}

/// Local build (`QUIP_OTSUKA_SWEEP_V6`): the V6 layout when the ELL qualifies,
/// else V5's (16-bit codes, or V4's int4 words), which the V6 kernel reads
/// with the V5/V4 gathers.
fn pack_ell_v6(ell: &[i32], deg: &[i32], n: usize, weak: bool) -> Vec<i32> {
    let packed = if weak {
        pack_ell6_weak(ell, deg, n)
    } else {
        pack_ell6_all(ell, deg, n)
    };
    packed.unwrap_or_else(|| {
        tracing::warn!(
            nodes = n,
            weak,
            "QUIP_OTSUKA_SWEEP_V6: ELL not V6 packable; V5 gathers for this graph"
        );
        pack_ell_v5(ell, deg, n)
    })
}

/// Dynamic shared bytes of one block: 7 per node with 16-bit parents; V6
/// adds the -s copy.
fn shared_bytes(graph: &PreparedGraph, fast: bool, v6: bool) -> usize {
    if fast {
        graph.original_graph().num_nodes() * 7 + crate::capacity::CLUSTER_DYNAMIC_PADDING
    } else if v6 {
        graph.dynamic_shared_bytes() + v6_shared_extra(graph.original_graph().num_nodes())
    } else {
        graph.dynamic_shared_bytes()
    }
}

impl ClusterKernel {
    pub(crate) fn load(device_index: usize) -> Result<Self, ClusterDeviceError> {
        Self::load_variant(
            device_index,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)] // one flag per NVRTC variant
    pub(crate) fn load_variant(
        device_index: usize,
        fast: bool,
        sweep_v2: bool,
        device_pa: bool,
        sweep_v3: bool,
        sweep_v4: bool,
        sweep_v5: bool,
        sweep_v6: bool,
        union_sv: bool,
    ) -> Result<Self, ClusterDeviceError> {
        // V6 builds on V5, V5 on V4, V4 on V3, which is the V2 sweep with int4
        // ELL gathers. The SV union pass replaces V4's.
        let sweep_v5 = sweep_v5 || sweep_v6;
        let sweep_v4 = sweep_v4 || sweep_v5 || union_sv;
        let sweep_v3 = sweep_v3 || sweep_v4;
        let sweep_v2 = sweep_v2 || sweep_v3;
        // The V2 sweep keeps 32-bit parents; it replaces FAST rather than stacking.
        let fast = fast && !sweep_v2;
        let mut defines: Vec<&str> = Vec::new();
        if fast {
            defines.push(FAST_DEFINE);
        }
        if sweep_v2 {
            defines.push(SWEEP_V2_DEFINE);
        }
        if sweep_v3 {
            defines.push(SWEEP_V3_DEFINE);
        }
        if sweep_v4 {
            defines.push(SWEEP_V4_DEFINE);
        }
        if sweep_v5 {
            defines.push(SWEEP_V5_DEFINE);
        }
        if sweep_v6 {
            defines.push(SWEEP_V6_DEFINE);
        }
        if union_sv {
            defines.push(UNION_SV_DEFINE);
        }
        // Local build, profiling only: extra NVRTC options, comma or space separated
        // (e.g. "-lineinfo,-DQUIP_OTSUKA_PHASE_TIMING=1").
        let extra = std::env::var("QUIP_OTSUKA_NVRTC_OPTS").unwrap_or_default();
        defines.extend(
            extra
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|o| !o.is_empty()),
        );
        let defines = defines.as_slice();
        let ctx = CudaContext::new(device_index)?;
        // Synchronous allocation can wait for the persistent SA kernel.
        // Refuse before JIT or stream creation, while SA is still idle.
        check_async_alloc(ctx.has_async_alloc())?;
        let arch = device_arch(&ctx);
        let module = jit_cache::load_or_compile(
            &ctx,
            "cluster",
            SOURCE,
            defines,
            &format!("sm_{arch}"),
            driver_version()?,
            0,
            || compile_for_arch(SOURCE, 0, arch, defines),
        )?;
        // Resolve the function before SA launches. Lazy module loading can
        // synchronize the primary context, even with nonblocking streams.
        let function = module.load_function("quip_cluster_anneal")?;
        // Local build: the population-annealing energy kernel, resolved up
        // front for the same reason. The stock path never launches it.
        let energy = module.load_function("quip_cluster_energy")?;
        // Local build: the device PA module only when asked for, so the stock
        // and host-PA paths compile and load exactly what they did before.
        let pa = if device_pa {
            let pa_module = jit_cache::load_or_compile(
                &ctx,
                "cluster_pa",
                PA_SOURCE,
                &[],
                &format!("sm_{arch}"),
                driver_version()?,
                0,
                || compile_for_arch(PA_SOURCE, 0, arch, &[]),
            )?;
            Some(PaKernels {
                energy_best: pa_module.load_function("quip_pa_energy_best")?,
                plan: pa_module.load_function("quip_pa_plan")?,
                gather: pa_module.load_function("quip_pa_gather")?,
                houdayer: pa_module.load_function("quip_pa_houdayer")?,
            })
        } else {
            None
        };
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
        Ok(Self {
            stream,
            function,
            energy,
            fast,
            pa,
            sweep_v3,
            sweep_v4,
            sweep_v5,
            sweep_v6,
        })
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

    #[allow(clippy::too_many_lines)] // the stock upload plus the local variants' tables
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
        let energy = kernel.energy.clone();
        let shared = shared_bytes(&graph, kernel.fast, kernel.sweep_v6);
        // Local build (V6): the -s copy must fit the opt-in budget too.
        if kernel.sweep_v6 {
            let budget = attribute(
                ctx,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?;
            if shared + CLUSTER_STATIC_SHARED_BYTES > budget {
                return Err(PrepareError::SharedMemory {
                    requested: shared + CLUSTER_STATIC_SHARED_BYTES,
                    budget,
                }
                .into());
            }
        }
        function.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(shared)
                .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds i32"))?,
        )?;
        // Local build: device PA needs the Houdayer queue in shared memory.
        // A graph too large for it falls back to host checkpoints.
        let pa = match &kernel.pa {
            Some(pa) => {
                let bytes = houdayer_shared(graph.original_graph().num_nodes());
                let optin = attribute(
                    ctx,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )?;
                if bytes + 4 * PA_HOUDAYER_THREADS as usize + 64 <= optin {
                    pa.houdayer.set_attribute(
                        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        i32::try_from(bytes).map_err(|_| {
                            ClusterDeviceError::Parameter("shared memory exceeds i32")
                        })?,
                    )?;
                    Some(pa.clone())
                } else {
                    None
                }
            }
            None => None,
        };
        let stream = Arc::clone(&kernel.stream);
        let d = graph.layout();
        // Local build (QUIP_OTSUKA_WEAK_SCALE): the sweeps' tables. The exact
        // ones (d.weak_ell, d.all_ell) stay for final passes and energies.
        let scaled = graph.anneal_ell();
        let (sweep_weak, sweep_all) =
            scaled.map_or((&d.weak_ell, &d.all_ell), |a| (&a.weak_ell, &a.all_ell));
        Ok(Self {
            all_ell_anneal: match scaled {
                Some(a) if !(kernel.sweep_v3 || kernel.sweep_v5) => {
                    Some(upload(&stream, &a.all_ell)?)
                }
                _ => None,
            },
            weak_ell_anneal: match scaled {
                Some(a) if !(kernel.sweep_v3 || kernel.sweep_v5) => {
                    Some(upload(&stream, &a.weak_ell)?)
                }
                _ => None,
            },
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
            weak_ell4: if kernel.sweep_v6 {
                Some(upload(
                    &stream,
                    &pack_ell_v6(
                        sweep_weak,
                        &d.weak_deg,
                        graph.original_graph().num_nodes(),
                        true,
                    ),
                )?)
            } else if kernel.sweep_v5 {
                Some(upload(
                    &stream,
                    &pack_ell_v5(sweep_weak, &d.weak_deg, graph.original_graph().num_nodes()),
                )?)
            } else if kernel.sweep_v3 {
                Some(upload(
                    &stream,
                    &pack_ell4(sweep_weak, graph.original_graph().num_nodes()),
                )?)
            } else {
                None
            },
            all_ell4: if kernel.sweep_v6 {
                Some(upload(
                    &stream,
                    &pack_ell_v6(
                        sweep_all,
                        &d.all_deg,
                        graph.original_graph().num_nodes(),
                        false,
                    ),
                )?)
            } else if kernel.sweep_v5 {
                Some(upload(
                    &stream,
                    &pack_ell_v5(sweep_all, &d.all_deg, graph.original_graph().num_nodes()),
                )?)
            } else if kernel.sweep_v3 {
                Some(upload(
                    &stream,
                    &pack_ell4(sweep_all, graph.original_graph().num_nodes()),
                )?)
            } else {
                None
            },
            tiles_v4: if kernel.sweep_v4 {
                Some(upload(
                    &stream,
                    &pack_tiles_v4(&d.tiles, &d.gauges, &d.tile_w)?,
                )?)
            } else {
                None
            },
            pairs_v4: if kernel.sweep_v4 {
                Some(upload(&stream, &pack_pairs_v4(&d.pairs, &d.link_w)?)?)
            } else {
                None
            },
            graph,
            stream,
            function,
            energy,
            shared,
            pa,
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
        self.run_work_chunked(work, stopped, cancelled, 1, &OtsukaTuning::default())
    }

    /// Local build: [`Self::run_work`] under opt-in knobs. The default tuning
    /// is exactly `run_work`. Chunked and population modes launch the kernel's
    /// own per-sweep geometric ladder between the work's end betas.
    ///
    /// # Errors
    /// Returns admission, device, or cancellation errors without partial output.
    #[doc(hidden)]
    pub fn run_work_tuned(
        &mut self,
        work: &super::sampler::ClusterWork,
        tuning: &OtsukaTuning,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        self.run_work_probed(work, tuning, stopped, cancelled, None, PROBE_EVERY)
    }

    /// Local build (all7, deep lane): [`Self::run_work_tuned`] with a probe.
    /// A population anneal calls `probe` with the lowest lineage best (milli,
    /// lane energy) at the first checkpoint after each `every` (the sampler
    /// passes [`super::deep::PROBE_EVERY`]);
    /// `true` ends the anneal there: the final descent runs and the reads
    /// (each replica's state or lineage best, the lower) come back as from a
    /// completed run. Without a probe, or on a path without checkpoints
    /// (stock launches, no population), this is exactly `run_work_tuned`.
    ///
    /// # Errors
    /// Returns admission, device, or cancellation errors without partial output.
    #[doc(hidden)]
    pub fn run_work_probed(
        &mut self,
        work: &super::sampler::ClusterWork,
        tuning: &OtsukaTuning,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: Option<&dyn Fn(i64) -> bool>,
        every: Duration,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        if tuning.stock_launches() || !work.geometric {
            self.run_work_chunked(work, stopped, cancelled, 1, tuning)
        } else {
            self.run_native(work, tuning, stopped, cancelled, probe.map(|p| (p, every)))
        }
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
        self.run_work_chunked(work, &|| false, &|| false, chunk, &OtsukaTuning::default())
    }

    fn run_work_chunked(
        &mut self,
        work: &super::sampler::ClusterWork,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        chunk: usize,
        tuning: &OtsukaTuning,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        let schedule = ClusterSchedule {
            sweeps: work.betas.len() * work.sweeps_per_beta,
            chunk: 1,
            single_every: tuning.single_every,
            rounds: tuning.rounds,
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
            shared_mem_bytes: u32::try_from(self.shared)
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

    /// Local build: chunked launches of the kernel's own geometric ladder,
    /// optionally as a population anneal (resampling plus host Houdayer moves
    /// at `tuning.resamples` checkpoints). Ends with one 256-pass descent.
    #[allow(clippy::too_many_lines)]
    fn run_native(
        &mut self,
        work: &super::sampler::ClusterWork,
        tuning: &OtsukaTuning,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: Option<(&dyn Fn(i64) -> bool, Duration)>,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        let total = work.betas.len() * work.sweeps_per_beta;
        let (Some(&hot), Some(&cold)) = (work.betas.first(), work.betas.last()) else {
            return Err(ClusterDeviceError::JobParameter("empty beta ladder"));
        };
        // Local build (all11): `QUIP_OTSUKA_BETA_KNEE`; (0, 0) when off, so
        // Params match all10 byte for byte.
        let (beta_knee, knee_frac) = tuning.knee_for(hot, cold);
        let schedule = ClusterSchedule {
            sweeps: total,
            single_every: tuning.single_every,
            rounds: tuning.rounds,
            chunk: tuning.chunk.max(1),
            beta_min: hot,
            beta_knee,
            knee_frac,
            beta_max: cold,
        };
        let params = schedule.params()?;
        let ctx = self.stream.context();
        check_shared(
            &self.graph,
            attribute(
                ctx,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?,
        )?;
        let anneals = work.reads;
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
        check_memory(
            &self.graph,
            anneals,
            ctx.mem_get_info()?.0.saturating_add(model_bytes),
        )?;
        let (spin_count, rng_count, blocks) = launch_shape(&self.graph, anneals)?;
        let (host_spins, host_rng) = initial_state(spin_count, rng_count, work.seed);
        let mut spins = self.stream.clone_htod(&host_spins)?;
        let mut rng = self.stream.clone_htod(&host_rng)?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: u32::try_from(self.shared)
                .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds u32"))?,
        };
        let population = tuning.population && anneals > 1;
        if population && tuning.device_pa {
            if let Some(pa) = self.pa.clone() {
                return self.run_device_pa(
                    &pa, work, tuning, params, config, spins, rng, stopped, cancelled, probe,
                );
            }
        }
        let every = if population {
            (total / tuning.resamples.max(1)).max(1)
        } else {
            total
        };
        // Within a checkpoint segment, chunk 1 means "the whole segment".
        let step = if tuning.chunk > 1 {
            tuning.chunk
        } else {
            usize::MAX
        };
        let mut pop = population.then(|| Population::new(anneals, work.seed, hot));
        let mut last_probe = std::time::Instant::now();
        let mut sweep = 0;
        while sweep < total {
            let end = (sweep / every + 1).saturating_mul(every).min(total);
            while sweep < end {
                stop_point(stopped)?;
                let count = step.min(end - sweep);
                self.launch(
                    params,
                    int_arg(sweep)?,
                    int_arg(count)?,
                    0,
                    config,
                    &mut spins,
                    &mut rng,
                )?;
                self.stream.synchronize()?;
                sweep += count;
            }
            if let Some(pop) = pop.as_mut().filter(|_| sweep < total) {
                stop_point(stopped)?;
                let beta = host_beta(&params, sweep);
                self.population_step(pop, &mut spins, beta, tuning.houdayer_beta)?;
                // Local build (all7): the deep lane's probe, on the host bests.
                if let Some((probe, _)) = probe.filter(|p| last_probe.elapsed() >= p.1) {
                    last_probe = std::time::Instant::now();
                    let lowest = (0..pop.slots())
                        .filter_map(|s| pop.best(s).map(|b| b.1))
                        .min();
                    if lowest.is_some_and(probe) {
                        break;
                    }
                }
            }
        }
        stop_point(stopped)?;
        self.launch(params, 0, 0, 1, config, &mut spins, &mut rng)?;
        let states = self.stream.clone_dtoh(&spins)?;
        self.stream.synchronize()?;
        let n = self.graph.original_graph().num_nodes();
        let mut result = ClusterRun {
            spins: Vec::with_capacity(anneals),
            energies: Vec::with_capacity(anneals),
        };
        for (slot, s) in states.chunks_exact(n).enumerate() {
            if cancelled() {
                return Err(ClusterDeviceError::Cancelled);
            }
            let mut original = self.graph.unpermute(s)?;
            let mut energy = self.graph.energy(&original)?;
            if let Some((best, best_energy)) = pop.as_ref().and_then(|p| p.best(slot)) {
                if best_energy < energy {
                    original = self.graph.unpermute(best)?;
                    energy = self.graph.energy(&original)?;
                }
            }
            result.energies.push(energy);
            result.spins.push(original);
        }
        Ok(result)
    }

    /// Local build: [`Self::run_native`]'s population anneal with every
    /// checkpoint on the device. The schedule, checkpoint sweeps, and resampling
    /// rule are the host path's; only the random streams differ. The host keeps
    /// at most [`PA_QUEUE_DEPTH`] segments queued, so a stop takes effect within
    /// two segments, and synchronizes once at the end.
    #[allow(clippy::too_many_arguments)]
    fn run_device_pa(
        &mut self,
        pa: &PaKernels,
        work: &super::sampler::ClusterWork,
        tuning: &OtsukaTuning,
        params: Params,
        config: LaunchConfig,
        spins: CudaSlice<i8>,
        mut rng: CudaSlice<u64>,
        stopped: &dyn Fn() -> bool,
        cancelled: &dyn Fn() -> bool,
        probe: Option<(&dyn Fn(i64) -> bool, Duration)>,
    ) -> Result<ClusterRun, ClusterDeviceError> {
        let total = work.betas.len() * work.sweeps_per_beta;
        let anneals = work.reads;
        let mut last_probe = std::time::Instant::now();
        let n = self.graph.original_graph().num_nodes();
        let mut bufs = PaBuffers::new(&self.stream, spins, anneals, n)?;
        let every = (total / tuning.resamples.max(1)).max(1);
        let step = if tuning.chunk > 1 {
            tuning.chunk
        } else {
            usize::MAX
        };
        let mut last_beta = params.beta_min;
        let mut queued = std::collections::VecDeque::with_capacity(PA_QUEUE_DEPTH + 1);
        let mut checkpoint_index: i32 = 0;
        // Local build (QUIP_OTSUKA_HOUDAYER_EVERY): pairing-only moves inside
        // segments. Their RNG indices live far from the checkpoints'.
        let intra_every = if tuning.houdayer_beta > 0.0 && anneals >= 2 {
            tuning.houdayer_every
        } else {
            0
        };
        let mut intra_index: i32 = INTRA_HOUDAYER_INDEX;
        let mut sweep = 0;
        while sweep < total {
            let end = (sweep / every + 1).saturating_mul(every).min(total);
            while sweep < end {
                stop_point(stopped)?;
                let mut count = step.min(end - sweep);
                let intra = intra_move(intra_every, sweep, &mut count, end, |b| {
                    host_beta(&params, b - 1) >= tuning.houdayer_beta
                });
                self.launch(
                    params,
                    int_arg(sweep)?,
                    int_arg(count)?,
                    0,
                    config,
                    &mut bufs.spins,
                    &mut rng,
                )?;
                sweep += count;
                if intra {
                    self.enqueue_pa_houdayer(pa, &mut bufs, work.seed, intra_index)?;
                    intra_index = intra_index.wrapping_add(1);
                }
            }
            if sweep >= total {
                break;
            }
            let beta = host_beta(&params, sweep);
            let delta = f64::from(beta - last_beta);
            last_beta = beta;
            let houdayer = tuning.houdayer_beta > 0.0 && beta >= tuning.houdayer_beta;
            self.enqueue_pa_checkpoint(
                pa,
                &mut bufs,
                delta,
                houdayer,
                work.seed,
                checkpoint_index,
            )?;
            checkpoint_index = checkpoint_index.wrapping_add(1);
            queued.push_back(self.stream.record_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            ))?);
            if queued.len() > PA_QUEUE_DEPTH {
                if let Some(event) = queued.pop_front() {
                    event.synchronize()?;
                }
            }
            // Local build (all7): the deep lane's probe. The copy waits for
            // every queued segment (one short bubble per probe); the lineage
            // bests are this checkpoint's.
            if let Some((probe, _)) = probe.filter(|p| last_probe.elapsed() >= p.1) {
                last_probe = std::time::Instant::now();
                let bests = self.stream.clone_dtoh(&bufs.best_e)?;
                self.stream.synchronize()?;
                queued.clear();
                if bests.iter().copied().min().is_some_and(probe) {
                    break;
                }
            }
        }
        stop_point(stopped)?;
        self.launch(params, 0, 0, 1, config, &mut bufs.spins, &mut rng)?;
        let states = self.stream.clone_dtoh(&bufs.spins)?;
        let best_states = self.stream.clone_dtoh(&bufs.best)?;
        let best_energies = self.stream.clone_dtoh(&bufs.best_e)?;
        self.stream.synchronize()?;
        let mut result = ClusterRun {
            spins: Vec::with_capacity(anneals),
            energies: Vec::with_capacity(anneals),
        };
        for (slot, s) in states.chunks_exact(n).enumerate() {
            if cancelled() {
                return Err(ClusterDeviceError::Cancelled);
            }
            let mut original = self.graph.unpermute(s)?;
            let mut e = self.graph.energy(&original)?;
            if best_energies[slot] < e {
                original = self
                    .graph
                    .unpermute(&best_states[slot * n..(slot + 1) * n])?;
                e = self.graph.energy(&original)?;
            }
            result.energies.push(e);
            result.spins.push(original);
        }
        Ok(result)
    }

    /// Local build: enqueue one device checkpoint on the lane's stream:
    /// energies and lineage bests; with `delta > 0` systematic resampling and
    /// the parent gather (buffers swap); with `houdayer` one Houdayer move per
    /// shuffled pair. No host synchronization.
    fn enqueue_pa_checkpoint(
        &self,
        pa: &PaKernels,
        bufs: &mut PaBuffers,
        delta: f64,
        houdayer: bool,
        seed: u64,
        checkpoint: i32,
    ) -> Result<(), ClusterDeviceError> {
        let n = self.graph.original_graph().num_nodes();
        let n_arg = int_arg(n)?;
        let slots = int_arg(bufs.anneals)?;
        let grid = u32::try_from(bufs.anneals)
            .map_err(|_| ClusterDeviceError::Parameter("anneals exceed grid width"))?;
        let houdayer = houdayer && bufs.anneals >= 2;
        let houdayer_arg = i32::from(houdayer);
        let per_replica = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&pa.energy_best);
        launch
            .arg(&n_arg)
            .arg(&self.all_deg)
            .arg(&self.all_ell)
            .arg(&self.h)
            .arg(&bufs.spins)
            .arg(&mut bufs.energy)
            .arg(&mut bufs.best)
            .arg(&mut bufs.best_e);
        // SAFETY: one block per replica over anneals * n spins and bests and
        // anneals energies; the ELL arrays are the anneal kernel's.
        unsafe { launch.launch(per_replica) }?;
        if delta > 0.0 || houdayer {
            let mut launch = self.stream.launch_builder(&pa.plan);
            launch
                .arg(&slots)
                .arg(&delta)
                .arg(&houdayer_arg)
                .arg(&seed)
                .arg(&checkpoint)
                .arg(&bufs.energy)
                .arg(&mut bufs.picks)
                .arg(&mut bufs.order)
                .arg(&mut bufs.weight);
            // SAFETY: one block reads `slots` energies and writes `slots`
            // weights, picks, and order entries, each allocated with `anneals`.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }
        if delta > 0.0 {
            let mut launch = self.stream.launch_builder(&pa.gather);
            launch
                .arg(&n_arg)
                .arg(&bufs.picks)
                .arg(&bufs.spins)
                .arg(&mut bufs.alt)
                .arg(&bufs.best)
                .arg(&mut bufs.best_alt)
                .arg(&bufs.best_e)
                .arg(&mut bufs.best_e_alt);
            // SAFETY: picks hold slot indices below anneals; every buffer holds
            // anneals replicas of n spins, or one energy each.
            unsafe { launch.launch(per_replica) }?;
            std::mem::swap(&mut bufs.spins, &mut bufs.alt);
            std::mem::swap(&mut bufs.best, &mut bufs.best_alt);
            std::mem::swap(&mut bufs.best_e, &mut bufs.best_e_alt);
        }
        if houdayer {
            let mut launch = self.stream.launch_builder(&pa.houdayer);
            launch
                .arg(&n_arg)
                .arg(&self.all_deg)
                .arg(&self.all_ell)
                .arg(&mut bufs.spins)
                .arg(&bufs.order)
                .arg(&seed)
                .arg(&checkpoint);
            // SAFETY: order is a permutation of the slots, so the two replicas
            // of each block are distinct; shared memory is houdayer_shared(n),
            // admitted when the device opened.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (grid / 2, 1, 1),
                    block_dim: (PA_HOUDAYER_THREADS, 1, 1),
                    shared_mem_bytes: u32::try_from(houdayer_shared(n))
                        .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds u32"))?,
                })
            }?;
        }
        Ok(())
    }

    /// Local build (`QUIP_OTSUKA_HOUDAYER_EVERY`): one pairing-only Houdayer
    /// move inside a segment: shuffle the pairing (no resampling: the plan
    /// kernel at zero step leaves every replica in place), swap one differing
    /// component per pair, then refresh energies and lineage bests so a
    /// replica the swap lowered keeps that state. No host synchronization.
    fn enqueue_pa_houdayer(
        &self,
        pa: &PaKernels,
        bufs: &mut PaBuffers,
        seed: u64,
        index: i32,
    ) -> Result<(), ClusterDeviceError> {
        if bufs.anneals < 2 {
            return Ok(());
        }
        let n = self.graph.original_graph().num_nodes();
        let n_arg = int_arg(n)?;
        let slots = int_arg(bufs.anneals)?;
        let grid = u32::try_from(bufs.anneals)
            .map_err(|_| ClusterDeviceError::Parameter("anneals exceed grid width"))?;
        let one: i32 = 1;
        let zero_step: f64 = 0.0;
        let mut launch = self.stream.launch_builder(&pa.plan);
        launch
            .arg(&slots)
            .arg(&zero_step)
            .arg(&one)
            .arg(&seed)
            .arg(&index)
            .arg(&bufs.energy)
            .arg(&mut bufs.picks)
            .arg(&mut bufs.order)
            .arg(&mut bufs.weight);
        // SAFETY: as in enqueue_pa_checkpoint; at zero step the kernel reads
        // no energies or weights and writes the identity picks and a shuffle
        // of `order`, each `slots` long.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        let mut launch = self.stream.launch_builder(&pa.houdayer);
        launch
            .arg(&n_arg)
            .arg(&self.all_deg)
            .arg(&self.all_ell)
            .arg(&mut bufs.spins)
            .arg(&bufs.order)
            .arg(&seed)
            .arg(&index);
        // SAFETY: as in enqueue_pa_checkpoint: order is a permutation of the
        // slots and the shared size was admitted when the device opened.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid / 2, 1, 1),
                block_dim: (PA_HOUDAYER_THREADS, 1, 1),
                shared_mem_bytes: u32::try_from(houdayer_shared(n))
                    .map_err(|_| ClusterDeviceError::Parameter("shared memory exceeds u32"))?,
            })
        }?;
        let mut launch = self.stream.launch_builder(&pa.energy_best);
        launch
            .arg(&n_arg)
            .arg(&self.all_deg)
            .arg(&self.all_ell)
            .arg(&self.h)
            .arg(&bufs.spins)
            .arg(&mut bufs.energy)
            .arg(&mut bufs.best)
            .arg(&mut bufs.best_e);
        // SAFETY: as in enqueue_pa_checkpoint, one block per replica.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    /// One population checkpoint: device energies, best tracking, systematic
    /// resampling at `beta`, then Houdayer pairs on the host from `houdayer_beta`.
    fn population_step(
        &self,
        pop: &mut Population,
        spins: &mut CudaSlice<i8>,
        beta: f32,
        houdayer_beta: f32,
    ) -> Result<(), ClusterDeviceError> {
        let n = self.graph.original_graph().num_nodes();
        let anneals = pop.slots();
        let mut out = self.stream.alloc_zeros::<i64>(anneals)?;
        let n_arg = int_arg(n)?;
        let mut launch = self.stream.launch_builder(&self.energy);
        launch
            .arg(&n_arg)
            .arg(&self.all_deg)
            .arg(&self.all_ell)
            .arg(&self.h)
            .arg(&*spins)
            .arg(&mut out);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(anneals)
                    .map_err(|_| ClusterDeviceError::Parameter("anneals exceed grid width"))?,
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the energy kernel reads n spins per block from an allocation
        // of anneals * n, and the same validated ELL arrays as the anneal kernel.
        unsafe { launch.launch(config) }?;
        let energies = self.stream.clone_dtoh(&out)?;
        let mut host = self.stream.clone_dtoh(&*spins)?;
        self.stream.synchronize()?;
        let layout = self.graph.layout();
        pop.checkpoint(
            &mut host,
            &energies,
            n,
            beta,
            (houdayer_beta > 0.0 && beta >= houdayer_beta)
                .then_some((layout.all_deg.as_slice(), layout.all_ell.as_slice())),
        );
        self.stream.memcpy_htod(&host, spins)?;
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
        // Local build: V3 sweeps read the int4 ELLs; final passes the plain ones.
        // Local build (QUIP_OTSUKA_WEAK_SCALE): sweeps read the scaled tables
        // (packed or plain); final passes always the exact plain ones.
        let all_ell = match (&self.all_ell4, &self.all_ell_anneal, final_pass) {
            (Some(a), _, 0) | (None, Some(a), 0) => a,
            _ => &self.all_ell,
        };
        let weak_ell = match (&self.weak_ell4, &self.weak_ell_anneal, final_pass) {
            (Some(w), _, 0) | (None, Some(w), 0) => w,
            _ => &self.weak_ell,
        };
        // Local build: V4 sweeps read packed tile and pair records.
        let tiles = match (&self.tiles_v4, final_pass) {
            (Some(t), 0) => t,
            _ => &self.tiles,
        };
        let pairs = match (&self.pairs_v4, final_pass) {
            (Some(p), 0) => p,
            _ => &self.pairs,
        };
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(&h[0])
            .arg(&h[1])
            .arg(&h[2])
            .arg(&h[5])
            .arg(tiles)
            .arg(&self.gauges)
            .arg(pairs)
            .arg(&self.weak_deg)
            .arg(weak_ell)
            .arg(&self.all_deg)
            .arg(all_ell)
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

fn stop_point(stopped: &dyn Fn() -> bool) -> Result<(), ClusterDeviceError> {
    if stopped() {
        Err(ClusterDeviceError::Cancelled)
    } else {
        Ok(())
    }
}

fn int_arg(value: usize) -> Result<i32, ClusterDeviceError> {
    i32::try_from(value).map_err(|_| ClusterDeviceError::Parameter("count exceeds i32"))
}

/// Local build: first RNG index of the in-segment Houdayer moves. The
/// checkpoints count up from 0; this sits 2^30 above, so no index repeats.
const INTRA_HOUDAYER_INDEX: i32 = 1 << 30;

/// Local build (`QUIP_OTSUKA_HOUDAYER_EVERY`): plan one launch of a segment
/// starting at `sweep`, `count` sweeps long, inside a segment ending at `end`.
/// With `every > 0`, the launch is cut at the first multiple of `every` that
/// lies strictly inside the segment and where `due(boundary)` holds (beta of
/// the boundary's last sweep at or above the Houdayer beta); the return says
/// whether a move follows the launch. A segment end is left alone: the
/// checkpoint there already pairs the replicas. `every == 0` never changes
/// `count`, so the stock launch sequence is untouched.
fn intra_move(
    every: usize,
    sweep: usize,
    count: &mut usize,
    end: usize,
    due: impl Fn(usize) -> bool,
) -> bool {
    if every == 0 {
        return false;
    }
    let mut b = (sweep / every + 1).saturating_mul(every);
    while b <= sweep + *count && b < end {
        if due(b) {
            *count = b - sweep;
            return true;
        }
        b = b.saturating_add(every);
    }
    false
}

/// The kernel's `beta_at` on the host. Local build (all11): with the knee
/// (`QUIP_OTSUKA_BETA_KNEE`), the same piecewise geometric branch and test as
/// the kernel; without it, the all10 formula unchanged.
#[allow(clippy::cast_precision_loss)]
fn host_beta(q: &Params, sweep: usize) -> f32 {
    let f = if q.sweeps > 1 {
        sweep as f32 / (q.sweeps - 1) as f32
    } else {
        1.0
    };
    if q.beta_knee > 0.0 && q.knee_frac > 0.0 && q.knee_frac < 1.0 {
        if f < q.knee_frac {
            return q.beta_min * (q.beta_knee / q.beta_min).powf(f / q.knee_frac);
        }
        return q.beta_knee
            * (q.beta_max / q.beta_knee).powf((f - q.knee_frac) / (1.0 - q.knee_frac));
    }
    q.beta_min * (q.beta_max / q.beta_min).powf(f)
}

/// Local build: device buffers of a device population anneal. `spins` is the
/// live state; `alt`, `best_alt`, and `best_e_alt` are gather targets.
struct PaBuffers {
    anneals: usize,
    spins: CudaSlice<i8>,
    alt: CudaSlice<i8>,
    best: CudaSlice<i8>,
    best_alt: CudaSlice<i8>,
    best_e: CudaSlice<i64>,
    best_e_alt: CudaSlice<i64>,
    energy: CudaSlice<i64>,
    picks: CudaSlice<i32>,
    order: CudaSlice<i32>,
    weight: CudaSlice<f64>,
}

impl PaBuffers {
    fn new(
        stream: &Arc<CudaStream>,
        spins: CudaSlice<i8>,
        anneals: usize,
        n: usize,
    ) -> Result<Self, ClusterDeviceError> {
        let order: Vec<i32> = (0..int_arg(anneals)?).collect();
        Ok(Self {
            anneals,
            spins,
            alt: stream.alloc_zeros::<i8>(anneals * n)?,
            best: stream.alloc_zeros::<i8>(anneals * n)?,
            best_alt: stream.alloc_zeros::<i8>(anneals * n)?,
            best_e: stream.clone_htod(&vec![i64::MAX; anneals])?,
            best_e_alt: stream.alloc_zeros::<i64>(anneals)?,
            energy: stream.alloc_zeros::<i64>(anneals)?,
            picks: stream.alloc_zeros::<i32>(anneals)?,
            order: stream.clone_htod(&order)?,
            weight: stream.alloc_zeros::<f64>(anneals)?,
        })
    }
}

/// Local build: host side of a population anneal (Hukushima-Iba resampling
/// with Houdayer moves, after `quip-solver-cpu` `anneal_population`). Each
/// slot keeps the lowest state its lineage showed at a checkpoint. The device
/// RNG streams stay with their blocks, so copies of one state diverge.
struct Population {
    best: Vec<Option<(Vec<i8>, i64)>>,
    last_beta: Option<f32>,
    rng: u64,
    seen: Vec<bool>,
    cluster: Vec<u32>,
    differ: Vec<u32>,
    order: Vec<usize>,
}

impl Population {
    fn new(slots: usize, seed: u64, beta0: f32) -> Self {
        Self {
            best: vec![None; slots],
            last_beta: Some(beta0),
            rng: seed ^ 0x5E_ED_0F_07_50_CA_00,
            seen: Vec::new(),
            cluster: Vec::new(),
            differ: Vec::new(),
            order: (0..slots).collect(),
        }
    }

    fn slots(&self) -> usize {
        self.best.len()
    }

    fn best(&self, slot: usize) -> Option<(&[i8], i64)> {
        self.best
            .get(slot)
            .and_then(Option::as_ref)
            .map(|(s, e)| (s.as_slice(), *e))
    }

    fn next(&mut self) -> u64 {
        // splitmix64
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_precision_loss)]
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }

    /// Track bests, resample at `beta`, then optional Houdayer moves over the
    /// device-order adjacency `(all_deg, all_ell)`.
    fn checkpoint(
        &mut self,
        host: &mut Vec<i8>,
        energies: &[i64],
        n: usize,
        beta: f32,
        houdayer: Option<(&[i32], &[i32])>,
    ) {
        let slots = self.slots();
        for (slot, &e) in energies.iter().enumerate().take(slots) {
            if self.best[slot].as_ref().is_none_or(|b| e < b.1) {
                self.best[slot] = Some((host[slot * n..(slot + 1) * n].to_vec(), e));
            }
        }
        let step = f64::from(beta - self.last_beta.unwrap_or(beta));
        self.last_beta = Some(beta);
        if step > 0.0 {
            let u = self.uniform();
            let picks = resample(energies, step, u);
            let mut next = Vec::with_capacity(host.len());
            for &i in &picks {
                next.extend_from_slice(&host[i * n..(i + 1) * n]);
            }
            *host = next;
            self.best = picks.iter().map(|&i| self.best[i].clone()).collect();
        }
        if let Some((deg, ell)) = houdayer {
            for k in (1..slots).rev() {
                let j = self.below(k + 1);
                self.order.swap(k, j);
            }
            for p in 0..slots / 2 {
                let (a, b) = (self.order[2 * p], self.order[2 * p + 1]);
                self.houdayer(host, a, b, n, deg, ell);
            }
        }
    }

    /// Grow one connected cluster of nodes where replicas `a` and `b` differ
    /// and swap it between them. The summed energy is unchanged.
    #[allow(clippy::many_single_char_names)]
    fn houdayer(
        &mut self,
        host: &mut [i8],
        a: usize,
        b: usize,
        n: usize,
        deg: &[i32],
        ell: &[i32],
    ) {
        let (oa, ob) = (a * n, b * n);
        self.differ.clear();
        self.differ.extend(
            (0..n)
                .filter(|&u| host[oa + u] != host[ob + u])
                .map(|u| u32::try_from(u).unwrap_or(u32::MAX)),
        );
        if self.differ.is_empty() {
            return;
        }
        self.seen.resize(n, false);
        let pick = self.below(self.differ.len());
        let root = self.differ[pick];
        self.cluster.clear();
        self.cluster.push(root);
        self.seen[root as usize] = true;
        let mut at = 0;
        while at < self.cluster.len() {
            let u = self.cluster[at] as usize;
            at += 1;
            for e in 0..usize::try_from(deg[u]).unwrap_or(0) {
                #[allow(clippy::cast_sign_loss)]
                let v = ((ell[e * n + u] as u32) >> 16) as usize;
                if !self.seen[v] && host[oa + v] != host[ob + v] {
                    self.seen[v] = true;
                    self.cluster.push(u32::try_from(v).unwrap_or(u32::MAX));
                }
            }
        }
        for &u in &self.cluster {
            let u = u as usize;
            self.seen[u] = false;
            host[oa + u] = -host[oa + u];
            host[ob + u] = -host[ob + u];
        }
    }
}

/// Systematic resampling, as `quip-solver-cpu`: the slot each copy takes, in
/// proportion to `exp(-step * (E - E_min) / 1000)`, with one uniform offset `u`.
#[allow(clippy::cast_precision_loss)]
fn resample(energy: &[i64], step: f64, u: f64) -> Vec<usize> {
    let low = energy.iter().copied().min().unwrap_or(0);
    let weights: Vec<f64> = energy
        .iter()
        .map(|&e| (-step * (e - low) as f64 / 1000.0).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let stride = total / energy.len() as f64;
    let mut mark = u * stride;
    let mut picks = Vec::with_capacity(energy.len());
    let mut reach = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        reach += w;
        while mark < reach && picks.len() < energy.len() {
            picks.push(i);
            mark += stride;
        }
    }
    picks.resize(energy.len(), energy.len().saturating_sub(1));
    picks
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

    /// Local build: a real x-w62-p1200 dimer instance on the live Aglais
    /// spec-120 topology (4575 nodes, 41474 edges), generated on the host.
    fn live_dimer_instance(
        nonce: [u8; 32],
    ) -> quip_solver_core::ising_generators::instance::Instance {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::{generate_instance, BaseTopology};
        use quip_solver_core::ising_generators::ice::{RECIPES, TWISTED_DIMER_V1};
        let blob = include_bytes!("../../tests/fixtures/aglais-z12t4-spec120.bin.gz");
        let topo = ising_generators::blob::decode(blob)
            .unwrap()
            .topology()
            .unwrap();
        let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
        let recipe = RECIPES.iter().find(|r| r.alias == "x-w62-p1200").unwrap();
        let digest = params_digest(recipe.params);
        let contracts = [TWISTED_DIMER_V1];
        let admitted = admit(
            &contracts,
            TWISTED_DIMER_V1.family_id.as_bytes(),
            TWISTED_DIMER_V1.revision,
            recipe.params,
            &digest,
        )
        .unwrap();
        let bounds = admitted.size_bounds(base.stats()).unwrap();
        generate_instance(&admitted, &base, &bounds, nonce).unwrap()
    }

    fn live_limits() -> crate::capacity::DeviceLimits {
        crate::capacity::DeviceLimits {
            shared_bytes_per_block: 48 << 10,
            shared_bytes_per_block_optin: 99 << 10,
            free_bytes: 24 << 30,
            sm_count: 170,
            threads_per_sm: 1536,
        }
    }

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// A packed ELL word: neighbour (high 16 bits) and coupling (low i16).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn word(w: i32) -> (usize, i32) {
        (((w as u32) >> 16) as usize, i32::from(w as i16))
    }

    /// The rows of an int4-packed ELL exactly as the V3 kernel walks them
    /// (`field4`/`field4s`): chunk count K from int4 0, `ceil(deg / 4)`
    /// chunks of node u at int4 `1 + k * n + u`, zero words past the degree.
    fn v3_row(packed: &[i32], deg: &[i32], n: usize, u: usize) -> Vec<i32> {
        let k_all = usize::try_from(packed[0]).unwrap();
        let kd = (usize::try_from(deg[u]).unwrap() + 3) / 4;
        assert!(
            kd <= k_all,
            "row {u}: {kd} chunks over the header's {k_all}"
        );
        (0..kd)
            .flat_map(|k| {
                let at = 4 * (1 + k * n + u);
                packed[at..at + 4].to_vec()
            })
            .collect()
    }

    /// Local build (QUIP_OTSUKA_TILE_MILLI): on real dimer1200 instances, for
    /// every tile and pair weight the knobs admit, the prepared layers rebuild
    /// the instance Hamiltonian exactly for random states: the weighted tile
    /// and pair layer plus the weak ELL (plain and V3 int4-packed) equals the
    /// miner's own scoring of the instance, and so do the full ELL and the
    /// rescoring the device run reports. The stock weights keep the weak
    /// layer at +-62 only; softened tiles move +-(1000 - tile) into it.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn soft_tile_layers_rebuild_the_instance_hamiltonian() {
        use crate::cluster::Weights;
        use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
        use quip_solver_core::quip_protocol::scoring::energy_from_milli;
        let mut x = 0x7113_5EED_u64;
        for inst in 0..3 {
            let mut nonce = [0_u8; 32];
            for c in nonce.chunks_exact_mut(8) {
                c.copy_from_slice(&xorshift(&mut x).to_le_bytes());
            }
            let instance = live_dimer_instance(nonce);
            let n = instance.h().len();
            assert_eq!((n, instance.edges().len()), (4575, 41474));
            let edges: Vec<(usize, usize)> = instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect();
            for (tile_milli, pair_milli) in [
                (1000, 1000),
                (1000, 250),
                (500, 250),
                (250, 250),
                (250, 1000),
                (125, 250),
                (1, 0),
            ] {
                let weights = Weights {
                    pair_milli,
                    tile_milli,
                };
                let p = PreparedGraph::from_instance(
                    &instance,
                    &TWISTED_DIMER_V1,
                    weights,
                    &live_limits(),
                    64,
                )
                .unwrap();
                let d = p.layout();
                assert_eq!(d.header[0] as usize, n);
                assert!(d.tile_w.iter().all(|&w| w == tile_milli));
                assert!(d.link_w.iter().all(|&w| w == pair_milli));
                let weak4 = pack_ell4(&d.weak_ell, n);
                let all4 = pack_ell4(&d.all_ell, n);
                // Weak couplings by value: +-62 only at the stock weights.
                let mut weak_j = std::collections::BTreeSet::new();
                for &w in &d.weak_ell {
                    if w != 0 {
                        weak_j.insert(word(w).1);
                    }
                }
                let mut want = std::collections::BTreeSet::from([-62, 62]);
                if tile_milli < 1000 {
                    want.extend([-(1000 - tile_milli), 1000 - tile_milli]);
                }
                if pair_milli < 1000 {
                    want.insert(pair_milli - 1000);
                }
                assert_eq!(weak_j, want, "tile {tile_milli} pair {pair_milli}");
                // Rows: the weak row of a node is a sub-multiset of its full row.
                let (ww, aw) = (d.header[3], d.header[4]);
                assert!(ww <= aw, "weak width {ww} over full width {aw}");
                if inst == 0 {
                    eprintln!(
                        "tile {tile_milli} pair {pair_milli}: weak width {ww} ({} int4 chunks), \
                         full width {aw} ({} chunks), V3 register rows hold 6 chunks",
                        weak4[0], all4[0]
                    );
                }
                for _ in 0..12 {
                    let s: Vec<i8> = (0..n)
                        .map(|_| {
                            if xorshift(&mut x) >> 33 & 1 == 1 {
                                1
                            } else {
                                -1
                            }
                        })
                        .collect();
                    let direct = energy_from_milli(&s, instance.h(), instance.j(), &edges);
                    assert_eq!(p.energy(&s).unwrap(), direct, "OtsukaGraph rescoring");
                    let dev = p.permute(&s).unwrap();
                    // Strong layer on device arrays, at the layout's weights.
                    let mut strong = 0_i64;
                    for (t, (nodes, g)) in d
                        .tiles
                        .chunks_exact(4)
                        .zip(d.gauges.chunks_exact(4))
                        .enumerate()
                    {
                        for a in 0..4 {
                            for b in a + 1..4 {
                                strong += i64::from(d.tile_w[t])
                                    * i64::from(g[a] * g[b])
                                    * i64::from(dev[nodes[a] as usize] * dev[nodes[b] as usize]);
                            }
                        }
                    }
                    for (k, pr) in d.pairs.chunks_exact(2).enumerate() {
                        strong -= i64::from(d.link_w[k])
                            * i64::from(dev[pr[0] as usize] * dev[pr[1] as usize]);
                    }
                    let ell_twice = |deg: &[i32], row: &dyn Fn(usize) -> Vec<i32>| -> i64 {
                        let mut t = 0_i64;
                        for u in 0..n {
                            let r = row(u);
                            for (e, &w) in r.iter().enumerate() {
                                if e >= usize::try_from(deg[u]).unwrap() {
                                    assert_eq!(w, 0, "padding past the degree");
                                    continue;
                                }
                                let (v, j) = word(w);
                                t += i64::from(j) * i64::from(dev[u] * dev[v]);
                            }
                        }
                        assert_eq!(t % 2, 0);
                        t / 2
                    };
                    let plain = |ell: &[i32], deg: &[i32]| {
                        ell_twice(deg, &|u| {
                            (0..usize::try_from(deg[u]).unwrap())
                                .map(|e| ell[e * n + u])
                                .collect()
                        })
                    };
                    let weak = plain(&d.weak_ell, &d.weak_deg);
                    let weak_v3 = ell_twice(&d.weak_deg, &|u| v3_row(&weak4, &d.weak_deg, n, u));
                    let all = plain(&d.all_ell, &d.all_deg);
                    let all_v3 = ell_twice(&d.all_deg, &|u| v3_row(&all4, &d.all_deg, n, u));
                    assert_eq!(weak, weak_v3, "V3 weak packing");
                    assert_eq!(all, direct, "full ELL");
                    assert_eq!(all_v3, direct, "V3 full packing");
                    assert_eq!(strong + weak, direct, "tile {tile_milli} pair {pair_milli}");
                    let (t, l, r) = p.split_energies(&s).unwrap();
                    assert_eq!((t + l, r), (strong, weak));
                }
            }
        }
    }

    /// Local build: the V3 cluster decision on a softened graph. Components
    /// come from random tile and pair bonds (unions link the larger root under
    /// the smaller, as the kernel's); the sign-tagged accumulate over the
    /// int4-packed weak rows (`field4s`/`wterm`, then `acc[root] += s[u] * f`)
    /// gives each component exactly minus half the change of the weak-layer
    /// energy when it flips, so `delta = -2 * acc` is the true weak cost, the
    /// same as the plain V2 accumulate.
    #[test]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn v3_component_fields_price_soft_tile_residuals_exactly() {
        use crate::cluster::Weights;
        use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
        let instance = live_dimer_instance([7; 32]);
        let n = instance.h().len();
        let mut x = 0xACC_u64;
        for (tile_milli, pair_milli) in [(1000, 1000), (1000, 250), (500, 250), (250, 250)] {
            let p = PreparedGraph::from_instance(
                &instance,
                &TWISTED_DIMER_V1,
                Weights {
                    pair_milli,
                    tile_milli,
                },
                &live_limits(),
                64,
            )
            .unwrap();
            let d = p.layout();
            let weak4 = pack_ell4(&d.weak_ell, n);
            for _ in 0..3 {
                let s: Vec<i8> = (0..n)
                    .map(|_| if xorshift(&mut x) & 1 == 1 { 1 } else { -1 })
                    .collect();
                // Random strong bonds -> components, roots = smallest node.
                let mut parent: Vec<usize> = (0..n).collect();
                fn find(p: &mut [usize], mut a: usize) -> usize {
                    while p[a] != a {
                        p[a] = p[p[a]];
                        a = p[a];
                    }
                    a
                }
                let mut unite = |a: usize, b: usize| {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb)] = ra.min(rb);
                    }
                };
                for t in d.tiles.chunks_exact(4) {
                    for a in 0..4 {
                        for b in a + 1..4 {
                            if xorshift(&mut x) % 3 == 0 {
                                unite(t[a] as usize, t[b] as usize);
                            }
                        }
                    }
                }
                for pr in d.pairs.chunks_exact(2) {
                    if xorshift(&mut x) % 2 == 0 {
                        unite(pr[0] as usize, pr[1] as usize);
                    }
                }
                let root: Vec<usize> = (0..n).map(|u| find(&mut parent, u)).collect();
                // V3: parent[i] = root | sign bit of s[i].
                let tagged: Vec<i32> = (0..n)
                    .map(|i| root[i] as i32 | if s[i] < 0 { i32::MIN } else { 0 })
                    .collect();
                let mut acc = vec![0_i64; n];
                let mut acc_v2 = vec![0_i64; n];
                for u in 0..n {
                    let r = root[u];
                    let mut f = 0_i32;
                    for w in v3_row(&weak4, &d.weak_deg, n, u) {
                        let (v, j) = word(w);
                        let pv = tagged[v];
                        f += if (pv & 0x7fff_ffff) as usize != r {
                            if pv < 0 {
                                -j
                            } else {
                                j
                            }
                        } else {
                            0
                        };
                    }
                    acc[r] += i64::from(s[u]) * i64::from(f);
                    let mut f2 = 0_i32;
                    for e in 0..d.weak_deg[u] as usize {
                        let (v, j) = word(d.weak_ell[e * n + u]);
                        if root[v] != r {
                            f2 += j * i32::from(s[v]);
                        }
                    }
                    acc_v2[r] += i64::from(s[u]) * i64::from(f2);
                }
                assert_eq!(acc, acc_v2, "V3 and V2 accumulates");
                let orig = p.unpermute(&s).unwrap();
                let weak0 = p.weak_energy(&orig).unwrap();
                let roots: Vec<usize> = (0..n).filter(|&u| root[u] == u).collect();
                assert!(roots.len() > 100);
                let mut checked = 0;
                for &r in roots.iter().step_by(roots.len() / 40) {
                    let flipped: Vec<i8> = (0..n)
                        .map(|u| if root[u] == r { -s[u] } else { s[u] })
                        .collect();
                    let weak1 = p.weak_energy(&p.unpermute(&flipped).unwrap()).unwrap();
                    assert_eq!(-2 * acc[r], weak1 - weak0, "tile {tile_milli} root {r}");
                    checked += 1;
                }
                assert!(checked >= 40);
            }
        }
    }

    /// Local build (QUIP_OTSUKA_HOUDAYER_EVERY): launches cut at in-segment
    /// multiples of the period once due; off (0) never changes a launch.
    #[test]
    fn intra_houdayer_cuts_launches_only_when_due() {
        // Off: any count is kept.
        let mut c = 256;
        assert!(!intra_move(0, 0, &mut c, 256, |_| true));
        assert_eq!(c, 256);
        // Every 8 from sweep 0 in a 256-sweep segment: cut at 8.
        let mut c = 256;
        assert!(intra_move(8, 0, &mut c, 256, |_| true));
        assert_eq!(c, 8);
        // From 8: next boundary 16.
        let mut c = 248;
        assert!(intra_move(8, 8, &mut c, 256, |_| true));
        assert_eq!(c, 8);
        // The segment end is the checkpoint's move: no cut at 256.
        let mut c = 8;
        assert!(!intra_move(8, 248, &mut c, 256, |_| true));
        assert_eq!(c, 8);
        // Not yet due (beta below the Houdayer beta): no cut.
        let mut c = 256;
        assert!(!intra_move(8, 0, &mut c, 256, |_| false));
        assert_eq!(c, 256);
        // Due from boundary 104 on: the first due boundary is the cut.
        let mut c = 256;
        assert!(intra_move(8, 0, &mut c, 256, |b| b >= 100));
        assert_eq!(c, 104);
        // A chunk boundary that is also a period boundary still moves.
        let mut c = 64;
        assert!(intra_move(64, 0, &mut c, 256, |_| true));
        assert_eq!(c, 64);
        // A launch shorter than the period with no boundary inside: kept.
        let mut c = 5;
        assert!(!intra_move(8, 1, &mut c, 256, |_| true));
        assert_eq!(c, 5);
        // Every sweep.
        let mut c = 256;
        assert!(intra_move(1, 10, &mut c, 256, |_| true));
        assert_eq!(c, 1);
    }

    /// The beta the in-segment rule tests is the kernel's own geometric ladder.
    #[test]
    fn host_beta_matches_the_geometric_ends() {
        let q = ClusterSchedule {
            sweeps: 131_072,
            beta_min: 1.0,
            beta_max: 12.0,
            ..Default::default()
        }
        .params()
        .unwrap();
        assert!((host_beta(&q, 0) - 1.0).abs() < 1e-6);
        assert!((host_beta(&q, 131_071) - 12.0).abs() < 1e-4);
        // beta 3 at f = ln 3 / ln 12, about 44% of the anneal.
        let f = (3.0_f32).ln() / (12.0_f32).ln();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let at = (f * 131_071.0) as usize;
        assert!(host_beta(&q, at - 2) < 3.0 && host_beta(&q, at + 2) >= 3.0);
    }

    /// Local build (all11): a line-for-line CPU port of the kernel's
    /// `beta_at` (`kernels/cluster.cu`), f32 throughout.
    #[allow(clippy::cast_precision_loss)]
    fn kernel_beta_at(q: &Params, sweep: i32) -> f32 {
        let f = if q.sweeps > 1 {
            sweep as f32 / (q.sweeps - 1) as f32
        } else {
            1.0
        };
        if q.beta_knee > 0.0 && q.knee_frac > 0.0 && q.knee_frac < 1.0 {
            if f < q.knee_frac {
                return q.beta_min * (q.beta_knee / q.beta_min).powf(f / q.knee_frac);
            }
            return q.beta_knee
                * (q.beta_max / q.beta_knee).powf((f - q.knee_frac) / (1.0 - q.knee_frac));
        }
        q.beta_min * (q.beta_max / q.beta_min).powf(f)
    }

    /// all10's `host_beta`, verbatim: the knee-off reference.
    #[allow(clippy::cast_precision_loss)]
    fn all10_host_beta(q: &Params, sweep: usize) -> f32 {
        let f = if q.sweeps > 1 {
            sweep as f32 / (q.sweeps - 1) as f32
        } else {
            1.0
        };
        q.beta_min * (q.beta_max / q.beta_min).powf(f)
    }

    fn knee_params(sweeps: usize, knee: f32, frac: f32) -> Params {
        ClusterSchedule {
            sweeps,
            beta_min: 1.0,
            beta_knee: knee,
            knee_frac: frac,
            beta_max: 12.0,
            ..Default::default()
        }
        .params()
        .unwrap()
    }

    fn params_bytes(p: &Params) -> [u8; std::mem::size_of::<Params>()] {
        // SAFETY: Params is repr(C), ten 32-bit scalars, no padding.
        unsafe { std::mem::transmute_copy(p) }
    }

    /// all11: the kernel source still holds the knee branch the port copies.
    #[test]
    fn kernel_beta_at_source_has_the_knee_branch() {
        for line in [
            "__device__ float beta_at(const Params& q, int sweep) {",
            "float f = q.sweeps > 1 ? (float)sweep / (float)(q.sweeps - 1) : 1.0f;",
            "if (q.beta_knee > 0.0f && q.knee_frac > 0.0f && q.knee_frac < 1.0f) {",
            "if (f < q.knee_frac) return q.beta_min * powf(q.beta_knee / q.beta_min, f / q.knee_frac);",
            "return q.beta_knee * powf(q.beta_max / q.beta_knee, (f - q.knee_frac) / (1.0f - q.knee_frac));",
            "return q.beta_min * powf(q.beta_max / q.beta_min, f);",
        ] {
            assert!(SOURCE.contains(line), "kernel beta_at changed: {line}");
        }
    }

    /// all11: with the knee, `host_beta` is the kernel's `beta_at` at the
    /// ends, at the knee, and at every sweep of a run; it rises monotonically
    /// and is continuous at the knee.
    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap
    )]
    fn host_beta_knee_matches_kernel_beta_at() {
        for (sweeps, knee, frac) in [
            (335_360_usize, 4.0_f32, 0.873_f32),
            (335_360, 4.0, 0.8),
            (524_288, 4.0, 0.873),
            (20_960, 4.0, 0.873),    // stage 1: 0.0625 of 1310 x 256
            (2_012_160, 4.0, 0.873), // deep lane: MULT_MAX 6
            (8_192, 3.0, 0.75),
            (1_000, 11.5, 0.1),
            (2, 4.0, 0.5),
        ] {
            let q = knee_params(sweeps, knee, frac);
            let last = sweeps - 1;
            let at_knee = (frac * last as f32) as usize;
            for s in [
                0,
                at_knee.saturating_sub(1),
                at_knee,
                (at_knee + 1).min(last),
                last,
            ] {
                assert_eq!(
                    host_beta(&q, s).to_bits(),
                    kernel_beta_at(&q, s as i32).to_bits(),
                    "sweeps {sweeps} sweep {s}"
                );
            }
            assert!((host_beta(&q, 0) - 1.0).abs() < 1e-6);
            assert!(
                (host_beta(&q, last) - 12.0).abs() < 1e-4,
                "{}",
                host_beta(&q, last)
            );
            // Every sweep: identical to the kernel port, never decreasing.
            let mut prev = 0.0_f32;
            let step = (sweeps / 200_000).max(1);
            for s in (0..sweeps).step_by(step).chain(std::iter::once(last)) {
                let b = host_beta(&q, s);
                assert_eq!(b.to_bits(), kernel_beta_at(&q, s as i32).to_bits());
                assert!(
                    b >= prev,
                    "sweeps {sweeps}: beta fell at {s}: {prev} -> {b}"
                );
                prev = b;
            }
            // The knee: below it before, at or above it after, and continuous.
            if sweeps >= 8_192 {
                let lo = host_beta(&q, at_knee.saturating_sub(1));
                let hi = host_beta(&q, at_knee + 2);
                assert!(lo < knee && hi >= knee * 0.9999, "{lo} {hi}");
                assert!(hi - lo < 1e-2 * knee, "jump at the knee: {lo} -> {hi}");
            }
        }
        // Live Kcut: about frac of the sweeps sit below the knee.
        let q = knee_params(335_360, 4.0, 0.873);
        let below = (0..335_360).filter(|&s| host_beta(&q, s) < 4.0).count();
        assert!((292_700..=292_800).contains(&below), "{below}");
    }

    /// all11: knee off (default tuning) keeps Params and the host schedule
    /// identical to all10: zero knee fields, same bytes, same betas.
    #[test]
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    fn knee_off_is_the_all10_schedule() {
        let off = OtsukaTuning::default();
        let (k, f) = off.knee_for(1.0, 12.0);
        assert_eq!((k.to_bits(), f.to_bits()), (0, 0), "+0.0 exactly");
        let env = OtsukaTuning::from_lookup(|_| None);
        assert_eq!(env.knee_for(1.0, 12.0), (0.0, 0.0));
        for sweeps in [1_usize, 2, 1_000, 335_360, 524_288] {
            let all10 = ClusterSchedule {
                sweeps,
                beta_min: 1.0,
                beta_knee: 0.0,
                knee_frac: 0.0,
                beta_max: 12.0,
                ..Default::default()
            }
            .params()
            .unwrap();
            let q = knee_params(sweeps, k, f);
            assert_eq!(params_bytes(&q), params_bytes(&all10));
            let step = (sweeps / 100_000).max(1);
            for s in (0..sweeps).step_by(step).chain(std::iter::once(sweeps - 1)) {
                assert_eq!(
                    host_beta(&q, s).to_bits(),
                    all10_host_beta(&all10, s).to_bits()
                );
                assert_eq!(
                    host_beta(&q, s).to_bits(),
                    kernel_beta_at(&q, s as i32).to_bits()
                );
            }
        }
        // A knee outside a job's ladder falls back to the all10 schedule too.
        let on = OtsukaTuning {
            beta_knee: 4.0,
            knee_frac: 0.873,
            ..OtsukaTuning::default()
        };
        assert_eq!(on.knee_for(1.0, 12.0), (4.0, 0.873));
        assert_eq!(on.knee_for(4.0, 12.0), (0.0, 0.0));
        assert_eq!(on.knee_for(1.0, 4.0), (0.0, 0.0));
        assert_eq!(on.knee_for(5.0, 5.0), (0.0, 0.0), "constant-beta launches");
    }

    /// Local build (V6): the kernel's `metropolis_v6` row sum (`sum6`) on a
    /// V6 all-ELL: entries index the spin window (s, -s at V6_NEG_OFF, zero).
    fn v6_metro_field(p: &[i32], n: usize, u: usize, window: &[i8]) -> i32 {
        let k_all = usize::try_from(p[0]).unwrap();
        let rows = 4 * (1 + k_all * n);
        let kd = usize::try_from(p[rows + u]).unwrap();
        assert!((1..=k_all).contains(&kd), "row {u}: {kd} chunks of {k_all}");
        let (mut strong, mut weak) = (0_i32, 0_i32);
        for k in 0..kd {
            let mut sum = 0_i32;
            for q in 0..4 {
                let w = p[4 * (1 + k * n + u) + q].cast_unsigned();
                for idx in [w & 0xffff, w >> 16] {
                    sum += i32::from(window[idx as usize]);
                }
            }
            if k == 0 {
                strong += sum
            } else {
                weak += sum
            }
        }
        p[2] * strong + p[3] * weak
    }

    /// Local build (V6): the kernel's `field6s_sh` / `aword6` on a V6 weak ELL
    /// with V6-tagged parents (root | sign << 31 | 1 << 30), row root `r`.
    fn v6_acc_field(p: &[i32], n: usize, u: usize, tagged: &[i32], r: i32) -> i32 {
        let k_all = usize::try_from(p[0]).unwrap();
        let kd = usize::try_from(p[4 * (1 + k_all * n) + u]).unwrap();
        let (r1, r2) = (r | 0x4000_0000, r | 0x4000_0000 | i32::MIN);
        let (mut strong, mut weak) = (0_i32, 0_i32);
        for k in 0..kd {
            let mut sum = 0_i32;
            for q in 0..4 {
                let x = p[4 * (1 + k * n + u) + q].cast_unsigned();
                assert_eq!(x & 3, 0, "parent byte offsets are word aligned");
                let pl = tagged[(x & 0xffff) as usize / 4];
                let ph = tagged[((x >> 14) & 0xfffc) as usize / 4];
                let tl = pl ^ ((x << 1) & 0x8000_0000).cast_signed();
                let th = ph ^ (x & 0x8000_0000).cast_signed();
                if pl != r1 && pl != r2 {
                    sum += tl >> 30;
                }
                if ph != r1 && ph != r2 {
                    sum += th >> 30;
                }
            }
            if k == 0 {
                strong += sum
            } else {
                weak += sum
            }
        }
        p[2] * strong + p[3] * weak
    }

    /// The spin window a V6 block holds: s, garbage where the CU-1 flag word
    /// and the queue head live (never read), -s, then the zero byte.
    fn v6_window(s: &[i8]) -> Vec<i8> {
        let n = s.len();
        let off = v6_neg_off(n);
        let mut w = vec![0x55_i8; off + n + 1];
        w[..n].copy_from_slice(s);
        for (v, &x) in s.iter().enumerate() {
            w[off + v] = -x;
        }
        w[off + n] = 0;
        w
    }

    /// Plain-ELL references: Metropolis field sum J s[v]; V3 accumulate
    /// field sum over neighbours of another root.
    fn plain_fields(
        ell: &[i32],
        deg: &[i32],
        n: usize,
        u: usize,
        s: &[i8],
        root: &[usize],
    ) -> (i32, i32) {
        let (mut m, mut a) = (0, 0);
        for e in 0..usize::try_from(deg[u]).unwrap() {
            let (v, j) = word(ell[e * n + u]);
            m += j * i32::from(s[v]);
            if root[v] != root[u] {
                a += j * i32::from(s[v]);
            }
        }
        (m, a)
    }

    fn v6_tagged(s: &[i8], root: &[usize]) -> Vec<i32> {
        (0..s.len())
            .map(|i| {
                i32::try_from(root[i]).unwrap() | 0x4000_0000 | if s[i] < 0 { i32::MIN } else { 0 }
            })
            .collect()
    }

    /// Local build (V6): on synthetic ELLs with two magnitudes (both signs,
    /// up to eight strong entries a row, empty rows, either class larger), the
    /// V6 Metropolis and accumulate gathers as the kernel decodes them equal
    /// the plain-ELL sums for random states and components; graphs outside the
    /// format keep V5's layout.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn pack_ell6_fields_match_plain_ell() {
        let mut x = 0x5EED_0006_u64;
        for (n, js_mag, jw_mag) in [
            (37_usize, 1000, 62),
            (53, 750, 93),
            (64, 1000, 68),
            (29, 45, 1000),
            (41, 45, 0),
        ] {
            let width = 20_usize;
            let mut ell = vec![0_i32; width * n];
            let mut deg = vec![0_i32; n];
            for u in 0..n {
                let d = if u % 11 == 3 {
                    0
                } else {
                    usize::try_from(xorshift(&mut x) % 21).unwrap()
                };
                // At most eight entries of the first magnitude per row (or any
                // number when it is the smaller one and the other fits).
                let mut strong = 0;
                for e in 0..d {
                    let first = jw_mag != 0 && strong < 8 && xorshift(&mut x) % 3 == 0;
                    strong += usize::from(first);
                    let mag = if first || jw_mag == 0 { js_mag } else { jw_mag };
                    let j = if xorshift(&mut x) & 1 == 1 { mag } else { -mag };
                    let v = u32::try_from(xorshift(&mut x) % n as u64).unwrap();
                    ell[e * n + u] =
                        (v << 16 | u32::from((j as i16).cast_unsigned())).cast_signed();
                }
                deg[u] = i32::try_from(d).unwrap();
            }
            let all = pack_ell6_all(&ell, &deg, n).expect("two magnitudes, <= 8 strong per row");
            let weak = pack_ell6_weak(&ell, &deg, n).expect("same rows");
            assert_eq!((all[1], weak[1]), (2, 2), "V6 header format");
            assert_eq!(
                (all[2], all[3]),
                (weak[2], weak[3]),
                "same classes in both encodings"
            );
            assert_eq!(
                all.len(),
                4 * (1 + usize::try_from(all[0]).unwrap() * n) + n
            );
            if jw_mag == 0 {
                assert_eq!(
                    (all[2], all[3]),
                    (0, js_mag),
                    "a single magnitude is all weak"
                );
            }
            for _ in 0..4 {
                let s: Vec<i8> = (0..n)
                    .map(|_| if xorshift(&mut x) & 1 == 1 { 1 } else { -1 })
                    .collect();
                let root: Vec<usize> = (0..n)
                    .map(|u| {
                        let r = usize::try_from(xorshift(&mut x) % 6).unwrap();
                        r.min(u)
                    })
                    .collect();
                let window = v6_window(&s);
                let tagged = v6_tagged(&s, &root);
                for u in 0..n {
                    let (m, a) = plain_fields(&ell, &deg, n, u, &s, &root);
                    assert_eq!(
                        v6_metro_field(&all, n, u, &window),
                        m,
                        "n {n} Metropolis field of node {u}"
                    );
                    let r = i32::try_from(root[u]).unwrap();
                    assert_eq!(
                        v6_acc_field(&weak, n, u, &tagged, r),
                        a,
                        "n {n} accumulate field of node {u}"
                    );
                }
            }
        }
        // Outside the format: V5's layout (header .y = 1), or V4's.
        let n = 16_usize;
        let mut ell = vec![0_i32; 10 * n];
        let mut deg = vec![0_i32; n];
        let put = |ell: &mut Vec<i32>, e: usize, u: usize, v: u32, j: i16| {
            ell[e * n + u] = (v << 16 | u32::from(j.cast_unsigned())).cast_signed();
        };
        put(&mut ell, 0, 0, 1, 1000);
        put(&mut ell, 1, 0, 2, 62);
        put(&mut ell, 2, 0, 3, 750);
        deg[0] = 3;
        assert!(pack_ell6_all(&ell, &deg, n).is_none(), "three magnitudes");
        assert_eq!(pack_ell_v6(&ell, &deg, n, false)[1], 1);
        assert_eq!(pack_ell_v6(&ell, &deg, n, true)[1], 1);
        // Nine of each magnitude in one row: neither class fits chunk 0.
        let mut ell = vec![0_i32; 18 * n];
        for e in 0..18 {
            put(
                &mut ell,
                e,
                1,
                u32::try_from(e % n).unwrap(),
                if e < 9 { 1000 } else { -62 },
            );
        }
        let mut deg = vec![0_i32; n];
        deg[1] = 18;
        assert!(
            pack_ell6_weak(&ell, &deg, n).is_none(),
            "nine strong entries either way"
        );
        assert_eq!(pack_ell_v6(&ell, &deg, n, true)[1], 1);
        // Node indices need 14 bits.
        assert!(ell6_rows(&[], &vec![0; 1 << 14], 1 << 14).is_none());
    }

    /// Local build (V6): on real dimer1200 instances at the live draw weights
    /// (tile 250, pair 250) and weak scales 1, 1.5 (live) and 2, both anneal
    /// ELLs qualify for V6, and the V6 gathers equal the V3 ones (plain-ELL
    /// sums) for random states and components from random strong bonds.
    #[test]
    fn v6_gathers_match_v3_on_live_instances() {
        use crate::cluster::Weights;
        use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
        let mut x = 0x0B5E_55ED_u64;
        for (inst, ws) in [(0_u8, 1.0), (1, 1.5), (2, 1.5), (3, 2.0)] {
            let instance = live_dimer_instance([inst * 17 + 5; 32]);
            let n = instance.h().len();
            let p = PreparedGraph::from_instance_scaled(
                &instance,
                &TWISTED_DIMER_V1,
                Weights {
                    pair_milli: 250,
                    tile_milli: 250,
                },
                &live_limits(),
                64,
                ws,
            )
            .unwrap();
            let d = p.layout();
            let (wl, al) = p
                .anneal_ell()
                .map_or((&d.weak_ell, &d.all_ell), |a| (&a.weak_ell, &a.all_ell));
            let all = pack_ell_v6(al, &d.all_deg, n, false);
            let weak = pack_ell_v6(wl, &d.weak_deg, n, true);
            assert_eq!(
                (all[1], weak[1]),
                (2, 2),
                "ws {ws}: live graphs take the V6 layout"
            );
            assert_eq!(
                all[2], 1000,
                "Metropolis strong class: the 1000-milli tile/pair couplings"
            );
            assert_eq!(
                weak[2], 750,
                "accumulate strong class: the tile/pair residuals"
            );
            let k = usize::try_from(all[0]).unwrap();
            assert!(k <= 4, "ws {ws}: {k} chunks");
            for _ in 0..2 {
                let s: Vec<i8> = (0..n)
                    .map(|_| if xorshift(&mut x) & 1 == 1 { 1 } else { -1 })
                    .collect();
                let mut parent: Vec<usize> = (0..n).collect();
                fn find(p: &mut [usize], mut a: usize) -> usize {
                    while p[a] != a {
                        p[a] = p[p[a]];
                        a = p[a];
                    }
                    a
                }
                for t in d.tiles.chunks_exact(4) {
                    if xorshift(&mut x) % 2 == 0 {
                        let (a, b) = (
                            find(&mut parent, t[0] as usize),
                            find(&mut parent, t[2] as usize),
                        );
                        parent[a.max(b)] = a.min(b);
                    }
                }
                for pr in d.pairs.chunks_exact(2) {
                    if xorshift(&mut x) % 3 != 0 {
                        let (a, b) = (
                            find(&mut parent, pr[0] as usize),
                            find(&mut parent, pr[1] as usize),
                        );
                        parent[a.max(b)] = a.min(b);
                    }
                }
                let root: Vec<usize> = (0..n).map(|u| find(&mut parent, u)).collect();
                let window = v6_window(&s);
                let tagged = v6_tagged(&s, &root);
                for u in 0..n {
                    let (m, _) = plain_fields(al, &d.all_deg, n, u, &s, &root);
                    let (_, a) = plain_fields(wl, &d.weak_deg, n, u, &s, &root);
                    assert_eq!(
                        v6_metro_field(&all, n, u, &window),
                        m,
                        "ws {ws} Metropolis node {u}"
                    );
                    let r = i32::try_from(root[u]).unwrap();
                    assert_eq!(
                        v6_acc_field(&weak, n, u, &tagged, r),
                        a,
                        "ws {ws} accumulate node {u}"
                    );
                }
            }
        }
    }

    /// Local build (V6): the kernel's shared layout constants and the host's.
    #[test]
    fn v6_shared_layout_matches_the_kernel() {
        assert!(SOURCE.contains("#define V6_NEG_OFF(n) ((((n) + 3) & ~3) + 8)"));
        assert!(SOURCE.contains("#define V3TAG 0x40000000"));
        assert!(SOURCE.contains("#define V3M 0x3fffffff"));
        for n in [1_usize, 2, 3, 4, 5, 4560, 4575, 16_383] {
            assert_eq!(v6_neg_off(n), ((n + 3) & !3) + 8);
            let base = 9 * n + crate::capacity::CLUSTER_DYNAMIC_PADDING;
            // The zero byte sits at s + off + n with s at 8 n.
            assert!(
                8 * n + v6_neg_off(n) + n < base + v6_shared_extra(n),
                "n {n}"
            );
            // The V4 queue head (the int after the flag word) ends before the copy.
            assert!(((n + 3) & !3) + 8 <= v6_neg_off(n));
        }
        // The live topology: 4560 nodes, 41056 + 4556 bytes per block; two
        // blocks (plus 1 KB driver reserve each) still fit an SM's 100 KB.
        assert_eq!(9 * 4560 + 16 + v6_shared_extra(4560), 45_612);
        assert!(2 * (45_612 + 1024) <= 100 << 10);
    }

    /// Local build (`QUIP_OTSUKA_UNION_SV`): a CPU model of the kernel's
    /// `union_sv` with racing threads. Phases run their items in a random
    /// order; a check-round read sees either the phase-start or the current
    /// parent (both are what a racing shared load can return); red.min and
    /// CAS are atomic. Returns every node's final root after the compress.
    fn union_sv_model(
        n: usize,
        list: &[(usize, usize)],
        overflow: &[(usize, usize)],
        rounds: usize,
        x: &mut u64,
    ) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        let find = |p: &mut Vec<usize>, mut a: usize| -> usize {
            // unite_v2's find_v2 with path halving (to an ancestor).
            loop {
                let par = p[a];
                if par == a {
                    return a;
                }
                let next = p[par];
                if next < par {
                    p[a] = next;
                }
                a = par;
            }
        };
        let unite = |p: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(p, a), find(p, b));
            if ra != rb {
                p[ra.max(rb)] = ra.min(rb);
            }
        };
        let shuffled = |len: usize, x: &mut u64| -> Vec<usize> {
            let mut o: Vec<usize> = (0..len).collect();
            for i in (1..len).rev() {
                o.swap(i, usize::try_from(xorshift(x) % (i as u64 + 1)).unwrap());
            }
            o
        };
        // Draw-phase overflow: united at once (unite_v2), before the barrier.
        for &(a, b) in overflow {
            unite(&mut p, a, b);
        }
        let jump = |p: &mut Vec<usize>, x: &mut u64| loop {
            let mut again = false;
            for i in shuffled(n, x) {
                let v = p[i];
                if p[v] != v {
                    p[i] = p[v];
                    again = true;
                }
            }
            if !again {
                break;
            }
        };
        let forest = |p: &Vec<usize>| {
            assert!(
                p.iter().enumerate().all(|(i, &v)| v <= i),
                "parents only decrease"
            )
        };
        if overflow.is_empty() {
            for k in shuffled(list.len(), x) {
                let (a, b) = list[k];
                if a != b {
                    let hi = a.max(b);
                    p[hi] = p[hi].min(a.min(b));
                }
            }
        }
        forest(&p);
        let mut converged = false;
        for _ in 0..rounds {
            jump(&mut p, x);
            let snap = p.clone();
            let mut hooked = false;
            for k in shuffled(list.len(), x) {
                let (a, b) = list[k];
                let ra = if xorshift(x) & 1 == 1 { snap[a] } else { p[a] };
                let rb = if xorshift(x) & 1 == 1 { snap[b] } else { p[b] };
                if ra != rb {
                    let hi = ra.max(rb);
                    p[hi] = p[hi].min(ra.min(rb));
                    hooked = true;
                }
            }
            forest(&p);
            if !hooked {
                converged = true;
                break;
            }
        }
        if !converged {
            jump(&mut p, x);
            for k in shuffled(list.len(), x) {
                let (a, b) = list[k];
                if p[a] != p[b] {
                    unite(&mut p, a, b);
                }
            }
            forest(&p);
        }
        // compress_v4: every node to its root.
        (0..n)
            .map(|mut i| {
                while p[i] != i {
                    i = p[i];
                }
                i
            })
            .collect()
    }

    /// Local build (`QUIP_OTSUKA_UNION_SV`): on the live topology's tiles and
    /// pairs (device order) with random kernel-like draws from sparse to
    /// percolating, with and without draw-phase overflow and for 0-3 hook
    /// rounds, the SV union model leaves every node under its component's
    /// smallest node: the same parents the V3/V4 union passes compress to.
    #[test]
    fn union_sv_roots_are_component_minima() {
        use crate::cluster::Weights;
        use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
        let instance = live_dimer_instance([42; 32]);
        let p = PreparedGraph::from_instance(
            &instance,
            &TWISTED_DIMER_V1,
            Weights {
                pair_milli: 250,
                tile_milli: 250,
            },
            &live_limits(),
            64,
        )
        .unwrap();
        let d = p.layout();
        let n = d.header[0] as usize;
        let mut x = 0x5_0005_u64;
        let mut cases = 0;
        for density in [1_u64, 3, 5, 7, 9] {
            for trial in 0..3 {
                // Tiles: 0, 1 or 2 bonds between up/down nodes; pairs: a bond with p ~ density / 10.
                let mut edges = Vec::new();
                for t in d.tiles.chunks_exact(4) {
                    let t: Vec<usize> = t.iter().map(|&v| v as usize).collect();
                    match xorshift(&mut x) % 10 {
                        r if r < density / 2 => {
                            edges.push((t[0], t[1]));
                            edges.push((t[2], t[3]));
                        }
                        r if r < density => edges.push((
                            t[usize::try_from(r % 4).unwrap()],
                            t[usize::try_from((r + 2) % 4).unwrap()],
                        )),
                        _ => {}
                    }
                }
                for pr in d.pairs.chunks_exact(2) {
                    if xorshift(&mut x) % 10 < density {
                        edges.push((pr[0] as usize, pr[1] as usize));
                    }
                }
                // Reference: union-find, root = smallest node.
                let mut uf: Vec<usize> = (0..n).collect();
                fn find(p: &mut [usize], mut a: usize) -> usize {
                    while p[a] != a {
                        p[a] = p[p[a]];
                        a = p[a];
                    }
                    a
                }
                for &(a, b) in &edges {
                    let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
                    uf[ra.max(rb)] = ra.min(rb);
                }
                let want: Vec<usize> = (0..n).map(|u| find(&mut uf, u)).collect();
                let biggest = {
                    let mut c = vec![0_usize; n];
                    for &r in &want {
                        c[r] += 1;
                    }
                    c.into_iter().max().unwrap()
                };
                for rounds in [0_usize, 1, 3] {
                    // No overflow, then an overflow of a random third of the edges.
                    let got = union_sv_model(n, &edges, &[], rounds, &mut x);
                    assert_eq!(
                        got, want,
                        "density {density} trial {trial} rounds {rounds}: largest {biggest}"
                    );
                    let cut = edges.len() / 3 + trial;
                    let got = union_sv_model(n, &edges[cut..], &edges[..cut], rounds, &mut x);
                    assert_eq!(
                        got, want,
                        "density {density} trial {trial} rounds {rounds} overflow {cut}"
                    );
                    cases += 2;
                }
            }
        }
        assert_eq!(cases, 90);
    }

    #[test]
    fn pack_ell4_chunks_rows_and_zero_pads() {
        // n = 3 nodes, width 5 (column-major e * n + u); row u holds 10*u + e + 1 for e < deg.
        let (n, width, deg) = (3_usize, 5_usize, [5_usize, 2, 0]);
        let mut ell = vec![0_i32; width * n];
        for u in 0..n {
            for e in 0..deg[u] {
                ell[e * n + u] = i32::try_from(10 * u + e + 1).unwrap();
            }
        }
        let p = pack_ell4(&ell, n);
        assert_eq!(p[0], 2, "chunk count in the header");
        assert_eq!(p.len(), 4 * (1 + 2 * n));
        for u in 0..n {
            for e in 0..8 {
                let want = if e < width { ell[e * n + u] } else { 0 };
                assert_eq!(
                    p[4 * (1 + (e / 4) * n + u) + e % 4],
                    want,
                    "node {u} entry {e}"
                );
            }
        }
    }

    /// Emulates the V5 kernel's 16-bit gathers (`field16_2_sh` and
    /// `field16s_sh`, padding correction included) against plain-ELL sums.
    #[test]
    fn pack_ell16_fields_match_plain_ell() {
        let n = 37_usize;
        let width = 20_usize;
        let js = [1000_i32, -1000, 62, -62];
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut ell = vec![0_i32; width * n];
        let mut deg = vec![0_i32; n];
        for u in 0..n {
            let d = if u == 3 {
                0
            } else {
                usize::try_from(next() % 21).unwrap()
            };
            deg[u] = i32::try_from(d).unwrap();
            for e in 0..d {
                let v = u32::try_from(next() % n as u64).unwrap();
                let j = js[usize::try_from(next() % 4).unwrap()];
                ell[e * n + u] = (v << 16 | u32::from((j as i16).cast_unsigned())).cast_signed();
            }
        }
        let p = pack_ell16(&ell, &deg, n).expect("four couplings qualify");
        assert_eq!((p[0], p[1]), (3, 1), "header: chunk count, 16-bit format");
        let jcode = |c: u32| -> i32 {
            let w = p[2 + (c as usize >> 1)].cast_unsigned();
            i32::from((w >> (16 * (c & 1))) as u16 as i16)
        };
        let spins: Vec<i32> = (0..n)
            .map(|_| if next() & 1 == 1 { 1 } else { -1 })
            .collect();
        // Parent words as the accumulate sees them: root | sign bit.
        let parent: Vec<i32> = (0..n)
            .map(|v| {
                let root = i32::try_from(next() % 5).unwrap();
                root | if spins[v] < 0 { i32::MIN } else { 0 }
            })
            .collect();
        let wterm = |v: usize, j: i32, r: i32| -> i32 {
            let pw = parent[v];
            if pw & 0x7fff_ffff != r {
                if pw < 0 {
                    -j
                } else {
                    j
                }
            } else {
                0
            }
        };
        for u in 0..n {
            let d = usize::try_from(deg[u]).unwrap();
            let r = parent[u] & 0x7fff_ffff; // the kernel's ra: the node's own root
            let (mut want_m, mut want_a) = (0_i32, 0_i32);
            for e in 0..d {
                let w = ell[e * n + u].cast_unsigned();
                let (v, j) = ((w >> 16) as usize, i32::from((w & 0xffff) as u16 as i16));
                want_m += j * spins[v];
                want_a += wterm(v, j, r);
            }
            let kd = (d + 7) / 8;
            let (mut got_m, mut got_a) = (0_i32, 0_i32);
            for k in 0..kd {
                for q in 0..4 {
                    let w = p[4 * (1 + k * n + u) + q].cast_unsigned();
                    for e16 in [w & 0xffff, w >> 16] {
                        let (v, c) = ((e16 >> 2) as usize, e16 & 3);
                        got_m += jcode(c) * spins[v];
                        got_a += wterm(v, jcode(c), r);
                    }
                }
            }
            let npad = i32::try_from(8 * kd - d).unwrap();
            got_m -= npad * jcode(0) * spins[u];
            for e in d..8 * kd {
                let w = p[4 * (1 + (e / 8) * n + u) + (e % 8) / 2].cast_unsigned();
                assert_eq!(
                    (w >> (16 * (e % 2))) & 0xffff,
                    u32::try_from(u << 2).unwrap(),
                    "padding of node {u}"
                );
            }
            assert_eq!(got_m, want_m, "Metropolis field of node {u}");
            assert_eq!(got_a, want_a, "accumulate field of node {u}");
        }
        // A fifth coupling keeps V4's int4 layout (header .y = 0).
        ell[0] = (1_u32 << 16 | 750).cast_signed();
        deg[0] = deg[0].max(1);
        assert!(pack_ell16(&ell, &deg, n).is_none());
        assert_eq!(pack_ell_v5(&ell, &deg, n)[1], 0);
    }
    #[test]
    fn pack_tiles_v4_nodes_gauge_bits_and_weight() {
        let tiles = [1, 2, 3, 65_535, 7, 0, 9, 4];
        let gauges = [1_i8, -1, 0, 1, -1, -1, 1, 1];
        let p = pack_tiles_v4(&tiles, &gauges, &[250, 300]).unwrap();
        assert_eq!(p.len(), 8);
        assert_eq!(p[0], 1 | 2 << 16);
        assert_eq!(p[1].cast_unsigned(), 3 | 65_535 << 16);
        assert_eq!(p[2], 0b1001 | 0b0010 << 4, "gauge 0 sets neither bit");
        assert_eq!(p[3], 250);
        assert_eq!(p[6], 0b1100 | 0b0011 << 4);
        assert_eq!(p[7], 300);
        assert!(pack_tiles_v4(&[0, 1, 2, 65_536], &[1; 4], &[1]).is_err());
        let pp = pack_pairs_v4(&[5, 6, 65_535, 0], &[250, -250]).unwrap();
        assert_eq!(pp, vec![5 | 6 << 16, 250, 65_535, -250]);
    }

    #[test]
    fn v4_up_mask_matches_spin_times_gauge() {
        // The kernel's tile_up: ((sp & pos) | (~sp & neg)) & 15 == bits of s * g > 0.
        for code in 0..81_u32 {
            let g: Vec<i8> = (0..4)
                .map(|k| [-1_i8, 0, 1][((code / 3_u32.pow(k)) % 3) as usize])
                .collect();
            let bits = pack_tiles_v4(&[0, 1, 2, 3], &g, &[1]).unwrap()[2].cast_unsigned();
            for sp in 0..16_u32 {
                let want = (0..4).fold(0, |m, k| {
                    let s = if sp >> k & 1 == 1 { 1 } else { -1 };
                    if s * i32::from(g[k as usize]) > 0 {
                        m | 1 << k
                    } else {
                        m
                    }
                });
                assert_eq!(
                    ((sp & bits) | (!sp & (bits >> 4))) & 15,
                    want,
                    "gauge {g:?} spins {sp:04b}"
                );
            }
        }
    }

    #[test]
    fn v4_pick_table_matches_fns() {
        // kernels/cluster.cu V4_PICK_LO / V4_PICK_HI replace __fns(mask, 0, k + 1).
        let grab = |name: &str| -> u64 {
            let line = SOURCE
                .lines()
                .find(|l| l.starts_with(&format!("#define {name} ")))
                .unwrap();
            let hex = line
                .split_whitespace()
                .nth(2)
                .unwrap()
                .trim_start_matches("0x")
                .trim_end_matches("ull");
            u64::from_str_radix(hex, 16).unwrap()
        };
        let (lo, hi) = (grab("V4_PICK_LO"), grab("V4_PICK_HI"));
        for mask in 0..16_u64 {
            let t = if mask & 8 != 0 { hi } else { lo };
            let set: Vec<u64> = (0..4).filter(|b| mask >> b & 1 == 1).collect();
            for (k, &pos) in set.iter().enumerate() {
                assert_eq!(
                    (t >> ((mask & 7) * 8 + 2 * k as u64)) & 3,
                    pos,
                    "mask {mask:04b} k {k}"
                );
            }
        }
    }
    use crate::capacity::DeviceLimits;
    use crate::cluster::{Model, Motifs, Weights};

    #[test]
    fn resample_keeps_count_and_favours_low_energy() {
        let picks = resample(&[0, 100_000, 0, 100_000], 1.0, 0.5);
        assert_eq!(picks.len(), 4);
        assert!(picks.iter().all(|&i| i == 0 || i == 2));
        // Zero step weights every slot equally: the identity.
        assert_eq!(resample(&[5, 9, 1], 0.0, 0.5), vec![0, 1, 2]);
    }

    /// Local build: the device checkpoint follows the host rules exactly:
    /// energies, systematic picks for the same uniform, the parent gather of
    /// states and lineage bests, the pairing shuffle, and Houdayer swaps of
    /// one whole differing component that keep each pair's summed energy.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::unreadable_literal,
        clippy::decimal_bitwise_operands
    )]
    fn device_pa_checkpoint_matches_host_rules() {
        let n = 600usize;
        let mut pop = Population::new(1, 0, 1.0);
        pop.rng = 12_345;
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for u in 0..n {
            let v = (u + 1) % n;
            edges.push((u.min(v), u.max(v)));
            seen.insert((u.min(v), u.max(v)));
        }
        while edges.len() < 1500 {
            let (u, v) = (pop.below(n), pop.below(n));
            if u != v && seen.insert((u.min(v), u.max(v))) {
                edges.push((u.min(v), u.max(v)));
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let j: Vec<i32> = edges
            .iter()
            .map(|_| (pop.below(1800) as i32 - 900) | 1)
            .collect();
        // The cluster contract admits zero fields only.
        let h = vec![0_i32; n];
        let edges32: Vec<(u32, u32)> = edges
            .iter()
            .map(|&(u, v)| (u32::try_from(u).unwrap(), u32::try_from(v).unwrap()))
            .collect();
        let limits = DeviceLimits {
            shared_bytes_per_block: 48 << 10,
            shared_bytes_per_block_optin: 100 << 10,
            free_bytes: 1 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        };
        let graph = PreparedGraph::new(
            &Model {
                h_milli: &h,
                edges: &edges32,
                j_milli: &j,
            },
            &Motifs {
                tiles: &[],
                gauges: &[],
                pairs: &[],
            },
            Weights::FULL,
            &limits,
            8,
        )
        .unwrap();
        let kernel =
            ClusterKernel::load_variant(0, false, false, true, false, false, false, false, false)
                .unwrap();
        let dev = ClusterDevice::open_with_kernel(graph, &kernel).unwrap();
        let pa = dev.pa.clone().expect("device PA admitted");
        let r = 8;
        // Correlated replicas so each pair differs on part of the graph.
        let base: Vec<i8> = (0..n)
            .map(|_| if pop.below(2) == 0 { 1 } else { -1 })
            .collect();
        let mut host = Vec::with_capacity(r * n);
        for _ in 0..r {
            host.extend(base.iter().map(|&x| if pop.below(10) < 3 { -x } else { x }));
        }
        let energy_of = |s: &[i8]| dev.graph.energy(&dev.graph.unpermute(s).unwrap()).unwrap();
        let spins = dev.stream.clone_htod(&host).unwrap();
        let mut bufs = PaBuffers::new(&dev.stream, spins, r, n).unwrap();
        let (seed, delta) = (99_u64, 0.004_f64);
        let mirror = |checkpoint: u64| {
            let mut p = Population::new(1, 0, 1.0);
            p.rng = 99 ^ 0x5E_ED_0F_07_50_CA_00 ^ checkpoint.wrapping_mul(0xD1B5_4A32_D192_ED03);
            p.next();
            p
        };

        dev.enqueue_pa_checkpoint(&pa, &mut bufs, delta, false, seed, 0)
            .unwrap();
        let energies = dev.stream.clone_dtoh(&bufs.energy).unwrap();
        let picks = dev.stream.clone_dtoh(&bufs.picks).unwrap();
        let after = dev.stream.clone_dtoh(&bufs.spins).unwrap();
        let best = dev.stream.clone_dtoh(&bufs.best).unwrap();
        let best_e = dev.stream.clone_dtoh(&bufs.best_e).unwrap();
        dev.stream.synchronize().unwrap();
        for i in 0..r {
            assert_eq!(
                energies[i],
                energy_of(&host[i * n..(i + 1) * n]),
                "energy {i}"
            );
        }
        let u = mirror(0).uniform();
        let expected: Vec<i32> = resample(&energies, delta, u)
            .into_iter()
            .map(|i| i32::try_from(i).unwrap())
            .collect();
        assert_eq!(picks, expected);
        assert!(
            expected.iter().any(|&p| p != expected[0]),
            "degenerate picks"
        );
        for (i, &p) in picks.iter().enumerate() {
            let p = usize::try_from(p).unwrap();
            assert_eq!(after[i * n..(i + 1) * n], host[p * n..(p + 1) * n]);
            assert_eq!(best[i * n..(i + 1) * n], host[p * n..(p + 1) * n]);
            assert_eq!(best_e[i], energies[p]);
        }

        dev.enqueue_pa_checkpoint(&pa, &mut bufs, 0.0, true, seed, 1)
            .unwrap();
        let order = dev.stream.clone_dtoh(&bufs.order).unwrap();
        let swapped = dev.stream.clone_dtoh(&bufs.spins).unwrap();
        dev.stream.synchronize().unwrap();
        let mut shuffle = mirror(1);
        let mut expect_order: Vec<i32> = (0..8).collect();
        for k in (1..r).rev() {
            let j = shuffle.below(k + 1);
            expect_order.swap(k, j);
        }
        assert_eq!(order, expect_order);
        let layout = dev.graph.layout();
        let (deg, ell) = (&layout.all_deg, &layout.all_ell);
        let mut moved = 0;
        for p in 0..r / 2 {
            let a = usize::try_from(order[2 * p]).unwrap();
            let b = usize::try_from(order[2 * p + 1]).unwrap();
            let (a0, b0) = (&after[a * n..(a + 1) * n], &after[b * n..(b + 1) * n]);
            let (a1, b1) = (&swapped[a * n..(a + 1) * n], &swapped[b * n..(b + 1) * n]);
            let flipped: std::collections::BTreeSet<usize> =
                (0..n).filter(|&x| a1[x] != a0[x]).collect();
            for x in 0..n {
                if flipped.contains(&x) {
                    assert!(
                        a0[x] != b0[x] && a1[x] == b0[x] && b1[x] == a0[x],
                        "pair {p} node {x}"
                    );
                } else {
                    assert_eq!((a1[x], b1[x]), (a0[x], b0[x]), "pair {p} node {x}");
                }
            }
            let differ = (0..n).any(|x| a0[x] != b0[x]);
            assert_eq!(flipped.is_empty(), !differ, "pair {p}");
            if let Some(&root) = flipped.iter().next() {
                // The flipped set is exactly the differing component of root.
                let mut comp = std::collections::BTreeSet::from([root]);
                let mut stack = vec![root];
                while let Some(x) = stack.pop() {
                    for e in 0..usize::try_from(deg[x]).unwrap() {
                        #[allow(clippy::cast_sign_loss)]
                        let y = ((ell[e * n + x] as u32) >> 16) as usize;
                        if a0[y] != b0[y] && comp.insert(y) {
                            stack.push(y);
                        }
                    }
                }
                assert_eq!(comp, flipped, "pair {p}");
                moved += 1;
            }
            assert_eq!(energy_of(a0) + energy_of(b0), energy_of(a1) + energy_of(b1));
        }
        assert!(moved > 0, "no Houdayer move happened");
    }

    #[test]
    fn houdayer_swaps_one_differing_component() {
        // Path 0-1-2-3 in slot-major ELL (n = 4, width 2), J = 1 milli.
        let n = 4;
        let w = |v: i32| (v << 16) | 1;
        let deg = vec![1, 2, 2, 1];
        let mut ell = vec![0; 2 * n];
        ell[0] = w(1);
        ell[1] = w(0);
        ell[n + 1] = w(2);
        ell[2] = w(1);
        ell[n + 2] = w(3);
        ell[3] = w(2);
        // a = ++++, b = -+-- : differing nodes {0} and {2,3}, two components.
        let mut host = vec![1, 1, 1, 1, -1, 1, -1, -1];
        let mut pop = Population::new(2, 7, 1.0);
        pop.houdayer(&mut host, 0, 1, n, &deg, &ell);
        assert!(
            host == [-1, 1, 1, 1, 1, 1, -1, -1] || host == [1, 1, -1, -1, -1, 1, 1, 1],
            "{host:?}"
        );
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
