//! Pack validated motifs and energy layers into bounded device arrays.

use std::collections::HashMap;

use super::{key, Edge, GraphError, Model, Motifs, OtsukaGraph, Weights};

use crate::capacity::{
    cluster_footprint, ClusterShape, DeviceLimits, Footprint, CLUSTER_DYNAMIC_PADDING,
    CLUSTER_SHARED_BYTES_PER_NODE, CLUSTER_STATIC_SHARED_BYTES,
};
use quip_solver_core::families::FamilyContext;
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::ice::twisted_dimer::local_gauges;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use thiserror::Error;

/// A packed neighbor uses 16 bits (`gpucluster.cu:123-124,226-228`).
/// Node positions 0 through 65535 are representable.
pub const MAX_PACKED_NODES: usize = 1 << 16;

/// ELL addressing multiplies slot by node count in signed int (`gpucluster.cu:123,226`).
pub const MAX_ELL_ENTRIES: usize = i32::MAX as usize;

/// Component sums use signed int atomic addition (`gpucluster.cu:233`).
const MAX_COMPONENT_FIELD: u64 = i32::MAX as u64;

/// A rejection before device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PrepareError {
    /// The model does not satisfy the CPU Otsuka graph contract.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Only twisted-ice-dimer revision 1 is supported.
    #[error("the cluster preparation supports only quip/twisted-ice-dimer revision 1")]
    UnsupportedFamily,
    /// The canonical metadata does not contain readable tiles, gauges, and pairs.
    #[error("the family metadata does not contain valid tiles, gauges, and pairs")]
    Metadata,
    /// A device index or array length exceeds the kernel integer width.
    #[error("{array} exceeds the cluster kernel index width")]
    IndexOverflow {
        /// Array or index that failed the width check.
        array: &'static str,
    },
    /// A component could overflow its signed 32-bit residual accumulator.
    #[error("the sum of absolute residual couplings exceeds the component accumulator")]
    AccumulatorOverflow,
    /// No anneals were requested.
    #[error("the cluster launch must contain at least one anneal")]
    EmptyLaunch,
    /// Dynamic and static shared memory exceed the opt-in per-block limit.
    #[error("cluster shared memory {requested} exceeds the per-block budget {budget}")]
    SharedMemory {
        /// Required bytes per block, including static storage.
        requested: usize,
        /// Opt-in bytes per block.
        budget: usize,
    },
    /// Global allocations exceed the usable device memory.
    #[error("cluster device memory {requested} exceeds the usable budget {budget}")]
    DeviceMemory {
        /// Required global bytes, including pool allowance.
        requested: usize,
        /// Usable global bytes after headroom.
        budget: usize,
    },
}

/// Why a state cannot be scored or permuted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SpinError {
    /// The spin count differs from the node count.
    #[error("the spin count differs from the node count")]
    Length,
    /// A spin is not a sign.
    #[error("spin {node} is not 1 or -1")]
    Value {
        /// Position in the supplied order.
        node: usize,
    },
}

/// Upload arrays for `Dev` in `gpucluster.cu:68-74`, in device node order.
/// All int arrays are 32-bit. Gauges upload as signed bytes, not file int32.
/// Tile, pair, and gauge arrays are flat for aligned CUDA allocation and
/// reinterpretation as `int4`, `int2`, and `char4` respectively.
#[derive(Debug)]
pub struct DeviceLayout {
    /// `[n, tiles, links, weak_width, all_width, colors]` (`gpucluster.cu:15-18`).
    pub header: [i32; 6],
    /// Four node indices per tile.
    pub tiles: Vec<i32>,
    /// Four local gauge signs per tile, in tile-position order.
    pub gauges: Vec<i8>,
    /// Two node indices per metadata pair.
    pub pairs: Vec<i32>,
    /// Nonzero residual degree per node.
    pub weak_deg: Vec<i32>,
    /// Slot-major packed residual adjacency, padded with zero words.
    pub weak_ell: Vec<i32>,
    /// Nonzero original degree per node.
    pub all_deg: Vec<i32>,
    /// Slot-major packed original adjacency, padded with zero words.
    pub all_ell: Vec<i32>,
    /// Contiguous color offsets, including zero and the final node count.
    pub color_off: Vec<i32>,
    /// Zero fields, uploaded explicitly for the kernel field pointer.
    pub h: Vec<i32>,
    /// One positive milli weight per tile.
    pub tile_w: Vec<i32>,
    /// One nonnegative milli bond weight per pair, subtracted from energy.
    pub link_w: Vec<i32>,
}

/// Local build (`QUIP_OTSUKA_WEAK_SCALE`): the full and residual ELL tables
/// with every weak coupling (`|J| <` [`WEAK_LIMIT_MILLI`] in the model)
/// multiplied by the scale. The degrees and slot layout are those of
/// [`DeviceLayout`]; only the coupling halves of the words differ. The anneal
/// sweeps read these; final descent, device energies, and every reported
/// energy use the exact tables.
#[derive(Debug)]
pub struct AnnealEll {
    /// The scale on the weak couplings.
    pub weak_scale: f64,
    /// Scaled residual adjacency, slot-major like [`DeviceLayout::weak_ell`].
    pub weak_ell: Vec<i32>,
    /// Scaled full adjacency, slot-major like [`DeviceLayout::all_ell`].
    pub all_ell: Vec<i32>,
}

/// Model couplings below this magnitude, in milli, are weak: the
/// twisted-dimer weak layer (45 or 62 milli) against the 1000-milli tiles
/// and pairs.
pub const WEAK_LIMIT_MILLI: i32 = 500;

/// The weak coupling `j` scaled by `scale` and rounded. Strong couplings and
/// zero pass unchanged; a weak coupling keeps its sign and stays nonzero and
/// weak, so the ELL degrees do not change.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn scale_weak(j: i32, scale: f64) -> i32 {
    if j == 0 || j.abs() >= WEAK_LIMIT_MILLI {
        return j;
    }
    let s = (f64::from(j) * scale).round().clamp(
        -f64::from(WEAK_LIMIT_MILLI - 1),
        f64::from(WEAK_LIMIT_MILLI - 1),
    ) as i32;
    if s == 0 {
        j.signum()
    } else {
        s
    }
}

/// A validated original graph, its permutations, and bounded device inputs.
#[derive(Debug)]
pub struct PreparedGraph {
    original: OtsukaGraph,
    layout: DeviceLayout,
    /// Local build (`QUIP_OTSUKA_WEAK_SCALE`): the ELL tables the anneal
    /// sweeps read when the weak couplings are scaled. `None` at scale 1.
    anneal: Option<AnnealEll>,
    device_to_original: Vec<usize>,
    original_to_device: Vec<usize>,
    footprint: Footprint,
    dynamic_shared_bytes: usize,
}

impl PreparedGraph {
    /// Validate canonical coefficients and verified motifs, then prepare arrays.
    ///
    /// This low-level entry checks the same graph rules as the CPU port.
    /// The caller verifies the family metadata before supplying motifs.
    /// `anneals` counts concurrent blocks at the fixed cluster thread count.
    ///
    /// # Errors
    /// Returns a graph, width, accumulator, launch, or capacity rejection before
    /// any device allocation. This module makes no CUDA calls.
    pub fn new(
        model: &Model<'_>,
        motifs: &Motifs<'_>,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        Self::with_weak_scale(model, motifs, weights, limits, anneals, 1.0)
    }

    /// Local build: [`Self::new`] with the anneal sweeps reading the weak
    /// couplings scaled by `weak_scale` ([`AnnealEll`]). At scale 1 this is
    /// [`Self::new`] exactly. The original graph, the exact ELL tables, and
    /// every energy stay the instance's own.
    ///
    /// # Errors
    /// The same as [`Self::new`].
    #[allow(clippy::too_many_lines)] // the stock preparation plus the scaled tables
    pub fn with_weak_scale(
        model: &Model<'_>,
        motifs: &Motifs<'_>,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
        weak_scale: f64,
    ) -> Result<Self, PrepareError> {
        let original = OtsukaGraph::new(model, motifs, weights)?;
        let n = original.n;
        if n > MAX_PACKED_NODES {
            return Err(PrepareError::IndexOverflow { array: "nodes" });
        }
        if anneals == 0 {
            return Err(PrepareError::EmptyLaunch);
        }
        check_residual_sum(original.weak_edges.iter().map(|edge| edge.2))?;
        let (color_off, device_to_original, original_to_device) = coloring(&original);
        let all_rows = rows(n, &original.edges, &original_to_device);
        // Keep original edge order for the residual ELL, as prep.py does.
        let residual: HashMap<_, _> = original
            .weak_edges
            .iter()
            .map(|&(u, v, j)| (key(u, v), j))
            .collect();
        let weak_edges: Vec<_> = original
            .edges
            .iter()
            .map(|&(u, v, _)| (u, v, residual.get(&key(u, v)).copied().unwrap_or(0)))
            .collect();
        let weak_rows = rows(n, &weak_edges, &original_to_device);
        let shape = ClusterShape {
            nodes: n,
            tiles: original.tiles.len(),
            links: original.pairs.len(),
            weak_width: weak_rows.iter().map(Vec::len).max().unwrap_or(0),
            all_width: all_rows.iter().map(Vec::len).max().unwrap_or(0),
            colors: color_off.len() - 1,
        };
        check_widths(shape)?;
        let dynamic_shared_bytes = n * CLUSTER_SHARED_BYTES_PER_NODE + CLUSTER_DYNAMIC_PADDING;
        let requested = dynamic_shared_bytes + CLUSTER_STATIC_SHARED_BYTES;
        if requested > limits.shared_bytes_per_block_optin {
            return Err(PrepareError::SharedMemory {
                requested,
                budget: limits.shared_bytes_per_block_optin,
            });
        }
        let footprint = cluster_footprint(shape, anneals);
        if footprint.total() > limits.usable_bytes() {
            return Err(PrepareError::DeviceMemory {
                requested: footprint.total(),
                budget: limits.usable_bytes(),
            });
        }
        let (weak_deg, weak_ell) = ell(&weak_rows, shape.weak_width)?;
        let (all_deg, all_ell) = ell(&all_rows, shape.all_width)?;
        // Local build: the scaled tables share the exact ones' degrees, because
        // scale_weak keeps every nonzero coupling nonzero.
        #[allow(clippy::float_cmp)]
        let anneal = if weak_scale == 1.0 {
            None
        } else {
            let scaled = |es: &[Edge], weak: &dyn Fn(usize) -> bool| -> Vec<Edge> {
                es.iter()
                    .enumerate()
                    .map(|(at, &(u, v, j))| {
                        (
                            u,
                            v,
                            if weak(at) {
                                scale_weak(j, weak_scale)
                            } else {
                                j
                            },
                        )
                    })
                    .collect()
            };
            let model_weak = |at: usize| original.edges[at].2.abs() < WEAK_LIMIT_MILLI;
            let all_scaled = scaled(&original.edges, &model_weak);
            let weak_scaled = scaled(&weak_edges, &model_weak);
            check_residual_sum(weak_scaled.iter().map(|edge| edge.2))?;
            let (deg_w, weak_ell_s) = ell(
                &rows(n, &weak_scaled, &original_to_device),
                shape.weak_width,
            )?;
            let (deg_a, all_ell_s) =
                ell(&rows(n, &all_scaled, &original_to_device), shape.all_width)?;
            debug_assert!(
                deg_w == weak_deg && deg_a == all_deg,
                "scaling keeps degrees"
            );
            if deg_w != weak_deg || deg_a != all_deg {
                return Err(PrepareError::IndexOverflow {
                    array: "scaled ELL degrees",
                });
            }
            Some(AnnealEll {
                weak_scale,
                weak_ell: weak_ell_s,
                all_ell: all_ell_s,
            })
        };
        let id = |u: usize| int_index(original_to_device[u], "motif node");
        let tiles = original
            .tiles
            .iter()
            .flat_map(|tile| tile.nodes)
            .map(id)
            .collect::<Result<_, _>>()?;
        let gauges = original.tiles.iter().flat_map(|tile| tile.gauge).collect();
        let pairs = original
            .pairs
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .map(id)
            .collect::<Result<_, _>>()?;
        let header = [
            shape.nodes,
            shape.tiles,
            shape.links,
            shape.weak_width,
            shape.all_width,
            shape.colors,
        ]
        .map(|value| int_index(value, "header"))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let layout = DeviceLayout {
            header: [
                header[0], header[1], header[2], header[3], header[4], header[5],
            ],
            tiles,
            gauges,
            pairs,
            weak_deg,
            weak_ell,
            all_deg,
            all_ell,
            color_off: color_off
                .into_iter()
                .map(|off| int_index(off, "color offsets"))
                .collect::<Result<_, _>>()?,
            h: vec![0; n],
            tile_w: vec![weights.tile_milli; shape.tiles],
            link_w: vec![weights.pair_milli; shape.links],
        };
        Ok(Self {
            original,
            layout,
            anneal,
            device_to_original,
            original_to_device,
            footprint,
            dynamic_shared_bytes,
        })
    }

    /// Prepare a session-validated canonical instance under its family contract.
    ///
    /// The caller must validate the instance against `contract` and its declared
    /// parameters first. Prefer [`Self::from_family`] for session jobs.
    ///
    /// # Errors
    /// Returns unsupported family or unreadable metadata, then the same errors
    /// as [`Self::new`]. The instance offset stays with the session for scoring.
    pub fn from_instance(
        instance: &Instance,
        contract: &Contract,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        Self::from_instance_scaled(instance, contract, weights, limits, anneals, 1.0)
    }

    /// Local build: [`Self::from_instance`] with the weak couplings scaled by
    /// `weak_scale` in the anneal tables ([`Self::with_weak_scale`]).
    ///
    /// # Errors
    /// The same as [`Self::from_instance`].
    pub fn from_instance_scaled(
        instance: &Instance,
        contract: &Contract,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
        weak_scale: f64,
    ) -> Result<Self, PrepareError> {
        if contract.family_id != TWISTED_DIMER_V1.family_id
            || contract.revision != TWISTED_DIMER_V1.revision
        {
            return Err(PrepareError::UnsupportedFamily);
        }
        let (tiles, gauges) = local_gauges(instance).ok_or(PrepareError::Metadata)?;
        let pairs = metadata_pairs(instance.metadata()).ok_or(PrepareError::Metadata)?;
        Self::with_weak_scale(
            &Model {
                h_milli: instance.h(),
                edges: instance.edges(),
                j_milli: instance.j(),
            },
            &Motifs {
                tiles: &tiles,
                gauges: &gauges,
                pairs: &pairs,
            },
            weights,
            limits,
            anneals,
            weak_scale,
        )
    }

    /// Read the bound canonical instance supplied by a lease or explicit job.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::from_instance`].
    pub fn from_family(
        family: &FamilyContext,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        Self::from_instance(
            family.instance(),
            family.contract(),
            weights,
            limits,
            anneals,
        )
    }

    /// Immutable upload arrays in device node order.
    #[must_use]
    pub fn layout(&self) -> &DeviceLayout {
        &self.layout
    }

    /// Local build: the scaled anneal tables, when the weak scale is not 1.
    #[must_use]
    pub fn anneal_ell(&self) -> Option<&AnnealEll> {
        self.anneal.as_ref()
    }

    /// Original coefficients and motifs for rescoring in original node order.
    #[must_use]
    pub fn original_graph(&self) -> &OtsukaGraph {
        &self.original
    }

    /// Device position to original position: `device[k] = original[perm[k]]`.
    #[must_use]
    pub fn device_to_original(&self) -> &[usize] {
        &self.device_to_original
    }

    /// Original position to device position, the explicit inverse permutation.
    #[must_use]
    pub fn original_to_device(&self) -> &[usize] {
        &self.original_to_device
    }

    /// Checked global allocation estimate for the requested concurrent anneals.
    #[must_use]
    pub fn footprint(&self) -> Footprint {
        self.footprint
    }

    /// Dynamic bytes per block. Admission also counts the static `changed` int.
    #[must_use]
    pub fn dynamic_shared_bytes(&self) -> usize {
        self.dynamic_shared_bytes
    }

    fn check_spins(&self, spins: &[i8]) -> Result<(), SpinError> {
        if spins.len() != self.original.n {
            return Err(SpinError::Length);
        }
        if let Some(node) = spins.iter().position(|&s| s != 1 && s != -1) {
            return Err(SpinError::Value { node });
        }
        Ok(())
    }

    /// Convert an original-order state to device order.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn permute(&self, spins: &[i8]) -> Result<Vec<i8>, SpinError> {
        self.check_spins(spins)?;
        Ok(self
            .device_to_original
            .iter()
            .map(|&old| spins[old])
            .collect())
    }

    /// Restore a device-order state to original order before returning spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn unpermute(&self, spins: &[i8]) -> Result<Vec<i8>, SpinError> {
        self.check_spins(spins)?;
        Ok(self
            .original_to_device
            .iter()
            .map(|&new| spins[new])
            .collect())
    }

    /// Original energy in milli, without the session-owned offset.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.energy(spins))
    }

    /// Weighted tile and pair energy in milli, on original-order spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn strong_energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.strong_energy(spins))
    }

    /// Non-motif energy plus softened motif residuals, on original-order spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn weak_energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.weak_energy(spins))
    }

    /// `(tile, link, residual)` energies in milli, on original-order spins.
    /// Fields are zero under the admitted contract.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn split_energies(&self, spins: &[i8]) -> Result<(i64, i64, i64), SpinError> {
        self.check_spins(spins)?;
        let link: i64 = self
            .original
            .pairs
            .iter()
            .map(|&(u, v)| {
                -i64::from(self.original.weights.pair_milli) * i64::from(spins[u] * spins[v])
            })
            .sum();
        let strong = self.original.strong_energy(spins);
        Ok((strong - link, link, self.original.energy(spins) - strong))
    }
}

fn int_index(value: usize, array: &'static str) -> Result<i32, PrepareError> {
    i32::try_from(value).map_err(|_| PrepareError::IndexOverflow { array })
}

fn check_widths(shape: ClusterShape) -> Result<(), PrepareError> {
    for (array, count) in [
        ("tile indices", shape.tiles.saturating_mul(4)),
        ("pair indices", shape.links.saturating_mul(2)),
        ("weak ELL", shape.nodes.saturating_mul(shape.weak_width)),
        ("all ELL", shape.nodes.saturating_mul(shape.all_width)),
        ("color offsets", shape.colors.saturating_add(1)),
    ] {
        if count > MAX_ELL_ENTRIES {
            return Err(PrepareError::IndexOverflow { array });
        }
    }
    Ok(())
}

fn check_residual_sum(couplings: impl Iterator<Item = i32>) -> Result<(), PrepareError> {
    let sum = couplings.fold(0_u64, |sum, j| {
        sum.saturating_add(u64::from(j.unsigned_abs()))
    });
    if sum > MAX_COMPONENT_FIELD {
        return Err(PrepareError::AccumulatorOverflow);
    }
    Ok(())
}

fn coloring(graph: &OtsukaGraph) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = graph.n;
    let mut adj = vec![Vec::new(); n];
    // Include zero edges, matching prep.py coloring and the canonical topology.
    for &(u, v, _) in &graph.edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&u| std::cmp::Reverse(adj[u].len()));
    let mut color = vec![usize::MAX; n];
    let mut seen = vec![usize::MAX; n];
    for u in order {
        for &v in &adj[u] {
            if color[v] != usize::MAX {
                seen[color[v]] = u;
            }
        }
        color[u] = seen.iter().position(|&mark| mark != u).unwrap_or(0);
    }
    let mut perm: Vec<_> = (0..n).collect();
    perm.sort_by_key(|&u| color[u]);
    let mut off = vec![0; color.iter().max().copied().unwrap_or(0) + 2];
    let mut inverse = vec![0; n];
    for (new, &old) in perm.iter().enumerate() {
        inverse[old] = new;
        off[color[old] + 1] += 1;
    }
    for c in 1..off.len() {
        off[c] += off[c - 1];
    }
    (off, perm, inverse)
}

fn rows(n: usize, edges: &[Edge], inverse: &[usize]) -> Vec<Vec<(usize, i32)>> {
    let mut rows = vec![Vec::new(); n];
    // prep.py concatenates the forward halves before the reverse halves.
    for reverse in [false, true] {
        for &(u, v, j) in edges {
            if j == 0 {
                continue;
            }
            let (u, v) = if reverse { (v, u) } else { (u, v) };
            rows[inverse[u]].push((inverse[v], j));
        }
    }
    rows
}

fn ell(rows: &[Vec<(usize, i32)>], width: usize) -> Result<(Vec<i32>, Vec<i32>), PrepareError> {
    let mut deg = Vec::with_capacity(rows.len());
    let mut table = vec![0; width * rows.len()];
    for (u, row) in rows.iter().enumerate() {
        deg.push(int_index(row.len(), "degree")?);
        for (slot, &(v, j)) in row.iter().enumerate() {
            let neighbor =
                u16::try_from(v).map_err(|_| PrepareError::IndexOverflow { array: "neighbor" })?;
            let coupling =
                i16::try_from(j).map_err(|_| PrepareError::IndexOverflow { array: "coupling" })?;
            let word =
                u32::from(neighbor) << 16 | u32::from(u16::from_ne_bytes(coupling.to_ne_bytes()));
            table[slot * rows.len() + u] = i32::from_ne_bytes(word.to_ne_bytes());
        }
    }
    Ok((deg, table))
}

/// Local build (CPU polish): the validated host graph of a family instance at
/// `weights`, with the same checks as [`PreparedGraph::from_instance`] and no
/// device layout.
///
/// # Errors
/// Unsupported family, unreadable metadata, or a graph rejection.
pub(crate) fn host_graph(
    instance: &Instance,
    contract: &Contract,
    weights: Weights,
) -> Result<OtsukaGraph, PrepareError> {
    if contract.family_id != TWISTED_DIMER_V1.family_id
        || contract.revision != TWISTED_DIMER_V1.revision
    {
        return Err(PrepareError::UnsupportedFamily);
    }
    let (tiles, gauges) = local_gauges(instance).ok_or(PrepareError::Metadata)?;
    let pairs = metadata_pairs(instance.metadata()).ok_or(PrepareError::Metadata)?;
    Ok(OtsukaGraph::new(
        &Model {
            h_milli: instance.h(),
            edges: instance.edges(),
            j_milli: instance.j(),
        },
        &Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &pairs,
        },
        weights,
    )?)
}

fn metadata_pairs(region: &[u8]) -> Option<Vec<(u32, u32)>> {
    let (count, mut rest) = region.split_first_chunk::<2>()?;
    for _ in 0..u16::from_le_bytes(*count) {
        let (header, tail) = rest.split_first_chunk::<6>()?;
        let len = usize::try_from(u32::from_le_bytes(header[2..6].try_into().ok()?)).ok()?;
        let (payload, next) = tail.split_at_checked(len)?;
        rest = next;
        if u16::from_le_bytes([header[0], header[1]]) != 3 {
            continue;
        }
        let (count, data) = payload.split_first_chunk::<4>()?;
        let (pairs, tail) = data.as_chunks::<8>();
        if !tail.is_empty() || usize::try_from(u32::from_le_bytes(*count)).ok()? != pairs.len() {
            return None;
        }
        return pairs
            .iter()
            .map(|p| {
                Some((
                    u32::from_le_bytes(p[..4].try_into().ok()?),
                    u32::from_le_bytes(p[4..].try_into().ok()?),
                ))
            })
            .collect();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_accumulator_rejects_the_first_overflow() {
        assert!(check_residual_sum([i32::MAX].into_iter()).is_ok());
        assert_eq!(
            check_residual_sum([i32::MAX, -1].into_iter()),
            Err(PrepareError::AccumulatorOverflow)
        );
        assert_eq!(
            check_residual_sum([i32::MIN].into_iter()),
            Err(PrepareError::AccumulatorOverflow)
        );
    }

    #[test]
    fn kernel_array_products_cannot_overflow_signed_addressing() {
        let shape = ClusterShape {
            nodes: 65_536,
            tiles: 0,
            links: 0,
            weak_width: 0,
            all_width: 32_768,
            colors: 1,
        };
        assert_eq!(
            check_widths(shape),
            Err(PrepareError::IndexOverflow { array: "all ELL" })
        );
        assert!(check_widths(ClusterShape {
            all_width: 32_767,
            ..shape
        })
        .is_ok());
        assert_eq!(
            check_widths(ClusterShape {
                tiles: usize::MAX,
                ..shape
            }),
            Err(PrepareError::IndexOverflow {
                array: "tile indices"
            })
        );
    }

    #[test]
    fn scale_weak_rounds_keeps_sign_and_leaves_strong() {
        assert_eq!(scale_weak(45, 1.5), 68);
        assert_eq!(scale_weak(-45, 1.5), -68);
        assert_eq!(scale_weak(45, 1.25), 56);
        assert_eq!(scale_weak(45, 2.0), 90);
        assert_eq!(scale_weak(45, 1.0), 45);
        assert_eq!(scale_weak(1000, 2.0), 1000);
        assert_eq!(scale_weak(-1000, 0.5), -1000);
        assert_eq!(scale_weak(750, 2.0), 750);
        assert_eq!(scale_weak(0, 2.0), 0);
        assert_eq!(scale_weak(1, 0.25), 1, "a weak coupling stays nonzero");
        assert_eq!(scale_weak(-1, 0.25), -1);
        assert_eq!(scale_weak(400, 2.0), 499, "a weak coupling stays weak");
    }

    /// A graph of two gauged tiles, one pair, and weak links, as the
    /// twisted-dimer cells are built (strong 1000, pair -1000, weak 45).
    #[allow(clippy::type_complexity)]
    fn two_tiles() -> (
        Vec<i32>,
        Vec<(u32, u32)>,
        Vec<i32>,
        Vec<[u32; 4]>,
        Vec<[i8; 4]>,
        Vec<(u32, u32)>,
    ) {
        let tiles = vec![[0, 1, 2, 3], [4, 5, 6, 7]];
        let gauges = vec![[1, -1, 1, 1], [1, 1, -1, 1]];
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for (t, g) in tiles.iter().zip(&gauges) {
            for a in 0..4 {
                for b in a + 1..4 {
                    edges.push((t[a], t[b]));
                    j.push(1000 * i32::from(g[a] * g[b]));
                }
            }
        }
        let pairs = vec![(3, 4)];
        edges.push((3, 4));
        j.push(-1000);
        for (u, v, w) in [(0, 4, 45), (1, 5, -45), (2, 6, 45), (0, 7, -45), (2, 5, 45)] {
            edges.push((u, v));
            j.push(w);
        }
        (vec![0; 8], edges, j, tiles, gauges, pairs)
    }

    fn limits() -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 48 << 10,
            shared_bytes_per_block_optin: 100 << 10,
            free_bytes: 1 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn coupling(word: i32) -> i32 {
        i32::from(i16::from_ne_bytes((word as u16).to_ne_bytes()))
    }

    #[test]
    fn weak_scale_one_is_the_stock_preparation() {
        let (h, edges, j, tiles, gauges, pairs) = two_tiles();
        let model = Model {
            h_milli: &h,
            edges: &edges,
            j_milli: &j,
        };
        let motifs = Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &pairs,
        };
        let w = Weights {
            pair_milli: 250,
            tile_milli: 250,
        };
        let stock = PreparedGraph::new(&model, &motifs, w, &limits(), 4).unwrap();
        let one = PreparedGraph::with_weak_scale(&model, &motifs, w, &limits(), 4, 1.0).unwrap();
        assert!(stock.anneal_ell().is_none() && one.anneal_ell().is_none());
        assert_eq!(stock.layout().all_ell, one.layout().all_ell);
        assert_eq!(stock.layout().weak_ell, one.layout().weak_ell);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn weak_scale_touches_only_the_anneal_tables_and_weak_couplings() {
        let (h, edges, j, tiles, gauges, pairs) = two_tiles();
        let model = Model {
            h_milli: &h,
            edges: &edges,
            j_milli: &j,
        };
        let motifs = Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &pairs,
        };
        let w = Weights {
            pair_milli: 250,
            tile_milli: 250,
        };
        let stock = PreparedGraph::new(&model, &motifs, w, &limits(), 4).unwrap();
        let scaled = PreparedGraph::with_weak_scale(&model, &motifs, w, &limits(), 4, 1.5).unwrap();
        // The exact tables, the graph, and its energies are unchanged.
        assert_eq!(stock.layout().all_ell, scaled.layout().all_ell);
        assert_eq!(stock.layout().weak_ell, scaled.layout().weak_ell);
        assert_eq!(stock.layout().all_deg, scaled.layout().all_deg);
        let spins = [1, -1, 1, 1, -1, 1, -1, -1];
        assert_eq!(
            stock.energy(&spins).unwrap(),
            scaled.energy(&spins).unwrap()
        );
        let a = scaled.anneal_ell().expect("scaled tables");
        assert!((a.weak_scale - 1.5).abs() < f64::EPSILON);
        let (mut weak_seen, mut strong_seen) = (0, 0);
        for (exact, anneal) in [
            (&stock.layout().all_ell, &a.all_ell),
            (&stock.layout().weak_ell, &a.weak_ell),
        ] {
            assert_eq!(exact.len(), anneal.len());
            for (&e, &s) in exact.iter().zip(anneal.iter()) {
                assert_eq!(e as u32 >> 16, s as u32 >> 16, "same neighbour");
                let (je, js) = (coupling(e), coupling(s));
                if je != 0 && je.abs() < WEAK_LIMIT_MILLI {
                    assert_eq!(js, scale_weak(je, 1.5));
                    assert_eq!(je.abs(), 45);
                    assert_eq!(js.abs(), 68);
                    weak_seen += 1;
                } else {
                    assert_eq!(
                        js, je,
                        "strong couplings and soft-weight residuals are kept"
                    );
                    strong_seen += usize::from(je != 0);
                }
            }
        }
        // Five weak edges, both directions, in both tables.
        assert_eq!(weak_seen, 20);
        assert!(strong_seen > 0);
    }

    #[test]
    fn pair_payload_lengths_and_truncation_are_checked() {
        let mut region = vec![1, 0, 3, 0, 12, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0];
        assert_eq!(metadata_pairs(&region), Some(vec![(1, 3)]));
        region[8] = 2;
        assert_eq!(metadata_pairs(&region), None);
        for end in 0..region.len() {
            assert_eq!(metadata_pairs(&region[..end]), None);
        }
    }
}
