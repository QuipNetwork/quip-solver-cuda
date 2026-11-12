//! The K4 tiles of a job: from `quip/charged-ice@1` metadata, or recovered
//! from the coupling graph of a plain job.

use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{CancelToken, IsingGraph};

use super::greedy::Ising;
use super::milli;

/// The metadata section that lists the tiles, in every ice family.
pub const TILES_SECTION: u16 = 1;

/// A kind 1 tile section: a `u32` count, then four `u32` positions per tile,
/// each tile ascending and the tiles ascending, all inside the model
/// (quip-solver-cpu 10e8029, src/postprocess/tile_worm_sampler.rs:139-160).
#[must_use]
pub fn parse_tiles(payload: &[u8], n: usize) -> Option<Vec<[u32; 4]>> {
    let (count, rest) = payload.split_first_chunk::<4>()?;
    let count = usize::try_from(u32::from_le_bytes(*count)).ok()?;
    let (rows, tail) = rest.as_chunks::<16>();
    if rows.len() != count || !tail.is_empty() {
        return None;
    }
    let mut tiles: Vec<[u32; 4]> = Vec::with_capacity(count);
    for row in rows {
        let at = |k: usize| {
            u32::from_le_bytes([row[4 * k], row[4 * k + 1], row[4 * k + 2], row[4 * k + 3]])
        };
        let tile = [at(0), at(1), at(2), at(3)];
        let ascending = tile.windows(2).all(|w| w[0] < w[1]);
        if !ascending
            || usize::try_from(tile[3]).ok()? >= n
            || tiles.last().is_some_and(|prev| *prev >= tile)
        {
            return None;
        }
        tiles.push(tile);
    }
    Some(tiles)
}

/// The tiles of a family instance, checked against the job graph
/// (quip-solver-cpu 10e8029, src/postprocess/tile_worm_sampler.rs:107-137).
///
/// # Errors
///
/// A description of the first check that fails: the instance does not fit
/// the graph, the coefficients differ, or the tile section is absent or
/// malformed.
pub fn instance_tiles(graph: &IsingGraph, instance: &Instance) -> Result<Vec<[u32; 4]>, String> {
    let n = graph.h.len();
    if instance.node_ids().len() != n || instance.edges().len() != graph.edges.len() {
        return Err("the instance does not fit its job graph".to_owned());
    }
    let same = |c: &f64, m: i32| milli(*c) == Some(i64::from(m));
    if !graph.h.iter().zip(instance.h()).all(|(c, &m)| same(c, m))
        || !graph.j.iter().zip(instance.j()).all(|(c, &m)| same(c, m))
        || !graph
            .edges
            .iter()
            .zip(instance.edges())
            .all(|(&(u, v), &(a, b))| u32::try_from(u) == Ok(a) && u32::try_from(v) == Ok(b))
    {
        return Err("the job coefficients differ from the instance".to_owned());
    }
    let payload = crate::families::flow_loops::section(instance.metadata(), TILES_SECTION)
        .ok_or_else(|| "the instance has no tile section".to_owned())?;
    parse_tiles(payload, n).ok_or_else(|| "the tile section is malformed".to_owned())
}

/// The largest degree at which [`recover_tiles`] runs. Charged ice has
/// degree at most 20. A denser model has no ice tiles to recover, and its K4
/// enumeration grows with the cube of the degree.
pub const RECOVER_MAX_DEGREE: usize = 32;

/// The coupling lookups [`recover_tiles`] may spend per edge.
pub const RECOVER_LOOKUPS_PER_EDGE: usize = 64;

/// Why [`recover_tiles`] returned no tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoverStop {
    /// A node has more than [`RECOVER_MAX_DEGREE`] neighbours.
    Degree {
        /// The largest degree.
        max: usize,
    },
    /// The enumeration used more than [`RECOVER_LOOKUPS_PER_EDGE`] lookups
    /// per edge.
    Budget,
    /// The cancel token abandoned the watermark.
    Cancelled,
}

impl std::fmt::Display for RecoverStop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Degree { max } => write!(
                f,
                "degree {max} is above the tile recovery limit of {RECOVER_MAX_DEGREE}"
            ),
            Self::Budget => write!(
                f,
                "tile recovery passed its budget of {RECOVER_LOOKUPS_PER_EDGE} lookups per edge"
            ),
            Self::Cancelled => write!(f, "tile recovery was cancelled"),
        }
    }
}

/// Recover the tiles of a plain job from its couplings.
///
/// A tile is a K4 clique whose six couplings share one nonzero magnitude and
/// whose signs are gauge-equivalent to an antiferromagnetic K4, the test of
/// [`IceTiles::new`](super::loop_descent::IceTiles::new). The cliques are
/// visited in ascending order of their sorted node tuples, the order of a
/// kind 1 section. A clique is kept when none of its nodes already has two
/// tiles. The visit order is lexicographic, so the filter runs as the
/// cliques are found, and a node with two tiles is not expanded further.
///
/// # Errors
///
/// [`RecoverStop`] when the model is too dense, the lookup budget runs out,
/// or the token abandons the watermark. The token is checked once per node.
pub fn recover_tiles(
    ising: &Ising,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<[u32; 4]>, RecoverStop> {
    let n = ising.num_nodes();
    let max = (0..n).map(|u| ising.row(u).len()).max().unwrap_or(0);
    if max > RECOVER_MAX_DEGREE {
        return Err(RecoverStop::Degree { max });
    }
    let edges: usize = (0..n).map(|u| ising.row(u).len()).sum::<usize>() / 2;
    let mut budget = edges.saturating_mul(RECOVER_LOOKUPS_PER_EDGE).max(1);
    let mut lookup = |u: u32, v: u32| {
        budget = budget.checked_sub(1).ok_or(RecoverStop::Budget)?;
        Ok(ising.coupling(u, v))
    };
    let mut slots = vec![0_u8; n];
    let mut tiles = Vec::new();
    for a in 0..n {
        if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
            return Err(RecoverStop::Cancelled);
        }
        let row = ising.row(a);
        let ra = &row[row.partition_point(|&(v, _)| (v as usize) <= a)..];
        let a32 = u32::try_from(a).map_err(|_| RecoverStop::Budget)?;
        for (ib, &(b, jab)) in ra.iter().enumerate() {
            if slots[a] >= 2 {
                break;
            }
            if jab == 0 || slots[b as usize] >= 2 {
                continue;
            }
            let mag = jab.abs();
            let gb = jab.signum();
            for (ic, &(c, jac)) in ra.iter().enumerate().skip(ib + 1) {
                if slots[a] >= 2 || slots[b as usize] >= 2 {
                    break;
                }
                if jac.abs() != mag || slots[c as usize] >= 2 {
                    continue;
                }
                // Gauge with g_a = +1: g_b and g_c follow from the a-row.
                let gc = jac.signum();
                match lookup(b, c)? {
                    Some(jbc) if jbc.abs() == mag && jbc.signum() == gb * gc => {}
                    _ => continue,
                }
                for &(d, jad) in &ra[ic + 1..] {
                    if jad.abs() != mag || slots[d as usize] >= 2 {
                        continue;
                    }
                    let gd = jad.signum();
                    let fits = |j: Option<i64>, g: i64| {
                        j.is_some_and(|j| j.abs() == mag && j.signum() == g)
                    };
                    if !fits(lookup(b, d)?, gb * gd) || !fits(lookup(c, d)?, gc * gd) {
                        continue;
                    }
                    let tile = [a32, b, c, d];
                    for &u in &tile {
                        slots[u as usize] += 1;
                    }
                    tiles.push(tile);
                    if slots[a] >= 2 || slots[b as usize] >= 2 || slots[c as usize] >= 2 {
                        break;
                    }
                }
            }
        }
    }
    Ok(tiles)
}
