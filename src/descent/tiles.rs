//! The K4 tiles of a job: from `quip/charged-ice@1` metadata, or recovered
//! from the coupling graph of a plain job.

use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::IsingGraph;

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

/// Recover the tiles of a plain job from its couplings.
///
/// A tile is a K4 clique whose six couplings share one nonzero magnitude and
/// whose signs are gauge-equivalent to an antiferromagnetic K4, the test of
/// [`IceTiles::new`](super::loop_descent::IceTiles::new). The cliques are
/// visited in ascending order of their sorted node tuples, the order of a
/// kind 1 section. A clique is kept when none of its nodes already has two
/// tiles.
#[must_use]
pub fn recover_tiles(ising: &Ising) -> Vec<[u32; 4]> {
    let n = ising.num_nodes();
    let mut found: Vec<[u32; 4]> = Vec::new();
    for a in 0..n {
        let up = |u: usize| {
            let row = ising.row(u);
            let at = row.partition_point(|&(v, _)| (v as usize) <= u);
            &row[at..]
        };
        let ra = up(a);
        for (ib, &(b, jab)) in ra.iter().enumerate() {
            if jab == 0 {
                continue;
            }
            let mag = jab.abs();
            for (ic, &(c, jac)) in ra.iter().enumerate().skip(ib + 1) {
                if jac.abs() != mag {
                    continue;
                }
                let Some(jbc) = ising.coupling(b, c) else {
                    continue;
                };
                if jbc.abs() != mag {
                    continue;
                }
                for &(d, jad) in &ra[ic + 1..] {
                    if jad.abs() != mag {
                        continue;
                    }
                    let (Some(jbd), Some(jcd)) = (ising.coupling(b, d), ising.coupling(c, d))
                    else {
                        continue;
                    };
                    if jbd.abs() != mag || jcd.abs() != mag {
                        continue;
                    }
                    // Gauge with g_a = +1: g_b, g_c, g_d follow from the a-row.
                    let (gb, gc, gd) = (jab.signum(), jac.signum(), jad.signum());
                    if jbc.signum() == gb * gc && jbd.signum() == gb * gd && jcd.signum() == gc * gd
                    {
                        let a = u32::try_from(a).unwrap_or(u32::MAX);
                        found.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    found.sort_unstable();
    let mut slots = vec![0_u8; n];
    let mut tiles = Vec::new();
    for tile in found {
        if tile.iter().any(|&u| slots[u as usize] >= 2) {
            continue;
        }
        for &u in &tile {
            slots[u as usize] += 1;
        }
        tiles.push(tile);
    }
    tiles
}
