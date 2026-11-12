//! Host descent after the MSA kernel, the `msa-nat-loop/1` policy of
//! `quip-cuda-msa-descent`.
//!
//! The policy matches the archived research pipeline
//! (new-pow/gpu/scripts/campaign_greedy.py:243-273):
//!
//! 1. Natural-move greedy descent on every raw read ([`greedy`]).
//! 2. The alternating natural-move and worm schedule ([`loop_descent`]) on
//!    the read with the lowest raw energy, the first on ties, seeded from the
//!    job seed.
//! 3. The loop state replaces the greedy state of that read when it is
//!    strictly lower, so the best returned energy is the greedy best unless
//!    the loop result is strictly lower.
//!
//! Every returned state is rescored exactly on the job graph. The tiles come
//! from `quip/charged-ice@1` metadata or are recovered from the couplings
//! ([`tiles`]).

pub mod greedy;
pub mod loop_descent;
mod sampler;
pub mod tiles;

pub use sampler::{family_tiles, MsaDescentSampler, MSA_DESCENT_SOLVER_ID};

use quip_solver_core::{CancelToken, IsingGraph};

use greedy::{natural_moves, DescentError, Greedy, Ising, MoveCounts, MoveSources};
use loop_descent::{nat_loop, IceTiles, LoopLimits, TileCounts, ROUNDS};

/// The policy id written to the job log.
pub const POLICY_ID: &str = "msa-nat-loop/1";

/// The exact milli value of a coefficient, or `None` when it is not a whole
/// number of milli inside the `i32` wire range.
#[must_use]
pub fn milli(c: f64) -> Option<i64> {
    let scaled = (c * 1000.0).round();
    if !scaled.is_finite() || scaled.abs() > f64::from(i32::MAX) {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is a whole number inside the i32 range"
    )]
    let m = scaled as i32;
    #[expect(
        clippy::float_cmp,
        reason = "a wire coefficient is exactly the decoded milli value over 1000"
    )]
    let exact = f64::from(m) / 1000.0 == c;
    exact.then_some(i64::from(m))
}

/// The integer milli model of a job graph.
///
/// # Errors
///
/// A description when a coefficient is not exact milli, a node index does
/// not fit `u32`, or the model fails [`Ising::new`].
pub fn job_ising(graph: &IsingGraph) -> Result<Ising, String> {
    let convert = |values: &[f64]| {
        values
            .iter()
            .map(|&c| milli(c).ok_or_else(|| format!("coefficient {c} is not exact milli")))
            .collect::<Result<Vec<i64>, String>>()
    };
    let h = convert(&graph.h)?;
    let j = convert(&graph.j)?;
    let edges = graph
        .edges
        .iter()
        .map(|&(u, v)| match (u32::try_from(u), u32::try_from(v)) {
            (Ok(u), Ok(v)) => Ok((u, v)),
            _ => Err("a node index does not fit u32".to_owned()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ising::new(&h, &edges, &j).map_err(|e| e.to_string())
}

/// Where the tiles of a job came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileSource {
    /// The kind 1 section of a validated `quip/charged-ice@1` instance.
    Metadata,
    /// Recovered from the couplings by [`tiles::recover_tiles`].
    Recovered,
}

impl TileSource {
    /// The log name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Recovered => "recovered",
        }
    }
}

/// The move list and tiles of one job.
#[derive(Clone, Debug)]
pub struct Plan {
    greedy: Greedy,
    tiles: IceTiles,
    source: TileSource,
    moves: MoveCounts,
}

impl Plan {
    /// Build the natural moves and the worm tiles of `graph`. With
    /// `metadata` the tiles are those of the family instance. Without it they
    /// are recovered from the couplings. The tile subsets of every candidate
    /// join the move list, as the research `tetrahedra` rows do
    /// (descent.py:126-129).
    ///
    /// # Errors
    ///
    /// A description when the graph has no exact integer model or a tile
    /// names a node outside it.
    pub fn new(graph: &IsingGraph, metadata: Option<Vec<[u32; 4]>>) -> Result<Self, String> {
        let ising = job_ising(graph)?;
        let (candidates, source) = match metadata {
            Some(tiles) => (tiles, TileSource::Metadata),
            None => (tiles::recover_tiles(&ising), TileSource::Recovered),
        };
        let tiles = IceTiles::new(&ising, &candidates).map_err(|e| e.to_string())?;
        let sources = MoveSources {
            tiles: &candidates,
            pairs: &[],
            listed: &[],
        };
        let (moves, counts) = natural_moves(&ising, &sources).map_err(|e| e.to_string())?;
        let greedy = Greedy::new(ising, &moves, true).map_err(|e| e.to_string())?;
        Ok(Self {
            greedy,
            tiles,
            source,
            moves: counts,
        })
    }

    /// The integer model.
    #[must_use]
    pub fn ising(&self) -> &Ising {
        self.greedy.ising()
    }

    /// The greedy move list.
    #[must_use]
    pub const fn greedy(&self) -> &Greedy {
        &self.greedy
    }

    /// The worm tiles.
    #[must_use]
    pub const fn tiles(&self) -> &IceTiles {
        &self.tiles
    }

    /// Where the tiles came from.
    #[must_use]
    pub const fn source(&self) -> TileSource {
        self.source
    }

    /// The natural move counts.
    #[must_use]
    pub const fn moves(&self) -> MoveCounts {
        self.moves
    }

    /// The tile filter counts.
    #[must_use]
    pub const fn tile_counts(&self) -> TileCounts {
        self.tiles.counts()
    }
}

/// The energies of one descended job, in milli.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    /// The lowest raw kernel energy.
    pub raw_best: i64,
    /// The lowest energy after greedy descent of every read.
    pub greedy_best: i64,
    /// The energy of the loop result, `None` when the model has no tiles.
    pub loop_best: Option<i64>,
    /// The read the loop started from: the lowest raw read, first on ties.
    pub loop_from: Option<usize>,
    /// The rounds the alternating schedule ran.
    pub loop_rounds: u32,
    /// The lowest returned energy.
    pub final_best: i64,
}

/// One descended job: the states in read order and the energies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descended {
    /// The returned states, one per raw read, in read order.
    pub reads: Vec<Vec<i8>>,
    /// The greedy energy of each read, in read order.
    pub greedy_energies: Vec<i64>,
    /// The loop state, `None` when the model has no tiles.
    pub loop_state: Option<Vec<i8>>,
    /// The energies.
    pub report: Report,
}

/// Greedy descent of each read on up to `threads` scoped threads. Each read
/// descends alone, so the result does not depend on the thread count.
fn greedy_all(
    greedy: &Greedy,
    raw: &[Vec<i8>],
    threads: usize,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<Vec<i8>>, DescentError> {
    let one = |s: &Vec<i8>| {
        let mut s = s.clone();
        greedy.descend(&mut s, cancel).map(|_| s)
    };
    let threads = threads.clamp(1, raw.len().max(1));
    if threads == 1 {
        return raw.iter().map(one).collect();
    }
    let chunk = raw.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = raw
            .chunks(chunk)
            .map(|c| scope.spawn(move || c.iter().map(one).collect::<Result<Vec<_>, _>>()))
            .collect();
        let mut out = Vec::with_capacity(raw.len());
        for handle in handles {
            match handle.join() {
                Ok(part) => out.extend(part?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(out)
    })
}

/// Run the `msa-nat-loop/1` policy on the raw reads of one job.
///
/// # Errors
///
/// [`DescentError`] when a read does not fit the model or the token abandons
/// the watermark, and [`DescentError::Empty`] when there are no reads.
pub fn descend(
    plan: &Plan,
    raw: &[Vec<i8>],
    seed: u64,
    threads: usize,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Descended, DescentError> {
    let ising = plan.ising();
    for s in raw {
        plan.greedy.check(s)?;
    }
    let raw_energies: Vec<i64> = raw.iter().map(|s| ising.energy(s)).collect();
    let (from, &raw_best) = raw_energies
        .iter()
        .enumerate()
        .min_by_key(|&(i, &e)| (e, i))
        .ok_or(DescentError::Empty)?;
    let mut reads = greedy_all(&plan.greedy, raw, threads, cancel)?;
    let greedy_energies: Vec<i64> = reads.iter().map(|s| ising.energy(s)).collect();
    let greedy_best = greedy_energies.iter().copied().min().unwrap_or(raw_best);
    let mut loop_state = None;
    let mut loop_best = None;
    let mut loop_rounds = 0;
    if !plan.tiles.tiled().is_empty() {
        let mut one = vec![raw[from].clone()];
        loop_rounds = nat_loop(
            &plan.greedy,
            &plan.tiles,
            &mut one,
            LoopLimits::default(),
            seed,
            ROUNDS,
            cancel,
        )?;
        let state = one.remove(0);
        let energy = ising.energy(&state);
        if energy < greedy_energies[from] {
            reads[from].clone_from(&state);
        }
        loop_best = Some(energy);
        loop_state = Some(state);
    }
    let final_best = loop_best.map_or(greedy_best, |e| e.min(greedy_best));
    Ok(Descended {
        reads,
        greedy_energies,
        loop_state,
        report: Report {
            raw_best,
            greedy_best,
            loop_best,
            loop_from: loop_best.map(|_| from),
            loop_rounds,
            final_best,
        },
    })
}
