//! CPU pair-worm postprocess (upstream `ice-pairs-pw/1`) applied to the best reads of a GPU run.

use quip_solver_cpu::postprocess::greedy::{natural_moves, Greedy, Ising, MoveSources};
use quip_solver_cpu::postprocess::pair_worm::{schedule_pw, PairWorm, Work};

/// A prepared postprocess for one instance.
pub struct PairWormPolish {
    greedy: Greedy,
    pw: PairWorm,
}

impl PairWormPolish {
    /// Build from the milli-coefficients of the instance and its tiles and pairs (original node order).
    #[must_use]
    pub fn new(
        h: &[i64],
        edges: &[(u32, u32)],
        j: &[i64],
        tiles: &[[u32; 4]],
        pairs: &[(u32, u32)],
    ) -> Option<Self> {
        let ising = Ising::new(h, edges, j).ok()?;
        let sources = MoveSources { tiles, pairs, listed: &[] };
        let (moves, _) = natural_moves(&ising, &sources).ok()?;
        let pw = PairWorm::new(&ising, tiles, pairs).ok()?;
        let greedy = Greedy::new(ising, &moves, true).ok()?;
        Some(Self { greedy, pw })
    }

    /// Run the schedule on `reads` (best first or any order). Returns the final state and its energy (milli).
    #[must_use]
    pub fn polish(&self, reads: &[Vec<i8>], seed: u64, work: Work) -> Option<(Vec<i8>, i64)> {
        let out = schedule_pw(&self.greedy, &self.pw, reads, seed, work, None).ok()?;
        Some((out.state, out.report.energy))
    }
}

pub use quip_solver_cpu::postprocess::pair_worm::Work as PwWork;
