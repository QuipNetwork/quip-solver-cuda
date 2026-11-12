//! Zero-temperature worm (loop) descent on gauged K4 tiles, and the
//! alternating natural-move and worm schedule.
//!
//! Port of the research code in new-pow/gpu/scripts: `loop_descent` in
//! greedy.c:75-228, `ice_tiles` in descent.py:271-308, and `nat_loop` in
//! descent.py:337-350. The port keeps the random stream draw for draw, so a
//! seed gives the same states as the research library.
//!
//! In the gauge frame `sigma = g s` a tile has charge `q = sum sigma`. A worm
//! flips a tiled spin `u0`. Then each charged end tile flips a corner
//! `y != enter` with `sigma(y) = sign(q)`, which moves the charge to the other
//! tile of `y`, until the head charge is 0 or `y` has no other tile. Among the
//! candidates the leg takes the one with the lowest flip delta, and ties go to
//! a reservoir draw. A closed worm is kept when its exact energy change is
//! below zero, or equal to zero while the read's zero-move budget lasts.
//! Otherwise it is undone. Sweeps visit every tiled spin in a fresh random
//! order and stop after a sweep with no strictly improving worm, or at the
//! sweep limit.

use quip_solver_core::CancelToken;

use super::greedy::{DescentError, Greedy, Ising, MoveError};

/// The longest worm path, in flips (descent.py:320).
pub const MAX_LEN: usize = 8192;
/// The sweep limit of one worm descent (descent.py:319).
pub const MAX_SWEEPS: u32 = 50;
/// The round limit of the alternating schedule (descent.py:338).
pub const ROUNDS: u32 = 10;
/// The zero-move flip budget of each read in the archived run
/// (`campaign_greedy.py:260`).
pub const ZERO_BUDGET: i64 = 0;
/// The state a zero seed maps to (greedy.c:107).
const ZERO_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// How the candidate tiles were filtered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileCounts {
    /// Candidates that passed the coupling and gauge test.
    pub tiles: usize,
    /// Candidates that failed it.
    pub dropped: usize,
    /// Tile memberships dropped because the spin already had two tiles.
    pub spins_over_two: usize,
}

/// The gauged K4 tiles of a model, ready for the worm.
#[derive(Clone, Debug)]
pub struct IceTiles {
    tiles: Vec<[u32; 4]>,
    gauge: Vec<[i8; 4]>,
    /// The first two tiles of each spin, in tile order.
    mtile: Vec<[Option<u32>; 2]>,
    /// The spins with at least one tile, in node order.
    tiled: Vec<u32>,
    counts: TileCounts,
}

/// The six corner pairs of a tile, in `itertools.combinations` order.
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn sign(x: i64) -> i8 {
    match x.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

impl IceTiles {
    /// Keep each candidate whose six couplings exist and whose signs are
    /// gauge-equivalent to an antiferromagnetic K4: `sign(J_ab) = g_a g_b`
    /// with `g = [1, sign(J_01), sign(J_02), sign(J_03)]`. Each spin keeps
    /// its first two tiles in candidate order (descent.py:271-308).
    ///
    /// # Errors
    ///
    /// [`MoveError::Node`] when a candidate names a node outside the model.
    pub fn new(ising: &Ising, candidates: &[[u32; 4]]) -> Result<Self, MoveError> {
        let n = ising.num_nodes();
        let mut tiles = Vec::new();
        let mut gauge = Vec::new();
        let mut dropped = 0;
        for tile in candidates {
            if let Some(&node) = tile.iter().find(|&&u| u as usize >= n) {
                return Err(MoveError::Node { node });
            }
            let js: Option<Vec<i64>> = PAIRS
                .iter()
                .map(|&(a, b)| ising.coupling(tile[a], tile[b]))
                .collect();
            let Some(js) = js else {
                dropped += 1;
                continue;
            };
            let g = [1, sign(js[0]), sign(js[1]), sign(js[2])];
            let ok = PAIRS
                .iter()
                .zip(&js)
                .all(|(&(a, b), &j)| sign(j) == g[a] * g[b]);
            if ok {
                tiles.push(*tile);
                gauge.push(g);
            } else {
                dropped += 1;
            }
        }
        let mut mtile = vec![[None; 2]; n];
        let mut over = 0;
        for (t, tile) in tiles.iter().enumerate() {
            let t = u32::try_from(t).map_err(|_| MoveError::Node { node: u32::MAX })?;
            for &u in tile {
                let slots = &mut mtile[u as usize];
                if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
                    *slot = Some(t);
                } else {
                    over += 1;
                }
            }
        }
        let tiled_spins = (0..n)
            .filter(|&u| mtile[u][0].is_some())
            .map(|u| u32::try_from(u).map_err(|_| MoveError::Node { node: u32::MAX }))
            .collect::<Result<_, _>>()?;
        let counts = TileCounts {
            tiles: tiles.len(),
            dropped,
            spins_over_two: over,
        };
        Ok(Self {
            tiles,
            gauge,
            mtile,
            tiled: tiled_spins,
            counts,
        })
    }

    /// The kept tiles, in candidate order.
    #[must_use]
    pub fn tiles(&self) -> &[[u32; 4]] {
        &self.tiles
    }

    /// The spins with at least one tile, in node order.
    #[must_use]
    pub fn tiled(&self) -> &[u32] {
        &self.tiled
    }

    /// The filter counts.
    #[must_use]
    pub const fn counts(&self) -> TileCounts {
        self.counts
    }

    fn charge(&self, s: &[i8], t: usize) -> i32 {
        self.tiles[t]
            .iter()
            .zip(&self.gauge[t])
            .map(|(&u, &g)| i32::from(g) * i32::from(s[u as usize]))
            .sum()
    }
}

/// The research xorshift64 step (greedy.c:75-81).
fn xs64(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Flip spin `u` and update the local fields.
fn flip(ising: &Ising, s: &mut [i8], f: &mut [i64], u: usize) {
    let old = i64::from(s[u]);
    s[u] = -s[u];
    for &(v, j) in ising.row(u) {
        f[v as usize] -= 2 * j * old;
    }
}

/// The limits of one worm descent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopLimits {
    /// The longest worm path, in flips.
    pub max_len: usize,
    /// The sweep limit.
    pub max_sweeps: u32,
    /// The zero-move flip budget of each read.
    pub zero_budget: i64,
}

impl Default for LoopLimits {
    fn default() -> Self {
        Self {
            max_len: MAX_LEN,
            max_sweeps: MAX_SWEEPS,
            zero_budget: ZERO_BUDGET,
        }
    }
}

/// One read under the worm: its spins, local fields, and the walk path.
struct Walk<'a> {
    ising: &'a Ising,
    tiles: &'a IceTiles,
    spins: &'a mut [i8],
    fields: Vec<i64>,
    path: Vec<usize>,
    max_len: usize,
}

impl Walk<'_> {
    fn flip(&mut self, u: usize) {
        flip(self.ising, self.spins, &mut self.fields, u);
    }

    /// The flip delta of spin `u`.
    fn delta(&self, u: usize) -> i64 {
        -2 * i64::from(self.spins[u]) * self.fields[u]
    }

    /// Walk one leg from `head`, entered through `start`. Adds each flip
    /// delta to `cum` and returns whether the leg closed (greedy.c:157-201).
    fn leg(&mut self, mut head: u32, start: usize, cum: &mut i64, rng: &mut u64) -> bool {
        let mut enter = start;
        loop {
            let charge = self.tiles.charge(self.spins, head as usize);
            if charge == 0 {
                return true;
            }
            if self.path.len() >= self.max_len {
                return false;
            }
            let want = if charge > 0 { 1 } else { -1 };
            let mut pick: Option<usize> = None;
            let mut best = 0;
            let mut ties = 0_u64;
            let corners = self.tiles.tiles[head as usize];
            let gauges = self.tiles.gauge[head as usize];
            for (&corner, &gauge) in corners.iter().zip(&gauges) {
                let y = corner as usize;
                if y == enter || i32::from(gauge) * i32::from(self.spins[y]) != want {
                    continue;
                }
                let d = self.delta(y);
                if pick.is_none() || d < best {
                    pick = Some(y);
                    best = d;
                    ties = 1;
                } else if d == best {
                    ties += 1;
                    if xs64(rng).is_multiple_of(ties) {
                        pick = Some(y);
                    }
                }
            }
            let Some(y) = pick else {
                return false;
            };
            *cum += best;
            self.flip(y);
            self.path.push(y);
            let [first, second] = self.tiles.mtile[y];
            let other = if first == Some(head) { second } else { first };
            let Some(other) = other else {
                return true;
            };
            head = other;
            enter = y;
        }
    }

    /// Flip `start`, walk both legs, and return whether the worm closed and
    /// its exact energy change (greedy.c:140-203).
    fn worm(&mut self, start: usize, rng: &mut u64) -> (bool, i64) {
        let mut cum = self.delta(start);
        self.flip(start);
        self.path.clear();
        self.path.push(start);
        let mut ends = self.tiles.mtile[start];
        if ends[1].is_some() && xs64(rng) & 1 == 1 {
            ends.swap(0, 1);
        }
        for head in ends.into_iter().flatten() {
            if !self.leg(head, start, &mut cum, rng) {
                return (false, cum);
            }
        }
        (true, cum)
    }

    /// Undo the last worm, in reverse order.
    fn undo(&mut self) {
        while let Some(u) = self.path.pop() {
            self.flip(u);
        }
    }
}

/// Run the worm descent on each read in place, one random stream across the
/// reads, and return the number of kept worms (greedy.c:102-228).
///
/// # Errors
///
/// [`DescentError::Length`] or [`DescentError::Value`] when a read does not
/// fit the model, and [`DescentError::Cancelled`] when the token abandons
/// the watermark. The token is checked before each sweep.
pub fn loop_descent(
    ising: &Ising,
    tiles: &IceTiles,
    reads: &mut [Vec<i8>],
    limits: LoopLimits,
    seed: u64,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<u64, DescentError> {
    let nodes = ising.num_nodes();
    let mut rng = if seed == 0 { ZERO_SEED } else { seed };
    let mut accepted = 0;
    let mut order: Vec<u32> = Vec::with_capacity(tiles.tiled.len());
    for spins in reads.iter_mut() {
        if spins.len() != nodes {
            return Err(DescentError::Length);
        }
        if let Some(node) = spins.iter().position(|&x| x != 1 && x != -1) {
            return Err(DescentError::Value { node });
        }
        let fields = (0..nodes).map(|i| ising.field(spins, i)).collect();
        let mut walk = Walk {
            ising,
            tiles,
            spins,
            fields,
            path: Vec::with_capacity(limits.max_len.max(1)),
            max_len: limits.max_len,
        };
        let mut budget = limits.zero_budget;
        // The order is not reset between sweeps (greedy.c:126-138).
        order.clear();
        order.extend_from_slice(&tiles.tiled);
        for _ in 0..limits.max_sweeps {
            if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
                return Err(DescentError::Cancelled);
            }
            let mut improved = false;
            for i in (1..order.len()).rev() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the remainder is at most i, a usize"
                )]
                let k = (xs64(&mut rng) % (i as u64 + 1)) as usize;
                order.swap(i, k);
            }
            for &start in &order {
                let (closed, cum) = walk.worm(start as usize, &mut rng);
                let len = i64::try_from(walk.path.len()).unwrap_or(i64::MAX);
                if closed && (cum < 0 || (cum == 0 && budget >= len)) {
                    accepted += 1;
                    if cum < 0 {
                        improved = true;
                    } else {
                        budget -= len;
                    }
                } else {
                    walk.undo();
                }
            }
            if !improved {
                break;
            }
        }
    }
    Ok(accepted)
}

/// Alternate natural-move descent and worm descent until the energies of all
/// reads stop changing, for at most `rounds` rounds (descent.py:337-350).
/// Round `r` seeds the worm with `seed + r`. Returns the rounds run.
///
/// # Errors
///
/// [`DescentError`] when a read does not fit the model or the token abandons
/// the watermark.
pub fn nat_loop(
    greedy: &Greedy,
    tiles: &IceTiles,
    reads: &mut [Vec<i8>],
    limits: LoopLimits,
    seed: u64,
    rounds: u32,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<u32, DescentError> {
    let ising = greedy.ising();
    for s in reads.iter_mut() {
        greedy.descend(s, cancel)?;
    }
    let mut energies: Vec<i64> = reads.iter().map(|s| ising.energy(s)).collect();
    let mut done = 0;
    for round in 1..=rounds {
        done = round;
        loop_descent(
            ising,
            tiles,
            reads,
            limits,
            seed.wrapping_add(u64::from(round)),
            cancel,
        )?;
        for s in reads.iter_mut() {
            greedy.descend(s, cancel)?;
        }
        let next: Vec<i64> = reads.iter().map(|s| ising.energy(s)).collect();
        if next == energies {
            break;
        }
        energies = next;
    }
    Ok(done)
}
