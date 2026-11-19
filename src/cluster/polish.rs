//! Local build (`BudFox`, 2026-11-14): host CPU polish of a family lease job's
//! reads after the GPU returns them, on a dedicated thread pool.
//!
//! Off by default: with `QUIP_OTSUKA_CPU_POLISH` unset (or 0) the sampler
//! builds no pool and every result goes out exactly as before.
//!
//! On, a lane hands each completed lease result to the pool and moves on to
//! its next GPU job. A pool thread:
//!
//! 1. (level 1+) runs a single-spin greedy descent of every read. The GPU's
//!    final descent does not always reach a single-flip minimum: a read can
//!    be a lineage best from a population checkpoint, taken before the
//!    descent.
//! 2. (level 2) continues the `TOPK` lowest reads with `SWEEPS` Otsuka
//!    cluster sweeps (the lane's tile and pair weights, one Metropolis sweep
//!    after each) at the fixed low temperature `1 / BETA`, keeps the lowest
//!    state visited, and descends it again.
//!
//! A read is replaced only by a strictly lower state, so no energy rises. The
//! energy of a replaced read is rescored from the job's own coefficients
//! (the same `energy_milli` the lane uses), so every reported energy is the
//! true energy of the returned spins. The read count and order are unchanged.
//!
//! When the pool's queue is full the lane sends the result unpolished at
//! once, so the GPU pipeline never waits on the CPU. A job whose watermark is
//! cancelled (round change) before or during its polish completes as
//! [`quip_solver_core::StreamOutcome::Cancelled`], the outcome the lane's own
//! `finish` gives a job cancelled while it ran. Only lease salts are
//! polished: an explicit job (a mempool order with a diversity gate) is
//! never changed.
//!
//! Measured (agent O, 2026-11-14) on our own deep on-chain dimer1200 proofs
//! (33 at -5065..-5080 units) and six 32-read proofs: level 1 adds +0.15
//! units on average, level 2 (beta 20, 1000 sweeps, top 4) another +0.33 to
//! +0.41, about 0.25 s of one core per job.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_CPU_POLISH` | 0 (off) | 1 = single-spin descent of every read; 2 = plus the low-temperature cluster continuation of the top reads. |
//! | `QUIP_OTSUKA_POLISH_THREADS` | 2 | Pool threads, `1..=32`. |
//! | `QUIP_OTSUKA_POLISH_TOPK` | 4 | Reads continued at level 2, `1..=256`. |
//! | `QUIP_OTSUKA_POLISH_SWEEPS` | 1000 | Cluster sweeps per continued read, `1..=1_000_000`. |
//! | `QUIP_OTSUKA_POLISH_BETA` | 20 | Inverse temperature of the continuation, per 1000 milli, `(0, 1000]`. |
//! | `QUIP_OTSUKA_POLISH_CUTOFF` | unset (all) | Polish only jobs whose best read is at or below this energy, in milli (for example `-5070000`). |
//! | `QUIP_OTSUKA_POLISH_QUEUE` | threads | Jobs that may wait for a pool thread, `1..=256`; a full queue sends unpolished. |

use super::{OtsukaGraph, PrepareError, Weights};
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::SamplerResult;
use std::sync::OnceLock;

/// Polish knobs. Present only when polish is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolishConfig {
    /// 1 = descent only; 2 = descent plus the cluster continuation.
    pub level: u8,
    /// Pool threads.
    pub threads: usize,
    /// Reads continued at level 2.
    pub topk: usize,
    /// Cluster sweeps per continued read.
    pub sweeps: usize,
    /// Inverse temperature of the continuation, per 1000 milli.
    pub beta: f64,
    /// Polish only jobs whose best read is at or below this, in milli.
    pub cutoff: Option<i64>,
    /// Jobs that may wait for a pool thread.
    pub queue: usize,
}

impl Default for PolishConfig {
    fn default() -> Self {
        Self {
            level: 2,
            threads: 2,
            topk: 4,
            sweeps: 1000,
            beta: 20.0,
            cutoff: None,
            queue: 2,
        }
    }
}

impl PolishConfig {
    /// Read the knobs through `get`. `None` (off) unless the level is 1 or 2.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let num = |name: &str, ok: &dyn Fn(f64) -> bool| -> Option<f64> {
            let raw = get(name)?;
            let parsed = raw
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && ok(*v));
            eprintln!(
                "quip-otsuka: {name}={raw:?} {}",
                if parsed.is_some() {
                    "applied"
                } else {
                    "ignored"
                }
            );
            parsed
        };
        let int = |v: f64| v.fract() == 0.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            let level = num("QUIP_OTSUKA_CPU_POLISH", &|v| {
                int(v) && (0.0..=2.0).contains(&v)
            })? as u8;
            if level == 0 {
                return None;
            }
            let d = Self::default();
            let threads = num("QUIP_OTSUKA_POLISH_THREADS", &|v| {
                int(v) && (1.0..=32.0).contains(&v)
            })
            .map_or(d.threads, |v| v as usize);
            let topk = num("QUIP_OTSUKA_POLISH_TOPK", &|v| {
                int(v) && (1.0..=256.0).contains(&v)
            })
            .map_or(d.topk, |v| v as usize);
            let sweeps = num("QUIP_OTSUKA_POLISH_SWEEPS", &|v| {
                int(v) && (1.0..=1_000_000.0).contains(&v)
            })
            .map_or(d.sweeps, |v| v as usize);
            let beta =
                num("QUIP_OTSUKA_POLISH_BETA", &|v| v > 0.0 && v <= 1000.0).unwrap_or(d.beta);
            let cutoff =
                num("QUIP_OTSUKA_POLISH_CUTOFF", &|v| int(v) && v.abs() < 9.0e15).map(|v| v as i64);
            let queue = num("QUIP_OTSUKA_POLISH_QUEUE", &|v| {
                int(v) && (1.0..=256.0).contains(&v)
            })
            .map_or(threads, |v| v as usize);
            Some(Self {
                level,
                threads,
                topk,
                sweeps,
                beta,
                cutoff,
                queue,
            })
        }
    }

    /// The process environment, read once.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        static ENV: OnceLock<Option<PolishConfig>> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }

    /// Whether a job whose best read is `best` should be polished.
    #[must_use]
    pub fn wants(&self, best: i64) -> bool {
        self.cutoff.is_none_or(|c| best <= c)
    }
}

/// xoshiro256** seeded through splitmix64: a small, fast host generator.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    #[allow(clippy::cast_precision_loss)]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Uniform in `0..n`, `n > 0`.
    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Adjacency in compressed rows.
#[derive(Debug)]
struct Rows {
    start: Vec<usize>,
    adj: Vec<(u32, i32)>,
}

impl Rows {
    fn build(n: usize, edges: &[(usize, usize, i32)]) -> Self {
        let mut start = vec![0; n + 1];
        for &(u, v, _) in edges {
            start[u + 1] += 1;
            start[v + 1] += 1;
        }
        for u in 0..n {
            start[u + 1] += start[u];
        }
        let mut fill = start.clone();
        let mut adj = vec![(0, 0); 2 * edges.len()];
        #[allow(clippy::cast_possible_truncation)]
        for &(u, v, j) in edges {
            adj[fill[u]] = (v as u32, j);
            fill[u] += 1;
            adj[fill[v]] = (u as u32, j);
            fill[v] += 1;
        }
        Self { start, adj }
    }

    #[inline]
    fn row(&self, u: usize) -> &[(u32, i32)] {
        &self.adj[self.start[u]..self.start[u + 1]]
    }
}

/// The host model a polish works on: every coupling for the local fields,
/// the strong motifs for the cluster draw, and the weak layer the components
/// are scored against.
#[derive(Debug)]
pub struct PolishGraph {
    n: usize,
    tiles: Vec<([usize; 4], [i8; 4])>,
    pairs: Vec<(usize, usize)>,
    weights: Weights,
    edges: Vec<(usize, usize, i32)>,
    all: Rows,
    weak_edges: Vec<(usize, usize, i32)>,
    weak: Rows,
}

impl PolishGraph {
    /// The polish model of a validated host graph.
    #[must_use]
    pub fn new(graph: &OtsukaGraph) -> Self {
        Self {
            n: graph.n,
            tiles: graph.tiles.iter().map(|t| (t.nodes, t.gauge)).collect(),
            pairs: graph.pairs.clone(),
            weights: graph.weights,
            edges: graph.edges.clone(),
            all: Rows::build(graph.n, &graph.edges),
            weak_edges: graph.weak_edges.clone(),
            weak: Rows::build(graph.n, &graph.weak_edges),
        }
    }

    /// The polish model of a canonical twisted-dimer instance, with the
    /// cluster draw at `weights`.
    ///
    /// # Errors
    /// Unsupported family, unreadable metadata, or a graph rejection.
    pub fn from_instance(
        instance: &Instance,
        contract: &Contract,
        weights: Weights,
    ) -> Result<Self, PrepareError> {
        Ok(Self::new(&super::prepare::host_graph(
            instance, contract, weights,
        )?))
    }

    /// The number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// The model energy of `s`, in milli (zero fields, no offset).
    #[must_use]
    pub fn energy(&self, s: &[i8]) -> i64 {
        self.edges
            .iter()
            .map(|&(u, v, j)| i64::from(j) * i64::from(s[u] * s[v]))
            .sum()
    }
}

/// Bond probabilities at one inverse temperature (Otsuka, PRB 90, 220406).
#[derive(Clone, Copy)]
struct Breakup {
    zero_ice: f64,
    single_ice: f64,
    zero_defect: f64,
    pair_link: f64,
}

impl Breakup {
    fn new(beta: f64, weights: Weights) -> Self {
        let tile_beta = beta * f64::from(weights.tile_milli) / 1000.0;
        let z = (-2.0 * tile_beta).exp();
        let w0 = z.powi(4);
        let w1 = (z - w0) / 3.0;
        Self {
            zero_ice: w0,
            single_ice: w0 + 4.0 * w1,
            zero_defect: (-6.0 * tile_beta).exp(),
            pair_link: -(-2.0 * beta * f64::from(weights.pair_milli) / 1000.0).exp_m1(),
        }
    }
}

/// A direct-mapped cache of `exp(-beta * delta / 1000)` at one beta.
struct Boltzmann {
    beta: f64,
    keys: [i64; 256],
    values: [f64; 256],
}

impl Boltzmann {
    fn new(beta: f64) -> Self {
        Self {
            beta,
            keys: [i64::MIN; 256],
            values: [0.0; 256],
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn get(&mut self, delta: i64) -> f64 {
        let slot = ((delta >> 1) as usize) % 256;
        if self.keys[slot] != delta {
            self.keys[slot] = delta;
            self.values[slot] = (-self.beta * delta as f64 / 1000.0).exp();
        }
        self.values[slot]
    }
}

/// Accept a move that costs `x = beta * delta` with probability `1 / (1 + e^x)`.
fn heat_bath(x: f64, rng: &mut Rng) -> bool {
    if x <= 0.0 {
        rng.unit() < 1.0 / (1.0 + x.exp())
    } else {
        let e = (-x).exp();
        rng.unit() < e / (1.0 + e)
    }
}

/// One spin state with its local fields and the cluster workspace.
struct Chain<'g> {
    g: &'g PolishGraph,
    s: Vec<i8>,
    field: Vec<i64>,
    parent: Vec<u32>,
    label: Vec<u32>,
    start: Vec<u32>,
    members: Vec<u32>,
    delta: Vec<i64>,
}

impl<'g> Chain<'g> {
    fn new(g: &'g PolishGraph, s: &[i8]) -> Self {
        let mut c = Self {
            g,
            s: s.to_vec(),
            field: vec![0; g.n],
            parent: vec![0; g.n],
            label: vec![0; g.n],
            start: Vec::with_capacity(g.n + 1),
            members: vec![0; g.n],
            delta: Vec::with_capacity(g.n),
        };
        c.set(s);
        c
    }

    fn set(&mut self, s: &[i8]) {
        self.s.copy_from_slice(s);
        for u in 0..self.g.n {
            self.field[u] = self
                .g
                .all
                .row(u)
                .iter()
                .map(|&(v, j)| i64::from(j) * i64::from(self.s[v as usize]))
                .sum();
        }
    }

    /// Flip `u`, patch its neighbours' fields, return the energy change.
    #[inline]
    fn flip(&mut self, u: usize) -> i64 {
        let twice = 2 * i64::from(self.s[u]);
        for &(v, j) in self.g.all.row(u) {
            self.field[v as usize] -= twice * i64::from(j);
        }
        self.s[u] = -self.s[u];
        -twice * self.field[u]
    }

    /// Single-spin greedy descent to a single-flip minimum. Returns the
    /// energy change (never positive).
    fn descend(&mut self) -> i64 {
        let mut change = 0;
        loop {
            let mut moved = false;
            for u in 0..self.g.n {
                if i64::from(self.s[u]) * self.field[u] > 0 {
                    change += self.flip(u);
                    moved = true;
                }
            }
            if !moved {
                return change;
            }
        }
    }

    /// One Metropolis sweep in node order. Returns the energy change.
    fn single_sweep(&mut self, cache: &mut Boltzmann, rng: &mut Rng) -> i64 {
        let mut change = 0;
        for u in 0..self.g.n {
            let delta = -2 * i64::from(self.s[u]) * self.field[u];
            if delta <= 0 || rng.unit() < cache.get(delta) {
                change += self.flip(u);
            }
        }
        change
    }

    fn find(&mut self, mut u: u32) -> u32 {
        loop {
            let up = self.parent[u as usize];
            if up == u {
                return u;
            }
            let grand = self.parent[up as usize];
            self.parent[u as usize] = grand;
            u = grand;
        }
    }

    fn join(&mut self, a: usize, b: usize) {
        #[allow(clippy::cast_possible_truncation)]
        let (a, b) = (self.find(a as u32), self.find(b as u32));
        if a < b {
            self.parent[b as usize] = a;
        } else if b < a {
            self.parent[a as usize] = b;
        }
    }

    /// Draw the tile and pair bonds, then list the components in the order
    /// of their smallest node.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names
    )]
    fn partition(&mut self, p: &Breakup, rng: &mut Rng) {
        for (u, parent) in self.parent.iter_mut().enumerate() {
            *parent = u as u32;
        }
        let g = self.g;
        for &(nodes, gauge) in &g.tiles {
            let mut up = [0; 4];
            let mut down = [0; 4];
            let (mut ups, mut downs) = (0, 0);
            for k in 0..4 {
                if self.s[nodes[k]] * gauge[k] > 0 {
                    up[ups] = nodes[k];
                    ups += 1;
                } else {
                    down[downs] = nodes[k];
                    downs += 1;
                }
            }
            if ups == 0 || ups == 4 {
                continue;
            }
            let x = rng.next_u64();
            let r = (x >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
            if ups == 2 {
                if r < p.zero_ice {
                    continue;
                }
                if r < p.single_ice {
                    self.join(up[(x & 1) as usize], down[((x >> 1) & 1) as usize]);
                } else {
                    let c = (x & 1) as usize;
                    self.join(up[0], down[c]);
                    self.join(up[1], down[1 - c]);
                }
            } else if r >= p.zero_defect {
                let a = up[rng.below(ups)];
                let b = down[rng.below(downs)];
                self.join(a, b);
            }
        }
        for &(u, v) in &g.pairs {
            if self.s[u] == self.s[v] && rng.unit() < p.pair_link {
                self.join(u, v);
            }
        }
        // parent[u] <= u and every root is its component's smallest node, so
        // one pass in node order labels each node from its (seen) parent.
        self.start.clear();
        self.start.push(0);
        for u in 0..self.parent.len() {
            let up = self.parent[u] as usize;
            let c = if up == u {
                self.start.push(0);
                (self.start.len() - 2) as u32
            } else {
                self.label[up]
            };
            self.label[u] = c;
            self.start[c as usize + 1] += 1;
        }
        for c in 1..self.start.len() {
            self.start[c] += self.start[c - 1];
        }
        let components = self.start.len() - 1;
        self.delta.clear();
        self.delta
            .extend(self.start[..components].iter().map(|&at| i64::from(at)));
        for u in 0..self.label.len() {
            let c = self.label[u] as usize;
            let at = self.delta[c] as usize;
            self.members[at] = u as u32;
            self.delta[c] += 1;
        }
    }

    /// One cluster sweep: bonds on the strong layer, then a heat-bath flip of
    /// each component against the weak layer. Returns the energy change.
    #[allow(clippy::cast_precision_loss)]
    fn cluster_sweep(&mut self, beta: f64, p: &Breakup, rng: &mut Rng) -> i64 {
        self.partition(p, rng);
        self.delta.fill(0);
        let g = self.g;
        for &(u, v, j) in &g.weak_edges {
            let (lu, lv) = (self.label[u], self.label[v]);
            if lu != lv {
                let t = -2 * i64::from(j) * i64::from(self.s[u] * self.s[v]);
                self.delta[lu as usize] += t;
                self.delta[lv as usize] += t;
            }
        }
        let mut change = 0;
        for c in 0..self.delta.len() {
            let d = self.delta[c];
            if !heat_bath(beta * d as f64 / 1000.0, rng) {
                continue;
            }
            for at in self.start[c] as usize..self.start[c + 1] as usize {
                let u = self.members[at] as usize;
                let su = i64::from(self.s[u]);
                for &(v, j) in g.weak.row(u) {
                    let lv = self.label[v as usize] as usize;
                    if lv != c {
                        self.delta[lv] += 4 * i64::from(j) * su * i64::from(self.s[v as usize]);
                    }
                }
                change += self.flip(u);
            }
            self.delta[c] = -d;
        }
        change
    }
}

/// What one polish did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PolishOutcome {
    /// The best read energy before, in milli.
    pub before: i64,
    /// The best read energy after, in milli.
    pub after: i64,
    /// Reads replaced by a lower state.
    pub improved: usize,
    /// Reads whose rescored energy disagreed with the polish model; they
    /// were left unchanged (expected 0).
    pub mismatched: usize,
}

/// Polish `reads` in place. `score` is the job's own energy function (the
/// one the lane rescored the reads with); a read is replaced only when its
/// polished state scores strictly lower under it, and its `energy_milli`
/// becomes that score. `stop` is polled between reads and every 64 sweeps;
/// `None` means it fired, and `reads` may then be partly polished (each read
/// still consistent, each energy still exact).
#[must_use]
pub fn polish_reads(
    graph: &PolishGraph,
    reads: &mut [SamplerResult],
    config: &PolishConfig,
    seed: u64,
    score: &dyn Fn(&[i8]) -> i64,
    stop: &dyn Fn() -> bool,
) -> Option<PolishOutcome> {
    let before = reads
        .iter()
        .map(|r| r.energy_milli)
        .min()
        .unwrap_or(i64::MAX);
    let mut out = PolishOutcome {
        before,
        after: before,
        improved: 0,
        mismatched: 0,
    };
    // The offset between the job's score and the polish model, per read: it
    // must not change when a read is replaced, or the read is left alone.
    let usable =
        |r: &SamplerResult| r.spins.len() == graph.n && r.spins.iter().all(|&x| x == 1 || x == -1);
    let accept = |r: &mut SamplerResult, s: Vec<i8>, model_gain: i64, out: &mut PolishOutcome| {
        if model_gain <= 0 {
            return;
        }
        let e = score(&s);
        if e == r.energy_milli - model_gain {
            r.spins = s;
            r.energy_milli = e;
            out.improved += 1;
        } else {
            out.mismatched += 1;
        }
    };
    let mut chain: Option<Chain<'_>> = None;
    for r in reads.iter_mut() {
        if stop() {
            return None;
        }
        if !usable(r) {
            continue;
        }
        let c = chain.get_or_insert_with(|| Chain::new(graph, &r.spins));
        c.set(&r.spins);
        let gain = -c.descend();
        let s = c.s.clone();
        accept(r, s, gain, &mut out);
    }
    if config.level >= 2 && config.sweeps > 0 {
        let mut order: Vec<usize> = (0..reads.len()).filter(|&i| usable(&reads[i])).collect();
        order.sort_by_key(|&i| reads[i].energy_milli);
        let p = Breakup::new(config.beta, graph.weights);
        let mut cache = Boltzmann::new(config.beta);
        for (k, &i) in order.iter().take(config.topk).enumerate() {
            if stop() {
                return None;
            }
            let mut rng = Rng::new(seed ^ (k as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            let start = reads[i].spins.clone();
            let c = chain.get_or_insert_with(|| Chain::new(graph, &start));
            c.set(&start);
            let mut e = 0_i64; // relative to the read
            let mut best = (start, 0_i64);
            for sweep in 0..config.sweeps {
                if sweep % 64 == 63 && stop() {
                    return None;
                }
                e += c.cluster_sweep(config.beta, &p, &mut rng);
                e += c.single_sweep(&mut cache, &mut rng);
                if e < best.1 {
                    best.0.copy_from_slice(&c.s);
                    best.1 = e;
                }
            }
            c.set(&best.0);
            let gain = -(best.1 + c.descend());
            let s = c.s.clone();
            accept(&mut reads[i], s, gain, &mut out);
        }
    }
    out.after = reads
        .iter()
        .map(|r| r.energy_milli)
        .min()
        .unwrap_or(i64::MAX);
    Some(out)
}
