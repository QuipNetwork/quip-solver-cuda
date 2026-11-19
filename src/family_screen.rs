//! Local build (`BudFox`, 2026-11-13): probe screen for family leases.
//!
//! Off by default. With `QUIP_FAMILY_SCREEN=1` the msa binary declares
//! `quip/twisted-ice-dimer@1` as a family it generates itself, so the session
//! hands those leases to [`crate::MsaSampler`]'s `sample_lease` instead of the
//! core expander, and this module works them:
//!
//! 1. A pool of host threads generates instances in salt order.
//! 2. Stage 0 (optional, `QUIP_SCREEN_STAGE0_KEEP` < 1): the generator
//!    threads score each instance with agentA's lean structural model
//!    ([`stage0_score`]) and pass on only the lowest-scoring fraction of every
//!    window of `64 / keep` salts, before any GPU work.
//!    With `QUIP_SCREEN_PROBE=0` steps 3 and 4 are skipped: one streaming
//!    session fully solves every stage-0 survivor as it arrives.
//! 3. The probe runs the stock msa streaming kernel on the batch at a short
//!    setting (default 64 reads x 512 sweeps) and records each salt's best
//!    probe energy.
//! 4. Survivors join a pool ranked deepest first. Each cycle solves the
//!    deepest `width * QUIP_SCREEN_FULL_PER_LANE` at the lease's own full
//!    setting, carries the next best as many into the next cycle, and reports
//!    the rest with `LeaseSink::screen`.
//!
//! Probe and full solves run as separate finite `run_stream` sessions, one
//! after the other, because a self-feeding session is keyed on `num_sweeps`.
//! Host generation keeps running while the GPU solves, so the next probe
//! batch is ready when the full phase ends. The canonical generator costs
//! about 13.5 ms per z(12,4) instance on one core, 9.5 ms of it nonce
//! independent; [`fast_dimer`] caches that part (about 2.5 ms per instance)
//! and is checked against the canonical generator on one salt in 64.

pub(crate) mod fast_dimer;
mod stage0;
mod stage0v2;

use crate::lease::{check_capacity, fault, run_full_batch};
use crate::{streaming, IsingGraph, KernelKind, MsaSampler, SampleParams};
use fast_dimer::FastDimer;
use quip_solver_core::families::FamilyClaim;
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::family::SaltFault;
use quip_solver_core::ising_generators::ice::{local_gauges, TWISTED_DIMER_V1};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    FamilyLease, Lease, LeaseSink, SampleError, SamplerResult, StreamJob, StreamOutcome,
    StreamResult, TopologyView,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::channel;

const DIMER: &[Contract] = &[TWISTED_DIMER_V1];

/// The family this build screens when `QUIP_FAMILY_SCREEN=1`.
pub(crate) const CLAIMS: &[FamilyClaim] = &[FamilyClaim {
    revisions: DIMER,
    only_params: &[],
}];

/// Stage 0: agentA's lean structural score of a twisted-dimer instance, lower
/// is predicted deeper. `tiles` holds the node positions of the instance's
/// selected tiles, which type its edges exactly (see [`stage0`]). Returns 0
/// when a tile names an edge the graph lacks. Runs only when
/// `QUIP_SCREEN_STAGE0_KEEP` is below 1.
#[must_use]
pub fn stage0_score(graph: &IsingGraph, tiles: &[[u32; 4]]) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // Milli coefficients are integers.
    let j: Vec<i32> = graph
        .j
        .iter()
        .map(|v| (v * 1000.0).round() as i32)
        .collect();
    let topo = stage0::Topo::shared(graph.h.len(), &graph.edges);
    stage0::lean_score(&topo, &j, tiles).unwrap_or(0.0)
}

/// Local build: [`stage0_score`] with the stage-0 model picked as
/// `QUIP_OTSUKA_STAGE0_MODEL` does: 2 = [`stage0v2`] (cycles of the
/// pair-contracted multigraph), 3 = the same score (model 3 is a model-1
/// prefilter in front of model 2; the prefilter is a gate, not a score),
/// anything else the lean model 1. For the harness's offline survivor
/// selection. Returns 0 when a score is missing.
#[must_use]
pub fn stage0_score_model(model: u8, graph: &IsingGraph, tiles: &[[u32; 4]]) -> f32 {
    if !matches!(model, 2 | 3) {
        return stage0_score(graph, tiles);
    }
    #[allow(clippy::cast_possible_truncation)] // Milli coefficients are integers.
    let j: Vec<i32> = graph
        .j
        .iter()
        .map(|v| (v * 1000.0).round() as i32)
        .collect();
    let topo = stage0v2::Topo2::shared(graph.h.len(), &graph.edges);
    stage0v2::score(&topo, &j, tiles).unwrap_or(0.0)
}

/// Stage 0 of a validated twisted-dimer instance whose edges are `edges`
/// (the job graph's edge list, the instance's own order). Tiles come from the
/// instance metadata. `None` when the metadata or a tile edge is missing.
pub(crate) fn instance_stage0(instance: &Instance, edges: &Edges) -> Option<f32> {
    let (tiles, _) = local_gauges(instance)?;
    let topo = stage0::Topo::shared(instance.node_ids().len(), edges);
    stage0::lean_score(&topo, instance.j(), &tiles)
}

/// Local build: stage 0, model 2 ([`stage0v2`]): cycles of the
/// pair-contracted multigraph, fitted on GPU Otsuka x512 labels. Same inputs
/// and convention as [`instance_stage0`] (lower is predicted deeper).
pub(crate) fn instance_stage0_v2(instance: &Instance, edges: &Edges) -> Option<f32> {
    let (tiles, _) = local_gauges(instance)?;
    let topo = stage0v2::Topo2::shared(instance.node_ids().len(), edges);
    stage0v2::score(&topo, instance.j(), &tiles)
}

/// Local build: the stage-0 score of `model` (1 = agentA lean, 2 = cycles of
/// the pair-contracted multigraph, 3 = model 2's score: model 3 only adds a
/// model-1 prefilter gate in front, see [`cascade_keeps`]). Any other value
/// scores with model 1.
pub(crate) fn instance_stage0_model(model: u8, instance: &Instance, edges: &Edges) -> Option<f32> {
    match model {
        2 | 3 => instance_stage0_v2(instance, edges),
        _ => instance_stage0(instance, edges),
    }
}

/// Local build (all6): [`instance_stage0_model`] on raw parts: `n` nodes,
/// the job graph's `edges`, the couplings `j` aligned with them, and the
/// node positions of the selected tiles. The local screen calls it with the
/// cached generator's couplings ([`fast_dimer::FastDimer::couplings`]), so a
/// salt is scored without the canonical instance. Same convention (lower is
/// predicted deeper); `None` when a tile edge is missing.
pub(crate) fn raw_stage0_model(
    model: u8,
    n: usize,
    edges: &Edges,
    j: &[i32],
    tiles: &[[u32; 4]],
) -> Option<f32> {
    match model {
        2 | 3 => stage0v2::score(&stage0v2::Topo2::shared(n, edges), j, tiles),
        _ => stage0::lean_score(&stage0::Topo::shared(n, edges), j, tiles),
    }
}

/// Local build (agent P): the two keeps of the stage-0 model-3 cascade for an
/// overall `keep` and a prefilter fraction `pre`, as `(prefilter, main)`.
///
/// Model 1 (lean, ~2 ms per salt) passes the lowest `prefilter` of salts;
/// model 2 (~15-19 ms) scores only those and passes the lowest `main` of
/// them, so about `prefilter * main = keep` of all salts reach a lane while
/// model 2 runs on `prefilter` of them. A keep at or above `pre` leaves
/// model 1 alone at the keep (`main` = 1, no model-2 gate).
///
/// Why: on the spec-120 topology (0x2d130ce6, `x-w45-p1350` params) live GPU
/// labels deepen steadily with a tighter model-2 keep (keep 0.005 vs 0.025:
/// mean -1.0, P(E <= -5075) x1.7), but model 2 on the job thread caps the
/// screen near 50-65 salts per second. Model 1's lowest 20% hold all of model
/// 2's lowest 0.5% and 98.5% of its lowest 1%.
#[must_use]
pub(crate) fn cascade_keeps(keep: f64, pre: f64) -> (f64, f64) {
    let keep = keep.clamp(f64::MIN_POSITIVE, 1.0);
    let prefilter = pre.clamp(f64::MIN_POSITIVE, 1.0).max(keep);
    (prefilter, (keep / prefilter).min(1.0))
}

/// Keeps about `keep` of a stream of stage-0 scores, one decision per score
/// with no buffering: a score passes when it is at or below the `keep`
/// quantile of the last [`Stage0Gate::HISTORY`] scores, itself included.
/// The cutoff is recomputed every [`Stage0Gate::REFRESH`] scores, and on
/// every score while fewer than `REFRESH` have been seen.
pub(crate) struct Stage0Gate {
    keep: f64,
    history: std::collections::VecDeque<f32>,
    cutoff: f32,
    since: usize,
}

impl Stage0Gate {
    pub(crate) const HISTORY: usize = 4096;
    pub(crate) const REFRESH: usize = 64;

    pub(crate) fn new(keep: f64) -> Self {
        Self {
            keep: keep.clamp(f64::MIN_POSITIVE, 1.0),
            history: std::collections::VecDeque::with_capacity(Self::HISTORY),
            cutoff: f32::INFINITY,
            since: 0,
        }
    }

    /// Record `score` and decide whether it passes.
    pub(crate) fn admit(&mut self, score: f32) -> bool {
        if self.keep >= 1.0 {
            return true;
        }
        if self.history.len() == Self::HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(score);
        self.since += 1;
        if self.history.len() < Self::REFRESH || self.since >= Self::REFRESH {
            self.since = 0;
            let mut sorted: Vec<f32> = self.history.iter().copied().collect();
            sorted.sort_by(f32::total_cmp);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let rank = ((sorted.len() as f64 * self.keep).ceil() as usize).clamp(1, sorted.len());
            self.cutoff = sorted[rank - 1];
        }
        score <= self.cutoff
    }
}

fn stage0_milli(n: usize, edges: &Edges, j: &[i32], tiles: &[[u32; 4]]) -> f32 {
    let topo = stage0::Topo::shared(n, edges);
    stage0::lean_score(&topo, j, tiles).unwrap_or(0.0)
}

/// Environment knobs, read once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Config {
    /// `QUIP_FAMILY_SCREEN=1`. Off: stock behaviour.
    pub(crate) enabled: bool,
    /// `QUIP_SCREEN_PROBE_READS`, default 64 (one msa replica word).
    pub(crate) probe_reads: usize,
    /// `QUIP_SCREEN_PROBE_SWEEPS`, default 512.
    pub(crate) probe_sweeps: usize,
    /// `QUIP_SCREEN_KEEP_MIN`: the fewest probes per full solve, default 4.
    /// A cycle waits for this many generated instances before probing.
    pub(crate) keep_min: usize,
    /// `QUIP_SCREEN_KEEP_MAX`: the most probes per full solve, default 16.
    /// A cycle probes at most this many, which also sizes the host buffer.
    pub(crate) keep_max: usize,
    /// `QUIP_SCREEN_FULL_PER_LANE`: full solves per lane per cycle, default 1.
    pub(crate) full_per_lane: usize,
    /// `QUIP_SCREEN_GEN_THREADS`: host generator threads, default 4.
    pub(crate) gen_threads: usize,
    /// `QUIP_SCREEN_STAGE0_KEEP`: fraction stage 0 keeps, default 1 (off).
    pub(crate) stage0_keep: f64,
    /// `QUIP_SCREEN_PROBE`, default 1. `0` skips the probe: survivors of
    /// stage 0 are ranked by their stage-0 score and solved without any GPU
    /// probe time.
    pub(crate) probe: bool,
    /// `QUIP_SCREEN_FAST_GEN`, default 1: the cached dimer generator, checked
    /// against the canonical one on one salt in 64. `0` uses only the canonical.
    pub(crate) fast_gen: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: false,
            probe_reads: 64,
            probe_sweeps: 512,
            keep_min: 4,
            keep_max: 16,
            full_per_lane: 1,
            gen_threads: 4,
            stage0_keep: 1.0,
            probe: true,
            fast_gen: true,
        }
    }
}

impl Config {
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let d = Self::default();
        let num = |key: &str, default: usize, min: usize, max: usize| {
            get(key)
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|v| (min..=max).contains(v))
                .unwrap_or(default)
        };
        let enabled = get("QUIP_FAMILY_SCREEN").is_some_and(|v| v.trim() == "1");
        let keep_min = num("QUIP_SCREEN_KEEP_MIN", d.keep_min, 1, 4096);
        let keep_max = num("QUIP_SCREEN_KEEP_MAX", d.keep_max, 1, 4096).max(keep_min);
        let stage0_keep = get("QUIP_SCREEN_STAGE0_KEEP")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| *v > 0.0 && *v <= 1.0)
            .unwrap_or(d.stage0_keep);
        Self {
            enabled,
            probe_reads: num("QUIP_SCREEN_PROBE_READS", d.probe_reads, 1, 128),
            probe_sweeps: num("QUIP_SCREEN_PROBE_SWEEPS", d.probe_sweeps, 1, 1 << 20),
            keep_min,
            keep_max,
            full_per_lane: num("QUIP_SCREEN_FULL_PER_LANE", d.full_per_lane, 1, 64),
            gen_threads: num("QUIP_SCREEN_GEN_THREADS", d.gen_threads, 1, 256),
            stage0_keep,
            probe: get("QUIP_SCREEN_PROBE").is_none_or(|v| v.trim() != "0"),
            fast_gen: get("QUIP_SCREEN_FAST_GEN").is_none_or(|v| v.trim() != "0"),
        }
    }
}

pub(crate) fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let config = Config::from_lookup(|key| std::env::var(key).ok());
        if config.enabled {
            eprintln!("quip-screen: family screen enabled {config:?}");
        }
        config
    })
}

/// Milli coefficients kept compactly between generation and use.
enum Coeffs {
    Zero(usize),
    I16(Box<[i16]>),
    I32(Box<[i32]>),
}

impl Coeffs {
    fn new(values: &[i32]) -> Self {
        if values.iter().all(|&v| v == 0) {
            return Self::Zero(values.len());
        }
        let narrow: Option<Vec<i16>> = values.iter().map(|&v| i16::try_from(v).ok()).collect();
        match narrow {
            Some(narrow) => Self::I16(narrow.into_boxed_slice()),
            None => Self::I32(values.into()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Zero(n) => *n,
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }

    fn get(&self, k: usize) -> i64 {
        match self {
            Self::Zero(_) => 0,
            Self::I16(v) => i64::from(v[k]),
            Self::I32(v) => i64::from(v[k]),
        }
    }

    /// The same conversion the core expander applies (`f64::from_milli`).
    fn to_f64(&self) -> Vec<f64> {
        (0..self.len())
            .map(|k| match self {
                Self::Zero(_) => 0.0,
                Self::I16(v) => f64::from(v[k]) / 1000.0,
                Self::I32(v) => f64::from(v[k]) / 1000.0,
            })
            .collect()
    }
}

/// One generated salt.
struct Candidate {
    index: u64,
    h: Coeffs,
    j: Coeffs,
    edges: Arc<[(usize, usize)]>,
    offset_milli: i64,
    stage0: f32,
}

impl Candidate {
    fn graph(&self) -> IsingGraph {
        IsingGraph {
            h: self.h.to_f64(),
            j: self.j.to_f64(),
            edges: Arc::clone(&self.edges),
        }
    }

    /// FAM-33 energy of `spins`, exact in milli, offset included.
    fn energy_milli(&self, spins: &[i8]) -> i64 {
        let mut e = self.offset_milli;
        for (i, &s) in spins.iter().enumerate().take(self.h.len()) {
            e += self.h.get(i) * i64::from(s);
        }
        for (k, &(u, v)) in self.edges.iter().enumerate() {
            e += self.j.get(k) * i64::from(spins[u]) * i64::from(spins[v]);
        }
        e
    }
}

enum Generated {
    Ready(Candidate),
    Skipped(u64),
    /// Salts stage 0 cut, one message per window.
    Stage0Rejected(Vec<u64>),
}

pub(crate) type Edges = Arc<[(usize, usize)]>;

/// Shares one edge allocation between instances whose edges are equal.
#[derive(Default)]
struct EdgeCache(Mutex<Option<Edges>>);

impl EdgeCache {
    fn edges(&self, fresh: &[(u32, u32)]) -> Arc<[(usize, usize)]> {
        let mut cached = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(edges) = cached.as_ref() {
            if edges.len() == fresh.len()
                && edges
                    .iter()
                    .zip(fresh)
                    .all(|(&(a, b), &(u, v))| a == u as usize && b == v as usize)
            {
                return Arc::clone(edges);
            }
        }
        let edges: Arc<[(usize, usize)]> = fresh
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        *cached = Some(Arc::clone(&edges));
        edges
    }
}

/// The cached generator and whether it still agrees with the canonical one.
struct Fast {
    dimer: FastDimer,
    trusted: AtomicBool,
}

/// One salt in every this many is also generated canonically and compared.
const FAST_CHECK_EVERY: u64 = 64;

fn same_instance(instance: &Instance, dimer: &FastDimer, j: &[i32]) -> bool {
    instance.j() == j
        && instance.edges() == dimer.edges()
        && instance.h().len() == dimer.nodes()
        && instance.h().iter().all(|&h| h == 0)
        && instance.offset_milli() == 0
}

/// Couplings from the cached generator, checked against the canonical one on
/// a sample of salts. `None` sends the salt to the canonical path.
fn fast_candidate(
    fast: &Fast,
    canonical: &dyn Fn([u8; 32]) -> Result<Instance, SaltFault>,
    nonce: [u8; 32],
    index: u64,
    cache: &EdgeCache,
    with_stage0: bool,
) -> Option<Candidate> {
    if !fast.trusted.load(Ordering::Relaxed) {
        return None;
    }
    let (j, tiles) = fast.dimer.couplings(nonce)?;
    if index.is_multiple_of(FAST_CHECK_EVERY) {
        let agrees =
            canonical(nonce).is_ok_and(|instance| same_instance(&instance, &fast.dimer, &j));
        if !agrees {
            fast.trusted.store(false, Ordering::Relaxed);
            tracing::error!(salt_index = index, "cached dimer generator disagrees with the canonical one; using the canonical generator");
            return None;
        }
    }
    let edges = cache.edges(fast.dimer.edges());
    let stage0 = if with_stage0 {
        stage0_milli(fast.dimer.nodes(), &edges, &j, &tiles)
    } else {
        0.0
    };
    Some(Candidate {
        index,
        h: Coeffs::Zero(fast.dimer.nodes()),
        j: Coeffs::new(&j),
        edges,
        offset_milli: 0,
        stage0,
    })
}

/// Check the cached generator on the lease's first salt before any salt of
/// the lease uses it.
fn fast_agrees_on_first_salt(fast: &Fast, family: &FamilyLease, lease: &Lease) -> bool {
    if lease.salt_count() == 0 {
        return true;
    }
    let nonce = lease.nonce(0);
    let agrees = match (family.generate(nonce), fast.dimer.couplings(nonce)) {
        (Ok(instance), Some((j, tiles))) => {
            same_instance(&instance, &fast.dimer, &j)
                && local_gauges(&instance).is_some_and(|(meta, _)| meta == tiles)
        }
        (Err(_), None) => true,
        _ => false,
    };
    if !agrees {
        tracing::error!(
            "cached dimer generator disagrees on salt 0; using the canonical generator"
        );
    }
    agrees
}

fn canonical_candidate(
    instance: &Instance,
    index: u64,
    cache: &EdgeCache,
    with_stage0: bool,
) -> Candidate {
    let edges = cache.edges(instance.edges());
    let stage0 = if with_stage0 {
        // The tile list is in the instance metadata of this family.
        local_gauges(instance).map_or(0.0, |(tiles, _)| {
            stage0_milli(instance.node_ids().len(), &edges, instance.j(), &tiles)
        })
    } else {
        0.0
    };
    Candidate {
        index,
        h: Coeffs::new(instance.h()),
        j: Coeffs::new(instance.j()),
        edges,
        offset_milli: instance.offset_milli(),
        stage0,
    }
}

fn generate(
    family: &FamilyLease,
    lease: &Lease,
    index: u64,
    cache: &EdgeCache,
    with_stage0: bool,
    fast: Option<&Fast>,
) -> Generated {
    let nonce = lease.nonce(index);
    let canonical = |nonce| family.generate(nonce);
    let fast =
        fast.and_then(|fast| fast_candidate(fast, &canonical, nonce, index, cache, with_stage0));
    let candidate = match fast {
        Some(candidate) => Ok(candidate),
        None => family
            .generate(nonce)
            .map(|instance| canonical_candidate(&instance, index, cache, with_stage0)),
    };
    match candidate {
        Ok(candidate) => Generated::Ready(candidate),
        Err(SaltFault::Skip(_)) => Generated::Skipped(index),
        Err(SaltFault::Defect(defect)) => {
            tracing::error!(
                ?defect,
                salt_index = index,
                "family construction broke its contract; skipping the salt"
            );
            Generated::Skipped(index)
        }
    }
}

/// The outcome of one selection step, as salt indices with probe energies.
#[derive(Debug, Default, PartialEq, Eq)]
struct Selection {
    solve: Vec<(i64, u64)>,
    carry: Vec<(i64, u64)>,
    evict: Vec<(i64, u64)>,
}

/// Rank `pool` deepest first (energy, then index). Solve the best `n_full`
/// plus every probe at or below `target`, carry the next `carry` into the
/// next cycle, and evict the rest.
fn select(
    mut pool: Vec<(i64, u64)>,
    n_full: usize,
    carry: usize,
    target: Option<i64>,
) -> Selection {
    pool.sort_unstable();
    let hits = target.map_or(0, |t| pool.partition_point(|&(e, _)| e <= t));
    let solve_n = n_full.max(hits).min(pool.len());
    let rest = pool.split_off(solve_n);
    let carry_n = carry.min(rest.len());
    let mut carried = rest;
    let evict = carried.split_off(carry_n);
    Selection {
        solve: pool,
        carry: carried,
        evict,
    }
}

#[derive(Default)]
struct Stats {
    generated: u64,
    skipped: u64,
    stage0_screened: u64,
    probed: u64,
    probe_screened: u64,
    solved: u64,
    pushed: u64,
    cycles: u64,
    best_probe: Option<i64>,
    best_full: Option<i64>,
    wait: Duration,
    probe: Duration,
    full: Duration,
}

/// Work one family lease with the probe screen.
#[allow(clippy::too_many_lines)] // Generator pool setup plus the lease summary.
pub(crate) fn sample(
    sampler: &MsaSampler,
    lease: &Lease,
    family: &FamilyLease,
    topology: &TopologyView,
    params: &SampleParams,
    sink: &LeaseSink,
) -> Result<(), SampleError> {
    check_capacity(topology, sampler.inner.device.max_nodes)?;
    let turn = sampler.turns.join();
    if !turn.wait(|| sink.is_stopped()) || sink.is_stopped() {
        return Ok(());
    }
    let config = *config();
    let started = Instant::now();
    let width = streaming::stream_width(&sampler.inner.device, KernelKind::Msa).max(1);
    let n_full = width * config.full_per_lane;
    let p_min = n_full * config.keep_min;
    let p_max = n_full * config.keep_max;
    let count = lease.salt_count();
    let next = AtomicU64::new(0);
    let quit = AtomicBool::new(false);
    let cache = EdgeCache::default();
    let mut stats = Stats::default();
    let with_stage0 = config.stage0_keep < 1.0;
    let fast = if config.fast_gen {
        FastDimer::new(
            family.family_id(),
            family.revision(),
            family.params(),
            family.topology(),
        )
        .map(|dimer| Fast {
            dimer,
            trusted: AtomicBool::new(true),
        })
        .filter(|fast| fast_agrees_on_first_salt(fast, family, lease))
    } else {
        None
    };
    let window = Stage0Window::new(config.stage0_keep);
    let running = std::sync::atomic::AtomicUsize::new(config.gen_threads);
    let result = std::thread::scope(|scope| {
        let (tx, rx) = sync_channel::<Generated>(p_max);
        for _ in 0..config.gen_threads {
            let tx = tx.clone();
            let (next, quit, cache, fast, window, running) =
                (&next, &quit, &cache, fast.as_ref(), &window, &running);
            scope.spawn(move || {
                loop {
                    if quit.load(Ordering::Relaxed) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= count {
                        break;
                    }
                    let sent = match generate(family, lease, index, cache, with_stage0, fast) {
                        Generated::Ready(candidate) if with_stage0 => window
                            .push(candidate)
                            .is_none_or(|full| window.cut(full, &tx)),
                        item => tx.send(item).is_ok(),
                    };
                    if !sent {
                        break;
                    }
                }
                // The last generator out cuts the partial window.
                if running.fetch_sub(1, Ordering::AcqRel) == 1
                    && with_stage0
                    && !quit.load(Ordering::Relaxed)
                {
                    let rest = window.take_rest();
                    if !rest.is_empty() {
                        window.cut(rest, &tx);
                    }
                }
            });
        }
        drop(tx);
        if config.probe {
            let outcome = cycles(
                sampler, params, sink, &config, &rx, n_full, p_min, p_max, &mut stats,
            );
            quit.store(true, Ordering::Relaxed);
            // Unblock generators waiting on a full buffer.
            drop(rx);
            outcome
        } else {
            // The feeder owns and drops the receiver when it stops.
            let outcome = stream_unprobed(sampler, params, sink, rx, &mut stats);
            quit.store(true, Ordering::Relaxed);
            outcome
        }
    });
    #[allow(clippy::cast_precision_loss)]
    let salts_per_s = stats.generated as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON);
    tracing::info!(
        turn = turn.id(),
        assigned = count,
        generated = stats.generated,
        skipped = stats.skipped,
        stage0_screened = stats.stage0_screened,
        probed = stats.probed,
        probe_screened = stats.probe_screened,
        solved = stats.solved,
        pushed = stats.pushed,
        cycles = stats.cycles,
        best_probe = ?stats.best_probe,
        best_full = ?stats.best_full,
        wait_ms = stats.wait.as_millis(),
        probe_ms = stats.probe.as_millis(),
        full_ms = stats.full.as_millis(),
        salts_per_s,
        "family screen lease finished"
    );
    result
}

/// The stage-0 cut, applied by the generator threads before the buffer, so
/// the buffer holds survivors only: of every `size` generated candidates,
/// the `ceil(keep * size)` with the lowest score go on.
struct Stage0Window {
    keep: f64,
    size: usize,
    pending: Mutex<Vec<Candidate>>,
}

impl Stage0Window {
    fn new(keep: f64) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let size = if keep < 1.0 {
            (64.0 / keep).ceil() as usize
        } else {
            1
        };
        Self {
            keep,
            size,
            pending: Mutex::new(Vec::new()),
        }
    }

    /// Add one candidate; returns a full window to cut, if this filled one.
    fn push(&self, candidate: Candidate) -> Option<Vec<Candidate>> {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        pending.push(candidate);
        (pending.len() >= self.size).then(|| std::mem::take(&mut *pending))
    }

    fn take_rest(&self) -> Vec<Candidate> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Send the stage-0 best of `window` on and its rejects as one message.
    fn cut(&self, mut window: Vec<Candidate>, tx: &SyncSender<Generated>) -> bool {
        window.sort_by(|a, b| a.stage0.total_cmp(&b.stage0).then(a.index.cmp(&b.index)));
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let keep = ((window.len() as f64 * self.keep).ceil() as usize).min(window.len());
        let rejected: Vec<u64> = window.split_off(keep).iter().map(|c| c.index).collect();
        for candidate in window {
            if tx.send(Generated::Ready(candidate)).is_err() {
                return false;
            }
        }
        rejected.is_empty() || tx.send(Generated::Stage0Rejected(rejected)).is_ok()
    }
}

/// Take a batch of stage-0 survivors: wait until `p_min` (or the end of the
/// lease), then take whatever else is ready, up to `p_max`. The flag is set
/// once every salt has been generated. `None`: stopped.
fn take_batch(
    rx: &Receiver<Generated>,
    sink: &LeaseSink,
    p_min: usize,
    p_max: usize,
    stats: &mut Stats,
) -> Option<(Vec<Candidate>, bool)> {
    let mut batch = Vec::new();
    let mut exhausted = false;
    let waited = Instant::now();
    while batch.len() < p_max {
        let got = if batch.len() < p_min {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(item) => Some(item),
                Err(RecvTimeoutError::Timeout) => {
                    if sink.is_stopped() {
                        return None;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    exhausted = true;
                    None
                }
            }
        } else {
            match rx.try_recv() {
                Ok(item) => Some(item),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    exhausted = true;
                    None
                }
            }
        };
        let Some(item) = got else { break };
        match item {
            Generated::Ready(candidate) => {
                stats.generated += 1;
                batch.push(candidate);
            }
            Generated::Skipped(index) => {
                stats.generated += 1;
                stats.skipped += 1;
                if sink.skip(index).is_err() {
                    return None;
                }
            }
            Generated::Stage0Rejected(indices) => {
                for index in indices {
                    stats.generated += 1;
                    stats.stage0_screened += 1;
                    if sink.screen(index, i64::MAX).is_err() {
                        return None;
                    }
                }
            }
        }
    }
    stats.wait += waited.elapsed();
    Some((batch, exhausted))
}

/// A stage-0 score as a rank key when the probe is off. Never reported.
#[allow(clippy::cast_possible_truncation)]
fn stage0_key(score: f32) -> i64 {
    (f64::from(score) * 1e6).round() as i64
}

#[allow(clippy::too_many_arguments)] // Lease context, knobs, and counters.
#[allow(clippy::too_many_lines)] // One screen cycle: batch, probe, select, solve.
fn cycles(
    sampler: &MsaSampler,
    params: &SampleParams,
    sink: &LeaseSink,
    config: &Config,
    rx: &Receiver<Generated>,
    n_full: usize,
    p_min: usize,
    p_max: usize,
    stats: &mut Stats,
) -> Result<(), SampleError> {
    let probe_params = SampleParams {
        num_reads: config.probe_reads,
        num_sweeps: config.probe_sweeps,
        ..params.clone()
    };
    // Without the probe, pool keys are stage-0 ranks, not energies: they
    // order the pool but are never reported or compared with the target.
    let probing = config.probe;
    let reported = |key: i64| if probing { key } else { i64::MAX };
    // Carried survivors: pool key and their candidate.
    let mut kept: HashMap<u64, Candidate> = HashMap::new();
    let mut carry: Vec<(i64, u64)> = Vec::new();
    let mut seed = seed_base();
    loop {
        if sink.is_stopped() {
            return Ok(());
        }
        let Some((batch, exhausted)) = take_batch(rx, sink, p_min, p_max, stats) else {
            break;
        };
        if batch.is_empty() && exhausted && carry.is_empty() {
            break;
        }
        stats.cycles += 1;
        let keys: Vec<Option<i64>> = if probing {
            let probe_start = Instant::now();
            let probed = run_batch(
                sampler,
                &batch,
                &probe_params,
                false,
                &mut seed,
                sink,
                |c, reads| {
                    reads
                        .iter()
                        .map(|r| r.energy_milli)
                        .min()
                        .map(|e| e.saturating_add(c.offset_milli))
                },
            )?;
            stats.probe += probe_start.elapsed();
            let Some(probed) = probed else {
                return Ok(());
            };
            stats.probed += probed.len() as u64;
            probed
        } else {
            batch.iter().map(|c| Some(stage0_key(c.stage0))).collect()
        };
        let mut pool = carry.clone();
        for (candidate, key) in batch.into_iter().zip(keys) {
            if let Some(key) = key {
                if probing {
                    stats.best_probe = Some(stats.best_probe.map_or(key, |b| b.min(key)));
                }
                pool.push((key, candidate.index));
                kept.insert(candidate.index, candidate);
            } else if sink.screen(candidate.index, i64::MAX).is_err() {
                return Ok(());
            }
        }
        let carry_n = if exhausted { 0 } else { n_full };
        let target = if probing {
            sink.target_energy_milli()
        } else {
            None
        };
        let selection = select(pool, n_full, carry_n, target);
        for &(key, index) in &selection.evict {
            kept.remove(&index);
            if sink.screen(index, reported(key)).is_err() {
                return Ok(());
            }
            stats.probe_screened += 1;
        }
        carry = selection.carry;
        // Full solves, deepest first.
        let (keys, survivors): (Vec<i64>, Vec<Candidate>) = selection
            .solve
            .iter()
            .filter_map(|&(key, index)| kept.remove(&index).map(|c| (key, c)))
            .unzip();
        stats.solved += survivors.len() as u64;
        let full_start = Instant::now();
        let full = run_batch(
            sampler,
            &survivors,
            params,
            true,
            &mut seed,
            sink,
            |c, reads| reads.iter().map(|r| c.energy_milli(&r.spins)).min(),
        );
        stats.full += full_start.elapsed();
        let Some(scores) = full? else {
            return Ok(());
        };
        stats.pushed += scores.iter().filter(|s| s.is_some()).count() as u64;
        // Paired stage-0, probe and full energies, for calibrating offline:
        // RUST_LOG=quip_solver_cuda::family_screen=debug.
        for ((candidate, &key), full) in survivors.iter().zip(&keys).zip(&scores) {
            if let Some(full) = full {
                let probe = probing.then_some(key);
                tracing::debug!(
                    salt_index = candidate.index,
                    probe = ?probe,
                    full,
                    stage0 = candidate.stage0,
                    "family screen pair"
                );
            }
        }
        if let Some(best) = scores.into_iter().flatten().min() {
            stats.best_full = Some(stats.best_full.map_or(best, |b| b.min(best)));
        }
        if exhausted && carry.is_empty() {
            break;
        }
    }
    // A stop or the end of the lease: carried salts were ranked but not solved.
    for (key, index) in carry {
        if sink.screen(index, reported(key)).is_err() {
            break;
        }
        stats.probe_screened += 1;
    }
    Ok(())
}

fn seed_base() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    #[allow(clippy::cast_possible_truncation)]
    let seed = (nanos as u64) ^ 0x9E37_79B9_7F4A_7C15;
    seed.max(1)
}

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// `QUIP_SCREEN_PROBE=0`: one streaming session for the whole lease, fed with
/// stage-0 survivors in the order they arrive, each fully solved. No probe,
/// no phases, so the session is built once and never drains between batches.
#[allow(clippy::too_many_lines)] // Feeder, reporter, and their shared state.
fn stream_unprobed(
    sampler: &MsaSampler,
    params: &SampleParams,
    sink: &LeaseSink,
    rx: Receiver<Generated>,
    stats: &mut Stats,
) -> Result<(), SampleError> {
    let width = streaming::stream_width(&sampler.inner.device, KernelKind::Msa).max(1);
    let (job_tx, job_rx) = channel(2 * width);
    let in_flight: Mutex<HashMap<u64, Candidate>> = Mutex::new(HashMap::new());
    let feed = Mutex::new(Stats::default());
    let started = Instant::now();
    let mut seed = seed_base();
    let outcome = std::thread::scope(|scope| {
        let (in_flight, feed) = (&in_flight, &feed);
        scope.spawn(move || {
            let mut counts = Stats::default();
            loop {
                if sink.is_stopped() {
                    break;
                }
                let item = match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok(item) => item,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                match item {
                    Generated::Ready(candidate) => {
                        counts.generated += 1;
                        counts.solved += 1;
                        let job = StreamJob {
                            job_id: candidate.index.to_le_bytes().to_vec(),
                            graph: candidate.graph(),
                            params: SampleParams {
                                seed: next_seed(&mut seed),
                                ..params.clone()
                            },
                            watermark: Some(1),
                        };
                        in_flight
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .insert(candidate.index, candidate);
                        if job_tx.blocking_send(job).is_err() {
                            break;
                        }
                    }
                    Generated::Skipped(index) => {
                        counts.generated += 1;
                        counts.skipped += 1;
                        if sink.skip(index).is_err() {
                            break;
                        }
                    }
                    Generated::Stage0Rejected(indices) => {
                        for index in indices {
                            counts.generated += 1;
                            counts.stage0_screened += 1;
                            if sink.screen(index, i64::MAX).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            *feed.lock().unwrap_or_else(PoisonError::into_inner) = counts;
            // Dropping job_tx closes intake; the session drains and returns.
        });
        run_full_batch(
            &sampler.inner,
            job_rx,
            4 * width,
            || sink.is_stopped(),
            |mut out, cancel| {
                let mut pushed = 0u64;
                let mut best: Option<i64> = None;
                let mut stopped = false;
                let mut failure = None;
                while let Some(result) = out.blocking_recv() {
                    if stopped || failure.is_some() {
                        continue;
                    }
                    if sink.is_stopped() {
                        stopped = true;
                        cancel.cancel_through(1);
                        continue;
                    }
                    let StreamResult {
                        job_id, outcome, ..
                    } = result;
                    let candidate = <[u8; 8]>::try_from(job_id.as_slice())
                        .ok()
                        .map(u64::from_le_bytes)
                        .and_then(|index| {
                            in_flight
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .remove(&index)
                        });
                    let Some(candidate) = candidate else {
                        failure = Some(fault("unknown family screen result"));
                        cancel.cancel_through(1);
                        continue;
                    };
                    match outcome {
                        StreamOutcome::Cancelled => stopped = true,
                        StreamOutcome::Completed(Err(error)) => {
                            failure = Some(match error {
                                SampleError::Capacity => SampleError::Capacity,
                                error => fault(error),
                            });
                            cancel.cancel_through(1);
                        }
                        StreamOutcome::Completed(Ok(reads)) => {
                            let exact: Vec<SamplerResult> = reads
                                .into_iter()
                                .map(|r| SamplerResult {
                                    energy_milli: candidate.energy_milli(&r.spins),
                                    spins: r.spins,
                                })
                                .collect();
                            let full = exact.iter().map(|r| r.energy_milli).min();
                            if sink.push(candidate.index, exact).is_err() {
                                stopped = true;
                                cancel.cancel_through(1);
                                continue;
                            }
                            pushed += 1;
                            if let Some(full) = full {
                                best = Some(best.map_or(full, |b| b.min(full)));
                                tracing::debug!(
                                    salt_index = candidate.index,
                                    probe = ?None::<i64>,
                                    full,
                                    stage0 = candidate.stage0,
                                    "family screen pair"
                                );
                            }
                        }
                    }
                }
                match failure {
                    Some(error) => Err(error),
                    None => Ok((pushed, best)),
                }
            },
        )
    });
    let counts = std::mem::take(&mut *feed.lock().unwrap_or_else(PoisonError::into_inner));
    stats.generated += counts.generated;
    stats.skipped += counts.skipped;
    stats.stage0_screened += counts.stage0_screened;
    stats.solved += counts.solved;
    stats.cycles += 1;
    stats.full += started.elapsed();
    let (pushed, best) = outcome?;
    stats.pushed += pushed;
    stats.best_full = best;
    Ok(())
}

/// Run `batch` through one finite streaming session at `params` and return
/// each candidate's `score`, in batch order. With `push`, each result is also
/// pushed to the sink with exact energies (a full solve); without it nothing
/// is reported (a probe). `Ok(None)`: the lease stopped.
#[allow(clippy::too_many_arguments)] // Batch, its settings, and the lease context.
fn run_batch(
    sampler: &MsaSampler,
    batch: &[Candidate],
    params: &SampleParams,
    push: bool,
    seed: &mut u64,
    sink: &LeaseSink,
    score: impl Fn(&Candidate, &[SamplerResult]) -> Option<i64> + Sync,
) -> Result<Option<Vec<Option<i64>>>, SampleError> {
    if batch.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let width = streaming::stream_width(&sampler.inner.device, KernelKind::Msa).max(1);
    let (job_tx, job_rx) = channel(2 * width);
    let seeds: Vec<u64> = (0..batch.len()).map(|_| next_seed(seed)).collect();
    let by_index: HashMap<u64, usize> = batch
        .iter()
        .enumerate()
        .map(|(k, c)| (c.index, k))
        .collect();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for (candidate, &seed) in batch.iter().zip(&seeds) {
                let job = StreamJob {
                    job_id: candidate.index.to_le_bytes().to_vec(),
                    graph: candidate.graph(),
                    params: SampleParams {
                        seed,
                        ..params.clone()
                    },
                    watermark: Some(1),
                };
                if job_tx.blocking_send(job).is_err() {
                    break;
                }
            }
        });
        run_full_batch(
            &sampler.inner,
            job_rx,
            batch.len(),
            || sink.is_stopped(),
            |mut out, cancel| {
                let mut scores = vec![None; batch.len()];
                let mut stopped = false;
                let mut failure = None;
                while let Some(result) = out.blocking_recv() {
                    if stopped || failure.is_some() {
                        continue;
                    }
                    if sink.is_stopped() {
                        stopped = true;
                        cancel.cancel_through(1);
                        continue;
                    }
                    let StreamResult {
                        job_id, outcome, ..
                    } = result;
                    let slot = <[u8; 8]>::try_from(job_id.as_slice())
                        .ok()
                        .map(u64::from_le_bytes)
                        .and_then(|index| by_index.get(&index).copied());
                    let Some(slot) = slot else {
                        failure = Some(fault("unknown family screen result"));
                        cancel.cancel_through(1);
                        continue;
                    };
                    match outcome {
                        StreamOutcome::Cancelled => stopped = true,
                        StreamOutcome::Completed(Err(error)) => {
                            failure = Some(match error {
                                SampleError::Capacity => SampleError::Capacity,
                                error => fault(error),
                            });
                            cancel.cancel_through(1);
                        }
                        StreamOutcome::Completed(Ok(reads)) => {
                            let candidate = &batch[slot];
                            scores[slot] = score(candidate, &reads);
                            if push {
                                let exact: Vec<SamplerResult> = reads
                                    .into_iter()
                                    .map(|r| SamplerResult {
                                        energy_milli: candidate.energy_milli(&r.spins),
                                        spins: r.spins,
                                    })
                                    .collect();
                                if sink.push(candidate.index, exact).is_err() {
                                    stopped = true;
                                    cancel.cancel_through(1);
                                }
                            }
                        }
                    }
                }
                if let Some(error) = failure {
                    return Err(error);
                }
                Ok(if stopped { None } else { Some(scores) })
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_off_and_parses_knobs() {
        let none = Config::from_lookup(|_| None);
        assert_eq!(none, Config::default());
        assert!(!none.enabled);
        let vars: HashMap<&str, &str> = [
            ("QUIP_FAMILY_SCREEN", "1"),
            ("QUIP_SCREEN_PROBE_SWEEPS", "1024"),
            ("QUIP_SCREEN_KEEP_MIN", "8"),
            ("QUIP_SCREEN_KEEP_MAX", "4"),
            ("QUIP_SCREEN_STAGE0_KEEP", "0.5"),
            ("QUIP_SCREEN_GEN_THREADS", "0"),
            ("QUIP_SCREEN_PROBE", "0"),
        ]
        .into_iter()
        .collect();
        let c = Config::from_lookup(|k| vars.get(k).map(|v| (*v).to_owned()));
        assert!(c.enabled);
        assert_eq!(c.probe_sweeps, 1024);
        assert_eq!((c.keep_min, c.keep_max), (8, 8));
        assert!((c.stage0_keep - 0.5).abs() < 1e-12);
        assert_eq!(c.gen_threads, 4, "out-of-range values keep the default");
        assert!(!c.probe);
        assert!(none.probe && none.fast_gen);
    }

    #[test]
    fn selection_solves_deepest_carries_next_and_keeps_target_hits() {
        let pool = vec![(-5, 1), (-9, 2), (-7, 3), (-1, 4), (-8, 5), (-3, 6)];
        let s = select(pool.clone(), 2, 2, None);
        assert_eq!(s.solve, vec![(-9, 2), (-8, 5)]);
        assert_eq!(s.carry, vec![(-7, 3), (-5, 1)]);
        assert_eq!(s.evict, vec![(-3, 6), (-1, 4)]);
        // Every probe at or below the target is solved, beyond n_full.
        let s = select(pool.clone(), 1, 0, Some(-7));
        assert_eq!(s.solve, vec![(-9, 2), (-8, 5), (-7, 3)]);
        assert!(s.carry.is_empty());
        assert_eq!(s.evict.len(), 3);
        // Ties break by salt index.
        let s = select(vec![(-2, 9), (-2, 3)], 1, 0, None);
        assert_eq!(s.solve, vec![(-2, 3)]);
        let s = select(Vec::new(), 4, 4, Some(0));
        assert_eq!(s, Selection::default());
    }

    #[test]
    fn coefficients_round_trip_and_score_exactly() {
        let j = [-1000, 62, -62, 1000];
        let c = Candidate {
            index: 0,
            h: Coeffs::new(&[0, 0, 0, 0]),
            j: Coeffs::new(&j),
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)].into(),
            offset_milli: 5,
            stage0: 0.0,
        };
        assert!(matches!(c.h, Coeffs::Zero(4)));
        assert!(matches!(c.j, Coeffs::I16(_)));
        assert!(matches!(Coeffs::new(&[40_000]), Coeffs::I32(_)));
        let g = c.graph();
        assert_eq!(g.j, vec![-1.0, 0.062, -0.062, 1.0]);
        // s = [+,+,-,-]: -1000*1 + 62*-1 + -62*1 + 1000*-1 + 5
        assert_eq!(c.energy_milli(&[1, 1, -1, -1]), -1000 - 62 - 62 - 1000 + 5);
    }

    /// Host cost of one twisted-dimer x-w62-p1200 instance on z(12,4), the
    /// number that bounds how many salts per second the screen can probe.
    /// CPU only: `cargo test --release --lib generation_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn generation_cost() {
        use quip_solver_core::ising_generators::descriptor::{
            admit, params_digest, AdvertisedLimits,
        };
        use quip_solver_core::ising_generators::family::{
            admit_topology, generate_instance, BaseTopology,
        };
        use quip_solver_core::ising_generators::ice::{topology::zephyr_lattice, RECIPES};
        let recipe = RECIPES[0];
        let (nodes, edges) = zephyr_lattice(12, 4).expect("z(12,4)");
        let topology = BaseTopology::new(nodes, &edges).expect("topology");
        let digest = params_digest(recipe.params);
        let admitted =
            admit(DIMER, b"quip/twisted-ice-dimer", 1, recipe.params, &digest).expect("admit");
        let limits = AdvertisedLimits {
            max_nodes: 1 << 30,
            max_edges: 1 << 30,
            max_degree: 0,
        };
        let bounds =
            admit_topology(&admitted, &topology, topology.order_digest(), &limits).expect("bounds");
        let n = 200u32;
        let started = Instant::now();
        let mut skipped = 0;
        let cache = EdgeCache::default();
        for k in 0..n {
            let mut nonce = [7u8; 32];
            nonce[..4].copy_from_slice(&k.to_le_bytes());
            match generate_instance(&admitted, &topology, &bounds, nonce) {
                Ok(instance) => {
                    drop((Coeffs::new(instance.j()), cache.edges(instance.edges())));
                }
                Err(_) => skipped += 1,
            }
        }
        let per = started.elapsed() / n;
        eprintln!(
            "twisted-dimer {}: {per:?} per instance on one core, {skipped} skipped",
            recipe.alias
        );
        // The nonce-independent part every instance recomputes: tile
        // candidates and their sides. A local generator could cache it.
        use quip_solver_core::ising_generators::ice::{candidates, sides, Lattice};
        let lattice = Lattice::Zephyr { m: 12, t: 4 };
        let started = Instant::now();
        for _ in 0..20 {
            let found = candidates(lattice, &topology).expect("candidates");
            drop(sides(lattice, &topology, &found).expect("sides"));
        }
        eprintln!(
            "  of which candidates + sides: {:?}",
            started.elapsed() / 20
        );
        let fast =
            FastDimer::new("quip/twisted-ice-dimer", 1, recipe.params, &topology).expect("fast");
        let started = Instant::now();
        for k in 0..n {
            let mut nonce = [7u8; 32];
            nonce[..4].copy_from_slice(&k.to_le_bytes());
            drop(fast.couplings(nonce).map(|(j, _)| Coeffs::new(&j)));
        }
        eprintln!(
            "  cached generator: {:?} per instance",
            started.elapsed() / n
        );
    }

    /// The cached generator reproduces the canonical couplings bit for bit, on
    /// the full z(12,4) and on a topology with nodes removed, for all three
    /// registered recipes.
    #[test]
    fn fast_dimer_matches_canonical() {
        use quip_solver_core::ising_generators::descriptor::{
            admit, params_digest, AdvertisedLimits,
        };
        use quip_solver_core::ising_generators::family::{
            admit_topology, generate_instance, BaseTopology,
        };
        use quip_solver_core::ising_generators::ice::{topology::zephyr_lattice, RECIPES};
        let (nodes, edges) = zephyr_lattice(12, 4).expect("z(12,4)");
        // Drop every 37th node, as a live chip has dead qubits.
        let dead: std::collections::HashSet<u32> = nodes.iter().copied().step_by(37).collect();
        let live: Vec<u32> = nodes
            .iter()
            .copied()
            .filter(|n| !dead.contains(n))
            .collect();
        let live_edges: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|(u, v)| !dead.contains(u) && !dead.contains(v))
            .collect();
        let limits = AdvertisedLimits {
            max_nodes: 1 << 30,
            max_edges: 1 << 30,
            max_degree: 0,
        };
        let mut compared = 0;
        for (node_ids, edges) in [(nodes.clone(), edges.clone()), (live, live_edges)] {
            let topology = BaseTopology::new(node_ids, &edges).expect("topology");
            for recipe in RECIPES {
                let digest = params_digest(recipe.params);
                let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, recipe.params, &digest)
                    .expect("admit");
                let bounds = admit_topology(&admitted, &topology, topology.order_digest(), &limits)
                    .expect("bounds");
                let fast = FastDimer::new("quip/twisted-ice-dimer", 1, recipe.params, &topology)
                    .expect("fast generator");
                for k in 0u32..12 {
                    let mut nonce = [0x5au8; 32];
                    nonce[..4].copy_from_slice(&k.to_le_bytes());
                    let canonical = generate_instance(&admitted, &topology, &bounds, nonce);
                    let quick = fast.couplings(nonce);
                    match canonical {
                        Ok(instance) => {
                            let (j, tiles) = quick.expect("fast generator returns an instance");
                            assert_eq!(
                                local_gauges(&instance).unwrap().0,
                                tiles,
                                "{} nonce {k}",
                                recipe.alias
                            );
                            assert!(
                                same_instance(&instance, &fast, &j),
                                "{} nonce {k}",
                                recipe.alias
                            );
                            compared += 1;
                        }
                        Err(_) => assert!(
                            quick.is_none(),
                            "{} nonce {k}: canonical skipped",
                            recipe.alias
                        ),
                    }
                }
            }
        }
        assert!(compared > 0);
        assert!(FastDimer::new(
            "quip/ice-glass",
            1,
            RECIPES[0].params,
            &BaseTopology::new(nodes, &edges).unwrap()
        )
        .is_none());
    }

    /// Local build (all6): the family parameters of the spec-120 topology
    /// 0x2d13 (weak 45, pairs 1350), read from its `RegisteredTopologies`
    /// meta. Not one of the three named recipes.
    const LIVE_W45_P1350: [u8; 34] = [
        0x01, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x04, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x46, 0x05, 0x00, 0x00,
    ];

    /// Local build (all6): on the live w45-p1350 parameters the cached
    /// generator reproduces the canonical instance, and stage 0 on its raw
    /// couplings ([`raw_stage0_model`], the local screen's
    /// `QUIP_OTSUKA_FAST_GEN` path) scores exactly what models 1 and 2 score
    /// on the canonical instance.
    #[test]
    fn fast_generator_and_raw_stage0_match_canonical_on_live_params() {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::{generate_instance, BaseTopology};
        let blob = include_bytes!("../tests/fixtures/aglais-z12t4-spec120.bin.gz");
        let topo = ising_generators::blob::decode(blob)
            .unwrap()
            .topology()
            .unwrap();
        let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
        let params = &LIVE_W45_P1350[..];
        let digest = params_digest(params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, params, &digest).unwrap();
        let bounds = admitted.size_bounds(base.stats()).unwrap();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, params, &base).expect("fast");
        let cache = EdgeCache::default();
        let shared = cache.edges(fast.edges());
        let mut compared = 0;
        for k in 0u32..16 {
            let mut nonce = [0x3cu8; 32];
            nonce[..4].copy_from_slice(&k.to_le_bytes());
            let Ok(instance) = generate_instance(&admitted, &base, &bounds, nonce) else {
                assert!(
                    fast.couplings(nonce).is_none(),
                    "nonce {k}: canonical skipped"
                );
                continue;
            };
            let (j, tiles) = fast.couplings(nonce).expect("fast instance");
            assert!(same_instance(&instance, &fast, &j), "nonce {k}");
            assert_eq!(local_gauges(&instance).unwrap().0, tiles, "nonce {k}");
            let own: Edges = instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect();
            for model in [1u8, 2, 3] {
                let canonical = instance_stage0_model(model, &instance, &own).unwrap();
                let raw = raw_stage0_model(model, fast.nodes(), &shared, &j, &tiles).unwrap();
                assert_eq!(
                    canonical.to_bits(),
                    raw.to_bits(),
                    "nonce {k} model {model}"
                );
            }
            compared += 1;
        }
        assert!(compared >= 12, "{compared}");
    }

    /// Local build (all6): host cost per salt of the local screen on the live
    /// w45-p1350 parameters, canonical vs cached generation, with models 1
    /// and 2. CPU only, one thread:
    /// `cargo test --release --lib local_screen_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn local_screen_cost() {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::{generate_instance, BaseTopology};
        let blob = include_bytes!("../tests/fixtures/aglais-z12t4-spec120.bin.gz");
        let topo = ising_generators::blob::decode(blob)
            .unwrap()
            .topology()
            .unwrap();
        let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
        let params = &LIVE_W45_P1350[..];
        let digest = params_digest(params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, params, &digest).unwrap();
        let bounds = admitted.size_bounds(base.stats()).unwrap();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, params, &base).expect("fast");
        let shared = EdgeCache::default().edges(fast.edges());
        let n = 100u32;
        let nonce = |k: u32| {
            let mut nonce = [0x77u8; 32];
            nonce[..4].copy_from_slice(&k.to_le_bytes());
            nonce
        };
        let started = Instant::now();
        let instances: Vec<Instance> = (0..n)
            .filter_map(|k| generate_instance(&admitted, &base, &bounds, nonce(k)).ok())
            .collect();
        let canonical = started.elapsed() / n;
        let started = Instant::now();
        let raw: Vec<_> = (0..n).filter_map(|k| fast.couplings(nonce(k))).collect();
        let cached = started.elapsed() / n;
        assert!(!instances.is_empty() && !raw.is_empty());
        // The stage-0 topologies are built once per edge set, outside the timing.
        for model in [1u8, 2] {
            std::hint::black_box(raw_stage0_model(
                model,
                fast.nodes(),
                &shared,
                &raw[0].0,
                &raw[0].1,
            ));
        }
        let started = Instant::now();
        for (j, tiles) in &raw {
            std::hint::black_box(raw_stage0_model(1, fast.nodes(), &shared, j, tiles));
        }
        let m1 = started.elapsed() / n;
        let started = Instant::now();
        for (j, tiles) in &raw {
            std::hint::black_box(raw_stage0_model(2, fast.nodes(), &shared, j, tiles));
        }
        let m2 = started.elapsed() / n;
        eprintln!(
            "local screen per salt: canonical generation {canonical:?}, cached {cached:?}, model 1 {m1:?}, model 2 {m2:?}"
        );
    }

    /// Host cost per salt of the stage-0 pipeline on the live topology: cached
    /// generation plus the lean score, as one generator thread runs it.
    /// CPU only: `cargo test --release --lib stage0_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn stage0_cost() {
        use quip_solver_core::ising_generators::ice::RECIPES;
        let (base, _) = live_dimer();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, RECIPES[0].params, &base).unwrap();
        let cache = EdgeCache::default();
        let edges = cache.edges(fast.edges());
        let started = Instant::now();
        let topo = stage0::Topo::shared(fast.nodes(), &edges);
        eprintln!(
            "stage-0 topology (once per edge set): {:?}",
            started.elapsed()
        );
        drop(topo);
        let n = 400u32;
        let (mut gen, mut score) = (Duration::ZERO, Duration::ZERO);
        let mut acc = 0f64;
        for k in 0..n {
            let mut nonce = [9u8; 32];
            nonce[..4].copy_from_slice(&k.to_le_bytes());
            let t0 = Instant::now();
            let (j, tiles) = fast.couplings(nonce).unwrap();
            let compact = Coeffs::new(&j);
            let t1 = Instant::now();
            acc += f64::from(stage0_milli(fast.nodes(), &edges, &j, &tiles));
            let t2 = Instant::now();
            drop(compact);
            gen += t1 - t0;
            score += t2 - t1;
        }
        eprintln!(
            "per salt on one core: cached generation {:?} + lean score {:?} = {:?} ({acc:.3})",
            gen / n,
            score / n,
            (gen + score) / n
        );
    }

    /// The live Aglais spec-120 topology (agentA `dl/zephyr-z12t4-live.bin.gz`)
    /// and the admitted x-w62-p1200 recipe.
    fn live_dimer() -> (
        quip_solver_core::ising_generators::family::BaseTopology,
        quip_solver_core::ising_generators::descriptor::SizeBounds,
    ) {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::BaseTopology;
        use quip_solver_core::ising_generators::ice::RECIPES;
        let blob = include_bytes!("../tests/fixtures/aglais-z12t4-spec120.bin.gz");
        let topo = ising_generators::blob::decode(blob)
            .unwrap()
            .topology()
            .unwrap();
        let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
        let recipe = RECIPES[0];
        assert_eq!(recipe.alias, "x-w62-p1200");
        let digest = params_digest(recipe.params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, recipe.params, &digest).unwrap();
        let bounds = admitted.size_bounds(base.stats()).unwrap();
        (base, bounds)
    }

    fn hex_nonce(hex: &str) -> [u8; 32] {
        let mut nonce = [0u8; 32];
        for (k, byte) in nonce.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap();
        }
        nonce
    }

    /// The lean score of agentA's model on the first rows of its
    /// `feat_all.tsv`, computed in Python from those rows and
    /// `screen_weights.json` (`bias + sum w * (x - mean) / sd`).
    #[allow(clippy::unreadable_literal)]
    const LEAN_REFERENCE: [(&str, f64); 6] = [
        (
            "000001aad2905c18c73eb6efec02e126230a5dd53e5c350b80888d5bbe08a7cd",
            -0.253900901697535,
        ),
        (
            "00041be00aced1f899f994713ad1e55c1aa6028d43acbb54e33aec0821a29a90",
            0.03348195635443595,
        ),
        (
            "00029453483ee1c274276acb426ce809838497fface230569d46dfc7edb6c513",
            -0.26303064778553326,
        ),
        (
            "0002250d0abead208d1024a3c3413b64a182e6b6f21338eeea22c5f47cfe5ec0",
            -0.14068192476475636,
        ),
        (
            "00044c7d8d461927b095665c7b533ba7f53b1b88ce43d73152a5fe6b63bbfe30",
            0.35257895155111507,
        ),
        (
            "0004ff9cee5fbf5333d4fd7ef562714774533ad7d1da13c12117e79861dfcbb7",
            0.08357218402832826,
        ),
    ];

    /// Stage 0 matches agentA's lean score on the canonical path (tiles from
    /// metadata), on the cached generator's path (tiles from `select`), and
    /// through the public `stage0_score` hook.
    #[test]
    fn stage0_matches_agent_a_lean_score() {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::generate_instance;
        use quip_solver_core::ising_generators::ice::RECIPES;
        let (base, bounds) = live_dimer();
        let params = RECIPES[0].params;
        let digest = params_digest(params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, params, &digest).unwrap();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, params, &base).unwrap();
        let fast = Fast {
            dimer: fast,
            trusted: AtomicBool::new(true),
        };
        let cache = EdgeCache::default();
        for (hex, reference) in LEAN_REFERENCE {
            let nonce = hex_nonce(hex);
            let instance = generate_instance(&admitted, &base, &bounds, nonce).unwrap();
            let canonical = canonical_candidate(&instance, 1, &cache, true);
            let check = |n| generate_instance(&admitted, &base, &bounds, n);
            // Index 0 also runs the 1-in-64 canonical comparison.
            let quick = fast_candidate(&fast, &check, nonce, 0, &cache, true);
            let quick = quick.expect("cached path");
            let (_, tiles) = fast.dimer.couplings(nonce).unwrap();
            let hook = stage0_score(&canonical.graph(), &tiles);
            for (label, got) in [
                ("canonical", canonical.stage0),
                ("cached", quick.stage0),
                ("hook", hook),
            ] {
                assert!(
                    (f64::from(got) - reference).abs() < 1e-5,
                    "{label} {hex}: {got} vs agentA {reference}"
                );
            }
        }
    }

    #[test]
    fn stage0_gate_keeps_about_the_requested_fraction_of_the_lowest() {
        let mut gate = Stage0Gate::new(0.1);
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
        let (mut kept, mut kept_low, n) = (0u32, 0u32, 20_000u32);
        for _ in 0..n {
            let x = (next_seed(&mut rng) >> 40) as f32 / (1u64 << 24) as f32;
            if gate.admit(x) {
                kept += 1;
                kept_low += u32::from(x <= 0.12);
            }
        }
        let fraction = f64::from(kept) / f64::from(n);
        assert!((0.09..=0.115).contains(&fraction), "kept {fraction}");
        assert!(kept_low * 100 >= kept * 97, "kept scores are the low ones");
        // keep = 1 passes everything.
        let mut all = Stage0Gate::new(1.0);
        assert!((0..500).all(|k| all.admit(k as f32)));
    }

    /// Model 3's cascade keeps: the prefilter takes `pre`, model 2 the rest
    /// of the keep, and a keep at or above `pre` is model 1 alone.
    #[test]
    fn stage0_cascade_keeps_split_the_keep() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let (pre, main) = cascade_keeps(0.01, 0.2);
        assert!(close(pre, 0.2) && close(main, 0.05), "{pre} {main}");
        assert!(close(pre * main, 0.01));
        let (pre, main) = cascade_keeps(0.025, 0.2);
        assert!(close(pre, 0.2) && close(main, 0.125));
        // keep >= pre: model 1 alone at the keep.
        let (pre, main) = cascade_keeps(0.3, 0.2);
        assert!(close(pre, 0.3) && close(main, 1.0));
        // Off stays off; out-of-range inputs are clamped.
        let (pre, main) = cascade_keeps(1.0, 0.2);
        assert!(close(pre, 1.0) && close(main, 1.0));
        let (pre, main) = cascade_keeps(0.01, 5.0);
        assert!(close(pre, 1.0) && close(main, 0.01));
    }

    /// The cascade over a synthetic stream: a cheap score `lean` correlated
    /// with the costly `full` one (as model 1 is with model 2, Spearman ~0.8
    /// on spec 120). The two gates keep about the requested fraction, the
    /// costly score runs on about `pre` of the salts, and the survivors are
    /// almost all of the stream's lowest `full` fraction.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn stage0_cascade_keeps_the_fraction_and_the_low_full_scores() {
        let (keep, pre) = (0.01, 0.2);
        let (pre_keep, main_keep) = cascade_keeps(keep, pre);
        let mut first = Stage0Gate::new(pre_keep);
        let mut second = Stage0Gate::new(main_keep);
        let mut rng = 0xD1B5_4A32_D192_ED03_u64;
        let mut unit = || (next_seed(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
        let n = 200_000u32;
        let (mut scored, mut kept) = (0u32, Vec::new());
        let mut fulls = Vec::with_capacity(n as usize);
        for _ in 0..n {
            // Box-Muller normals: full ~ N(0,1), lean = 0.8 full + 0.6 noise.
            let gauss = |u: f64, v: f64| {
                (-2.0 * u.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            };
            let full = gauss(unit(), unit());
            let lean = 0.8 * full + 0.6 * gauss(unit(), unit());
            fulls.push(full);
            if !first.admit(lean as f32) {
                continue;
            }
            scored += 1;
            if second.admit(full as f32) {
                kept.push(full);
            }
        }
        let fraction = kept.len() as f64 / f64::from(n);
        assert!((0.008..=0.012).contains(&fraction), "kept {fraction}");
        let scored = f64::from(scored) / f64::from(n);
        assert!((0.18..=0.22).contains(&scored), "model 2 ran on {scored}");
        // Survivors against the stream's own lowest-1% cutoff of `full`.
        fulls.sort_by(f64::total_cmp);
        let cutoff = fulls[(fulls.len() as f64 * 0.015) as usize];
        let low = kept.iter().filter(|&&f| f <= cutoff).count();
        assert!(
            low * 100 >= kept.len() * 85,
            "{low} of {} survivors are low",
            kept.len()
        );
    }

    /// The score from instance metadata equals the one from the generator's
    /// tile list (the Otsuka path scores the session's FamilyContext instance).
    #[test]
    fn instance_stage0_matches_agent_a() {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::generate_instance;
        use quip_solver_core::ising_generators::ice::RECIPES;
        let (base, bounds) = live_dimer();
        let params = RECIPES[0].params;
        let digest = params_digest(params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, params, &digest).unwrap();
        for (hex, reference) in LEAN_REFERENCE {
            let instance = generate_instance(&admitted, &base, &bounds, hex_nonce(hex)).unwrap();
            let edges: Edges = instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect();
            let got = instance_stage0(&instance, &edges).unwrap();
            assert!(
                (f64::from(got) - reference).abs() < 1e-5,
                "{hex}: {got} vs {reference}"
            );
        }
    }

    /// Model 2 on the first U-window salts of agent H's labelled set,
    /// computed in Python from the offline feature rows and
    /// `screen2_weights.json` (`bias + sum w * x`).
    #[allow(clippy::unreadable_literal)]
    const V2_REFERENCE: [(&str, f64); 6] = [
        (
            "00224aa75e5f1466f5aed15e03bf41d0f59746715bc1c33ad399c6d66e5f4b0b",
            0.5266234744009921,
        ),
        (
            "0057da37de0a2d838a58c9f6b8d150c602f26e598b768884740ead08a346697a",
            -0.29680472952736636,
        ),
        (
            "0073f6ef681cb6d7e4ee45b9ea1a4ce7d8595e7d0ea945cd3506d63ebe10ff5f",
            -0.1480589770570191,
        ),
        (
            "0079f97289e6cf34907d5c8cf4d73f3a3c25c2a1934bd6beef551aa04a3470ca",
            0.6123646446181397,
        ),
        (
            "007cafd3b757a25c1c889a95fd4f38cb68bb6a4808c40944a348dbec8f5a2d4e",
            0.5404323516859222,
        ),
        (
            "008612d8dcd438ec9efe7522f31cd9fc9a815fae2f67f573607354d56f153cb1",
            0.34938013145011837,
        ),
    ];

    /// Model 2 matches agent H's offline score on the canonical path (tiles
    /// from metadata, the Otsuka sampler's input), through the model switch,
    /// and on the cached generator's tile list.
    #[test]
    fn stage0_v2_matches_agent_h_reference() {
        use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
        use quip_solver_core::ising_generators::family::generate_instance;
        use quip_solver_core::ising_generators::ice::RECIPES;
        let (base, bounds) = live_dimer();
        let params = RECIPES[0].params;
        let digest = params_digest(params);
        let admitted = admit(DIMER, b"quip/twisted-ice-dimer", 1, params, &digest).unwrap();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, params, &base).unwrap();
        let cache = EdgeCache::default();
        let fast_edges = cache.edges(fast.edges());
        for (hex, reference) in V2_REFERENCE {
            let nonce = hex_nonce(hex);
            let instance = generate_instance(&admitted, &base, &bounds, nonce).unwrap();
            let edges: Edges = instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect();
            let canonical = instance_stage0_v2(&instance, &edges).unwrap();
            let switched = instance_stage0_model(2, &instance, &edges).unwrap();
            // Model 3 ranks with model 2's score (its prefilter is a gate).
            assert_eq!(
                instance_stage0_model(3, &instance, &edges)
                    .unwrap()
                    .to_bits(),
                switched.to_bits()
            );
            // The harness entry point (otsuka_vs_msa stage0 --stage0-model)
            // scores the same graph from milli-float couplings.
            {
                let (tiles, _) =
                    quip_solver_core::ising_generators::ice::local_gauges(&instance).unwrap();
                let graph = IsingGraph::new(
                    instance
                        .h()
                        .iter()
                        .map(|&h| f64::from(h) / 1000.0)
                        .collect(),
                    instance
                        .j()
                        .iter()
                        .map(|&j| f64::from(j) / 1000.0)
                        .collect(),
                    edges.to_vec(),
                );
                assert_eq!(
                    stage0_score_model(2, &graph, &tiles).to_bits(),
                    canonical.to_bits()
                );
                assert_eq!(
                    stage0_score_model(3, &graph, &tiles).to_bits(),
                    canonical.to_bits()
                );
                assert_eq!(
                    stage0_score_model(1, &graph, &tiles).to_bits(),
                    instance_stage0(&instance, &edges).unwrap().to_bits()
                );
            }
            let lean = instance_stage0_model(1, &instance, &edges).unwrap();
            assert_eq!(
                lean.to_bits(),
                instance_stage0(&instance, &edges).unwrap().to_bits()
            );
            let (j, tiles) = fast.couplings(nonce).unwrap();
            let topo = stage0v2::Topo2::shared(fast.nodes(), &fast_edges);
            let cached = stage0v2::score(&topo, &j, &tiles).unwrap();
            for (label, got) in [
                ("canonical", canonical),
                ("switch", switched),
                ("cached", cached),
            ] {
                assert!(
                    (f64::from(got) - reference).abs() < 2e-5,
                    "{label} {hex}: {got} vs agent H {reference}"
                );
            }
        }
    }

    /// Brute force over the pair-contracted multigraph of a small random
    /// instance: every 2-, 3- and 4-cycle by depth-first search over edge
    /// sequences, deduplicated by edge set, classified by pair traversals,
    /// weak edges and frustration. `stage0v2::features` must agree exactly.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::items_after_statements,
        clippy::too_many_arguments
    )]
    fn stage0_v2_features_match_brute_force_cycles() {
        use std::collections::HashMap;
        let mut seed = 0x0123_4567_89AB_CDEF_u64;
        let mut covered = [0u32; 3];
        for round in 0..12 {
            let n = 16usize;
            let mut keep = |p: u64| next_seed(&mut seed) % 100 < p;
            // Two tiles on 0..4 and 3..7 sharing node 3, every other edge with
            // probability ~0.55.
            let tiles: Vec<[u32; 4]> = vec![[0, 1, 2, 3], [3, 4, 5, 6]];
            let mut edge_set = std::collections::BTreeSet::new();
            for t in &tiles {
                for a in 0..4 {
                    for b in a + 1..4 {
                        let (u, v) = (t[a] as usize, t[b] as usize);
                        edge_set.insert((u.min(v), u.max(v)));
                    }
                }
            }
            for u in 0..n {
                for v in u + 1..n {
                    if keep(55) {
                        edge_set.insert((u, v));
                    }
                }
            }
            let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
            let tile_edge = |u: usize, v: usize| {
                tiles
                    .iter()
                    .any(|t| t.contains(&(u as u32)) && t.contains(&(v as u32)))
            };
            // Disjoint pairs on non-tile edges, then weak +-62 elsewhere and
            // gauged +-1000 on tile edges.
            let mut used = vec![false; n];
            let mut j = vec![0i32; edges.len()];
            for (k, &(u, v)) in edges.iter().enumerate() {
                if tile_edge(u, v) {
                    j[k] = if next_seed(&mut seed) & 1 == 0 {
                        1000
                    } else {
                        -1000
                    };
                } else if !used[u] && !used[v] && next_seed(&mut seed).is_multiple_of(3) {
                    used[u] = true;
                    used[v] = true;
                    j[k] = -1000;
                } else {
                    j[k] = if next_seed(&mut seed) & 1 == 0 {
                        62
                    } else {
                        -62
                    };
                }
            }
            let shared: Edges = edges.clone().into();
            let topo = stage0v2::Topo2::new(n, shared);
            let got = stage0v2::features(&topo, &j, &tiles).unwrap();
            // Brute force.
            let mut partner = vec![usize::MAX; n];
            let mut expect = [0u32; stage0v2::NF];
            let membership = |x: usize| tiles.iter().filter(|t| t.contains(&(x as u32))).count();
            for (k, &(u, v)) in edges.iter().enumerate() {
                if !tile_edge(u, v) && j[k] == -1000 {
                    partner[u] = v;
                    partner[v] = u;
                    let (a, b) = (
                        membership(u).min(membership(v)),
                        membership(u).max(membership(v)),
                    );
                    let cat = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
                        .iter()
                        .position(|&c| c == (a, b))
                        .unwrap();
                    expect[cat] += 1;
                }
            }
            let sid = |x: usize| {
                if partner[x] == usize::MAX {
                    x
                } else {
                    x.min(partner[x])
                }
            };
            // contracted edges: (su, sv, u, v, weak, positive)
            let ce: Vec<(usize, usize, usize, usize, bool, bool)> = edges
                .iter()
                .enumerate()
                .filter(|&(k, &(u, v))| tile_edge(u, v) || j[k] != -1000)
                .map(|(k, &(u, v))| (sid(u), sid(v), u, v, !tile_edge(u, v), j[k] > 0))
                .collect();
            let mut cycles: HashMap<Vec<usize>, (usize, usize, bool, usize)> = HashMap::new();
            // DFS: (start super, current super, current end node, start end node, edges, visited supers)
            fn walk(
                ce: &[(usize, usize, usize, usize, bool, bool)],
                start: usize,
                at: usize,
                at_end: usize,
                first_end: usize,
                path: &mut Vec<usize>,
                seen: &mut Vec<usize>,
                np: usize,
                out: &mut HashMap<Vec<usize>, (usize, usize, bool, usize)>,
            ) {
                for (k, &(su, sv, u, v, _, _)) in ce.iter().enumerate() {
                    if path.contains(&k) {
                        continue;
                    }
                    let (next, near, far) = if su == at {
                        (sv, u, v)
                    } else if sv == at {
                        (su, v, u)
                    } else {
                        continue;
                    };
                    let first = path.is_empty();
                    let turn = usize::from(!first && near != at_end);
                    let start_end = if first { near } else { first_end };
                    if next == start && path.len() + 1 >= 2 {
                        let np = np + turn + usize::from(far != start_end);
                        let mut key: Vec<usize> = path.clone();
                        key.push(k);
                        let nw = key.iter().filter(|&&e| ce[e].4).count();
                        let f = key.iter().filter(|&&e| ce[e].5).count() % 2 == 1;
                        let len = key.len();
                        key.sort_unstable();
                        out.insert(key, (np, nw, f, len));
                    }
                    if path.len() + 1 < 4 && !seen.contains(&next) {
                        path.push(k);
                        seen.push(next);
                        walk(ce, start, next, far, start_end, path, seen, np + turn, out);
                        seen.pop();
                        path.pop();
                    }
                }
            }
            let supers: std::collections::BTreeSet<usize> = (0..n).map(sid).collect();
            for &s in &supers {
                let mut visited = vec![s];
                walk(
                    &ce,
                    s,
                    s,
                    usize::MAX,
                    usize::MAX,
                    &mut Vec::new(),
                    &mut visited,
                    0,
                    &mut cycles,
                );
            }
            for &(np, nw, f, len) in cycles.values() {
                let at = match len {
                    2 => 6 + ((np - 1) * 3 + nw) * 2,
                    3 => 6 + 12 + (np * 4 + nw) * 2,
                    _ => 6 + 12 + 32 + (np * 5 + nw) * 2,
                };
                expect[at] += 1;
                expect[at + 1] += u32::from(f);
            }
            assert_eq!(got.to_vec(), expect.to_vec(), "round {round}");
            for (class, at) in [(0, 6), (1, 6 + 12), (2, 6 + 12 + 32)] {
                let width = [12, 32, 50][class];
                let traversing: u32 = got[at..at + width]
                    .chunks(2)
                    .enumerate()
                    .filter(|(k, _)| class == 0 || k / [3, 4, 5][class] > 0)
                    .map(|(_, c)| c[0])
                    .sum();
                covered[class] += traversing;
            }
        }
        assert!(
            covered.iter().all(|&c| c > 0),
            "pair-traversing cycles exercised: {covered:?}"
        );
    }

    /// Host cost per salt of model 2 on the live topology, one core.
    /// CPU only: `cargo test --release --lib stage0_v2_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn stage0_v2_cost() {
        use quip_solver_core::ising_generators::ice::RECIPES;
        let (base, _) = live_dimer();
        let fast = FastDimer::new("quip/twisted-ice-dimer", 1, RECIPES[0].params, &base).unwrap();
        let cache = EdgeCache::default();
        let edges = cache.edges(fast.edges());
        let topo = stage0v2::Topo2::shared(fast.nodes(), &edges);
        let n = 200u32;
        let inputs: Vec<_> = (0..n)
            .map(|k| {
                let mut nonce = [9u8; 32];
                nonce[..4].copy_from_slice(&k.to_le_bytes());
                fast.couplings(nonce).unwrap()
            })
            .collect();
        let t0 = Instant::now();
        let mut acc = 0f64;
        for (j, tiles) in &inputs {
            acc += f64::from(
                stage0::lean_score(&stage0::Topo::shared(fast.nodes(), &edges), j, tiles).unwrap(),
            );
        }
        let lean = t0.elapsed() / n;
        let t1 = Instant::now();
        for (j, tiles) in &inputs {
            acc += f64::from(stage0v2::score(&topo, j, tiles).unwrap());
        }
        let v2 = t1.elapsed() / n;
        eprintln!(
            "per salt on one core: lean {lean:?}, model 2 {v2:?} ({:.1} salts/s) ({acc:.3})",
            1.0 / v2.as_secs_f64()
        );
    }

    #[test]
    fn enabled_declaration_validates() {
        let declaration = quip_solver_core::families::Declaration {
            solver_id: "",
            generated: CLAIMS,
            accepted: &[],
        };
        let table = declaration.validate().expect("declaration validates");
        assert!(table.declares_generated("quip/twisted-ice-dimer", 1));
        let generic = quip_solver_core::families::Declaration::GENERIC
            .validate()
            .unwrap();
        assert!(!generic.declares_generated("quip/twisted-ice-dimer", 1));
    }
}
