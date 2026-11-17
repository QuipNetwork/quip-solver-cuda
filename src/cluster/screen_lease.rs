//! Lease-local salt screening for `quip-cuda-otsuka` (opt-in).
//!
//! A family lease holds tens of thousands of salts. Their instances differ by several units in how
//! low a good anneal gets, and a few structural counts predict part of that before any solve. With
//! screening on, the miner generates every salt of a lease on CPU threads, scores it, and spends the
//! GPU only on the best ones:
//!
//! 1. Score every salt (lower is better) from four counts of its canonical instance: nodes in no
//!    tile and no dimer pair (`free`), dimer pairs between two tiled nodes (`tiled_pairs`), and the
//!    mean weak energy over the ice states inside each tile (`f_mean`) and across tiles that share a
//!    variable (`m2`). Fitted on 304 winning Aglais twisted-ice instances against random ones, the
//!    score puts held-out winners 2.0 standard deviations from random salts.
//! 2. Keep the best `keep` salts of the lease; report the rest with `LeaseSink::screen`.
//! 3. Give the best `probe_count` kept salts a short GPU solve each, then one long anneal to the best
//!    of those. A long anneal (many sweeps, few reads) reaches energies that many short runs do not.
//!
//! One lease generates at a time with all `threads`; the last generator thread hands the turn on as
//! soon as generation ends. Only the lease holding the best candidates waits for the GPU; every other
//! lease returns once screened, so leases waiting on the GPU never hold the session's credits.
//! Every wait and every GPU solve stops as soon as the lease does (the session allows a stopped lease
//! thread one second to return).
//!
//! Settings (environment):
//!
//! - `QUIP_OTSUKA_SCREEN="threads,warm,keep"` turns screening on: generator threads, salts screened
//!   before the GPU starts, candidates kept per lease. Unset = the default session path.
//! - `QUIP_OTSUKA_SCREEN_RUNGS="probe_reads,probe_sweeps,probe_count,final_reads,final_sweeps"`,
//!   default `160,8192,8,256,262144`.
//! - `QUIP_OTSUKA_SCREEN_SCORE="w_free,w_tiled_pairs,w_fmean,w_m2"`, default `1,0.773,20.8,22.2`.
//! - `QUIP_OTSUKA_SCREEN_HOLDERS=<n>`: leases allowed to wait for the GPU, default 1.

use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    IsingGraph, Lease, LeaseSink, LeaseStopped, SampleError, SampleParams, SamplerResult,
};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// GPU budgets of the two solve stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rungs {
    /// Reads of each short solve.
    pub probe_reads: usize,
    /// Sweeps of each short solve.
    pub probe_sweeps: usize,
    /// Candidates that get a short solve per batch.
    pub probe_count: usize,
    /// Reads of the long anneal.
    pub final_reads: usize,
    /// Sweeps of the long anneal.
    pub final_sweeps: usize,
}

/// Default solve stages.
pub const DEFAULT_RUNGS: Rungs = Rungs {
    probe_reads: 160,
    probe_sweeps: 8192,
    probe_count: 8,
    final_reads: 256,
    final_sweeps: 262_144,
};

/// Screen settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenCfg {
    /// Generator threads.
    pub threads: usize,
    /// Salts screened before the lease may start GPU work.
    pub warm: u64,
    /// Candidates kept per lease.
    pub keep: usize,
    /// Score weights: free nodes, tiled pairs, `f_mean` (units), `m2` (units).
    pub weights: [f64; 4],
    /// GPU solve stages.
    pub rungs: Rungs,
    /// Leases allowed to wait for the GPU.
    pub holders: usize,
}

/// Default score weights.
pub const DEFAULT_WEIGHTS: [f64; 4] = [1.0, 0.773, 20.8, 22.2];

fn floats(v: &str) -> Vec<f64> {
    v.split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .collect()
}

fn counts(v: &str) -> Vec<usize> {
    v.split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect()
}

/// The settings from the environment; `None` when screening is off.
#[must_use]
pub fn screen_cfg() -> Option<ScreenCfg> {
    let get = |key: &str| std::env::var(key).ok();
    parse_cfg(
        get("QUIP_OTSUKA_SCREEN").as_deref(),
        get("QUIP_OTSUKA_SCREEN_SCORE").as_deref(),
        get("QUIP_OTSUKA_SCREEN_RUNGS").as_deref(),
        get("QUIP_OTSUKA_SCREEN_HOLDERS").as_deref(),
    )
}

/// [`screen_cfg`] from the four setting strings. `None` when `screen` is missing, short, or has
/// zero threads or zero kept candidates. A malformed score, rungs or holders value falls back to
/// its default.
#[must_use]
pub fn parse_cfg(
    screen: Option<&str>,
    score: Option<&str>,
    rungs: Option<&str>,
    holders: Option<&str>,
) -> Option<ScreenCfg> {
    let f = counts(screen?);
    if f.len() < 3 || f[0] == 0 || f[2] == 0 {
        return None;
    }
    let weights = score
        .map(floats)
        .filter(|g| g.len() == 4)
        .map_or(DEFAULT_WEIGHTS, |g| [g[0], g[1], g[2], g[3]]);
    let rungs = rungs
        .map(counts)
        .filter(|g| g.len() == 5 && g.iter().all(|&x| x > 0))
        .map_or(DEFAULT_RUNGS, |g| Rungs {
            probe_reads: g[0],
            probe_sweeps: g[1],
            probe_count: g[2],
            final_reads: g[3],
            final_sweeps: g[4],
        });
    let holders = holders
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    Some(ScreenCfg {
        threads: f[0],
        warm: f[1] as u64,
        keep: f[2],
        weights,
        rungs,
        holders,
    })
}

/// A stopped lease refuses further reports; nothing is left to do with that salt.
fn settled(_: Result<(), LeaseStopped>) {}

/// A score in thousandths, for ordering candidates.
#[expect(
    clippy::cast_possible_truncation,
    reason = "scores are a few thousand, far inside i64"
)]
fn score_milli(s: f64) -> i64 {
    (1000.0 * s) as i64
}

/// Whole milliseconds of `d`, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// One GPU solve at a time across all leases.
static GPU: Mutex<()> = Mutex::new(());

/// A lease waiting for the GPU with its best candidate score.
struct Holder {
    id: u64,
    best: i64,
    abandon: Arc<AtomicBool>,
}
static HOLDERS: Mutex<Vec<Holder>> = Mutex::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The generation turn. A flag, not a guard, so the lease's last generator thread can release it
/// the moment generation ends, while the lease's own thread is busy in a GPU solve.
static GEN_BUSY: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Take the generation turn, giving up when the lease stops.
fn gen_acquire(stopped: &dyn Fn() -> bool) -> bool {
    let (m, c) = &GEN_BUSY;
    let mut busy = m.lock().unwrap_or_else(PoisonError::into_inner);
    while *busy {
        if stopped() {
            return false;
        }
        busy = c
            .wait_timeout(busy, Duration::from_millis(5))
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
    *busy = true;
    true
}

fn gen_release() {
    let (m, c) = &GEN_BUSY;
    *m.lock().unwrap_or_else(PoisonError::into_inner) = false;
    c.notify_all();
}

/// Take `m`, giving up as soon as the lease stops.
fn lock_or_stop<'a>(
    m: &'a Mutex<()>,
    stopped: &dyn Fn() -> bool,
) -> Option<std::sync::MutexGuard<'a, ()>> {
    loop {
        match m.try_lock() {
            Ok(g) => return Some(g),
            Err(std::sync::TryLockError::Poisoned(e)) => return Some(e.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => {
                if stopped() {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

struct Cand {
    score: i64,
    energy: i64,
    idx: u64,
    inst: Arc<Instance>,
}

#[derive(Default)]
struct Shared {
    cands: Vec<Cand>,
    screened: u64,
    skipped: u64,
}

/// The structural score of an instance (lower is better) and the energy of its all-up state
/// (milli, offset excluded), which is what a screened-out salt reports.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one pass builds the adjacency, finds tiles and pairs, and sums the weak fits"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "node, edge and tile indices of an admitted instance are far below u32::MAX"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "node counts and milli sums are far below 2^52"
)]
pub fn score(inst: &Instance, weights: &[f64; 4]) -> (f64, i64) {
    let n = inst.node_ids().len();
    let edges = inst.edges();
    let j = inst.j();
    let mut deg = vec![0u32; n + 1];
    for &(u, v) in edges {
        deg[u as usize + 1] += 1;
        deg[v as usize + 1] += 1;
    }
    for u in 0..n {
        deg[u + 1] += deg[u];
    }
    let off = deg;
    let mut pos: Vec<u32> = off[..n].to_vec();
    let mut nb = vec![0u32; 2 * edges.len()];
    let mut wt = vec![0i32; 2 * edges.len()];
    for (&(u, v), &w) in edges.iter().zip(j) {
        let (u, v) = (u as usize, v as usize);
        nb[pos[u] as usize] = v as u32;
        wt[pos[u] as usize] = w;
        pos[u] += 1;
        nb[pos[v] as usize] = u as u32;
        wt[pos[v] as usize] = w;
        pos[v] += 1;
    }
    let strong = |w: i32| w == 1000 || w == -1000;
    // A strong edge is a tile edge when its ends have two common strong neighbours that are joined by a
    // strong edge (the edge lies in a K4); every other strong edge is a dimer pair.
    let mut partner = vec![u32::MAX; n];
    let mut mark = vec![u32::MAX; n];
    let mut tiled = vec![false; n];
    let is_strong_edge = |a: usize, b: usize| {
        (off[a] as usize..off[a + 1] as usize).any(|k| nb[k] as usize == b && strong(wt[k]))
    };
    for u in 0..n {
        for k in off[u] as usize..off[u + 1] as usize {
            if strong(wt[k]) {
                mark[nb[k] as usize] = u as u32;
            }
        }
        for k in off[u] as usize..off[u + 1] as usize {
            let v = nb[k] as usize;
            if !strong(wt[k]) || v < u {
                continue;
            }
            let common: Vec<usize> = (off[v] as usize..off[v + 1] as usize)
                .filter(|&k2| strong(wt[k2]) && mark[nb[k2] as usize] == u as u32)
                .map(|k2| nb[k2] as usize)
                .collect();
            let mut in_tile = false;
            'outer: for a in 0..common.len() {
                for b in a + 1..common.len() {
                    if is_strong_edge(common[a], common[b]) {
                        in_tile = true;
                        break 'outer;
                    }
                }
            }
            if in_tile {
                tiled[u] = true;
                tiled[v] = true;
            } else {
                partner[u] = v as u32;
                partner[v] = u as u32;
            }
        }
    }
    let free = (0..n)
        .filter(|&u| !tiled[u] && partner[u] == u32::MAX)
        .count();
    let pair_tt = (0..n)
        .filter(|&u| {
            partner[u] != u32::MAX
                && (partner[u] as usize) > u
                && tiled[u]
                && tiled[partner[u] as usize]
        })
        .count();
    // First-order fit of the weak couplings to the strong structure:
    // f_mean = sum over tiles of the mean weak energy inside the tile over its six ice states;
    // m2 = the same for weak couplings between two tiles that share a variable. Pair partners are one variable.
    let (mut f_mean, mut m2) = (0.0f64, 0.0f64);
    {
        let strong_j = |a: usize, b: usize| -> i32 {
            for k in off[a] as usize..off[a + 1] as usize {
                if nb[k] as usize == b && strong(wt[k]) {
                    return wt[k];
                }
            }
            0
        };
        let rep = |u: usize| {
            if partner[u] != u32::MAX && (partner[u] as usize) < u {
                partner[u] as usize
            } else {
                u
            }
        };
        let mut ice_tiles: Vec<([usize; 4], [f64; 4])> = Vec::with_capacity(n / 2);
        let mut sn: Vec<usize> = Vec::with_capacity(16);
        for u in 0..n {
            if !tiled[u] {
                continue;
            }
            sn.clear();
            sn.extend(
                (off[u] as usize..off[u + 1] as usize)
                    .filter(|&k| strong(wt[k]) && nb[k] as usize > u)
                    .map(|k| nb[k] as usize),
            );
            for a in 0..sn.len() {
                for b in a + 1..sn.len() {
                    if strong_j(sn[a], sn[b]) == 0 {
                        continue;
                    }
                    for c in b + 1..sn.len() {
                        if strong_j(sn[a], sn[c]) != 0 && strong_j(sn[b], sn[c]) != 0 {
                            let g = |k: usize| if strong_j(u, k) > 0 { 1.0 } else { -1.0 };
                            ice_tiles.push((
                                [rep(u), rep(sn[a]), rep(sn[b]), rep(sn[c])],
                                [1.0, g(sn[a]), g(sn[b]), g(sn[c])],
                            ));
                        }
                    }
                }
            }
        }
        let mut wl: Vec<(u64, i32)> = Vec::with_capacity(edges.len());
        for (&(u, v), &w) in edges.iter().zip(j) {
            if strong(w) {
                continue;
            }
            let (a, b) = (rep(u as usize), rep(v as usize));
            if a != b {
                wl.push(((a.min(b) * n + a.max(b)) as u64, w));
            }
        }
        wl.sort_unstable_by_key(|e| e.0);
        let mut keys: Vec<u64> = Vec::with_capacity(wl.len());
        let mut vals: Vec<i32> = Vec::with_capacity(wl.len());
        for (k, w) in wl {
            if keys.last() == Some(&k) {
                if let Some(last) = vals.last_mut() {
                    *last += w;
                }
            } else {
                keys.push(k);
                vals.push(w);
            }
        }
        let wget = |a: usize, b: usize| -> f64 {
            if a == b {
                return 0.0;
            }
            keys.binary_search(&((a.min(b) * n + a.max(b)) as u64))
                .map_or(0.0, |i| f64::from(vals[i]))
        };
        let mut cnt = vec![0u8; n];
        let mut memb = vec![[(0u32, 0u8); 4]; n];
        for (t, (r, g)) in ice_tiles.iter().enumerate() {
            for a in 0..4 {
                let c = cnt[r[a]] as usize;
                if c < 4 {
                    memb[r[a]][c] = (t as u32, a as u8);
                    cnt[r[a]] += 1;
                }
                for c2 in a + 1..4 {
                    f_mean -= wget(r[a], r[c2]) * g[a] * g[c2] / 3.0;
                }
            }
        }
        for w in 0..n {
            let c = cnt[w] as usize;
            for i1 in 0..c {
                for i2 in i1 + 1..c {
                    let (t1, a1) = memb[w][i1];
                    let (t2, a2) = memb[w][i2];
                    if t1 == t2 {
                        continue;
                    }
                    let (r1, g1) = &ice_tiles[t1 as usize];
                    let (r2, g2) = &ice_tiles[t2 as usize];
                    let gw = g1[a1 as usize] * g2[a2 as usize];
                    for i in 0..4 {
                        if i == a1 as usize {
                            continue;
                        }
                        for jx in 0..4 {
                            if jx == a2 as usize {
                                continue;
                            }
                            m2 += wget(r1[i], r2[jx]) * g1[i] * gw * g2[jx] / 9.0;
                        }
                    }
                }
            }
        }
    }
    let s = weights[0] * free as f64
        + weights[1] * pair_tt as f64
        + weights[2] * f_mean / 1000.0
        + weights[3] * m2 / 1000.0;
    // Every spin +1: the fields and the couplings each count once.
    let all_up: i64 = inst.h().iter().chain(j).map(|&w| i64::from(w)).sum();
    (s, all_up)
}

fn seed_of(nonce: &[u8; 32], salt: u64) -> u64 {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    for c in nonce.chunks(8) {
        let mut b = [0u8; 8];
        b[..c.len()].copy_from_slice(c);
        x = (x ^ u64::from_le_bytes(b)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 31;
    }
    x
}

/// Work one family lease: screen every salt on the CPU, solve the best on the GPU.
///
/// `solve` runs one generated salt (its instance and the lease's contract) with a cancel check,
/// returning reads with exact energies.
///
/// # Errors
/// A fatal device error from `solve`.
#[expect(
    clippy::too_many_lines,
    reason = "generation, holder hand-off and the two GPU stages share one lease's state and stop check"
)]
pub fn run_lease<F>(
    lease: &Lease,
    params: &SampleParams,
    sink: &LeaseSink,
    cfg: &ScreenCfg,
    solve: F,
) -> Result<(), SampleError>
where
    F: Fn(
        &IsingGraph,
        &SampleParams,
        &Instance,
        &'static Contract,
        &dyn Fn() -> bool,
    ) -> Result<Vec<SamplerResult>, SampleError>,
{
    let Some(family) = lease.family() else {
        return Err(SampleError::Capacity);
    };
    let count = lease.salt_count();
    let t0 = Instant::now();
    let next = AtomicU64::new(0);
    let workers_done = AtomicUsize::new(0);
    let shared = Mutex::new(Shared::default());
    let mut solved = 0u64;
    let mut best_solved = i64::MAX;
    let mut fatal: Option<SampleError> = None;
    let nworkers = cfg.threads.max(1);
    let stopped = || sink.is_stopped();
    if !gen_acquire(&stopped) {
        return Ok(());
    }
    let mut generating = true;
    let t_gen = Instant::now();
    let my_abandon = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        for _ in 0..nworkers {
            scope.spawn(|| {
                loop {
                    if stopped() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    if let Ok(inst) = family.generate(lease.nonce(i)) {
                        let (s, all_up) = score(&inst, &cfg.weights);
                        let score = score_milli(s);
                        let energy = all_up + inst.offset_milli();
                        let evicted = {
                            let mut st = shared.lock().unwrap_or_else(PoisonError::into_inner);
                            st.screened += 1;
                            let cand = Cand {
                                score,
                                energy,
                                idx: i,
                                inst: Arc::new(inst),
                            };
                            if st.cands.len() < cfg.keep {
                                st.cands.push(cand);
                                None
                            } else {
                                let (wi, worst) = st
                                    .cands
                                    .iter()
                                    .enumerate()
                                    .max_by_key(|(_, c)| c.score)
                                    .map_or((0, i64::MIN), |(k, c)| (k, c.score));
                                if cand.score < worst {
                                    Some(std::mem::replace(&mut st.cands[wi], cand))
                                } else {
                                    Some(cand)
                                }
                            }
                        };
                        if let Some(c) = evicted {
                            settled(sink.screen(c.idx, c.energy));
                        }
                    } else {
                        shared
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .skipped += 1;
                        settled(sink.skip(i));
                    }
                }
                if workers_done.fetch_add(1, Ordering::AcqRel) + 1 == nworkers {
                    gen_release();
                }
            });
        }
        // The lease thread drives the GPU.
        let build = |c: &Cand| {
            let inst = &c.inst;
            let edges: Vec<(usize, usize)> = inst
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect();
            let h: Vec<f64> = inst.h().iter().map(|&x| f64::from(x) / 1000.0).collect();
            let jv: Vec<f64> = inst.j().iter().map(|&x| f64::from(x) / 1000.0).collect();
            IsingGraph::new(h, jv, edges)
        };
        let my_id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut am_holder = false;
        loop {
            if stopped() {
                break;
            }
            let gen_done = workers_done.load(Ordering::Acquire) == nworkers;
            if gen_done && generating {
                generating = false;
                tracing::debug!(
                    ms = millis(t_gen.elapsed()),
                    "otsuka screen generation done"
                );
            }
            {
                let my_best = shared
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .cands
                    .iter()
                    .map(|c| c.score)
                    .min();
                let ready = gen_done
                    || shared
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .screened
                        >= cfg.warm;
                if am_holder && my_abandon.load(Ordering::Acquire) {
                    am_holder = false;
                    tracing::info!(
                        lease = my_id,
                        "otsuka screen: a better lease took the GPU turn"
                    );
                }
                if am_holder {
                    if let Some(b) = my_best {
                        let mut h = HOLDERS.lock().unwrap_or_else(PoisonError::into_inner);
                        if let Some(e) = h.iter_mut().find(|e| e.id == my_id) {
                            e.best = b;
                        }
                    }
                } else if let (true, Some(b)) = (ready, my_best) {
                    let mut h = HOLDERS.lock().unwrap_or_else(PoisonError::into_inner);
                    // drop replaced holders and holders whose lease has ended (their flag has no other owner)
                    h.retain(|e| {
                        !e.abandon.load(Ordering::Acquire) && Arc::strong_count(&e.abandon) > 1
                    });
                    if h.len() < cfg.holders {
                        h.push(Holder {
                            id: my_id,
                            best: b,
                            abandon: Arc::clone(&my_abandon),
                        });
                        am_holder = true;
                    } else if let Some(wi) = h
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, e)| e.best)
                        .map(|(i, _)| i)
                    {
                        if b < h[wi].best {
                            h[wi].abandon.store(true, Ordering::Release);
                            h.remove(wi);
                            h.push(Holder {
                                id: my_id,
                                best: b,
                                abandon: Arc::clone(&my_abandon),
                            });
                            am_holder = true;
                        }
                    }
                }
                if !am_holder {
                    if gen_done {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                }
            }
            let take: Option<Vec<Cand>> = {
                let mut st = shared.lock().unwrap_or_else(PoisonError::into_inner);
                if st.cands.is_empty() {
                    if gen_done {
                        break;
                    }
                    None
                } else if !gen_done && st.screened < cfg.warm {
                    None
                } else {
                    let want = cfg.rungs.probe_count;
                    st.cands.sort_by_key(|c| c.score);
                    let n = want.min(st.cands.len());
                    Some(st.cands.drain(..n).collect())
                }
            };
            let Some(batch) = take else {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            };
            let Some(_gpu) = lock_or_stop(&GPU, &stopped) else {
                for c in &batch {
                    settled(sink.screen(c.idx, c.energy));
                }
                break;
            };
            // one solve: returns the reads with the offset added, or None after reporting the salt
            let solve_one = |c: &Cand,
                             reads: usize,
                             sweeps: usize,
                             tag: &str|
             -> Result<Option<Vec<SamplerResult>>, SampleError> {
                let graph = build(c);
                let mut p = params.clone();
                if reads > 0 {
                    p.num_reads = reads;
                    p.num_sweeps = sweeps;
                }
                p.seed = seed_of(&lease.nonce(c.idx), c.idx ^ 0x5eed ^ (sweeps as u64));
                let ts = Instant::now();
                match solve(&graph, &p, &c.inst, family.contract(), &stopped) {
                    Ok(mut rs) if !rs.is_empty() => {
                        let off = c.inst.offset_milli();
                        for r in &mut rs {
                            r.energy_milli += off;
                        }
                        let best = rs.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX);
                        let screened = shared
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .screened;
                        tracing::info!(
                            salt = c.idx,
                            probe = c.score,
                            best,
                            ms = millis(ts.elapsed()),
                            screened,
                            stage = tag,
                            "otsuka screen solve"
                        );
                        Ok(Some(rs))
                    }
                    Ok(_) => Ok(None),
                    Err(e) if e.is_fatal() => Err(e),
                    Err(e) => {
                        if stopped() {
                            tracing::info!(
                                salt = c.idx,
                                ms = millis(ts.elapsed()),
                                "otsuka screen solve cancelled: the lease stopped"
                            );
                        } else {
                            tracing::warn!(error = %e, salt = c.idx, "otsuka screen solve failed; screening the salt");
                        }
                        Ok(None)
                    }
                }
            };
            let mut probed: Vec<(i64, usize, Vec<SamplerResult>)> = Vec::new();
            for (k, c) in batch.iter().enumerate() {
                if stopped() || fatal.is_some() {
                    settled(sink.screen(c.idx, c.energy));
                    continue;
                }
                match solve_one(c, cfg.rungs.probe_reads, cfg.rungs.probe_sweeps, "probe")
                    .unwrap_or_else(|e| {
                        fatal = Some(e);
                        None
                    }) {
                    Some(rs) => {
                        let b = rs.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX);
                        probed.push((b, k, rs));
                    }
                    None => {
                        settled(sink.screen(c.idx, c.energy));
                    }
                }
            }
            probed.sort_by_key(|x| x.0);
            let mut it = probed.into_iter();
            let top = it.next();
            for (b, k, rs) in it {
                best_solved = best_solved.min(b);
                solved += 1;
                settled(sink.push(batch[k].idx, rs));
            }
            if let Some((b, k, mut rs)) = top {
                let c = &batch[k];
                if !stopped() && fatal.is_none() {
                    if let Some(more) =
                        solve_one(c, cfg.rungs.final_reads, cfg.rungs.final_sweeps, "final")
                            .unwrap_or_else(|e| {
                                fatal = Some(e);
                                None
                            })
                    {
                        rs.extend(more);
                    }
                }
                best_solved = best_solved.min(rs.iter().map(|r| r.energy_milli).min().unwrap_or(b));
                solved += 1;
                settled(sink.push(c.idx, rs));
            }
            if fatal.is_some() {
                break;
            }
        }
        if fatal.is_some() {
            next.store(count, Ordering::Relaxed);
        }
    });
    drop(my_abandon);
    // Leave the holder list: drop replaced holders and holders whose lease has ended.
    HOLDERS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|e| !e.abandon.load(Ordering::Acquire) && Arc::strong_count(&e.abandon) > 1);
    let st = shared.into_inner().unwrap_or_else(PoisonError::into_inner);
    for c in st.cands {
        settled(sink.screen(c.idx, c.energy));
    }
    let ms = millis(t0.elapsed());
    tracing::info!(
        salts = count,
        screened = st.screened,
        skipped = st.skipped,
        solved,
        best = best_solved,
        rate = st.screened.saturating_mul(1000) / ms.max(1),
        stopped = stopped(),
        ms,
        "otsuka screen lease"
    );
    match fatal {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screening_is_off_without_a_usable_setting() {
        assert_eq!(parse_cfg(None, None, None, None), None);
        assert_eq!(parse_cfg(Some(""), None, None, None), None);
        assert_eq!(parse_cfg(Some("20,10000"), None, None, None), None);
        assert_eq!(parse_cfg(Some("0,10000,48"), None, None, None), None);
        assert_eq!(parse_cfg(Some("20,10000,0"), None, None, None), None);
    }

    #[test]
    fn defaults_fill_the_optional_settings() {
        let cfg = parse_cfg(Some("20, 10000, 48"), None, None, None).unwrap();
        assert_eq!(
            cfg,
            ScreenCfg {
                threads: 20,
                warm: 10_000,
                keep: 48,
                weights: DEFAULT_WEIGHTS,
                rungs: DEFAULT_RUNGS,
                holders: 1,
            }
        );
    }

    #[test]
    fn optional_settings_are_read_and_bad_ones_fall_back() {
        let cfg = parse_cfg(
            Some("8,500,16"),
            Some("1,0,2.5,3"),
            Some("64,1024,4,128,65536"),
            Some("3"),
        )
        .unwrap();
        assert_eq!(
            cfg.weights.map(f64::to_bits),
            [1.0, 0.0, 2.5, 3.0].map(f64::to_bits)
        );
        assert_eq!(
            cfg.rungs,
            Rungs {
                probe_reads: 64,
                probe_sweeps: 1024,
                probe_count: 4,
                final_reads: 128,
                final_sweeps: 65_536,
            }
        );
        assert_eq!(cfg.holders, 3);

        // Four score weights and five positive rung counts, or the defaults; at least one holder.
        let cfg = parse_cfg(
            Some("8,500,16"),
            Some("1,2,3"),
            Some("64,0,4,128,65536"),
            Some("0"),
        )
        .unwrap();
        assert_eq!(
            cfg.weights.map(f64::to_bits),
            DEFAULT_WEIGHTS.map(f64::to_bits)
        );
        assert_eq!(cfg.rungs, DEFAULT_RUNGS);
        assert_eq!(cfg.holders, 1);
    }
}
