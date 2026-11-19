//! Local build (all6, `BudFox`, 2026-11-15): a best-first pool of screened
//! lease salts (`QUIP_OTSUKA_POOL`), off by default.
//!
//! all5 hands every salt that passes stage 0 to the lanes in arrival order,
//! through a queue of two salts per lane, and parks the screen workers while
//! that queue is full (back-pressure). The lanes consume about 0.6 kept salts
//! per second, so at `QUIP_OTSUKA_STAGE0_KEEP` 0.01 the screen only needs
//! ~60 salts/s and the CPU slots sit idle most of the time.
//!
//! With the pool the screen never waits for the lanes. Every salt that passes
//! stage 0 is offered to a bounded pool ordered by its stage-0 score (lower
//! is predicted deeper), and a lane that needs work takes the best one. A full
//! pool admits a salt only in place of a worse one. The spare CPU therefore
//! buys selectivity: the lanes anneal about `lane rate / screen rate` of the
//! salts (~0.3% with six threads) instead of `STAGE0_KEEP`. On the G1 labels
//! (14/11 18:17-00:03, stage 1 off, n = 2845) the model-2 percentile band
//! (0, 0.25%] reached <= -5080 at 8.3% vs 3.4% for (0.25%, 1%].
//!
//! Lease accounting: an entry is a salt its lease is still waiting for (the
//! lease thread counts it outstanding). Every entry leaves exactly one way:
//! - a lane takes it and solves it;
//! - it is evicted by a better salt, refused at a full pool, or ages out
//!   (`QUIP_OTSUKA_POOL_MAX_AGE_MS`): the caller releases it unsolved, which
//!   the lease thread pushes as a salt with no reads (as a stage-0 reject);
//! - its lease stopped (round change, shutdown): it is dropped, exactly as a
//!   queued salt of a stopped lease is dropped by the lanes in all5.
//!
//! Aging bounds how long an entry keeps its lease (and that lease's credit)
//! alive: the coordinator sizes leases to ~60 s of salts and the session's
//! credit window is ~2 x lanes + queue depth, so entries must not pin old
//! leases for long.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError, TryLockError};
use std::time::{Duration, Instant};

/// The pool knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    /// Entries held at most.
    pub capacity: usize,
    /// An entry older than this is released unsolved on the next offer.
    pub max_age: Duration,
}

impl PoolConfig {
    /// Default `QUIP_OTSUKA_POOL_MAX_AGE_MS`.
    pub const MAX_AGE_MS: u64 = 120_000;

    /// `None` (off) unless `QUIP_OTSUKA_POOL` is an integer in `1..=256`.
    /// Every variable that is set logs `applied` or `ignored`.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let int = |name: &str, lo: u64, hi: u64| -> Option<u64> {
            let raw = get(name)?;
            let parsed = raw
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|v| (lo..=hi).contains(v));
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
        // Parse both first, so a set knob is always logged.
        let capacity = int("QUIP_OTSUKA_POOL", 0, 256);
        let max_age =
            int("QUIP_OTSUKA_POOL_MAX_AGE_MS", 1000, 1_800_000).unwrap_or(Self::MAX_AGE_MS);
        let capacity = usize::try_from(capacity?).ok().filter(|&c| c > 0)?;
        Some(Self {
            capacity,
            max_age: Duration::from_millis(max_age),
        })
    }

    /// [`Self::from_lookup`] on the process environment, read once.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        static ENV: OnceLock<Option<PoolConfig>> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }
}

struct Entry<T> {
    score: f32,
    seq: u64,
    at: Instant,
    item: T,
}

#[derive(Default)]
struct State<T> {
    entries: Vec<Entry<T>>,
    seq: u64,
}

/// What an offer left for the caller.
pub struct Spill<T> {
    /// Whether the offered item is now in the pool.
    pub admitted: bool,
    /// Items to release unsolved: an evicted entry, aged entries, or the
    /// offered item itself when refused. Their leases are live.
    pub release: Vec<T>,
    /// Entries of stopped leases, to drop.
    pub stale: Vec<T>,
}

#[derive(Default)]
struct Counters {
    offered: AtomicU64,
    admitted: AtomicU64,
    refused: AtomicU64,
    evicted: AtomicU64,
    aged: AtomicU64,
    stale: AtomicU64,
    taken: AtomicU64,
    /// Sum of taken scores, in micro-units, for the mean.
    taken_score_micro: AtomicI64,
    taken_wait_ms: AtomicU64,
}

#[derive(Clone, Copy)]
struct Snapshot {
    at: Instant,
    offered: u64,
    admitted: u64,
    refused: u64,
    evicted: u64,
    aged: u64,
    stale: u64,
    taken: u64,
    taken_score_micro: i64,
    taken_wait_ms: u64,
    screened: u64,
}

/// How often the pool logs its `otsuka pool` line.
const LOG_EVERY: Duration = Duration::from_mins(1);

/// A bounded pool of scored items, lowest score first out.
pub struct Pool<T> {
    config: PoolConfig,
    state: Mutex<State<T>>,
    ready: Condvar,
    counters: Counters,
    last: Mutex<Option<Snapshot>>,
}

impl<T> Pool<T> {
    /// An empty pool.
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config: PoolConfig {
                capacity: config.capacity.max(1),
                ..config
            },
            state: Mutex::new(State {
                entries: Vec::new(),
                seq: 0,
            }),
            ready: Condvar::new(),
            counters: Counters::default(),
            last: Mutex::new(None),
        }
    }

    /// The knobs.
    #[must_use]
    pub fn config(&self) -> PoolConfig {
        self.config
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Entries held now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state().entries.len()
    }

    /// Whether the pool holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether an offer of `score` would be admitted now (room, or better
    /// than the worst entry). A cheap check before a caller builds the item;
    /// the offer decides again.
    #[must_use]
    pub fn would_admit(&self, score: f32) -> bool {
        let state = self.state();
        state.entries.len() < self.config.capacity
            || worst(&state.entries).is_some_and(|w| score < state.entries[w].score)
    }

    /// Count an offer the caller refused itself after [`Self::would_admit`]
    /// said no (it never built the item).
    pub fn refused(&self) {
        self.counters.offered.fetch_add(1, Ordering::Relaxed);
        self.counters.refused.fetch_add(1, Ordering::Relaxed);
    }

    /// Offer `item` scored `score` at `now`. Entries of stopped leases
    /// (`gone`) go first, then entries older than the max age; a full pool
    /// then admits the item only in place of its worst entry, when the item
    /// scores strictly lower.
    pub fn offer(&self, item: T, score: f32, now: Instant, gone: &dyn Fn(&T) -> bool) -> Spill<T> {
        let c = &self.counters;
        c.offered.fetch_add(1, Ordering::Relaxed);
        let mut spill = Spill {
            admitted: false,
            release: Vec::new(),
            stale: Vec::new(),
        };
        let mut state = self.state();
        let mut i = 0;
        while i < state.entries.len() {
            let entry = &state.entries[i];
            if gone(&entry.item) {
                spill.stale.push(state.entries.swap_remove(i).item);
            } else if now.saturating_duration_since(entry.at) > self.config.max_age {
                spill.release.push(state.entries.swap_remove(i).item);
                c.aged.fetch_add(1, Ordering::Relaxed);
            } else {
                i += 1;
            }
        }
        c.stale
            .fetch_add(spill.stale.len() as u64, Ordering::Relaxed);
        if state.entries.len() >= self.config.capacity {
            match worst(&state.entries) {
                Some(w) if score < state.entries[w].score => {
                    spill.release.push(state.entries.swap_remove(w).item);
                    c.evicted.fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    c.refused.fetch_add(1, Ordering::Relaxed);
                    spill.release.push(item);
                    return spill;
                }
            }
        }
        state.seq += 1;
        let seq = state.seq;
        state.entries.push(Entry {
            score,
            seq,
            at: now,
            item,
        });
        drop(state);
        c.admitted.fetch_add(1, Ordering::Relaxed);
        spill.admitted = true;
        self.ready.notify_one();
        spill
    }

    /// Take the best (lowest score, then oldest) entry whose lease is live.
    /// Entries of stopped leases are dropped on the way.
    pub fn take(&self, gone: &dyn Fn(&T) -> bool) -> Option<T> {
        let now = Instant::now();
        let mut dropped = Vec::new();
        let taken = {
            let mut state = self.state();
            let mut i = 0;
            while i < state.entries.len() {
                if gone(&state.entries[i].item) {
                    dropped.push(state.entries.swap_remove(i).item);
                } else {
                    i += 1;
                }
            }
            best(&state.entries).map(|b| state.entries.swap_remove(b))
        };
        let c = &self.counters;
        c.stale.fetch_add(dropped.len() as u64, Ordering::Relaxed);
        drop(dropped);
        let entry = taken?;
        c.taken.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        c.taken_score_micro
            .fetch_add((f64::from(entry.score) * 1e6) as i64, Ordering::Relaxed);
        let waited =
            u64::try_from(now.saturating_duration_since(entry.at).as_millis()).unwrap_or(u64::MAX);
        c.taken_wait_ms.fetch_add(waited, Ordering::Relaxed);
        Some(entry.item)
    }

    /// Sleep until an offer is admitted or `timeout` passes. Spurious
    /// wakeups are fine: a caller loops on [`Self::take`].
    pub fn wait(&self, timeout: Duration) {
        let state = self.state();
        if state.entries.is_empty() {
            drop(
                self.ready
                    .wait_timeout(state, timeout)
                    .unwrap_or_else(PoisonError::into_inner),
            );
        }
    }

    /// Remove every entry (a new session: every earlier lease is gone).
    pub fn clear(&self) -> Vec<T> {
        let mut state = self.state();
        state.entries.drain(..).map(|e| e.item).collect()
    }

    /// The best and worst scores held, for the log line.
    fn range(&self) -> (usize, Option<f32>, Option<f32>) {
        let state = self.state();
        let lo = best(&state.entries).map(|i| state.entries[i].score);
        let hi = worst(&state.entries).map(|i| state.entries[i].score);
        (state.entries.len(), lo, hi)
    }

    fn snapshot(&self, at: Instant, screened: u64) -> Snapshot {
        let c = &self.counters;
        Snapshot {
            at,
            offered: c.offered.load(Ordering::Relaxed),
            admitted: c.admitted.load(Ordering::Relaxed),
            refused: c.refused.load(Ordering::Relaxed),
            evicted: c.evicted.load(Ordering::Relaxed),
            aged: c.aged.load(Ordering::Relaxed),
            stale: c.stale.load(Ordering::Relaxed),
            taken: c.taken.load(Ordering::Relaxed),
            taken_score_micro: c.taken_score_micro.load(Ordering::Relaxed),
            taken_wait_ms: c.taken_wait_ms.load(Ordering::Relaxed),
            screened,
        }
    }

    /// Log the `otsuka pool` line when [`LOG_EVERY`] has passed since the
    /// last one: entries held, best and worst score held (the worst is the
    /// admission cutoff once the pool is full), and per-second counts.
    /// `screened` is the screen's running count of salts through stage 0, so
    /// the line can show the share of screened salts the lanes took. Never
    /// blocks.
    #[allow(clippy::cast_precision_loss)] // Rates for a log line.
    pub fn maybe_log(&self, screened: u64) {
        let mut last = match self.last.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        let now = Instant::now();
        let Some(prev) = *last else {
            *last = Some(self.snapshot(now, screened));
            return;
        };
        let secs = now.duration_since(prev.at).as_secs_f64();
        if secs < LOG_EVERY.as_secs_f64() {
            return;
        }
        let s = self.snapshot(now, screened);
        let screened_d = screened.saturating_sub(prev.screened);
        let taken = s.taken - prev.taken;
        let (held, lo, hi) = self.range();
        let per = |a: u64, b: u64| (a - b) as f64 / secs;
        tracing::info!(
            capacity = self.config.capacity,
            held,
            best = lo.map_or(f64::NAN, f64::from),
            worst = hi.map_or(f64::NAN, f64::from),
            offered_per_s = per(s.offered, prev.offered),
            admitted_per_s = per(s.admitted, prev.admitted),
            refused_per_s = per(s.refused, prev.refused),
            evicted_per_s = per(s.evicted, prev.evicted),
            aged_per_s = per(s.aged, prev.aged),
            stale_per_s = per(s.stale, prev.stale),
            taken_per_s = per(s.taken, prev.taken),
            taken_mean_score = if taken == 0 {
                f64::NAN
            } else {
                (s.taken_score_micro - prev.taken_score_micro) as f64 / taken as f64 / 1e6
            },
            taken_mean_wait_ms = if taken == 0 {
                0.0
            } else {
                (s.taken_wait_ms - prev.taken_wait_ms) as f64 / taken as f64
            },
            taken_share = if screened_d == 0 {
                0.0
            } else {
                taken as f64 / screened_d as f64
            },
            "otsuka pool"
        );
        *last = Some(s);
    }
}

/// Index of the lowest (score, seq) entry.
fn best<T>(entries: &[Entry<T>]) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.score.total_cmp(&b.1.score).then(a.1.seq.cmp(&b.1.seq)))
        .map(|(i, _)| i)
}

/// Index of the highest (score, seq) entry: the newest of the worst, so an
/// older entry of equal score is kept.
fn worst<T>(entries: &[Entry<T>]) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.score.total_cmp(&b.1.score).then(a.1.seq.cmp(&b.1.seq)))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn config(capacity: usize) -> PoolConfig {
        PoolConfig {
            capacity,
            max_age: Duration::from_hours(1),
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)] // The pool's `&dyn Fn(&T)`.
    fn never(_: &u32) -> bool {
        false
    }

    #[test]
    fn knobs_default_off_and_parse() {
        let none = |_: &str| None;
        assert_eq!(PoolConfig::from_lookup(none), None, "unset is off");
        let get = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        assert_eq!(
            PoolConfig::from_lookup(get(&[("QUIP_OTSUKA_POOL", "0")])),
            None
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[("QUIP_OTSUKA_POOL", "x")])),
            None
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[("QUIP_OTSUKA_POOL", "257")])),
            None
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[("QUIP_OTSUKA_POOL", "16")])),
            Some(PoolConfig {
                capacity: 16,
                max_age: Duration::from_millis(PoolConfig::MAX_AGE_MS)
            })
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[
                ("QUIP_OTSUKA_POOL", "8"),
                ("QUIP_OTSUKA_POOL_MAX_AGE_MS", "30000")
            ])),
            Some(PoolConfig {
                capacity: 8,
                max_age: Duration::from_secs(30)
            })
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[
                ("QUIP_OTSUKA_POOL", "8"),
                ("QUIP_OTSUKA_POOL_MAX_AGE_MS", "5")
            ]))
            .map(|c| c.max_age),
            Some(Duration::from_millis(PoolConfig::MAX_AGE_MS)),
            "an out-of-range age is ignored"
        );
        assert_eq!(
            PoolConfig::from_lookup(get(&[("QUIP_OTSUKA_POOL_MAX_AGE_MS", "30000")])),
            None,
            "an age alone does not turn the pool on"
        );
    }

    #[test]
    fn lanes_take_the_lowest_score_first_and_ties_in_arrival_order() {
        let pool = Pool::new(config(8));
        let now = Instant::now();
        for (item, score) in [(1, 3.0), (2, -1.0), (3, 2.0), (4, -1.0), (5, 0.5)] {
            assert!(pool.offer(item, score, now, &never).admitted);
        }
        let order: Vec<u32> = std::iter::from_fn(|| pool.take(&never)).collect();
        assert_eq!(order, [2, 4, 5, 3, 1]);
        assert!(pool.is_empty());
    }

    #[test]
    fn a_full_pool_admits_only_a_strictly_better_salt_and_returns_the_loser() {
        let pool = Pool::new(config(2));
        let now = Instant::now();
        assert!(pool.offer(1, 1.0, now, &never).admitted);
        assert!(pool.offer(2, 2.0, now, &never).admitted);
        assert!(!pool.would_admit(2.0), "equal to the worst is not better");
        let refused = pool.offer(3, 2.0, now, &never);
        assert!(!refused.admitted);
        assert_eq!(
            refused.release,
            [3],
            "the refused salt comes back to release"
        );
        assert!(pool.would_admit(1.5));
        let evicting = pool.offer(4, 1.5, now, &never);
        assert!(evicting.admitted);
        assert_eq!(evicting.release, [2], "the worst entry is evicted");
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.take(&never), Some(1));
        assert_eq!(pool.take(&never), Some(4));
        assert_eq!(pool.take(&never), None);
    }

    #[test]
    fn stale_entries_are_dropped_not_released() {
        let pool = Pool::new(config(4));
        let now = Instant::now();
        for item in 0..4 {
            assert!(
                pool.offer(item, f32::from(u8::try_from(item).unwrap()), now, &never)
                    .admitted
            );
        }
        // Items 0 and 1 belong to a lease that stopped.
        let stopped = |item: &u32| *item < 2;
        let spill = pool.offer(9, 10.0, now, &stopped);
        assert!(spill.admitted, "room freed by the stale entries");
        assert!(spill.release.is_empty());
        let mut stale = spill.stale;
        stale.sort_unstable();
        assert_eq!(stale, [0, 1]);
        assert_eq!(
            pool.take(&|i: &u32| *i == 2),
            Some(3),
            "a stale best is skipped"
        );
        assert_eq!(pool.take(&never), Some(9));
        assert_eq!(pool.take(&never), None);
    }

    #[test]
    fn old_entries_age_out_to_release_on_the_next_offer() {
        let pool = Pool::new(PoolConfig {
            capacity: 4,
            max_age: Duration::from_secs(10),
        });
        let t0 = Instant::now();
        assert!(pool.offer(1, 5.0, t0, &never).admitted);
        assert!(
            pool.offer(2, 6.0, t0 + Duration::from_secs(8), &never)
                .admitted
        );
        let spill = pool.offer(3, 7.0, t0 + Duration::from_secs(11), &never);
        assert!(spill.admitted);
        assert_eq!(spill.release, [1], "only the entry older than the max age");
        assert_eq!(pool.take(&never), Some(2));
    }

    #[test]
    fn a_waiting_lane_wakes_on_an_offer() {
        let pool = Arc::new(Pool::new(config(4)));
        let lane = {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || {
                let started = Instant::now();
                pool.wait(Duration::from_secs(5));
                (started.elapsed(), pool.take(&never))
            })
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(pool.offer(7, 0.0, Instant::now(), &never).admitted);
        let (waited, taken) = lane.join().unwrap();
        assert!(
            waited < Duration::from_secs(2),
            "woken by the offer: {waited:?}"
        );
        assert_eq!(taken, Some(7));
    }

    /// An item that records its id when dropped, so a test sees every
    /// exit (taken, released, stale, cleared) exactly once.
    struct Tracked(u32, Arc<Mutex<Vec<u32>>>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0);
        }
    }

    /// Four producers and three consumers: every item leaves the pool exactly
    /// once (taken, released, stale or cleared), nothing is lost or
    /// duplicated, and the consumers stop cleanly once the producers are
    /// done. After half-time a "stopped lease" (ids ending in 7) goes stale.
    #[test]
    fn concurrent_offers_and_takes_account_for_every_item_once() {
        let pool = Arc::new(Pool::new(config(6)));
        let per = 2000u32;
        let producers = 4u32;
        let over = Arc::new(AtomicBool::new(false));
        let halftime = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(Mutex::new(Vec::<u32>::new()));
        let stale = {
            let halftime = Arc::clone(&halftime);
            Arc::new(move |t: &Tracked| halftime.load(Ordering::Relaxed) && t.0 % 10 == 7)
        };
        let mut handles = Vec::new();
        for p in 0..producers {
            let (pool, stale, halftime, dropped) = (
                Arc::clone(&pool),
                Arc::clone(&stale),
                Arc::clone(&halftime),
                Arc::clone(&dropped),
            );
            handles.push(std::thread::spawn(move || {
                let mut x = u64::from(p) * 7919 + 1;
                for k in 0..per {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if p == 0 && k == per / 2 {
                        halftime.store(true, Ordering::Relaxed);
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let score = (x % 1000) as f32;
                    let item = Tracked(p * per + k, Arc::clone(&dropped));
                    let spill = pool.offer(item, score, Instant::now(), &*stale);
                    drop(spill);
                }
            }));
        }
        let taken = Arc::new(AtomicU64::new(0));
        let mut consumers = Vec::new();
        for _ in 0..3 {
            let (pool, stale, over, taken) = (
                Arc::clone(&pool),
                Arc::clone(&stale),
                Arc::clone(&over),
                Arc::clone(&taken),
            );
            consumers.push(std::thread::spawn(move || loop {
                match pool.take(&*stale) {
                    Some(item) => {
                        assert!(!stale(&item), "a stale entry is never taken");
                        taken.fetch_add(1, Ordering::Relaxed);
                    }
                    None if over.load(Ordering::Acquire) => break,
                    None => pool.wait(Duration::from_millis(5)),
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        over.store(true, Ordering::Release);
        for c in consumers {
            c.join().unwrap();
        }
        drop(pool.clear());
        let mut ids = dropped.lock().unwrap().clone();
        let total = (producers * per) as usize;
        assert_eq!(ids.len(), total, "every item left exactly once");
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "no item left twice");
        assert!(taken.load(Ordering::Relaxed) > 0);
        assert_eq!(
            pool.counters.taken.load(Ordering::Relaxed),
            taken.load(Ordering::Relaxed)
        );
    }
}
