//! Local build (all12): the ease-race booster.
//!
//! Off by default: with `QUIP_OTSUKA_EASE_BOOST` unset (or 0) the sampler
//! builds nothing new, every result is sent exactly as in all11, and the
//! deep lanes never look at the ease queue. It needs the deep lane
//! (`QUIP_OTSUKA_DEEP_LANES`), whose lanes, probe and run-time estimate it
//! reuses.
//!
//! Why: a round's threshold holds for 100 blocks (~606 s) and then eases
//! ~0.124 u per block. The coordinator holds a result that is not yet below
//! the threshold and submits it at lag 0 once it qualifies, so in the ease
//! phase the deepest held result wins. 6 of 24 losses (4 h forensics) were
//! near-misses, our round best 0.01-0.27 u shallower than the rival's at
//! minute ~10. A long fresh-seed re-anneal of the same instance is worth
//! ~+4.3 u over a fresh rerun.
//!
//! Constraint: core's `LeaseSink` accepts ONE push per salt (a repeat is
//! ignored and logged), and the coordinator may not change. A salt that has
//! been delivered can no longer be improved. So the booster acts at the last
//! point before delivery: every local lease result, after the CPU polish
//! (its exact energies), passes [`Ease::consider`]. When it is the round's
//! best so far, at least `T0` into the round, above the live target and
//! within `GAP` of it, it is held (not sent) in an [`EaseGroup`] and `LANES`
//! long fresh-seed anneals of the same instance (each its own seed, never
//! repeated for that instance this round) are queued for the deep lanes,
//! ahead of the deep lane's own long runs. The group is delivered once,
//! through the normal path (merge, polish, send), when:
//! - a run's probed lineage best is releasable (below the target, or within
//!   `RELEASE` of it once the ease may have begun): the run stops there;
//! - every run ended (or was skipped);
//! - the watcher sees the held best releasable: queued runs are skipped and
//!   running ones harvest at their next probe (after [`HARVEST_GRACE`]
//!   without one, the held reads go out and the runs are cancelled).
//!
//! The delivered result passes [`Ease::consider`] again, so a round best that
//! is still short keeps being boosted with new seeds (bounded by `MAX_RUNS`
//! per instance and round). A deeper result (from any path) retargets:
//! earlier groups' queued runs are skipped, their running runs finish.
//! A lease stop (round change, shutdown) drops held groups and stops runs at
//! their next chunk, as for the deep lane.
//!
//! Release rule. The target is static during the hold phase and only rises
//! (eases) afterwards, so holding a result above it costs nothing until the
//! ease nears it. A held best is released when it is at or below the
//! target, or within `RELEASE` of it once the target has risen this round or
//! the round is `SAFE` old (just before the first ease block). `RELEASE` 250
//! is two ease blocks (~12 s) of warning.
//!
//! Length. A run takes the largest multiple of the job's sweeps in
//! `1..=MULT_MAX` that ends by the horizon: `END` into the round when held
//! before it, else `POST` from now; and never after the estimated release
//! (`SAFE` or the ease start, plus (gap - RELEASE) at ~19 milli/s). The run
//! time per multiple is the deep lane's measured estimate. No fit: no hold.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_EASE_BOOST` | 0 | 1: on (needs the deep lane). |
//! | `QUIP_OTSUKA_EASE_T0_S` | 420 | No hold earlier in the round. |
//! | `QUIP_OTSUKA_EASE_GAP_MILLI` | 1000 | Hold only within this above the live target. |
//! | `QUIP_OTSUKA_EASE_LANES` | deep lanes | Runs per hold (`1..=` deep lanes). |
//! | `QUIP_OTSUKA_EASE_END_S` | 640 | Runs held before this end by it. |
//! | `QUIP_OTSUKA_EASE_POST_S` | 90 | Run length after `END`. |
//! | `QUIP_OTSUKA_EASE_RELEASE_MILLI` | 250 | Release margin once the ease may have begun. |
//! | `QUIP_OTSUKA_EASE_SAFE_S` | 590 | The ease may begin after this (release margin applies). |
//! | `QUIP_OTSUKA_EASE_MULT_MAX` | 16 | Longest run, job-sweep multiple `1..=16`. |
//! | `QUIP_OTSUKA_EASE_MAX_RUNS` | 48 | Runs per instance and round. |
//!
//! Logs (info): `otsuka ease boost` per hold, `otsuka ease result` per
//! delivered group, `otsuka ease dropped` per dropped group, and
//! `otsuka ease round` once per round.

use super::deep::{Deep, End, HARVEST_GRACE};
use super::stage1::merge_better;
use super::stage2::{best_of, Payload};
use quip_solver_core::{StreamOutcome, StreamResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// `XORed` into the job seed for the boost runs' draws.
pub const EASE_SEED_SALT: u64 = 0xEA5E_B005_7A11_0012;

/// The ease rate assumed for the release estimate: 0.124 u per ~6.4 s
/// block, ~1.16 u/min.
pub const EASE_RATE_MILLI_PER_S: i64 = 19;

/// The ease booster's knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EaseConfig {
    /// No hold before this much of the round has passed.
    pub t0: Duration,
    /// Hold only within this above the live target.
    pub gap_milli: i64,
    /// Runs queued per hold.
    pub lanes: usize,
    /// Runs held before this time end by it.
    pub end: Duration,
    /// Run length after `end`.
    pub post: Duration,
    /// Release margin once the ease may have begun.
    pub release_milli: i64,
    /// The ease may begin after this time into the round.
    pub safe: Duration,
    /// Longest run (job-sweep multiple).
    pub mult_max: usize,
    /// Runs per instance and round.
    pub max_runs: u64,
}

/// Why a result was not held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// Before `T0`.
    Early,
    /// No live target yet.
    NoTarget,
    /// Not the round's best so far.
    NotBest,
    /// At or below the target: it can win now.
    Qualifies,
    /// More than `GAP` above the target.
    Far,
    /// Releasable now: holding it would only delay it.
    Releasing,
    /// This instance used its `MAX_RUNS` this round.
    Spent,
    /// Not even one job length fits before the horizon.
    NoTime,
}

/// A hold decision: the run multiple and count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Job-sweep multiple of each run.
    pub mult: usize,
    /// Runs to queue.
    pub runs: usize,
    /// Best minus target at the decision.
    pub gap: i64,
    /// Estimated release, time into the round.
    pub eta: Duration,
}

impl EaseConfig {
    /// Default `T0`.
    pub const T0: Duration = Duration::from_mins(7);
    /// Default `GAP_MILLI`.
    pub const GAP_MILLI: i64 = 1000;
    /// Default `END`.
    pub const END: Duration = Duration::from_secs(640);
    /// Default `POST`.
    pub const POST: Duration = Duration::from_secs(90);
    /// Default `RELEASE_MILLI`: two ease blocks.
    pub const RELEASE_MILLI: i64 = 250;
    /// Default `SAFE`: before the first ease block (~606 s).
    pub const SAFE: Duration = Duration::from_secs(590);
    /// Default `MULT_MAX`.
    pub const MULT_MAX: usize = 16;
    /// Default `MAX_RUNS`.
    pub const MAX_RUNS: u64 = 48;

    /// `None` (off) unless `QUIP_OTSUKA_EASE_BOOST=1`. `deep_lanes` is the
    /// deep lane count (LMAX with the adaptive deep lane), the default and
    /// cap of `LANES`; 0 (deep lane off) keeps the booster off. Every
    /// variable that is set logs `applied` or `ignored`.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>, deep_lanes: usize) -> Option<Self> {
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
        let on = num("QUIP_OTSUKA_EASE_BOOST", &|v| {
            (0.0..=1.0).contains(&v) && int(v)
        })
        .is_some_and(|v| v > 0.5);
        if !on {
            return None;
        }
        if deep_lanes == 0 {
            eprintln!("quip-otsuka: QUIP_OTSUKA_EASE_BOOST needs the deep lane (QUIP_OTSUKA_DEEP_LANES); ease boost off");
            return None;
        }
        let secs = |name: &str, default: Duration| {
            num(name, &|v| (0.0..=3600.0).contains(&v)).map_or(default, Duration::from_secs_f64)
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = |name: &str, lo: f64, hi: f64| {
            num(name, &|v| (lo..=hi).contains(&v) && int(v)).map(|v| v as u64)
        };
        #[allow(clippy::cast_possible_truncation)]
        let milli = |name: &str, lo: f64, default: i64| {
            num(name, &|v| (lo..=20_000.0).contains(&v) && int(v)).map_or(default, |v| v as i64)
        };
        let t0 = secs("QUIP_OTSUKA_EASE_T0_S", Self::T0);
        let gap_milli = milli("QUIP_OTSUKA_EASE_GAP_MILLI", 1.0, Self::GAP_MILLI);
        #[allow(clippy::cast_possible_truncation)]
        let lanes = count("QUIP_OTSUKA_EASE_LANES", 1.0, 8.0)
            .map_or(deep_lanes, |l| l as usize)
            .min(deep_lanes);
        let end = secs("QUIP_OTSUKA_EASE_END_S", Self::END);
        let post = secs("QUIP_OTSUKA_EASE_POST_S", Self::POST);
        let release_milli = milli("QUIP_OTSUKA_EASE_RELEASE_MILLI", 0.0, Self::RELEASE_MILLI);
        let safe = secs("QUIP_OTSUKA_EASE_SAFE_S", Self::SAFE);
        #[allow(clippy::cast_possible_truncation)]
        let mult_max =
            count("QUIP_OTSUKA_EASE_MULT_MAX", 1.0, 16.0).map_or(Self::MULT_MAX, |m| m as usize);
        let max_runs = count("QUIP_OTSUKA_EASE_MAX_RUNS", 1.0, 10_000.0).unwrap_or(Self::MAX_RUNS);
        Some(Self {
            t0,
            gap_milli,
            lanes,
            end,
            post,
            release_milli,
            safe,
            mult_max,
            max_runs,
        })
    }

    /// [`Self::from_lookup`] on the process environment.
    #[must_use]
    pub fn from_env(deep_lanes: usize) -> Option<Self> {
        Self::from_lookup(|name| std::env::var(name).ok(), deep_lanes)
    }

    /// Whether the ease may have begun: the target rose this round, or the
    /// round is `SAFE` old.
    #[must_use]
    pub fn easing(&self, elapsed: Duration, eased: bool) -> bool {
        eased || elapsed >= self.safe
    }

    /// Whether a best (reported energy) should be delivered now.
    #[must_use]
    pub fn releasable(
        &self,
        best: i64,
        target: Option<i64>,
        elapsed: Duration,
        eased: bool,
    ) -> bool {
        target.is_some_and(|t| {
            best <= t
                || (best <= t.saturating_add(self.release_milli) && self.easing(elapsed, eased))
        })
    }

    /// The estimated time into the round when a best `gap` above the target
    /// becomes releasable.
    #[must_use]
    pub fn eta(&self, elapsed: Duration, gap: i64, eased: bool) -> Duration {
        let ease_from = if self.easing(elapsed, eased) {
            elapsed
        } else {
            self.safe
        };
        let over = gap.saturating_sub(self.release_milli);
        if over <= 0 {
            return ease_from;
        }
        #[allow(clippy::cast_precision_loss)]
        let wait = Duration::try_from_secs_f64(over as f64 / EASE_RATE_MILLI_PER_S as f64)
            .unwrap_or(Duration::MAX);
        ease_from.saturating_add(wait)
    }

    /// The run multiple for a hold `elapsed` into the round, `gap` above the
    /// target, at `run` wall time per job length: the largest in
    /// `1..=MULT_MAX` (and `<= cap`) that ends by the horizon, `None` when
    /// not even one fits.
    #[must_use]
    pub fn mult(
        &self,
        elapsed: Duration,
        gap: i64,
        run: Duration,
        eased: bool,
        cap: usize,
    ) -> Option<usize> {
        let horizon = if elapsed < self.end {
            self.end
        } else {
            elapsed.saturating_add(self.post)
        }
        .min(self.eta(elapsed, gap, eased));
        (1..=self.mult_max.min(cap)).rev().find(|&m| {
            u32::try_from(m)
                .ok()
                .and_then(|m| run.checked_mul(m))
                .and_then(|d| elapsed.checked_add(d))
                .is_some_and(|end| end <= horizon)
        })
    }

    /// The hold decision, a pure function: a result of best `best`
    /// (reported energy) `elapsed` into the round, the round's best so far
    /// `round_best` (this result not included), the live `target`, whether
    /// it `eased` this round, the run time per job length `run`, runs this
    /// instance already took `used`, and the largest multiple the job allows
    /// `cap`.
    ///
    /// # Errors
    /// The [`Skip`] reason when the result is to be sent, not held.
    #[allow(clippy::too_many_arguments)]
    pub fn decide(
        &self,
        elapsed: Duration,
        best: i64,
        round_best: Option<i64>,
        target: Option<i64>,
        eased: bool,
        run: Duration,
        used: u64,
        cap: usize,
    ) -> Result<Plan, Skip> {
        if elapsed < self.t0 {
            return Err(Skip::Early);
        }
        let Some(t) = target else {
            return Err(Skip::NoTarget);
        };
        if round_best.is_some_and(|r| best > r) {
            return Err(Skip::NotBest);
        }
        let gap = best.saturating_sub(t);
        if gap <= 0 {
            return Err(Skip::Qualifies);
        }
        if gap > self.gap_milli {
            return Err(Skip::Far);
        }
        if self.releasable(best, target, elapsed, eased) {
            return Err(Skip::Releasing);
        }
        if used >= self.max_runs {
            return Err(Skip::Spent);
        }
        let mult = self
            .mult(elapsed, gap, run, eased, cap)
            .ok_or(Skip::NoTime)?;
        let runs = u64::try_from(self.lanes.max(1))
            .unwrap_or(1)
            .min(self.max_runs - used);
        Ok(Plan {
            mult,
            runs: usize::try_from(runs).unwrap_or(1),
            gap,
            eta: self.eta(elapsed, gap, eased),
        })
    }
}

/// The seed of boost run `n` of a job seeded `seed` (splitmix64).
#[must_use]
pub fn ease_seed(seed: u64, n: u64) -> u64 {
    let mut z =
        (seed ^ EASE_SEED_SALT).wrapping_add(n.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

/// One round's booster state.
#[derive(Default)]
struct RoundState {
    epoch: u64,
    /// Lowest reported best through [`Ease::consider`] this round.
    best: Option<i64>,
    /// Lowest live target seen this round (the hold value).
    min_target: Option<i64>,
    /// The current hold: its retire flag, held best and job seed.
    current: Option<(Arc<AtomicBool>, i64, u64)>,
    /// Boost runs issued per job seed (instance) this round.
    issued: HashMap<u64, u64>,
    stats: RoundStats,
}

#[derive(Default, Clone, Copy)]
struct RoundStats {
    holds: u64,
    runs: u64,
    retargets: u64,
    results: u64,
    probed: u64,
    harvested: u64,
    released: u64,
    forced: u64,
    dropped: u64,
    improved: u64,
    gain_milli: i64,
    best_held: Option<i64>,
    best_after: Option<i64>,
}

/// A hold granted by [`Ease::consider`].
pub struct Grant {
    /// The run multiple and count.
    pub plan: Plan,
    /// The first boost index (seed `ease_seed(job seed, first + k)`).
    pub first: u64,
    /// The group's retire flag.
    pub retired: Arc<AtomicBool>,
    /// Time into the round.
    pub elapsed: Duration,
    /// The live target.
    pub target: i64,
    /// Whether an earlier hold was retired for this one.
    pub retarget: bool,
}

/// Booster state shared by the lanes, the polish pool and the watcher.
pub struct Ease {
    config: EaseConfig,
    active: AtomicBool,
    round: Mutex<RoundState>,
}

impl Ease {
    /// Fresh state for `config` (inactive until a session turns it on).
    #[must_use]
    pub fn new(config: EaseConfig) -> Self {
        Self {
            config,
            active: AtomicBool::new(false),
            round: Mutex::new(RoundState::default()),
        }
    }

    /// The knobs.
    #[must_use]
    pub const fn config(&self) -> &EaseConfig {
        &self.config
    }

    /// Whether this session runs the booster (the deep lane is live).
    pub fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Turn the booster on or off for a session.
    pub fn set_active(&self, on: bool) {
        self.active.store(on, Ordering::Relaxed);
    }

    /// Roll the round state to `epoch` (logging the old round's summary)
    /// and fold `target` into the round's lowest target.
    #[allow(clippy::unused_self)] // Kept a method next to its callers.
    fn sync(&self, st: &mut RoundState, epoch: u64, elapsed: Duration, target: Option<i64>) {
        if st.epoch != epoch {
            let s = st.stats;
            tracing::info!(
                round = st.epoch,
                holds = s.holds,
                runs = s.runs,
                retargets = s.retargets,
                results = s.results,
                probed = s.probed,
                harvested = s.harvested,
                released = s.released,
                forced = s.forced,
                dropped = s.dropped,
                improved = s.improved,
                gain_milli = s.gain_milli,
                best_held = s.best_held,
                best_after = s.best_after,
                round_best = st.best,
                min_target = st.min_target,
                next_round_s = elapsed.as_secs_f64(),
                "otsuka ease round"
            );
            if let Some((flag, _, _)) = st.current.take() {
                flag.store(true, Ordering::Relaxed);
            }
            *st = RoundState {
                epoch,
                ..RoundState::default()
            };
        }
        if let Some(t) = target {
            st.min_target = Some(st.min_target.map_or(t, |m| m.min(t)));
        }
    }

    /// Time into the round, the live target, and whether it rose this round.
    fn context(
        &self,
        st: &mut RoundState,
        deep: &Deep,
        now: Instant,
    ) -> (Duration, Option<i64>, bool) {
        let (elapsed, epoch) = deep.round_now(now);
        let target = deep.live_target();
        self.sync(st, epoch, elapsed, target);
        let eased = target.zip(st.min_target).is_some_and(|(t, m)| t > m);
        (elapsed, target, eased)
    }

    /// The watcher's tick: roll the round and track the target.
    pub fn tick(&self, deep: &Deep, now: Instant) {
        let mut st = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        self.context(&mut st, deep, now);
    }

    /// Whether a best (reported energy) should be delivered now.
    pub fn release_now(&self, deep: &Deep, best: i64, now: Instant) -> bool {
        let mut st = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        let (elapsed, target, eased) = self.context(&mut st, deep, now);
        drop(st);
        self.config.releasable(best, target, elapsed, eased)
    }

    /// A result of best `best` (reported energy, exact) for the job seeded
    /// `seed` is about to be sent: hold it for a boost (`Ok`) or send it.
    /// Every result feeds the round best; a deeper one retires the current
    /// hold (its queued runs are skipped). `cap` is the largest multiple
    /// the job's sweeps allow.
    ///
    /// # Errors
    /// The [`Skip`] reason when the result is to be sent, not held.
    pub fn consider(
        &self,
        deep: &Deep,
        best: i64,
        seed: u64,
        cap: usize,
        now: Instant,
    ) -> Result<Grant, Skip> {
        let mut st = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        let (elapsed, target, eased) = self.context(&mut st, deep, now);
        let used = st.issued.get(&seed).copied().unwrap_or(0);
        let decision = self.config.decide(
            elapsed,
            best,
            st.best,
            target,
            eased,
            deep.run_estimate(),
            used,
            cap,
        );
        st.best = Some(st.best.map_or(best, |b| b.min(best)));
        // A deeper result (held or not) retargets: the current hold's queued
        // runs are skipped, its running runs finish.
        // A re-boost of the same instance (after its delivery) is not a
        // retarget.
        let mut retarget = false;
        if st
            .current
            .as_ref()
            .is_some_and(|(_, b, _)| best < *b || decision.is_ok())
        {
            if let Some((flag, _, held)) = st.current.take() {
                flag.store(true, Ordering::Relaxed);
                if held != seed {
                    st.stats.retargets += 1;
                    retarget = true;
                }
            }
        }
        let plan = decision?;
        let retired = Arc::new(AtomicBool::new(false));
        st.current = Some((Arc::clone(&retired), best, seed));
        st.issued.insert(seed, used + plan.runs as u64);
        st.stats.holds += 1;
        st.stats.runs += plan.runs as u64;
        st.stats.best_held = Some(st.stats.best_held.map_or(best, |b| b.min(best)));
        Ok(Grant {
            plan,
            first: used,
            retired,
            elapsed,
            target: target.unwrap_or_default(),
            retarget,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn finished(
        &self,
        facts: &Facts,
        long: Option<i64>,
        after: i64,
        end: End,
        runs: usize,
        deep: &Deep,
    ) {
        let target = deep.live_target();
        {
            let mut st = self.round.lock().unwrap_or_else(PoisonError::into_inner);
            if st.epoch == facts.epoch {
                let s = &mut st.stats;
                s.results += 1;
                match end {
                    End::Crossed => s.probed += 1,
                    End::Harvested => s.harvested += 1,
                    End::Released => s.released += 1,
                    End::Forced => s.forced += 1,
                    End::Last => {}
                }
                let gain = facts.best.saturating_sub(after).max(0);
                s.improved += u64::from(gain > 0);
                s.gain_milli += gain;
                s.best_after = Some(s.best_after.map_or(after, |b| b.min(after)));
            }
        }
        tracing::info!(
            before = facts.best,
            long,
            after,
            delivered = true,
            end = ?end,
            runs,
            mult = facts.mult,
            round_s = facts.round_s,
            held_s = facts.held_at.elapsed().as_secs_f64(),
            live_target = target,
            gap_after = target.map(|t| after.saturating_sub(t)),
            "otsuka ease result"
        );
    }

    fn dropped(&self, facts: &Facts, long: Option<i64>, runs: usize) {
        {
            let mut st = self.round.lock().unwrap_or_else(PoisonError::into_inner);
            if st.epoch == facts.epoch {
                st.stats.dropped += 1;
            }
        }
        tracing::info!(
            before = facts.best,
            long,
            runs,
            mult = facts.mult,
            round_s = facts.round_s,
            held_s = facts.held_at.elapsed().as_secs_f64(),
            "otsuka ease dropped"
        );
    }
}

struct Facts {
    best: i64,
    round_s: f64,
    held_at: Instant,
    mult: usize,
    epoch: u64,
}

/// One held round best: its delivered-to-be reads, merged with each boost
/// run's.
pub struct EaseGroup<J, F, D> {
    facts: Facts,
    /// The instance offset (reported = lane energy + offset).
    offset: i64,
    /// The lease's stop flag (round change, shutdown).
    stop: Arc<AtomicBool>,
    /// Set by a retarget: queued runs are skipped.
    retired: Arc<AtomicBool>,
    /// Set once delivered or dropped: running runs stop at their next chunk.
    done: AtomicBool,
    harvest: AtomicBool,
    harvest_since: Mutex<Option<Instant>>,
    /// A run's probe released the group.
    probed: AtomicBool,
    /// Runs queued or running, not yet finished.
    outstanding: AtomicUsize,
    /// Runs running now.
    running: AtomicUsize,
    /// Runs launched.
    launched: AtomicUsize,
    /// Lowest finished run best (reported); `i64::MAX` none.
    long_best: AtomicI64,
    payload: Mutex<Option<Payload<J, F, D>>>,
}

impl<J, F, D> EaseGroup<J, F, D> {
    /// A group holding `payload` (best `best`, reported) for `runs` boost
    /// runs of `mult` job lengths, granted `elapsed` into round `epoch`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        best: i64,
        offset: i64,
        elapsed: Duration,
        epoch: u64,
        mult: usize,
        runs: usize,
        stop: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
        payload: Payload<J, F, D>,
    ) -> Self {
        Self {
            facts: Facts {
                best,
                round_s: elapsed.as_secs_f64(),
                held_at: Instant::now(),
                mult,
                epoch,
            },
            offset,
            stop,
            retired,
            done: AtomicBool::new(false),
            harvest: AtomicBool::new(false),
            harvest_since: Mutex::new(None),
            probed: AtomicBool::new(false),
            outstanding: AtomicUsize::new(runs.max(1)),
            running: AtomicUsize::new(0),
            launched: AtomicUsize::new(0),
            long_best: AtomicI64::new(i64::MAX),
            payload: Mutex::new(Some(payload)),
        }
    }

    /// Whether a running run should stop (or a queued one never launch).
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.done.load(Ordering::Relaxed)
    }

    /// Whether a queued run would be skipped (lanes take these first).
    pub fn skippable(&self) -> bool {
        self.stopped()
            || self.retired.load(Ordering::Relaxed)
            || self.harvest.load(Ordering::Relaxed)
    }

    /// Whether the group still holds its salt.
    pub fn pending(&self) -> bool {
        self.payload
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// Mark one queued run as launched. False when it is to be skipped:
    /// the lane then calls [`Self::finish_run`] with `None`.
    pub fn begin(&self) -> bool {
        let slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() || self.skippable() {
            return false;
        }
        self.running.fetch_add(1, Ordering::AcqRel);
        self.launched.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// A running run's probe: `device_best` is its lineage best (lane
    /// energy). True stops the run there: a harvest was asked, or the best
    /// is releasable.
    pub fn probe(&self, ease: &Ease, deep: &Deep, device_best: i64) -> bool {
        if self.harvest.load(Ordering::Relaxed) {
            return true;
        }
        if ease.release_now(
            deep,
            device_best.saturating_add(self.offset),
            Instant::now(),
        ) {
            self.probed.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn reported_best(&self, payload: &Payload<J, F, D>) -> Option<i64> {
        best_of(&payload.result).map(|b| b.saturating_add(self.offset))
    }

    fn long(&self) -> Option<i64> {
        Some(self.long_best.load(Ordering::Relaxed)).filter(|&b| b != i64::MAX)
    }

    /// One run ended (`Some`, with its wall time) or was skipped (`None`).
    /// Returns the payload to deliver (reads merged) when a probe released
    /// the group, the last run ended, or a harvest's last running run
    /// returned; or with a fatal error as its outcome. `None` otherwise,
    /// once delivered, or when the lease stopped (dropped).
    pub fn finish_run(
        &self,
        ease: &Ease,
        deep: &Deep,
        run: Option<(StreamResult, Duration)>,
    ) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let left = self
            .outstanding
            .fetch_sub(1, Ordering::AcqRel)
            .saturating_sub(1);
        let running = if run.is_some() {
            self.running
                .fetch_sub(1, Ordering::AcqRel)
                .saturating_sub(1)
        } else {
            self.running.load(Ordering::Acquire)
        };
        let payload = slot.as_mut()?;
        if self.stop.load(Ordering::Relaxed) {
            self.done.store(true, Ordering::Relaxed);
            ease.dropped(
                &self.facts,
                self.long(),
                self.launched.load(Ordering::Relaxed),
            );
            drop(slot.take());
            return None;
        }
        if let Some((run, wall)) = run {
            payload.result.device_access_time_us = payload
                .result
                .device_access_time_us
                .saturating_add(run.device_access_time_us);
            match run.outcome {
                StreamOutcome::Completed(Ok(reads)) => {
                    if let Some(b) = reads
                        .iter()
                        .map(|r| r.energy_milli.saturating_add(self.offset))
                        .min()
                    {
                        self.long_best.fetch_min(b, Ordering::Relaxed);
                    }
                    if let StreamOutcome::Completed(Ok(merged)) = &mut payload.result.outcome {
                        merge_better(merged, reads);
                    }
                    let early =
                        self.probed.load(Ordering::Relaxed) || self.harvest.load(Ordering::Relaxed);
                    if !early {
                        deep.note_run(wall, self.facts.mult);
                    }
                }
                StreamOutcome::Completed(Err(error)) if error.is_fatal() => {
                    payload.result.outcome = StreamOutcome::Completed(Err(error));
                    self.done.store(true, Ordering::Relaxed);
                    ease.dropped(
                        &self.facts,
                        self.long(),
                        self.launched.load(Ordering::Relaxed),
                    );
                    return slot.take();
                }
                StreamOutcome::Completed(Err(error)) => {
                    tracing::debug!(%error, "otsuka ease run failed");
                }
                _ => {}
            }
        }
        let probed = self.probed.load(Ordering::Relaxed);
        let harvest = self.harvest.load(Ordering::Relaxed);
        if !(probed || left == 0 || (harvest && running == 0)) {
            return None;
        }
        let end = if probed {
            End::Crossed
        } else if harvest {
            End::Harvested
        } else {
            End::Last
        };
        self.done.store(true, Ordering::Relaxed);
        let after = self.reported_best(payload).unwrap_or(i64::MAX);
        ease.finished(
            &self.facts,
            self.long(),
            after,
            end,
            self.launched.load(Ordering::Relaxed),
            deep,
        );
        slot.take()
    }

    /// The watcher's check at `now`. A group of a stopped lease is dropped.
    /// When the held best is releasable: with nothing running it is
    /// delivered now (queued runs are skipped); otherwise the running runs
    /// are asked to harvest, and the held reads go out here only if no
    /// harvest came within [`HARVEST_GRACE`].
    pub fn release_due(&self, ease: &Ease, deep: &Deep, now: Instant) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let payload = slot.as_ref()?;
        if self.stop.load(Ordering::Relaxed) {
            self.done.store(true, Ordering::Relaxed);
            ease.dropped(
                &self.facts,
                self.long(),
                self.launched.load(Ordering::Relaxed),
            );
            drop(slot.take());
            return None;
        }
        let best = self.reported_best(payload)?;
        if !ease.release_now(deep, best, now) {
            return None;
        }
        let end = if self.running.load(Ordering::Acquire) > 0 {
            self.harvest.store(true, Ordering::Relaxed);
            let mut since = self
                .harvest_since
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let asked = *since.get_or_insert(now);
            if now.saturating_duration_since(asked) < HARVEST_GRACE {
                return None;
            }
            End::Forced
        } else {
            End::Released
        };
        self.done.store(true, Ordering::Relaxed);
        ease.finished(
            &self.facts,
            self.long(),
            best,
            end,
            self.launched.load(Ordering::Relaxed),
            deep,
        );
        slot.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::deep::DeepConfig;
    use quip_solver_core::SamplerResult;

    fn lookup(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    fn defaults(lanes: usize) -> EaseConfig {
        EaseConfig {
            t0: Duration::from_secs(420),
            gap_milli: 1000,
            lanes,
            end: Duration::from_secs(640),
            post: Duration::from_secs(90),
            release_milli: 250,
            safe: Duration::from_secs(590),
            mult_max: 16,
            max_runs: 48,
        }
    }

    const fn s(secs: u64) -> Duration {
        Duration::from_secs(secs)
    }

    #[test]
    fn off_by_default_and_knobs_parse() {
        assert_eq!(
            EaseConfig::from_lookup(lookup(&[]), 3),
            None,
            "off unless set"
        );
        for off in ["0", "2", "x", "0.5"] {
            assert_eq!(
                EaseConfig::from_lookup(lookup(&[("QUIP_OTSUKA_EASE_BOOST", off)]), 3),
                None,
                "{off}"
            );
        }
        assert_eq!(
            EaseConfig::from_lookup(lookup(&[("QUIP_OTSUKA_EASE_T0_S", "100")]), 3),
            None,
            "a knob alone does not turn it on"
        );
        assert_eq!(
            EaseConfig::from_lookup(lookup(&[("QUIP_OTSUKA_EASE_BOOST", "1")]), 0),
            None,
            "needs the deep lane"
        );
        assert_eq!(
            EaseConfig::from_lookup(lookup(&[("QUIP_OTSUKA_EASE_BOOST", "1")]), 3),
            Some(defaults(3)),
            "defaults, lanes = deep lanes"
        );
        let set = EaseConfig::from_lookup(
            lookup(&[
                ("QUIP_OTSUKA_EASE_BOOST", "1"),
                ("QUIP_OTSUKA_EASE_T0_S", "300"),
                ("QUIP_OTSUKA_EASE_GAP_MILLI", "1500"),
                ("QUIP_OTSUKA_EASE_LANES", "2"),
                ("QUIP_OTSUKA_EASE_END_S", "620"),
                ("QUIP_OTSUKA_EASE_POST_S", "60"),
                ("QUIP_OTSUKA_EASE_RELEASE_MILLI", "400"),
                ("QUIP_OTSUKA_EASE_SAFE_S", "580"),
                ("QUIP_OTSUKA_EASE_MULT_MAX", "8"),
                ("QUIP_OTSUKA_EASE_MAX_RUNS", "12"),
            ]),
            3,
        )
        .unwrap();
        assert_eq!(
            set,
            EaseConfig {
                t0: s(300),
                gap_milli: 1500,
                lanes: 2,
                end: s(620),
                post: s(60),
                release_milli: 400,
                safe: s(580),
                mult_max: 8,
                max_runs: 12,
            }
        );
        let bad = EaseConfig::from_lookup(
            lookup(&[
                ("QUIP_OTSUKA_EASE_BOOST", "1"),
                ("QUIP_OTSUKA_EASE_GAP_MILLI", "0"),
                ("QUIP_OTSUKA_EASE_LANES", "0"),
                ("QUIP_OTSUKA_EASE_MULT_MAX", "17"),
                ("QUIP_OTSUKA_EASE_T0_S", "-1"),
                ("QUIP_OTSUKA_EASE_MAX_RUNS", "1.5"),
            ]),
            3,
        )
        .unwrap();
        assert_eq!(bad, defaults(3), "out-of-range knobs keep their defaults");
        let capped = EaseConfig::from_lookup(
            lookup(&[
                ("QUIP_OTSUKA_EASE_BOOST", "1"),
                ("QUIP_OTSUKA_EASE_LANES", "5"),
            ]),
            2,
        )
        .unwrap();
        assert_eq!(capped.lanes, 2, "never more runs per hold than deep lanes");
    }

    #[test]
    fn ease_seeds_differ_per_run_and_from_the_job_deep_and_stage2() {
        let mut seen = std::collections::HashSet::new();
        for n in 0..64 {
            assert!(seen.insert(ease_seed(42, n)), "run {n} repeats a seed");
        }
        assert!(!seen.contains(&42));
        assert!(!seen.contains(&crate::cluster::deep::deep_seed(42)));
        for j in 1..=8 {
            assert!(!seen.contains(&crate::cluster::stage2::rerun_seed(42, j)));
        }
        assert!(!seen.contains(&0));
    }

    #[test]
    fn trigger_is_a_pure_function_of_time_best_target_and_gap() {
        let c = defaults(3);
        let run = s(25);
        let t = -5_087_000;
        let d = |elapsed: u64, best: i64, rb: Option<i64>, eased: bool| {
            c.decide(s(elapsed), best, rb, Some(t), eased, run, 0, 16)
        };
        assert_eq!(d(419, t + 500, None, false), Err(Skip::Early));
        assert_eq!(
            c.decide(s(450), t + 500, None, None, false, run, 0, 16),
            Err(Skip::NoTarget)
        );
        assert_eq!(d(450, t + 500, Some(t + 400), false), Err(Skip::NotBest));
        assert_eq!(
            d(450, t, None, false),
            Err(Skip::Qualifies),
            "at the target wins now"
        );
        assert_eq!(d(450, t - 1, None, false), Err(Skip::Qualifies));
        assert_eq!(d(450, t + 1001, None, false), Err(Skip::Far));
        let p = d(450, t + 1000, None, false).unwrap();
        assert_eq!((p.gap, p.runs), (1000, 3));
        assert!(
            d(450, t + 500, Some(t + 500), false).is_ok(),
            "ties the round best"
        );
        // In the hold phase a result 1 milli above the target is held ...
        assert!(d(450, t + 1, None, false).is_ok());
        // ... but not once the ease may have begun (SAFE or a risen target).
        assert_eq!(d(590, t + 250, None, false), Err(Skip::Releasing));
        assert_eq!(d(500, t + 250, None, true), Err(Skip::Releasing));
        assert!(
            d(600, t + 900, None, true).is_ok(),
            "beyond the margin in the ease"
        );
        assert_eq!(
            c.decide(s(450), t + 500, None, Some(t), false, run, 48, 16),
            Err(Skip::Spent)
        );
        let p = c
            .decide(s(450), t + 500, None, Some(t), false, run, 46, 16)
            .unwrap();
        assert_eq!(p.runs, 2, "the last runs of the budget");
        // A long job length leaves no time.
        assert_eq!(
            c.decide(s(580), t + 300, None, Some(t), false, s(30), 0, 16),
            Err(Skip::NoTime)
        );
    }

    #[test]
    fn release_rule() {
        let c = defaults(3);
        let t = Some(-1000);
        assert!(!c.releasable(0, None, s(900), true), "no target, hold");
        assert!(c.releasable(-1000, t, s(10), false), "at the target");
        assert!(!c.releasable(-999, t, s(589), false), "hold phase: hold");
        assert!(
            c.releasable(-750, t, s(590), false),
            "SAFE: within the margin"
        );
        assert!(!c.releasable(-749, t, s(590), false));
        assert!(c.releasable(-750, t, s(100), true), "a risen target");
    }

    #[test]
    fn mult_is_sized_to_end_before_the_horizon_and_the_release() {
        let c = defaults(3);
        let run = s(25);
        // 420 s, gap 1000: release eta = 590 + 750/19 = ~629.5 s < END 640.
        assert_eq!(c.eta(s(420), 1000, false).as_secs(), 629);
        assert_eq!(
            c.mult(s(420), 1000, run, false, 16),
            Some(8),
            "420 + 8 x 25 = 620"
        );
        // Gap within the margin: release at SAFE.
        assert_eq!(c.eta(s(420), 200, false), s(590));
        assert_eq!(
            c.mult(s(420), 200, run, false, 16),
            Some(6),
            "420 + 6 x 25 = 570"
        );
        // Capped by MULT_MAX and by the job.
        let short = EaseConfig { mult_max: 4, ..c };
        assert_eq!(short.mult(s(420), 1000, run, false, 16), Some(4));
        assert_eq!(c.mult(s(420), 1000, run, false, 3), Some(3));
        // After END: POST (90 s) from now, still before the release.
        assert_eq!(
            c.mult(s(650), 1000, run, true, 16),
            Some(1),
            "eta 650 + 39.5 s"
        );
        let wide = EaseConfig {
            gap_milli: 5000,
            ..c
        };
        assert_eq!(
            wide.mult(s(650), 5000, run, true, 16),
            Some(3),
            "650 + 90 = 740"
        );
        // Nothing fits.
        assert_eq!(c.mult(s(600), 300, run, true, 16), None);
        // Overflow is no fit, not a panic.
        assert_eq!(c.mult(s(10), 1000, Duration::MAX, false, 16), None);
    }

    fn deep() -> Deep {
        Deep::new(
            DeepConfig {
                lanes: 3,
                mult: 4,
                start: Duration::ZERO,
                q: 0.5,
                record: true,
                guard_milli: 1250,
                release_milli: 500,
                after_stage2: false,
                slack_milli: 0,
                log_drops: false,
                adaptive: None,
            },
            Instant::now(),
        )
    }

    /// A deep state 450 s into its round, target `t`.
    fn deep_at(t: i64) -> (Deep, Instant) {
        let d = deep();
        let now = Instant::now() + s(450);
        d.new_round(now.checked_sub(s(450)).unwrap());
        d.set_live_target(Some(t));
        (d, now)
    }

    #[test]
    fn retarget_and_dedupe() {
        let e = Ease::new(defaults(3));
        let t = -5_000_000;
        let (d, now) = deep_at(t);
        // DEFAULT_RUN 40 s: 450 + 4 x 40 = 610 > eta 590 + ..: fits some.
        let g1 = e.consider(&d, t + 800, 7, 16, now).unwrap();
        assert_eq!(g1.first, 0);
        assert!(!g1.retarget);
        // A worse result is sent and retires nothing.
        assert_eq!(
            e.consider(&d, t + 900, 8, 16, now).err(),
            Some(Skip::NotBest)
        );
        assert!(!g1.retired.load(Ordering::Relaxed));
        // The same instance again (after delivery): new seeds, old retired.
        let g2 = e.consider(&d, t + 700, 7, 16, now).unwrap();
        assert_eq!(
            g2.first, 3,
            "seed indices continue: no (instance, seed) twice"
        );
        assert!(!g2.retarget && g1.retired.load(Ordering::Relaxed));
        // A deeper result on another instance retargets.
        let g3 = e.consider(&d, t + 600, 9, 16, now).unwrap();
        assert_eq!(g3.first, 0);
        assert!(g3.retarget && g2.retired.load(Ordering::Relaxed));
        // A deeper result that is not held (it qualifies) retires too.
        assert_eq!(
            e.consider(&d, t - 1, 10, 16, now).err(),
            Some(Skip::Qualifies)
        );
        assert!(g3.retired.load(Ordering::Relaxed));
        // Nothing is held after that: the round best qualifies.
        assert_eq!(
            e.consider(&d, t + 100, 11, 16, now).err(),
            Some(Skip::NotBest)
        );
        // A new round forgets the best and the seed counts.
        d.new_round(now);
        d.set_live_target(Some(t));
        assert_eq!(e.consider(&d, t + 900, 7, 16, now).err(), Some(Skip::Early));
        let later = now + s(450);
        let g4 = e.consider(&d, t + 800, 7, 16, later).unwrap();
        assert_eq!(g4.first, 0, "a new round's instance starts over");
    }

    #[test]
    fn max_runs_bounds_reboosting_one_instance() {
        let e = Ease::new(EaseConfig {
            max_runs: 5,
            ..defaults(3)
        });
        let t = -5_000_000;
        let (d, now) = deep_at(t);
        assert_eq!(e.consider(&d, t + 800, 1, 16, now).unwrap().plan.runs, 3);
        assert_eq!(e.consider(&d, t + 800, 1, 16, now).unwrap().plan.runs, 2);
        assert_eq!(e.consider(&d, t + 800, 1, 16, now).err(), Some(Skip::Spent));
    }

    fn result(energies: &[i64]) -> StreamResult {
        StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Ok(energies
                .iter()
                .map(|&e| SamplerResult {
                    energy_milli: e,
                    spins: vec![1],
                })
                .collect())),
            device_access_time_us: 1,
        }
    }

    type G = EaseGroup<(), (), ()>;

    fn group(best: i64, runs: usize) -> (G, Arc<AtomicBool>, Arc<AtomicBool>) {
        let stop = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicBool::new(false));
        let g = G::new(
            best,
            0,
            s(450),
            0,
            4,
            runs,
            Arc::clone(&stop),
            Arc::clone(&retired),
            Payload {
                job: (),
                family: (),
                dest: (),
                result: result(&[best, best + 10]),
            },
        );
        (g, stop, retired)
    }

    fn best(p: &Payload<(), (), ()>) -> i64 {
        best_of(&p.result).unwrap()
    }

    #[test]
    fn group_delivers_once_after_its_last_run_with_reads_merged() {
        let e = Ease::new(defaults(2));
        let t = -5_000;
        let (d, _) = deep_at(t);
        let (g, _, _) = group(t + 800, 2);
        assert!(g.begin() && g.begin());
        assert!(g
            .finish_run(&e, &d, Some((result(&[t + 900]), s(1))))
            .is_none());
        let p = g
            .finish_run(&e, &d, Some((result(&[t + 700, t + 999]), s(1))))
            .unwrap();
        assert_eq!(best(&p), t + 700, "merged");
        assert!(!g.pending() && g.stopped());
        assert!(!g.begin(), "never again");
    }

    #[test]
    fn retired_group_skips_queued_runs_and_still_delivers() {
        let e = Ease::new(defaults(3));
        let (d, _) = deep_at(-5_000);
        let (g, _, retired) = group(-4_000, 3);
        assert!(g.begin());
        retired.store(true, Ordering::Relaxed);
        assert!(g.skippable() && !g.stopped(), "the running run continues");
        assert!(!g.begin());
        assert!(g.finish_run(&e, &d, None).is_none());
        assert!(!g.begin());
        assert!(g.finish_run(&e, &d, None).is_none());
        let p = g
            .finish_run(&e, &d, Some((result(&[-4_100]), s(1))))
            .unwrap();
        assert_eq!(best(&p), -4_100);
    }

    #[test]
    fn probe_releases_below_the_target_and_lease_stop_drops() {
        let e = Ease::new(defaults(2));
        let t = -5_000;
        let (d, _) = deep_at(t);
        let (g, _, _) = group(t + 800, 2);
        assert!(g.begin() && g.begin());
        assert!(
            !g.probe(&e, &d, t + 1),
            "hold phase: above the target runs on"
        );
        assert!(g.probe(&e, &d, t), "at the target: stop and deliver");
        let p = g.finish_run(&e, &d, Some((result(&[t]), s(1)))).unwrap();
        assert_eq!(best(&p), t);
        assert!(g.stopped(), "the other run stops at its next chunk");
        assert!(g
            .finish_run(&e, &d, Some((result(&[t - 5]), s(1))))
            .is_none());

        let (g, stop, _) = group(t + 800, 1);
        assert!(g.begin());
        stop.store(true, Ordering::Relaxed);
        assert!(g.stopped());
        assert!(
            g.finish_run(&e, &d, Some((result(&[t - 5]), s(1))))
                .is_none(),
            "dropped"
        );
        assert!(!g.pending());
    }

    #[test]
    fn watcher_releases_harvests_and_forces() {
        let e = Ease::new(defaults(2));
        let t = -5_000;
        let (d, now) = deep_at(t);
        // Not releasable: nothing.
        let (g, _, _) = group(t + 800, 2);
        assert!(g.release_due(&e, &d, now).is_none() && g.pending());
        // The target eases to within the margin: nothing running -> Released.
        d.set_live_target(Some(t + 600));
        let p = g.release_due(&e, &d, now).unwrap();
        assert_eq!(best(&p), t + 800);
        assert!(!g.begin(), "queued runs are skipped");
        // Running: harvest asked, delivered by the run (Harvested) ...
        d.set_live_target(Some(t));
        let (g, _, _) = group(t + 800, 2);
        assert!(g.begin());
        d.set_live_target(Some(t + 600));
        assert!(g.release_due(&e, &d, now).is_none());
        assert!(g.probe(&e, &d, t + 5_000), "harvest stops the run");
        assert!(
            g.finish_run(&e, &d, None).is_none(),
            "the queued run is skipped"
        );
        let p = g
            .finish_run(&e, &d, Some((result(&[t + 790]), s(1))))
            .unwrap();
        assert_eq!(best(&p), t + 790);
        // ... or forced after the grace.
        let (g, _, _) = group(t + 800, 1);
        assert!(g.begin());
        assert!(g.release_due(&e, &d, now).is_none());
        assert!(g.release_due(&e, &d, now + HARVEST_GRACE).is_some());
        assert!(g.stopped());
        assert!(g.finish_run(&e, &d, Some((result(&[t]), s(1)))).is_none());
    }
}
