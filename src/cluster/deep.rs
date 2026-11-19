//! Local build (all7, agent AH): the late-round deep lane.
//!
//! Off by default: with `QUIP_OTSUKA_DEEP_LANES` unset (or 0) the sampler
//! never builds a [`Deep`], no lane is reserved, and every lane runs exactly
//! as in all6. It needs the local lease path (`QUIP_OTSUKA_SCREEN_THREADS`
//! above 1), where the sampler owns a salt's single delivery.
//!
//! Why (a study of a rival miner): the rival's hold-phase win hazard rises with
//! time in the round (0.006/min at minutes 0-2, 0.07-0.08 at minutes 6-8)
//! while ours is flat; its winning proofs carry 114-523 s of device time and
//! cross the threshold by thin margins with lag 0. That is the signature of
//! a long anneal on a selected instance that is checked while it runs and
//! submitted the moment it crosses. Our lanes run many ~47 s anneals.
//!
//! What: `L` of the lanes (the first `L`) are deep lanes. A deep lane takes
//! ordinary work like any lane until a candidate appears. A candidate is the
//! FIRST full run of a local lease salt that, at least `START` after the
//! round began, is at or below the rolling `Q` quantile of first-run bests
//! (no trigger before 32 of them) and, with `RECORD` on, at or below the
//! round's best first run so far. Its reads are held (not delivered) in a
//! [`DeepGroup`], and one long anneal of the same instance (`MULT` times the
//! job's sweeps, a fresh seed, same reads and ladder ends) is queued for the
//! deep lanes only. The held salt is delivered once (polish, sort, push),
//! with the long run's reads merged in (read count unchanged):
//! - at once when the long run's lineage best, read at a PA checkpoint about
//!   every second ([`PROBE_EVERY`]), reaches the live target plus `RELEASE`:
//!   the run stops there, takes its final descent and is delivered (the
//!   submit-on-crossing of the rival's proofs);
//! - when the target eases to within `RELEASE` of the held best: a queued
//!   run is not started; a running one is asked to harvest at its next probe
//!   (or, after [`HARVEST_GRACE`] without one, the first run's reads go out
//!   and the run is cancelled);
//! - at the end of the long run otherwise;
//! - never, when the lease stopped (new round, shutdown): dropped, and a
//!   running long anneal stops at its next chunk, as every lane does.
//!
//! A first run at or below the live target plus `GUARD` is never held. At
//! most `L` groups are held at once, so the deep work never takes more than
//! `L` lanes and stage 2 and the pool keep the others.
//!
//! The round start is the last lease cut (a lease stopped with salts still
//! unclaimed or unsolved, which happens at a round change), or the session
//! start. It only gates `START` and the round record.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_DEEP_LANES` | unset (off) | Deep lanes `L`, `1..=8`, fewer than `QUIP_OTSUKA_CONCURRENCY`. 0 is off. |
//! | `QUIP_OTSUKA_DEEP_MULT` | 4 | Long-run sweeps as a multiple of the job's, `2..=16`. |
//! | `QUIP_OTSUKA_DEEP_START_S` | 60 | No trigger earlier in the round, `0..=1800` s. |
//! | `QUIP_OTSUKA_DEEP_Q` | 0.10 | Trigger quantile of the last 256 first-run bests, `0 < Q < 1`. |
//! | `QUIP_OTSUKA_DEEP_RECORD` | 1 | 1: also at or below the round's best first run so far. |
//! | `QUIP_OTSUKA_DEEP_GUARD_MILLI` | 1250 | Never hold a first best at or below target plus this, `0..=20000`. |
//! | `QUIP_OTSUKA_DEEP_RELEASE_MILLI` | 500 | Deliver once the held or probed best is within this of the target, `0..=20000`. |
//! | `QUIP_OTSUKA_DEEP_AFTER_STAGE2` | 0 | 1: offer a first run to stage 2 first, the deep lane takes what stage 2 does not hold. |
//!
//! Every finished group logs `otsuka deep` at info (first, long, merged,
//! how it ended, round and run seconds), and every 16th a summary.
//!
//! ## v2 (all9, agent AJ): adaptive lane allocation
//!
//! Off unless `QUIP_OTSUKA_DEEP_ADAPTIVE=1` (with the deep lane on). Off, every
//! decision, launch and log line is the all8 one.
//!
//! Why (live, 17/11 00:30-03:42, one deep lane, MULT 4): the deep lane gave
//! 3 of our 5 wins from about 0.76 lane-hours, ~4 crossings per lane-hour
//! against ~0.26 for ordinary lane work (48 h before). Its weak spots were:
//! - holds 60-120 s into the round took the round's still-weak record (0
//!   of 6 reached -5087; 5 of 11 held later did) and blocked the only deep
//!   lane for ~135 s;
//! - while it ran, a better record could not be held (cap 1);
//! - 6 holds of 22 were dropped at a round change. A result for an old
//!   round is useless: every nonce derives from the round's
//!   `last_proof_block_hash` and the `LeaseSink` refuses pushes once the
//!   lease stops. So the only fix is not to start what cannot end in time.
//!
//! What, with it on:
//! - **Allocation.** The cap on held groups is `alloc` in `0..=LMAX`, not
//!   `LANES`. It wants 0 before `RAMP0` and after `LATE`, 1 at `RAMP0`,
//!   rising linearly to `LMAX` at `RAMP1`. While the round's best first run
//!   is less than `GAP_FAR` above the live target (fresh work is close), it
//!   keeps at most `max(1, LMAX - 1)`. It rises by at most one per `STEP`,
//!   falls at once, and is 0 at a new round. Running long runs are never
//!   cut by a fall; it only gates new holds. Every change logs
//!   `otsuka deep alloc` (from, to, round_s, gap_milli, run_s, reason).
//! - **Length.** The long run takes the largest multiple in
//!   `MULT_MIN..=MULT_MAX` that ends by `HORIZON` into the round (the
//!   hold phase is 100 blocks, ~606 s; a run ending a little later can
//!   still win as the target eases), else `MULT_MIN`. The run time per
//!   multiple is measured: a moving average of first full runs and of
//!   long runs that ran to their end (40 s until the first one).
//! - **Order.** A deep lane takes the queued long run of the best first run,
//!   not the oldest.
//! - **Drops** log `otsuka deep dropped` (first, mult, round_s, held_s,
//!   ran_s, last probe) so a later block can see what the cut runs had.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_DEEP_ADAPTIVE` | 0 | 1: adaptive allocation and length (above). |
//! | `QUIP_OTSUKA_DEEP_LMAX` | `LANES` | Deep-capable lanes, `1..=8`, fewer than `QUIP_OTSUKA_CONCURRENCY`. |
//! | `QUIP_OTSUKA_DEEP_RAMP0_S` | 120 | No hold earlier in the round; 1 group from here. |
//! | `QUIP_OTSUKA_DEEP_RAMP1_S` | 180 | `LMAX` groups from here (linear in between). |
//! | `QUIP_OTSUKA_DEEP_LATE_S` | 540 | No hold later in the round. |
//! | `QUIP_OTSUKA_DEEP_HORIZON_S` | 620 | A long run should end by then. |
//! | `QUIP_OTSUKA_DEEP_MULT_MIN` | 3 | Shortest long run, `2..=16`. |
//! | `QUIP_OTSUKA_DEEP_MULT_MAX` | `MULT` | Longest long run, `2..=16`. |
//! | `QUIP_OTSUKA_DEEP_GAP_FAR_MILLI` | 2000 | Below this gap (round best minus target) keep one lane fewer. 0: never. |
//! | `QUIP_OTSUKA_DEEP_STEP_S` | 15 | Slowest rise of `alloc`, one per this. |
//! | `QUIP_OTSUKA_DEEP_SLACK_MILLI` | 0 | With `RECORD`, also hold a first up to this above the round record (any mode). |
//! | `QUIP_OTSUKA_DEEP_LOG_DROPS` | 0 | 1: log every dropped group (any mode; on with `ADAPTIVE`). |

use super::stage1::{merge_better, RollingCutoff};
use super::stage2::{best_of, Payload};
use quip_solver_core::{StreamOutcome, StreamResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// `XORed` into the job seed for the long run's draws.
pub const DEEP_SEED_SALT: u64 = 0x0DEE_9A11_C0FF_EE5D;

/// How often a running long anneal reads its lineage bests back (at the next
/// PA checkpoint after this much time). One small copy and one stream sync.
pub const PROBE_EVERY: Duration = Duration::from_secs(1);

/// How long a requested harvest may take before the watcher delivers the
/// first run's reads instead and cancels the long run.
pub const HARVEST_GRACE: Duration = Duration::from_secs(3);

/// all9: the run time per job length assumed until one is measured.
pub const DEFAULT_RUN: Duration = Duration::from_secs(40);

/// all9: the adaptive allocation knobs (`QUIP_OTSUKA_DEEP_ADAPTIVE=1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adaptive {
    /// No hold before this; one group from here.
    pub ramp0: Duration,
    /// `lanes` (LMAX) groups from here, linear in between.
    pub ramp1: Duration,
    /// No hold after this.
    pub late: Duration,
    /// A long run should end by this time into the round.
    pub horizon: Duration,
    /// Shortest long run (job-sweep multiple).
    pub mult_min: usize,
    /// Longest long run.
    pub mult_max: usize,
    /// Below this gap (round best first minus live target) keep one lane
    /// fewer; 0 never.
    pub gap_far_milli: i64,
    /// Slowest rise of the allocation, one per this.
    pub step: Duration,
}

impl Adaptive {
    /// Default `RAMP0`.
    pub const RAMP0: Duration = Duration::from_mins(2);
    /// Default `RAMP1`.
    pub const RAMP1: Duration = Duration::from_mins(3);
    /// Default `LATE`.
    pub const LATE: Duration = Duration::from_mins(9);
    /// Default `HORIZON`: the hold phase is 100 blocks of ~6.06 s (606 s);
    /// the agent-AJ model puts the MULT 4 to 3 switch near 495 s (34 s
    /// runs), i.e. an end by ~630 s.
    pub const HORIZON: Duration = Duration::from_secs(620);
    /// Default `MULT_MIN`.
    pub const MULT_MIN: usize = 3;
    /// Default `GAP_FAR_MILLI`.
    pub const GAP_FAR_MILLI: i64 = 2000;
    /// Default `STEP`.
    pub const STEP: Duration = Duration::from_secs(15);

    /// Groups wanted `elapsed` into the round with `lmax` deep lanes, and
    /// why. `gap` is the round's best first run minus the live target
    /// (`None`: unknown, treated as far).
    #[must_use]
    pub fn want(&self, lmax: usize, elapsed: Duration, gap: Option<i64>) -> (usize, &'static str) {
        if lmax == 0 || elapsed < self.ramp0 {
            return (0, "early");
        }
        if elapsed > self.late {
            return (0, "late");
        }
        let ramp = if elapsed >= self.ramp1 || self.ramp1 <= self.ramp0 {
            lmax
        } else {
            let span = self.ramp1.saturating_sub(self.ramp0).as_secs_f64();
            let into = elapsed.saturating_sub(self.ramp0).as_secs_f64();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let extra = ((lmax - 1) as f64 * into / span).floor() as usize;
            (1 + extra).min(lmax)
        };
        let near = self.gap_far_milli > 0 && gap.is_some_and(|g| g < self.gap_far_milli);
        let cap = (lmax - 1).max(1);
        if near && ramp > cap {
            return (cap, "near");
        }
        (ramp, if ramp == lmax { "full" } else { "ramp" })
    }

    /// The long run's multiple `elapsed` into the round, for a run time of
    /// `run` per job length: the largest in `mult_min..=mult_max` that ends
    /// by `horizon`, else `mult_min`.
    #[must_use]
    pub fn mult(&self, elapsed: Duration, run: Duration) -> usize {
        (self.mult_min..=self.mult_max.max(self.mult_min))
            .rev()
            .find(|&m| {
                u32::try_from(m)
                    .ok()
                    .and_then(|m| run.checked_mul(m))
                    .and_then(|d| elapsed.checked_add(d))
                    .is_some_and(|end| end <= self.horizon)
            })
            .unwrap_or(self.mult_min)
    }
}

/// all9: the queued item whose key is lowest (ties: the oldest), removed.
pub fn take_best<T>(queue: &mut VecDeque<T>, key: impl Fn(&T) -> i64) -> Option<T> {
    let best = queue
        .iter()
        .enumerate()
        .min_by_key(|(i, item)| (key(item), *i))
        .map(|(i, _)| i)?;
    queue.remove(best)
}

/// A knob reader: the parsed value of a variable when set and accepted.
type Knob<'a> = dyn Fn(&str, &dyn Fn(f64) -> bool) -> Option<f64> + 'a;

/// The deep-lane knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepConfig {
    /// Deep lanes (the first `lanes` lanes), also the cap on held groups.
    pub lanes: usize,
    /// Long-run sweeps as a multiple of the job's.
    pub mult: usize,
    /// No trigger before this much of the round has passed.
    pub start: Duration,
    /// Trigger quantile of the first-run bests.
    pub q: f64,
    /// Also require the round's best first run so far.
    pub record: bool,
    /// Never hold a first best at or below the live target plus this.
    pub guard_milli: i64,
    /// Deliver once the held or probed best is within this of the target.
    pub release_milli: i64,
    /// Offer a first run to stage 2 first; the deep lane gets what stage 2
    /// does not hold. Off: the deep lane first (both always see every first
    /// run, so their windows and the round record stay complete).
    pub after_stage2: bool,
    /// all9: with `record`, also hold a first up to this above the round
    /// record. 0 is the all8 rule.
    pub slack_milli: i64,
    /// all9: log every dropped group.
    pub log_drops: bool,
    /// all9: adaptive allocation and length; `None` is all8. With it,
    /// `lanes` is LMAX.
    pub adaptive: Option<Adaptive>,
}

impl DeepConfig {
    /// Default long-run multiple.
    pub const MULT: usize = 4;
    /// Default start delay.
    pub const START: Duration = Duration::from_mins(1);
    /// Default trigger quantile.
    pub const Q: f64 = 0.10;
    /// Default guard.
    pub const GUARD_MILLI: i64 = 1250;
    /// Default release margin.
    pub const RELEASE_MILLI: i64 = 500;

    /// `None` (off) unless `QUIP_OTSUKA_DEEP_LANES` is valid and above 0.
    /// Every variable that is set logs `applied` or `ignored`.
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
        let lanes = num("QUIP_OTSUKA_DEEP_LANES", &|v| {
            (0.0..=8.0).contains(&v) && int(v)
        })
        .map(|v| v as usize);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mult = num("QUIP_OTSUKA_DEEP_MULT", &|v| {
            (2.0..=16.0).contains(&v) && int(v)
        })
        .map_or(Self::MULT, |v| v as usize);
        let start = num("QUIP_OTSUKA_DEEP_START_S", &|v| (0.0..=1800.0).contains(&v))
            .map_or(Self::START, Duration::from_secs_f64);
        let q = num("QUIP_OTSUKA_DEEP_Q", &|v| v > 0.0 && v < 1.0).unwrap_or(Self::Q);
        let record = num("QUIP_OTSUKA_DEEP_RECORD", &|v| {
            (0.0..=1.0).contains(&v) && int(v)
        })
        .is_none_or(|v| v > 0.5);
        #[allow(clippy::cast_possible_truncation)]
        let guard_milli = num("QUIP_OTSUKA_DEEP_GUARD_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(Self::GUARD_MILLI, |v| v as i64);
        #[allow(clippy::cast_possible_truncation)]
        let release_milli = num("QUIP_OTSUKA_DEEP_RELEASE_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(Self::RELEASE_MILLI, |v| v as i64);
        let after_stage2 = num("QUIP_OTSUKA_DEEP_AFTER_STAGE2", &|v| {
            (0.0..=1.0).contains(&v) && int(v)
        })
        .is_some_and(|v| v > 0.5);
        let flag = |name: &str| {
            num(name, &|v| (0.0..=1.0).contains(&v) && int(v)).is_some_and(|v| v > 0.5)
        };
        #[allow(clippy::cast_possible_truncation)]
        let slack_milli = num("QUIP_OTSUKA_DEEP_SLACK_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(0, |v| v as i64);
        let log_drops = flag("QUIP_OTSUKA_DEEP_LOG_DROPS");
        let adaptive = flag("QUIP_OTSUKA_DEEP_ADAPTIVE").then(|| Self::adaptive_from(&num, mult));
        let mut lanes = lanes.filter(|&l| l > 0)?;
        if let Some((Some(lmax), _)) = adaptive {
            lanes = lmax;
        }
        Some(Self {
            lanes,
            mult,
            start,
            q,
            record,
            guard_milli,
            release_milli,
            after_stage2,
            slack_milli,
            log_drops: log_drops || adaptive.is_some(),
            adaptive: adaptive.map(|(_, a)| a),
        })
    }

    /// all9: the adaptive knobs (and `LMAX`, when set), read only with
    /// `QUIP_OTSUKA_DEEP_ADAPTIVE=1`.
    fn adaptive_from(num: &Knob<'_>, mult: usize) -> (Option<usize>, Adaptive) {
        let int = |v: f64| v.fract() == 0.0;
        let secs = |name: &str, default: Duration| {
            num(name, &|v| (0.0..=3600.0).contains(&v)).map_or(default, Duration::from_secs_f64)
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = |name: &str, lo: f64, hi: f64| {
            num(name, &|v| (lo..=hi).contains(&v) && int(v)).map(|v| v as usize)
        };
        let lmax = count("QUIP_OTSUKA_DEEP_LMAX", 1.0, 8.0);
        let ramp0 = secs("QUIP_OTSUKA_DEEP_RAMP0_S", Adaptive::RAMP0);
        let ramp1 = secs("QUIP_OTSUKA_DEEP_RAMP1_S", Adaptive::RAMP1).max(ramp0);
        let late = secs("QUIP_OTSUKA_DEEP_LATE_S", Adaptive::LATE);
        let horizon = secs("QUIP_OTSUKA_DEEP_HORIZON_S", Adaptive::HORIZON);
        let mult_max = count("QUIP_OTSUKA_DEEP_MULT_MAX", 2.0, 16.0).unwrap_or(mult);
        let mult_min = count("QUIP_OTSUKA_DEEP_MULT_MIN", 2.0, 16.0)
            .unwrap_or(Adaptive::MULT_MIN)
            .min(mult_max);
        #[allow(clippy::cast_possible_truncation)]
        let gap_far_milli = num("QUIP_OTSUKA_DEEP_GAP_FAR_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(Adaptive::GAP_FAR_MILLI, |v| v as i64);
        let step = secs("QUIP_OTSUKA_DEEP_STEP_S", Adaptive::STEP).min(Duration::from_mins(10));
        (
            lmax,
            Adaptive {
                ramp0,
                ramp1,
                late,
                horizon,
                mult_min,
                mult_max,
                gap_far_milli,
                step,
            },
        )
    }

    /// [`Self::from_lookup`] on the process environment, read once.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        static ENV: OnceLock<Option<DeepConfig>> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }
}

/// The seed of the long run of a job seeded `seed` (splitmix64).
#[must_use]
pub fn deep_seed(seed: u64) -> u64 {
    let mut z = (seed ^ DEEP_SEED_SALT).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

struct Window {
    firsts: RollingCutoff,
    seen: usize,
    shape: Option<(usize, usize)>,
    /// The round this record belongs to.
    epoch: u64,
    record: Option<i64>,
}

struct Round {
    start: Instant,
    epoch: u64,
}

/// all9: the adaptive allocation's state.
struct Alloc {
    cur: usize,
    changed: Instant,
    epoch: u64,
}

#[derive(Default)]
struct Stats {
    groups: AtomicU64,
    crossed: AtomicU64,
    harvested: AtomicU64,
    released: AtomicU64,
    forced: AtomicU64,
    improved: AtomicU64,
    gain_milli: AtomicU64,
    dropped: AtomicU64,
}

/// How a group ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum End {
    /// The long run ran to its end.
    Last,
    /// The long run's probed best reached the target: delivered at once.
    Crossed,
    /// The target eased to the held best; the running long run harvested.
    Harvested,
    /// The target eased to the held best before the long run started.
    Released,
    /// A requested harvest did not come within [`HARVEST_GRACE`].
    Forced,
}

/// Deep-lane state shared by every lane.
pub struct Deep {
    config: DeepConfig,
    window: Mutex<Window>,
    round: Mutex<Round>,
    live_target: AtomicI64,
    has_target: AtomicBool,
    stats: Stats,
    /// all9: the adaptive allocation (unused without `adaptive`).
    alloc: Mutex<Alloc>,
    /// all9: moving average of the wall time of one job length, in µs; 0
    /// until measured (adaptive only).
    run_us: AtomicU64,
}

impl Deep {
    /// First-run bests the trigger looks back over.
    pub const HISTORY: usize = 256;
    /// Trigger cutoff refresh period.
    pub const REFRESH: usize = 16;
    /// No trigger before this many first-run bests of the current shape.
    pub const WARMUP: usize = 32;
    const LOG_EVERY: u64 = 16;

    /// Fresh state for `config`; the round starts `now`.
    #[must_use]
    pub fn new(config: DeepConfig, now: Instant) -> Self {
        Self {
            config,
            window: Mutex::new(Window {
                firsts: RollingCutoff::new(config.q, Self::HISTORY, Self::REFRESH),
                seen: 0,
                shape: None,
                epoch: 0,
                record: None,
            }),
            round: Mutex::new(Round {
                start: now,
                epoch: 0,
            }),
            live_target: AtomicI64::new(0),
            has_target: AtomicBool::new(false),
            stats: Stats::default(),
            alloc: Mutex::new(Alloc {
                cur: 0,
                changed: now,
                epoch: 0,
            }),
            run_us: AtomicU64::new(0),
        }
    }

    /// The knobs.
    #[must_use]
    pub const fn config(&self) -> &DeepConfig {
        &self.config
    }

    /// A lease was cut (round change, shutdown) at `at`, or a session
    /// starts: a new round begins, with no record.
    pub fn new_round(&self, at: Instant) {
        let mut round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        round.start = at;
        round.epoch += 1;
    }

    /// all12: time since the round began, and the round's epoch.
    #[must_use]
    pub fn round_now(&self, now: Instant) -> (Duration, u64) {
        let round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        (now.saturating_duration_since(round.start), round.epoch)
    }

    /// Time since the round began.
    #[must_use]
    pub fn round_elapsed(&self, now: Instant) -> Duration {
        let round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        now.saturating_duration_since(round.start)
    }

    /// Record the live session target; `None` keeps the last one.
    pub fn set_live_target(&self, target: Option<i64>) {
        if let Some(t) = target {
            self.live_target.store(t, Ordering::Relaxed);
            self.has_target.store(true, Ordering::Release);
        }
    }

    /// The last live session target seen.
    #[must_use]
    pub fn live_target(&self) -> Option<i64> {
        self.has_target
            .load(Ordering::Acquire)
            .then(|| self.live_target.load(Ordering::Relaxed))
    }

    /// Whether a best (reported energy) is within the release margin of the
    /// live target: deliver it now.
    #[must_use]
    pub fn releasable(&self, best: i64) -> bool {
        self.live_target()
            .is_some_and(|t| best <= t.saturating_add(self.config.release_milli))
    }

    /// Record a first full run's best (reported energy) that another stage
    /// (stage 2) holds: the window and the round record see it, no decision.
    pub fn observe(&self, shape: (usize, usize), best: i64, now: Instant) {
        self.record_first(shape, best, now);
    }

    /// Feed the window and the round record; returns (in the quantile after
    /// the warm-up, at or below the round record, time into the round).
    fn record_first(
        &self,
        shape: (usize, usize),
        best: i64,
        now: Instant,
    ) -> (bool, bool, Duration) {
        let (elapsed, epoch) = {
            let round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
            (now.saturating_duration_since(round.start), round.epoch)
        };
        let mut w = self.window.lock().unwrap_or_else(PoisonError::into_inner);
        if w.shape != Some(shape) {
            w.firsts = RollingCutoff::new(self.config.q, Self::HISTORY, Self::REFRESH);
            w.seen = 0;
            w.shape = Some(shape);
            w.record = None;
        }
        if w.epoch != epoch {
            w.epoch = epoch;
            w.record = None;
        }
        w.seen += 1;
        let admitted = w.firsts.admit(best) && w.seen > Self::WARMUP;
        // all9: SLACK widens the hold rule only; the record stays the best.
        let record = w
            .record
            .is_none_or(|r| best <= r.saturating_add(self.config.slack_milli));
        if w.record.is_none_or(|r| best <= r) {
            w.record = Some(best);
        }
        (admitted, record, elapsed)
    }

    /// all9: the round's best first run, if one was recorded this round.
    fn round_record(&self) -> Option<i64> {
        let epoch = self
            .round
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .epoch;
        let w = self.window.lock().unwrap_or_else(PoisonError::into_inner);
        w.record.filter(|_| w.epoch == epoch)
    }

    /// all9: the measured wall time of one job length (first full runs and
    /// long runs that ran to their end), or [`DEFAULT_RUN`].
    #[must_use]
    pub fn run_estimate(&self) -> Duration {
        match self.run_us.load(Ordering::Relaxed) {
            0 => DEFAULT_RUN,
            us => Duration::from_micros(us),
        }
    }

    /// all9: a run of `mult` job lengths took `wall` (adaptive only; one
    /// eighth weight per sample).
    pub fn note_run(&self, wall: Duration, mult: usize) {
        if self.config.adaptive.is_none() || mult == 0 {
            return;
        }
        let per = u64::try_from(wall.as_micros() / mult as u128)
            .unwrap_or(u64::MAX)
            .max(1);
        let mut old = self.run_us.load(Ordering::Relaxed);
        loop {
            let new = if old == 0 {
                per
            } else {
                old - old / 8 + per / 8
            };
            match self
                .run_us
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(now) => old = now,
            }
        }
    }

    /// all9: the long run's multiple for a hold `elapsed` into the round:
    /// `MULT` without `adaptive`.
    #[must_use]
    pub fn mult_for(&self, elapsed: Duration) -> usize {
        self.config
            .adaptive
            .map_or(self.config.mult, |a| a.mult(elapsed, self.run_estimate()))
    }

    /// all9: the cap on held groups at `now`: `lanes` without `adaptive`;
    /// with it, the slewed allocation (logged when it changes).
    pub fn allocation(&self, now: Instant) -> usize {
        let Some(a) = self.config.adaptive else {
            return self.config.lanes;
        };
        let (elapsed, epoch) = {
            let round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
            (now.saturating_duration_since(round.start), round.epoch)
        };
        let gap = self
            .round_record()
            .zip(self.live_target())
            .map(|(r, t)| r.saturating_sub(t));
        let (want, why) = a.want(self.config.lanes, elapsed, gap);
        let mut st = self.alloc.lock().unwrap_or_else(PoisonError::into_inner);
        let from = st.cur;
        let mut reason = why;
        if st.epoch != epoch {
            // The reset dates from the round start, however late it is seen.
            st.epoch = epoch;
            st.cur = 0;
            st.changed = now.checked_sub(elapsed).unwrap_or(now);
            reason = "new_round";
        }
        if want < st.cur {
            st.cur = want;
            st.changed = now;
        } else if want > st.cur && now.saturating_duration_since(st.changed) >= a.step {
            st.cur += 1;
            st.changed = now;
        }
        let to = st.cur;
        drop(st);
        if to != from {
            tracing::info!(
                from,
                to,
                round_s = elapsed.as_secs_f64(),
                gap_milli = gap,
                run_s = self.run_estimate().as_secs_f64(),
                mult = a.mult(elapsed, self.run_estimate()),
                reason,
                "otsuka deep alloc"
            );
        }
        to
    }

    /// Record a first full run's best (reported energy) and decide whether
    /// to hold it for a long run. `held` is the number of deep groups held
    /// now. Every first-run best feeds the window and the round record,
    /// triggered or not; a new shape empties the window.
    pub fn should_deep(&self, shape: (usize, usize), best: i64, now: Instant, held: usize) -> bool {
        let (admitted, record, elapsed) = self.record_first(shape, best, now);
        admitted
            && (record || !self.config.record)
            && elapsed >= self.config.start
            && held < self.allocation(now)
            // Can win now or very soon: never delay it.
            && self
                .live_target()
                .is_none_or(|t| best > t.saturating_add(self.config.guard_milli))
    }

    #[allow(clippy::cast_precision_loss)] // Seconds for a log line.
    fn finished(&self, group: &GroupFacts, long: Option<i64>, merged: i64, end: End) {
        let st = &self.stats;
        let groups = st.groups.fetch_add(1, Ordering::Relaxed) + 1;
        let counter = match end {
            End::Last => None,
            End::Crossed => Some(&st.crossed),
            End::Harvested => Some(&st.harvested),
            End::Released => Some(&st.released),
            End::Forced => Some(&st.forced),
        };
        if let Some(c) = counter {
            c.fetch_add(1, Ordering::Relaxed);
        }
        let gain = u64::try_from(group.first_best.saturating_sub(merged)).unwrap_or(0);
        st.improved
            .fetch_add(u64::from(gain > 0), Ordering::Relaxed);
        st.gain_milli.fetch_add(gain, Ordering::Relaxed);
        if self.config.adaptive.is_some() {
            // all9: the same line with the run's multiple.
            tracing::info!(
                first = group.first_best,
                long,
                merged,
                end = ?end,
                round_s = group.round_s,
                held_s = group.held_at.elapsed().as_secs_f64(),
                live_target = self.live_target(),
                mult = group.mult,
                "otsuka deep"
            );
        } else {
            tracing::info!(
                first = group.first_best,
                long,
                merged,
                end = ?end,
                round_s = group.round_s,
                held_s = group.held_at.elapsed().as_secs_f64(),
                live_target = self.live_target(),
                "otsuka deep"
            );
        }
        if groups.is_multiple_of(Self::LOG_EVERY) {
            tracing::info!(
                groups,
                crossed = st.crossed.load(Ordering::Relaxed),
                harvested = st.harvested.load(Ordering::Relaxed),
                released = st.released.load(Ordering::Relaxed),
                forced = st.forced.load(Ordering::Relaxed),
                dropped = st.dropped.load(Ordering::Relaxed),
                improved = st.improved.load(Ordering::Relaxed),
                mean_gain_milli = st.gain_milli.load(Ordering::Relaxed) / groups,
                "otsuka deep summary"
            );
        }
    }

    /// A group was dropped (its lease stopped, or a fatal long run). all9:
    /// with `log_drops`, one line on what it had.
    fn dropped(&self, group: &GroupFacts, ran: Option<Duration>, last_probe: Option<i64>) {
        self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        if self.config.log_drops {
            tracing::info!(
                first = group.first_best,
                mult = group.mult,
                round_s = group.round_s,
                held_s = group.held_at.elapsed().as_secs_f64(),
                ran_s = ran.map(|d| d.as_secs_f64()),
                last_probe,
                live_target = self.live_target(),
                "otsuka deep dropped"
            );
        }
    }
}

struct GroupFacts {
    first_best: i64,
    round_s: f64,
    held_at: Instant,
    /// all9: the long run's job-sweep multiple (0: not given).
    mult: usize,
}

/// One held salt: the first run's reads, waiting for its long run.
pub struct DeepGroup<J, F, D> {
    facts: GroupFacts,
    /// The instance offset (reported = lane energy + offset).
    offset: i64,
    /// The long run launched (a queued one is released without a launch).
    started: AtomicBool,
    /// Set once delivered or dropped: the long run stops at its next chunk.
    done: AtomicBool,
    /// The watcher asked the running long run to stop and deliver.
    harvest: AtomicBool,
    harvest_since: Mutex<Option<Instant>>,
    /// The long run stopped early because its probed best crossed.
    crossed: AtomicBool,
    /// The lease's stop flag (round change, shutdown).
    stop: Arc<AtomicBool>,
    payload: Mutex<Option<Payload<J, F, D>>>,
    /// all9: when the long run launched.
    run_started: Mutex<Option<Instant>>,
    /// all9: the last probed lineage best (reported energy); `i64::MAX` none.
    last_probe: AtomicI64,
}

impl<J, F, D> DeepGroup<J, F, D> {
    /// A group for a first run of best `first_best` (reported energy) over
    /// an instance of offset `offset`, held `round_elapsed` into the round.
    #[must_use]
    pub fn new(
        first_best: i64,
        offset: i64,
        round_elapsed: Duration,
        stop: Arc<AtomicBool>,
        payload: Payload<J, F, D>,
    ) -> Self {
        Self {
            facts: GroupFacts {
                first_best,
                round_s: round_elapsed.as_secs_f64(),
                held_at: Instant::now(),
                mult: 0,
            },
            offset,
            started: AtomicBool::new(false),
            done: AtomicBool::new(false),
            harvest: AtomicBool::new(false),
            harvest_since: Mutex::new(None),
            crossed: AtomicBool::new(false),
            stop,
            payload: Mutex::new(Some(payload)),
            run_started: Mutex::new(None),
            last_probe: AtomicI64::new(i64::MAX),
        }
    }

    /// all9: record the long run's job-sweep multiple (for the logs and the
    /// run-time estimate).
    #[must_use]
    pub const fn with_mult(mut self, mult: usize) -> Self {
        self.facts.mult = mult;
        self
    }

    /// all9: the first run's best (reported energy), the queue's order.
    #[must_use]
    pub const fn first_best(&self) -> i64 {
        self.facts.first_best
    }

    fn ran(&self) -> Option<Duration> {
        self.run_started
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map(|t| t.elapsed())
    }

    fn last_probe(&self) -> Option<i64> {
        Some(self.last_probe.load(Ordering::Relaxed)).filter(|&p| p != i64::MAX)
    }

    /// The instance offset.
    #[must_use]
    pub const fn offset(&self) -> i64 {
        self.offset
    }

    /// Whether the long run should stop (or never launch).
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.done.load(Ordering::Relaxed)
    }

    /// Whether the group still holds its salt (counts toward the cap).
    pub fn pending(&self) -> bool {
        self.payload
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// Mark the long run as launched. False when the group is already
    /// delivered or its lease stopped: the lane then skips it.
    pub fn start(&self) -> bool {
        let slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() || self.stopped() {
            return false;
        }
        self.started.store(true, Ordering::Relaxed);
        *self
            .run_started
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        true
    }

    /// The running long run's probe: `device_best` is its lineage best
    /// (lane energy). True stops the run there and delivers it: the watcher
    /// asked for a harvest, or the best reached the release margin.
    pub fn probe(&self, deep: &Deep, device_best: i64) -> bool {
        if self.harvest.load(Ordering::Relaxed) {
            return true;
        }
        self.last_probe
            .fetch_min(device_best.saturating_add(self.offset), Ordering::Relaxed);
        if deep.releasable(device_best.saturating_add(self.offset)) {
            self.crossed.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn reported_best(&self, payload: &Payload<J, F, D>) -> Option<i64> {
        best_of(&payload.result).map(|b| b.saturating_add(self.offset))
    }

    /// The long run ended (completed, harvested, cancelled or failed): the
    /// payload to deliver, its reads merged, or with a fatal error as its
    /// outcome. `None` when already delivered or the lease stopped.
    pub fn complete(&self, deep: &Deep, run: StreamResult) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let payload = slot.as_mut()?;
        self.done.store(true, Ordering::Relaxed);
        if self.stop.load(Ordering::Relaxed) {
            deep.dropped(&self.facts, self.ran(), self.last_probe());
            drop(slot.take());
            return None;
        }
        payload.result.device_access_time_us = payload
            .result
            .device_access_time_us
            .saturating_add(run.device_access_time_us);
        let mut long = None;
        match run.outcome {
            StreamOutcome::Completed(Ok(reads)) => {
                long = reads
                    .iter()
                    .map(|r| r.energy_milli.saturating_add(self.offset))
                    .min();
                if let StreamOutcome::Completed(Ok(merged)) = &mut payload.result.outcome {
                    merge_better(merged, reads);
                }
            }
            StreamOutcome::Completed(Err(error)) if error.is_fatal() => {
                payload.result.outcome = StreamOutcome::Completed(Err(error));
                deep.dropped(&self.facts, self.ran(), self.last_probe());
                return slot.take();
            }
            StreamOutcome::Completed(Err(error)) => {
                tracing::debug!(%error, "otsuka deep long run failed; first run delivered");
            }
            _ => {}
        }
        let merged = self.reported_best(payload).unwrap_or(i64::MAX);
        let end = if self.crossed.load(Ordering::Relaxed) {
            End::Crossed
        } else if self.harvest.load(Ordering::Relaxed) {
            End::Harvested
        } else {
            End::Last
        };
        // all9: a long run that ran to its end times the job length.
        if let (End::Last, Some(_), Some(ran)) = (end, long, self.ran()) {
            deep.note_run(ran, self.facts.mult);
        }
        deep.finished(&self.facts, long, merged, end);
        slot.take()
    }

    /// The watcher's check at `now`. When the live target has eased to
    /// within the release margin of the held best: a group whose long run
    /// has not started is delivered now (the run is skipped); a running one
    /// is asked to harvest, and delivered here with the first run's reads
    /// only if no harvest came within [`HARVEST_GRACE`]. A group whose lease
    /// stopped is dropped here, freeing its place under the cap.
    pub fn release_due(&self, deep: &Deep, now: Instant) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let payload = slot.as_ref()?;
        if self.stop.load(Ordering::Relaxed) {
            self.done.store(true, Ordering::Relaxed);
            deep.dropped(&self.facts, self.ran(), self.last_probe());
            drop(slot.take());
            return None;
        }
        let best = self.reported_best(payload)?;
        if !deep.releasable(best) {
            return None;
        }
        let end = if self.started.load(Ordering::Relaxed) {
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
        deep.finished(&self.facts, None, best, end);
        slot.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::{SampleError, SamplerResult};

    fn lookup(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    fn done(energies: &[i64]) -> StreamResult {
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

    fn config() -> DeepConfig {
        DeepConfig {
            lanes: 1,
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
        }
    }

    type G = DeepGroup<(), (), ()>;

    fn group(first: &[i64]) -> (G, Arc<AtomicBool>) {
        let stop = Arc::new(AtomicBool::new(false));
        let best = first.iter().copied().min().unwrap();
        let g = G::new(
            best,
            0,
            Duration::ZERO,
            Arc::clone(&stop),
            Payload {
                job: (),
                family: (),
                dest: (),
                result: done(first),
            },
        );
        (g, stop)
    }

    fn energies(p: Payload<(), (), ()>) -> Vec<i64> {
        let mut e: Vec<i64> = match p.result.outcome {
            StreamOutcome::Completed(Ok(r)) => r.iter().map(|r| r.energy_milli).collect(),
            _ => panic!("completed"),
        };
        e.sort_unstable();
        e
    }

    fn warm(d: &Deep, now: Instant) {
        warm_at(d, now, 0);
    }

    fn warm_at(d: &Deep, now: Instant, value: i64) {
        for _ in 0..Deep::WARMUP {
            assert!(
                !d.should_deep((1, 1), value, now, 0),
                "warm-up never triggers"
            );
        }
    }

    #[test]
    fn off_unless_lanes_is_valid_and_positive_and_defaults_apply() {
        assert_eq!(DeepConfig::from_lookup(lookup(&[])), None);
        assert_eq!(
            DeepConfig::from_lookup(lookup(&[("QUIP_OTSUKA_DEEP_MULT", "8")])),
            None,
            "MULT alone does not turn it on"
        );
        for bad in ["0", "9", "-1", "1.5", "x"] {
            assert_eq!(
                DeepConfig::from_lookup(lookup(&[("QUIP_OTSUKA_DEEP_LANES", bad)])),
                None,
                "{bad}"
            );
        }
        assert_eq!(
            DeepConfig::from_lookup(lookup(&[("QUIP_OTSUKA_DEEP_LANES", "1")])),
            Some(DeepConfig {
                lanes: 1,
                mult: 4,
                start: Duration::from_mins(1),
                q: 0.10,
                record: true,
                guard_milli: 1250,
                release_milli: 500,
                after_stage2: false,
                slack_milli: 0,
                log_drops: false,
                adaptive: None,
            })
        );
        let set = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "2"),
            ("QUIP_OTSUKA_DEEP_MULT", "8"),
            ("QUIP_OTSUKA_DEEP_START_S", "90"),
            ("QUIP_OTSUKA_DEEP_Q", "0.05"),
            ("QUIP_OTSUKA_DEEP_RECORD", "0"),
            ("QUIP_OTSUKA_DEEP_GUARD_MILLI", "2000"),
            ("QUIP_OTSUKA_DEEP_RELEASE_MILLI", "0"),
            ("QUIP_OTSUKA_DEEP_AFTER_STAGE2", "1"),
        ]))
        .unwrap();
        assert_eq!(
            set,
            DeepConfig {
                lanes: 2,
                mult: 8,
                start: Duration::from_secs(90),
                q: 0.05,
                record: false,
                guard_milli: 2000,
                release_milli: 0,
                after_stage2: true,
                slack_milli: 0,
                log_drops: false,
                adaptive: None,
            }
        );
        let bad = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "1"),
            ("QUIP_OTSUKA_DEEP_MULT", "1"),
            ("QUIP_OTSUKA_DEEP_Q", "1"),
            ("QUIP_OTSUKA_DEEP_RECORD", "2"),
        ]))
        .unwrap();
        assert_eq!(
            bad,
            DeepConfig {
                lanes: 1,
                ..DeepConfig::from_lookup(lookup(&[("QUIP_OTSUKA_DEEP_LANES", "1")])).unwrap()
            },
            "out-of-range knobs keep their defaults"
        );
    }

    #[test]
    fn deep_seed_differs_from_the_job_and_the_stage2_reruns() {
        let s = deep_seed(42);
        assert_ne!(s, 42);
        assert_ne!(s, 0);
        for j in 1..=8 {
            assert_ne!(s, super::super::stage2::rerun_seed(42, j));
        }
    }

    #[test]
    fn trigger_needs_warmup_quantile_record_start_cap_and_guard() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                start: Duration::from_mins(1),
                ..config()
            },
            t0,
        );
        warm(&d, t0);
        assert!(!d.should_deep((1, 1), -10, t0, 0), "before START");
        let late = t0 + Duration::from_secs(61);
        assert!(d.should_deep((1, 1), -20, late, 0), "a new low after START");
        assert!(!d.should_deep((1, 1), -15, late, 0), "not a round record");
        assert!(
            !d.should_deep((1, 1), -30, late, 1),
            "the cap (1 lane) is full"
        );
        assert!(d.should_deep((1, 1), -30, late, 0), "ties the record (-30)");
        assert!(!d.should_deep((1, 1), 50, late, 0), "above the quantile");
        // A new round clears the record, and starts the START clock again.
        d.new_round(late);
        assert!(
            !d.should_deep((1, 1), -25, late + Duration::from_secs(1), 0),
            "before START"
        );
        assert!(
            d.should_deep((1, 1), -26, late + Duration::from_secs(61), 0),
            "the old round's record no longer counts"
        );
        // A new shape restarts the warm-up.
        assert!(!d.should_deep((2, 2), -5_000, late + Duration::from_secs(61), 0));
    }

    #[test]
    fn guard_never_holds_a_first_near_the_target() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                q: 0.99,
                record: false,
                ..config()
            },
            t0,
        );
        warm_at(&d, t0, 10_000);
        assert!(d.should_deep((1, 1), -100, t0, 0), "no target yet: hold");
        d.set_live_target(Some(-1_000));
        assert!(!d.should_deep((1, 1), -2_000, t0, 0), "below the target");
        assert!(
            !d.should_deep((1, 1), -1_000 + 1_250, t0, 0),
            "at target + guard"
        );
        assert!(
            d.should_deep((1, 1), -1_000 + 1_251, t0, 0),
            "just outside the guard"
        );
    }

    #[test]
    fn observed_firsts_set_the_record_and_fill_the_window() {
        let t0 = Instant::now();
        let d = Deep::new(config(), t0);
        for _ in 0..=Deep::WARMUP {
            d.observe((1, 1), 0, t0);
        }
        d.observe((1, 1), -50, t0);
        assert!(
            !d.should_deep((1, 1), -40, t0, 0),
            "the observed -50 is the record"
        );
        assert!(
            d.should_deep((1, 1), -60, t0, 0),
            "a new record after observed warm-up"
        );
    }

    #[test]
    fn record_off_takes_any_first_in_the_quantile() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                record: false,
                ..config()
            },
            t0,
        );
        warm(&d, t0);
        assert!(d.should_deep((1, 1), -20, t0, 0));
        assert!(
            d.should_deep((1, 1), -15, t0, 0),
            "not a record, still in the quantile"
        );
    }

    #[test]
    fn complete_merges_the_long_run_and_delivers_once() {
        let d = Deep::new(config(), Instant::now());
        let (g, _stop) = group(&[-10, -5, -1]);
        assert!(g.start());
        let p = g.complete(&d, done(&[-12, -2, 4])).expect("delivered");
        assert_eq!(
            energies(p),
            vec![-12, -10, -5],
            "read count kept, best merged"
        );
        assert!(!g.pending() && g.stopped());
        assert!(g.complete(&d, done(&[-99])).is_none(), "never twice");
        assert!(!g.start(), "a delivered group never launches again");
    }

    #[test]
    fn probe_crosses_at_the_release_margin_with_the_offset() {
        let d = Deep::new(config(), Instant::now());
        let stop = Arc::new(AtomicBool::new(false));
        let g: G = DeepGroup::new(
            2_000,
            1_000,
            Duration::ZERO,
            stop,
            Payload {
                job: (),
                family: (),
                dest: (),
                result: done(&[1_000]),
            },
        );
        assert!(!g.probe(&d, -1_000_000), "no target yet: run on");
        d.set_live_target(Some(500));
        // Lane 0 + offset 1000 = 1000 reported, target + release = 1000.
        assert!(g.probe(&d, 0));
        assert!(!g.probe(&Deep::new(config(), Instant::now()), 0));
        assert!(g.start());
        let p = g.complete(&d, done(&[0])).unwrap();
        assert_eq!(energies(p), vec![0]);
    }

    #[test]
    fn watcher_releases_a_queued_group_and_harvests_a_running_one() {
        let d = Deep::new(config(), Instant::now());
        d.set_live_target(Some(-5_000));
        let now = Instant::now();
        // Queued (not started): released at once, the run is then skipped.
        let (g, _stop) = group(&[-4_000]);
        assert!(g.release_due(&d, now).is_none(), "far from the target");
        d.set_live_target(Some(-4_500));
        assert!(g.release_due(&d, now).is_some(), "eased to within 500");
        assert!(!g.start(), "the queued long run never launches");
        assert!(g.complete(&d, done(&[-9_000])).is_none());

        // Running: a harvest is requested, the run's probe stops it, and
        // the run delivers its merged reads.
        let (g, _stop) = group(&[-4_000]);
        assert!(g.start());
        assert!(
            g.release_due(&d, now).is_none(),
            "harvest asked, not forced"
        );
        assert!(g.probe(&d, 0), "the probe sees the request");
        let p = g.complete(&d, done(&[-4_200])).expect("harvested");
        assert_eq!(energies(p), vec![-4_200]);
        assert!(g.release_due(&d, now).is_none(), "once");

        // Running, but no harvest within the grace: the first reads go out
        // and the long run is cancelled.
        let (g, _stop) = group(&[-4_000]);
        assert!(g.start());
        assert!(g.release_due(&d, now).is_none());
        assert!(g.release_due(&d, now + HARVEST_GRACE / 2).is_none());
        let p = g.release_due(&d, now + HARVEST_GRACE).expect("forced");
        assert_eq!(energies(p), vec![-4_000]);
        assert!(g.stopped(), "the long run stops at its next chunk");
        assert!(g.complete(&d, done(&[-9_000])).is_none(), "never twice");
    }

    #[test]
    fn a_stopped_lease_drops_the_group() {
        let d = Deep::new(config(), Instant::now());
        let (g, stop) = group(&[-10]);
        assert!(g.start());
        stop.store(true, Ordering::Relaxed);
        assert!(g.stopped());
        let cancelled = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Cancelled,
            device_access_time_us: 0,
        };
        assert!(g.complete(&d, cancelled).is_none());
        assert!(!g.pending());

        let (g, stop) = group(&[-10]);
        stop.store(true, Ordering::Relaxed);
        assert!(!g.start(), "a stopped lease's run never launches");
        assert!(g.release_due(&d, Instant::now()).is_none());
        assert!(!g.pending(), "dropped by the watcher, off the cap");
    }

    #[test]
    fn a_failed_long_run_delivers_the_first_reads_and_a_fatal_one_its_error() {
        let d = Deep::new(config(), Instant::now());
        let (g, _stop) = group(&[-10]);
        assert!(g.start());
        let busy = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Err(SampleError::DeviceBusy)),
            device_access_time_us: 0,
        };
        assert_eq!(energies(g.complete(&d, busy).unwrap()), vec![-10]);

        let (g, _stop) = group(&[-10]);
        assert!(g.start());
        let fault = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Err(SampleError::DeviceFault("x".into()))),
            device_access_time_us: 0,
        };
        let p = g.complete(&d, fault).unwrap();
        assert!(matches!(
            p.result.outcome,
            StreamOutcome::Completed(Err(ref e)) if e.is_fatal()
        ));
    }

    // ---- all9 (v2) ----

    fn adaptive() -> Adaptive {
        Adaptive {
            ramp0: Duration::from_mins(2),
            ramp1: Duration::from_mins(3),
            late: Duration::from_mins(9),
            horizon: Duration::from_mins(10),
            mult_min: 3,
            mult_max: 4,
            gap_far_milli: 2000,
            step: Duration::from_secs(15),
        }
    }

    #[test]
    fn v2_knobs_are_off_unless_set_and_parse_in_range() {
        let v1 = DeepConfig::from_lookup(lookup(&[("QUIP_OTSUKA_DEEP_LANES", "1")])).unwrap();
        assert_eq!(
            (v1.slack_milli, v1.log_drops, v1.adaptive),
            (0, false, None)
        );
        let on = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "1"),
            ("QUIP_OTSUKA_DEEP_ADAPTIVE", "1"),
            ("QUIP_OTSUKA_DEEP_LMAX", "3"),
        ]))
        .unwrap();
        assert_eq!(on.lanes, 3, "LMAX replaces LANES");
        assert!(on.log_drops, "adaptive logs drops");
        assert_eq!(
            on.adaptive,
            Some(Adaptive {
                mult_max: 4,
                horizon: Adaptive::HORIZON,
                ..adaptive()
            })
        );
        assert_eq!(
            DeepConfig::from_lookup(lookup(&[
                ("QUIP_OTSUKA_DEEP_ADAPTIVE", "1"),
                ("QUIP_OTSUKA_DEEP_LMAX", "2"),
            ])),
            None,
            "ADAPTIVE never turns the lane on by itself"
        );
        let lmax_alone = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "1"),
            ("QUIP_OTSUKA_DEEP_LMAX", "3"),
        ]))
        .unwrap();
        assert_eq!(lmax_alone, v1, "LMAX without ADAPTIVE is all8");
        let set = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "2"),
            ("QUIP_OTSUKA_DEEP_MULT", "5"),
            ("QUIP_OTSUKA_DEEP_ADAPTIVE", "1"),
            ("QUIP_OTSUKA_DEEP_RAMP0_S", "90"),
            ("QUIP_OTSUKA_DEEP_RAMP1_S", "30"),
            ("QUIP_OTSUKA_DEEP_LATE_S", "500"),
            ("QUIP_OTSUKA_DEEP_HORIZON_S", "620"),
            ("QUIP_OTSUKA_DEEP_MULT_MIN", "6"),
            ("QUIP_OTSUKA_DEEP_GAP_FAR_MILLI", "0"),
            ("QUIP_OTSUKA_DEEP_STEP_S", "5"),
            ("QUIP_OTSUKA_DEEP_SLACK_MILLI", "700"),
        ]))
        .unwrap();
        assert_eq!(set.lanes, 2);
        assert_eq!(set.slack_milli, 700);
        assert_eq!(
            set.adaptive,
            Some(Adaptive {
                ramp0: Duration::from_secs(90),
                ramp1: Duration::from_secs(90),
                late: Duration::from_secs(500),
                horizon: Duration::from_secs(620),
                mult_min: 5,
                mult_max: 5,
                gap_far_milli: 0,
                step: Duration::from_secs(5),
            }),
            "RAMP1 at least RAMP0, MULT_MIN at most MULT_MAX (MULT)"
        );
        let drops = DeepConfig::from_lookup(lookup(&[
            ("QUIP_OTSUKA_DEEP_LANES", "1"),
            ("QUIP_OTSUKA_DEEP_LOG_DROPS", "1"),
            ("QUIP_OTSUKA_DEEP_SLACK_MILLI", "-5"),
            ("QUIP_OTSUKA_DEEP_ADAPTIVE", "2"),
        ]))
        .unwrap();
        assert_eq!(
            (drops.log_drops, drops.slack_milli, drops.adaptive),
            (true, 0, None),
            "bad values keep the all8 behaviour"
        );
    }

    #[test]
    fn want_ramps_stops_late_and_keeps_a_lane_back_when_the_target_is_near() {
        let a = adaptive();
        let s = Duration::from_secs;
        assert_eq!(a.want(2, s(0), None), (0, "early"));
        assert_eq!(a.want(2, s(119), None), (0, "early"));
        assert_eq!(a.want(2, s(120), None), (1, "ramp"));
        assert_eq!(a.want(2, s(179), None), (1, "ramp"));
        assert_eq!(a.want(2, s(180), None), (2, "full"));
        assert_eq!(a.want(2, s(540), Some(5000)), (2, "full"));
        assert_eq!(a.want(2, s(541), None), (0, "late"));
        assert_eq!(a.want(2, s(300), Some(1999)), (1, "near"));
        assert_eq!(a.want(2, s(300), Some(2000)), (2, "full"));
        assert_eq!(a.want(1, s(300), Some(0)), (1, "full"), "one lane stays");
        assert_eq!(a.want(3, s(150), None), (2, "ramp"));
        assert_eq!(a.want(3, s(200), Some(-500)), (2, "near"));
        assert_eq!(a.want(0, s(300), None).0, 0);
        let never = Adaptive {
            gap_far_milli: 0,
            ..a
        };
        assert_eq!(never.want(2, s(300), Some(-9_000)), (2, "full"));
    }

    #[test]
    fn mult_is_the_longest_that_ends_by_the_horizon() {
        let a = adaptive();
        let s = Duration::from_secs;
        assert_eq!(a.mult(s(120), s(33)), 4, "120 + 132 <= 600");
        assert_eq!(a.mult(s(468), s(33)), 4);
        assert_eq!(a.mult(s(469), s(33)), 3);
        assert_eq!(a.mult(s(501), s(33)), 3);
        assert_eq!(a.mult(s(502), s(33)), 3, "nothing fits: the shortest");
        let wide = Adaptive { mult_max: 6, ..a };
        assert_eq!(wide.mult(s(120), s(33)), 6);
        assert_eq!(wide.mult(s(300), s(33)), 6, "300 + 198 <= 600");
        assert_eq!(wide.mult(s(420), s(33)), 5);
        assert_eq!(a.mult(s(0), Duration::MAX), 3, "no overflow");
    }

    #[test]
    fn take_best_takes_the_lowest_key_then_the_oldest() {
        let mut q: VecDeque<(i64, u8)> = [(5, 0), (-3, 1), (-3, 2), (9, 3)].into();
        assert_eq!(take_best(&mut q, |x| x.0), Some((-3, 1)));
        assert_eq!(take_best(&mut q, |x| x.0), Some((-3, 2)));
        assert_eq!(take_best(&mut q, |x| x.0), Some((5, 0)));
        assert_eq!(take_best(&mut q, |x| x.0), Some((9, 3)));
        assert_eq!(take_best(&mut q, |x| x.0), None);
    }

    #[test]
    fn without_adaptive_the_cap_is_lanes_and_the_length_mult() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                lanes: 2,
                ..config()
            },
            t0,
        );
        for secs in [0, 59, 600, 3600] {
            let at = t0 + Duration::from_secs(secs);
            assert_eq!(d.allocation(at), 2);
            assert_eq!(d.mult_for(Duration::from_secs(secs)), 4);
        }
        d.note_run(Duration::from_secs(1), 1);
        assert_eq!(d.run_estimate(), DEFAULT_RUN, "no timing without adaptive");
    }

    #[test]
    fn run_estimate_is_a_moving_average_per_job_length() {
        let d = Deep::new(
            DeepConfig {
                adaptive: Some(adaptive()),
                ..config()
            },
            Instant::now(),
        );
        assert_eq!(d.run_estimate(), DEFAULT_RUN);
        d.note_run(Duration::from_secs(132), 4);
        assert_eq!(d.run_estimate(), Duration::from_secs(33));
        d.note_run(Duration::from_secs(41), 1);
        assert_eq!(d.run_estimate(), Duration::from_secs(34), "1/8 weight");
        d.note_run(Duration::from_secs(9), 0);
        assert_eq!(
            d.run_estimate(),
            Duration::from_secs(34),
            "a zero multiple is ignored"
        );
        assert_eq!(
            d.mult_for(Duration::from_secs(470)),
            3,
            "470 + 4 x 34 > 600"
        );
        assert_eq!(d.mult_for(Duration::from_secs(400)), 4);
    }

    #[test]
    fn allocation_rises_one_per_step_falls_at_once_and_restarts_each_round() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                lanes: 2,
                start: Duration::ZERO,
                adaptive: Some(adaptive()),
                ..config()
            },
            t0,
        );
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert_eq!(d.allocation(at(60)), 0, "early");
        assert_eq!(d.allocation(at(120)), 1);
        assert_eq!(d.allocation(at(180)), 2, "15 s after the last rise");
        d.new_round(at(200));
        assert_eq!(d.allocation(at(201)), 0, "a new round starts at 0");
        assert_eq!(d.allocation(at(320)), 1);
        assert_eq!(d.allocation(at(381)), 2, "full");
        // The round's best first is within GAP_FAR of the target: one fewer.
        d.observe((1, 1), -10_000, at(381));
        d.set_live_target(Some(-11_000));
        assert_eq!(d.allocation(at(382)), 1, "near: falls at once");
        d.set_live_target(Some(-13_000));
        assert_eq!(
            d.allocation(at(383)),
            1,
            "far again, but a rise waits a step"
        );
        assert_eq!(d.allocation(at(397)), 2);
        assert_eq!(d.allocation(at(741)), 0, "late");
        // A new round seen late: the reset dates from the round start, so
        // the first rise does not wait a step.
        d.new_round(at(800));
        assert_eq!(d.allocation(at(920)), 1);
        assert_eq!(d.allocation(at(925)), 1);
        assert_eq!(d.allocation(at(980)), 2);
    }

    #[test]
    fn adaptive_holds_only_under_the_allocation() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                lanes: 2,
                q: 0.99,
                record: false,
                start: Duration::ZERO,
                adaptive: Some(adaptive()),
                ..config()
            },
            t0,
        );
        warm_at(&d, t0, 10_000);
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert!(!d.should_deep((1, 1), -1, at(100), 0), "before RAMP0");
        assert!(d.should_deep((1, 1), -1, at(130), 0), "one group");
        assert!(
            !d.should_deep((1, 1), -1, at(131), 1),
            "ramp: one at a time"
        );
        assert!(d.should_deep((1, 1), -1, at(190), 1), "two after RAMP1");
        assert!(!d.should_deep((1, 1), -1, at(191), 2), "LMAX");
        assert!(!d.should_deep((1, 1), -1, at(560), 0), "after LATE");
    }

    #[test]
    fn slack_widens_the_hold_but_not_the_record() {
        let t0 = Instant::now();
        let d = Deep::new(
            DeepConfig {
                lanes: 2,
                q: 0.99,
                slack_milli: 100,
                ..config()
            },
            t0,
        );
        warm_at(&d, t0, 10_000);
        assert!(d.should_deep((1, 1), -1_000, t0, 0), "a record");
        assert!(d.should_deep((1, 1), -950, t0, 0), "within the slack");
        assert!(!d.should_deep((1, 1), -899, t0, 0), "beyond the slack");
        assert!(
            !d.should_deep((1, 1), -890, t0, 0),
            "the record stayed -1000 (-950 + 100 would take -890)"
        );
        assert!(d.should_deep((1, 1), -900, t0, 0), "at record + slack");
    }

    #[test]
    fn a_group_keeps_its_lowest_probe_and_run_time_for_the_drop_line() {
        let d = Deep::new(
            DeepConfig {
                log_drops: true,
                ..config()
            },
            Instant::now(),
        );
        let (g, stop) = group(&[-10]);
        let g = g.with_mult(3);
        assert_eq!(g.facts.mult, 3);
        assert_eq!(g.first_best(), -10);
        assert_eq!((g.ran(), g.last_probe()), (None, None));
        assert!(g.start());
        assert!(!g.probe(&d, 7));
        assert!(!g.probe(&d, 9));
        assert_eq!(g.last_probe(), Some(7), "lowest probe kept");
        assert!(g.ran().is_some());
        stop.store(true, Ordering::Relaxed);
        let cancelled = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Cancelled,
            device_access_time_us: 0,
        };
        assert!(g.complete(&d, cancelled).is_none());
        assert_eq!(d.stats.dropped.load(Ordering::Relaxed), 1);
    }
}
