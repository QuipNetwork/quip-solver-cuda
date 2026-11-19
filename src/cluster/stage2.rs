//! Local build (all5, from agent Z's sketch): stage 2, re-anneal an unusually
//! deep instance with fresh seeds before reporting it.
//!
//! Off by default: with `QUIP_OTSUKA_STAGE2_Q` unset the sampler never builds
//! a [`Stage2`] and every lane delivers exactly as before. It needs the local
//! lease path (`QUIP_OTSUKA_SCREEN_THREADS` above 1): only there does the
//! sampler own a salt's delivery.
//!
//! Why: on dimer1200 the full-run best of one instance is roughly
//! `mu_instance + eps_run`. Replicate anneals of the same 170 instances
//! (agent D, old topology) correlate r = 0.37-0.51 at 8k-32k sweeps and 0.63
//! at 65k; the live G1 -> C1-64 shift implies, among stage-1 keepers, an
//! instance share of ~0.39 of the full-run variance. A first full run in the
//! top decile therefore predicts a deeper-than-average instance, and one
//! more full anneal of it (cost 1 full, no short run) reaches a given deep
//! level more often per GPU second than a fresh stage-1 keeper (1.83 fulls).
//! Agent Z's model: Q 0.10, K 2 costs ~+10.7% GPU for x1.35-1.77 at -5085.
//!
//! How: a lane that finishes the FIRST full anneal of a local lease salt
//! (stage 1 promoted it, or stage 1 is off) offers its best energy to the
//! trigger. A triggered salt is not delivered. A [`Group`] keeps its reads,
//! job, family and destination, and `K` rerun items (the same instance, the
//! job's parameters, fresh seeds) go to the lanes' priority queue, ahead of
//! fresh screened salts. Each rerun merges its reads into the group
//! ([`super::stage1::merge_better`], read count unchanged). The group is
//! delivered (polish, sort, push) once:
//! - when its last rerun ends;
//! - at once when a rerun leaves the merged best within `RELEASE` of the live
//!   session target;
//! - by the sampler's 50 ms watcher, when the target eases to within
//!   `RELEASE` of the held best while reruns still run (the coordinator then
//!   submits it, or stashes it until the target eases further);
//! - never, when the lease stopped (round change, shutdown): dropped.
//!
//! Why delivery waits: core's `LeaseSink` accepts one push per salt index, so
//! a salt cannot be reported twice. To never delay a result that can already
//! win, a first run whose best is at or below `live target + GUARD` is
//! delivered at once and not deepened (in the ease phase the target rises
//! ~0.124 unit per ~6.2 s block, so GUARD 1250 milli ~ 10 blocks ~ 60 s,
//! the wall time of one rerun round at 6 lanes).
//!
//! Energies here are reported energies: lane energy plus the instance offset
//! (FAM-33), the units of the session target. The offset is 0 for
//! twisted-ice-dimer, so this matters only for correctness.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_STAGE2_Q` | unset (off) | Trigger: first-run best at or below this quantile of the last 256 first-run bests, `0 < Q < 1` (0.10 = top decile). No trigger before 32 firsts (per topology shape). |
//! | `QUIP_OTSUKA_STAGE2_K` | 2 | Extra full anneals per triggered salt, `1..=8`. |
//! | `QUIP_OTSUKA_STAGE2_GUARD_MILLI` | 1250 | Never hold a first best at or below the live target plus this, `0..=20000`. |
//! | `QUIP_OTSUKA_STAGE2_MAX_ACTIVE` | 3 | Groups held at once (caps the GPU share), `1..=64`. |
//! | `QUIP_OTSUKA_STAGE2_RELEASE_MILLI` | 500 | Deliver a held group early once its best is at or below the live target plus this (pre-polish; the polish adds ~0.55 unit), `0..=20000`. |
//!
//! Calibration comes free: every completed rerun logs `first`, `rerun` and
//! `merged` at info (`otsuka stage2 pair`). The regression slope of rerun on
//! first is unbiased under selection on `first`, so one evening of stage 2
//! measures the live instance share. A summary line (`otsuka stage2
//! summary`) follows every 32 finished groups.

use super::stage1::{merge_better, RollingCutoff};
use quip_solver_core::{SamplerResult, StreamOutcome, StreamResult};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// `XORed` into the job seed (with the rerun index) for a rerun's draws.
pub const STAGE2_SEED_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// The stage-2 knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage2Config {
    /// Trigger quantile of the first-run bests.
    pub q: f64,
    /// Reruns per triggered salt.
    pub k: usize,
    /// Never hold a first best at or below the live target plus this.
    pub guard_milli: i64,
    /// Groups held at once.
    pub max_active: usize,
    /// Deliver a held group once its best is within this of the target.
    pub release_milli: i64,
}

impl Stage2Config {
    /// Default reruns per triggered salt.
    pub const K: usize = 2;
    /// Default guard.
    pub const GUARD_MILLI: i64 = 1250;
    /// Default cap on held groups.
    pub const MAX_ACTIVE: usize = 3;
    /// Default early-release margin.
    pub const RELEASE_MILLI: i64 = 500;

    /// `None` (off) unless `QUIP_OTSUKA_STAGE2_Q` is valid. Every variable
    /// that is set logs `applied` or `ignored`, like the other knobs.
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
        // Parse every knob first, so a set knob is always logged.
        let q = num("QUIP_OTSUKA_STAGE2_Q", &|v| v > 0.0 && v < 1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = num("QUIP_OTSUKA_STAGE2_K", &|v| {
            (1.0..=8.0).contains(&v) && int(v)
        })
        .map_or(Self::K, |v| v as usize);
        #[allow(clippy::cast_possible_truncation)]
        let guard_milli = num("QUIP_OTSUKA_STAGE2_GUARD_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(Self::GUARD_MILLI, |v| v as i64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_active = num("QUIP_OTSUKA_STAGE2_MAX_ACTIVE", &|v| {
            (1.0..=64.0).contains(&v) && int(v)
        })
        .map_or(Self::MAX_ACTIVE, |v| v as usize);
        #[allow(clippy::cast_possible_truncation)]
        let release_milli = num("QUIP_OTSUKA_STAGE2_RELEASE_MILLI", &|v| {
            (0.0..=20_000.0).contains(&v) && int(v)
        })
        .map_or(Self::RELEASE_MILLI, |v| v as i64);
        Some(Self {
            q: q?,
            k,
            guard_milli,
            max_active,
            release_milli,
        })
    }

    /// [`Self::from_lookup`] on the process environment, read once.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        static ENV: OnceLock<Option<Stage2Config>> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }
}

/// The seed of rerun `j` (1-based) of a job seeded `seed` (splitmix64).
#[must_use]
pub fn rerun_seed(seed: u64, j: u64) -> u64 {
    let mut z = (seed ^ STAGE2_SEED_SALT).wrapping_add(j.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

/// The lowest energy of a completed result, `None` otherwise or without reads.
#[must_use]
pub fn best_of(result: &StreamResult) -> Option<i64> {
    match &result.outcome {
        StreamOutcome::Completed(Ok(reads)) => {
            reads.iter().map(|r: &SamplerResult| r.energy_milli).min()
        }
        _ => None,
    }
}

struct Window {
    firsts: RollingCutoff,
    seen: usize,
    shape: Option<(usize, usize)>,
}

#[derive(Default)]
struct Stats {
    groups: AtomicU64,
    early: AtomicU64,
    released: AtomicU64,
    improved: AtomicU64,
    gain_milli: AtomicU64,
    dropped: AtomicU64,
}

/// How a group ended, for the counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum End {
    /// Every rerun ran.
    Last,
    /// A rerun brought the best within the release margin.
    Early,
    /// The watcher released it as the target eased.
    Released,
}

/// Stage-2 state shared by every lane.
pub struct Stage2 {
    config: Stage2Config,
    window: Mutex<Window>,
    /// The live session target (`LeaseSink::target_energy_milli`), refreshed
    /// by every lease thread each stop poll; valid once `has_target`.
    live_target: AtomicI64,
    has_target: AtomicBool,
    stats: Stats,
}

impl Stage2 {
    /// First-run bests the trigger looks back over.
    pub const HISTORY: usize = 256;
    /// Trigger cutoff refresh period.
    pub const REFRESH: usize = 16;
    /// No trigger before this many first-run bests of the current shape.
    pub const WARMUP: usize = 32;
    const LOG_EVERY: u64 = 32;

    /// Fresh state for `config`.
    #[must_use]
    pub fn new(config: Stage2Config) -> Self {
        Self {
            config,
            window: Mutex::new(Window {
                firsts: RollingCutoff::new(config.q, Self::HISTORY, Self::REFRESH),
                seen: 0,
                shape: None,
            }),
            live_target: AtomicI64::new(0),
            has_target: AtomicBool::new(false),
            stats: Stats::default(),
        }
    }

    /// The knobs.
    #[must_use]
    pub const fn config(&self) -> &Stage2Config {
        &self.config
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

    /// Whether a held best (reported energy) should be delivered now.
    #[must_use]
    pub fn releasable(&self, best: i64) -> bool {
        self.live_target()
            .is_some_and(|t| best <= t.saturating_add(self.config.release_milli))
    }

    /// Local build (all7): record a first full run's best (reported energy)
    /// that the deep lane holds: the window sees it, no decision.
    pub fn observe(&self, shape: (usize, usize), best: i64) {
        let mut w = self.window.lock().unwrap_or_else(PoisonError::into_inner);
        if w.shape != Some(shape) {
            w.firsts = RollingCutoff::new(self.config.q, Self::HISTORY, Self::REFRESH);
            w.seen = 0;
            w.shape = Some(shape);
        }
        w.seen += 1;
        w.firsts.push(best);
    }

    /// Record a first full run's best (reported energy) and decide whether
    /// to deepen it, before the cap on held groups. Every first-run best
    /// feeds the window, triggered or not. A new shape (topology change)
    /// empties the window first, as stage 1 does.
    pub fn should_deepen(&self, shape: (usize, usize), best: i64) -> bool {
        let admitted = {
            let mut w = self.window.lock().unwrap_or_else(PoisonError::into_inner);
            if w.shape != Some(shape) {
                w.firsts = RollingCutoff::new(self.config.q, Self::HISTORY, Self::REFRESH);
                w.seen = 0;
                w.shape = Some(shape);
            }
            w.seen += 1;
            let admitted = w.firsts.admit(best);
            admitted && w.seen > Self::WARMUP
        };
        if !admitted {
            return false;
        }
        // Can win now or very soon: never delay it.
        self.live_target()
            .is_none_or(|t| best > t.saturating_add(self.config.guard_milli))
    }

    fn finished(&self, first: i64, final_best: i64, end: End) {
        let st = &self.stats;
        let groups = st.groups.fetch_add(1, Ordering::Relaxed) + 1;
        st.early
            .fetch_add(u64::from(end == End::Early), Ordering::Relaxed);
        st.released
            .fetch_add(u64::from(end == End::Released), Ordering::Relaxed);
        let gain = u64::try_from(first.saturating_sub(final_best)).unwrap_or(0);
        st.improved
            .fetch_add(u64::from(gain > 0), Ordering::Relaxed);
        st.gain_milli.fetch_add(gain, Ordering::Relaxed);
        if groups.is_multiple_of(Self::LOG_EVERY) {
            tracing::info!(
                groups,
                early = st.early.load(Ordering::Relaxed),
                released = st.released.load(Ordering::Relaxed),
                dropped = st.dropped.load(Ordering::Relaxed),
                improved = st.improved.load(Ordering::Relaxed),
                mean_gain_milli = st.gain_milli.load(Ordering::Relaxed) / groups,
                cutoff = self
                    .window
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .firsts
                    .cutoff(),
                live_target = self.live_target(),
                "otsuka stage2 summary"
            );
        }
    }

    fn dropped(&self) {
        self.stats.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

/// What a group delivers once: the first run's job, family and destination
/// (generic so this module does not name the sampler's private types), and
/// the merged result (lane energies, offset not added).
pub struct Payload<J, F, D> {
    /// The first run's job.
    pub job: J,
    /// The first run's family facts.
    pub family: F,
    /// Where the salt's single result goes.
    pub dest: D,
    /// The first run's reads, merged with every rerun's so far.
    pub result: StreamResult,
}

/// One triggered salt: the first run's reads, merged with each rerun's.
pub struct Group<J, F, D> {
    /// The first run's best, reported energy.
    first_best: i64,
    /// The instance offset (reported = lane energy + offset).
    offset: i64,
    remaining: AtomicUsize,
    /// Set once delivered or dropped: later reruns are dropped by the lane
    /// without a launch, and a running one stops at its next chunk.
    done: AtomicBool,
    /// The lease's stop flag (round change, shutdown).
    stop: Arc<AtomicBool>,
    payload: Mutex<Option<Payload<J, F, D>>>,
}

impl<J, F, D> Group<J, F, D> {
    /// A group for a first run of best `first_best` (reported energy) over
    /// an instance of offset `offset`, waiting for `k` reruns.
    #[must_use]
    pub fn new(
        first_best: i64,
        offset: i64,
        k: usize,
        stop: Arc<AtomicBool>,
        payload: Payload<J, F, D>,
    ) -> Self {
        Self {
            first_best,
            offset,
            remaining: AtomicUsize::new(k.max(1)),
            done: AtomicBool::new(false),
            stop,
            payload: Mutex::new(Some(payload)),
        }
    }

    /// Whether a rerun of this group should stop (or never launch).
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

    fn reported_best(&self, payload: &Payload<J, F, D>) -> Option<i64> {
        best_of(&payload.result).map(|b| b.saturating_add(self.offset))
    }

    /// Merge one rerun's outcome. Returns the payload to deliver when this
    /// was the last rerun, or when the merged best is now releasable (early
    /// delivery), or with a fatal rerun error as its outcome (the lease
    /// thread then ends the session, as for a first run's fault). `None`
    /// otherwise, once delivered, or when the lease stopped (dropped).
    pub fn complete(
        &self,
        stage2: &Stage2,
        rerun: StreamResult,
        rerun_index: u64,
    ) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let last = self.remaining.fetch_sub(1, Ordering::AcqRel) <= 1;
        let payload = slot.as_mut()?;
        if self.stop.load(Ordering::Relaxed) {
            self.done.store(true, Ordering::Relaxed);
            stage2.dropped();
            drop(slot.take());
            return None;
        }
        payload.result.device_access_time_us = payload
            .result
            .device_access_time_us
            .saturating_add(rerun.device_access_time_us);
        match rerun.outcome {
            StreamOutcome::Completed(Ok(reads)) => {
                let rerun_best = reads
                    .iter()
                    .map(|r| r.energy_milli.saturating_add(self.offset))
                    .min();
                if let StreamOutcome::Completed(Ok(merged)) = &mut payload.result.outcome {
                    merge_better(merged, reads);
                }
                tracing::info!(
                    first = self.first_best,
                    rerun = rerun_best,
                    rerun_index,
                    merged = self.reported_best(payload),
                    "otsuka stage2 pair"
                );
            }
            StreamOutcome::Completed(Err(error)) if error.is_fatal() => {
                payload.result.outcome = StreamOutcome::Completed(Err(error));
                self.done.store(true, Ordering::Relaxed);
                stage2.dropped();
                return slot.take();
            }
            StreamOutcome::Completed(Err(error)) => {
                tracing::debug!(%error, rerun_index, "otsuka stage2 rerun failed");
            }
            // A rerun stopped by an early delivery or a lease stop.
            _ => {}
        }
        let best = self.reported_best(payload).unwrap_or(i64::MAX);
        let early = !last && stage2.releasable(best);
        if last || early {
            self.done.store(true, Ordering::Relaxed);
            let end = if last { End::Last } else { End::Early };
            stage2.finished(self.first_best, best, end);
            return slot.take();
        }
        None
    }

    /// The watcher's check: the payload to deliver when the live target has
    /// eased to within the release margin of the held best. A group whose
    /// lease stopped is dropped here, freeing its place under the cap.
    pub fn release_due(&self, stage2: &Stage2) -> Option<Payload<J, F, D>> {
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let payload = slot.as_ref()?;
        if self.stop.load(Ordering::Relaxed) {
            self.done.store(true, Ordering::Relaxed);
            stage2.dropped();
            drop(slot.take());
            return None;
        }
        let best = self.reported_best(payload)?;
        if !stage2.releasable(best) {
            return None;
        }
        self.done.store(true, Ordering::Relaxed);
        stage2.finished(self.first_best, best, End::Released);
        slot.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::SampleError;

    fn lookup(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    fn reads(energies: &[i64]) -> Vec<SamplerResult> {
        energies
            .iter()
            .map(|&e| SamplerResult {
                energy_milli: e,
                spins: vec![1],
            })
            .collect()
    }

    fn done(energies: &[i64]) -> StreamResult {
        StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Ok(reads(energies))),
            device_access_time_us: 1,
        }
    }

    fn config(q: f64) -> Stage2Config {
        Stage2Config {
            q,
            k: 2,
            guard_milli: 1250,
            max_active: 3,
            release_milli: 500,
        }
    }

    fn group(first: &[i64], k: usize) -> (Group<(), (), ()>, Arc<AtomicBool>) {
        let stop = Arc::new(AtomicBool::new(false));
        let best = first.iter().copied().min().unwrap();
        let g = Group::new(
            best,
            0,
            k,
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

    #[test]
    fn off_unless_q_is_valid_and_defaults_apply() {
        assert_eq!(Stage2Config::from_lookup(lookup(&[])), None);
        assert_eq!(
            Stage2Config::from_lookup(lookup(&[("QUIP_OTSUKA_STAGE2_K", "4")])),
            None,
            "K alone does not turn stage 2 on"
        );
        for bad in ["0", "1", "-0.1", "x", "NaN"] {
            assert_eq!(
                Stage2Config::from_lookup(lookup(&[("QUIP_OTSUKA_STAGE2_Q", bad)])),
                None,
                "{bad}"
            );
        }
        assert_eq!(
            Stage2Config::from_lookup(lookup(&[("QUIP_OTSUKA_STAGE2_Q", "0.1")])),
            Some(config(0.1))
        );
        let set = Stage2Config::from_lookup(lookup(&[
            ("QUIP_OTSUKA_STAGE2_Q", "0.05"),
            ("QUIP_OTSUKA_STAGE2_K", "3"),
            ("QUIP_OTSUKA_STAGE2_GUARD_MILLI", "2000"),
            ("QUIP_OTSUKA_STAGE2_MAX_ACTIVE", "9"),
            ("QUIP_OTSUKA_STAGE2_RELEASE_MILLI", "0"),
        ]))
        .unwrap();
        assert_eq!(
            set,
            Stage2Config {
                q: 0.05,
                k: 3,
                guard_milli: 2000,
                max_active: 9,
                release_milli: 0,
            }
        );
        let bad = Stage2Config::from_lookup(lookup(&[
            ("QUIP_OTSUKA_STAGE2_Q", "0.1"),
            ("QUIP_OTSUKA_STAGE2_K", "9"),
            ("QUIP_OTSUKA_STAGE2_MAX_ACTIVE", "1.5"),
        ]))
        .unwrap();
        assert_eq!(bad, config(0.1), "out-of-range knobs keep their defaults");
    }

    #[test]
    fn rerun_seeds_differ_from_each_other_and_the_job() {
        let seeds: std::collections::HashSet<u64> =
            (1..=8).map(|j| rerun_seed(42, j)).chain([42]).collect();
        assert_eq!(seeds.len(), 9);
        assert!(!seeds.contains(&0));
    }

    #[test]
    fn trigger_waits_for_warmup_then_takes_the_low_tail() {
        let s = Stage2::new(config(0.1));
        let shape = (10, 20);
        for i in 0..i64::try_from(Stage2::WARMUP).unwrap() {
            // Every value is a new minimum: admitted, but still warming up.
            assert!(!s.should_deepen(shape, -i), "warm-up {i}");
        }
        assert!(s.should_deepen(shape, -1_000), "a new low after warm-up");
        assert!(!s.should_deepen(shape, 0), "an ordinary first");
        assert!(
            !s.should_deepen((11, 20), -2_000),
            "a new shape starts a new warm-up"
        );
    }

    #[test]
    fn observed_firsts_count_toward_warmup_and_the_cutoff() {
        let s = Stage2::new(config(0.5));
        let shape = (10, 20);
        for _ in 0..=Stage2::WARMUP {
            s.observe(shape, -1_000);
        }
        assert!(!s.should_deepen(shape, 0), "above the observed lows");
        assert!(
            s.should_deepen(shape, -2_000),
            "warm-up passed on observed firsts"
        );
    }

    #[test]
    fn guard_never_holds_a_first_near_the_target() {
        let s = Stage2::new(config(0.5));
        let shape = (1, 1);
        for _ in 0..=Stage2::WARMUP {
            s.should_deepen(shape, 0);
        }
        assert!(s.should_deepen(shape, -100), "no target yet: deepen");
        s.set_live_target(Some(-3_000));
        assert!(!s.should_deepen(shape, -3_500), "below the target");
        assert!(!s.should_deepen(shape, -3_000 + 1_250), "at target + guard");
        assert!(
            s.should_deepen(shape, -3_000 + 1_251),
            "just outside the guard"
        );
        s.set_live_target(None);
        assert_eq!(s.live_target(), Some(-3_000), "None keeps the last target");
        s.set_live_target(Some(i64::MAX));
        assert!(
            !s.should_deepen(shape, -9_000),
            "a target that accepts everything never holds a result"
        );
    }

    #[test]
    fn group_merges_reruns_and_delivers_once_after_the_last() {
        let s = Stage2::new(config(0.1));
        let (g, _stop) = group(&[-10, -5, -1], 2);
        assert!(g.complete(&s, done(&[-3, -2, 0]), 1).is_none());
        assert!(g.pending() && !g.stopped());
        let p = g
            .complete(&s, done(&[-12, 4, 4]), 2)
            .expect("last rerun delivers");
        let mut energies: Vec<i64> = match p.result.outcome {
            StreamOutcome::Completed(Ok(r)) => r.iter().map(|r| r.energy_milli).collect(),
            _ => panic!("completed"),
        };
        energies.sort_unstable();
        assert_eq!(energies, vec![-12, -10, -5], "read count kept, best merged");
        assert!(!g.pending() && g.stopped());
        assert!(g.complete(&s, done(&[-99]), 3).is_none(), "never twice");
    }

    #[test]
    fn group_delivers_early_when_a_rerun_reaches_the_target() {
        let s = Stage2::new(config(0.1));
        s.set_live_target(Some(-1_000));
        let (g, _stop) = group(&[-100], 3);
        assert!(g.complete(&s, done(&[-200]), 1).is_none());
        let p = g.complete(&s, done(&[-1_400]), 2).expect("within release");
        assert_eq!(best_of(&p.result), Some(-1_400));
        assert!(g.stopped(), "the third rerun is dropped");
        assert!(g.complete(&s, done(&[-2_000]), 3).is_none());
    }

    #[test]
    fn watcher_releases_when_the_target_eases_and_drops_stopped_groups() {
        let s = Stage2::new(config(0.1));
        s.set_live_target(Some(-5_000));
        let (g, _stop) = group(&[-4_000], 2);
        assert!(g.release_due(&s).is_none(), "far from the target");
        s.set_live_target(Some(-4_600));
        assert!(g.release_due(&s).is_none(), "600 above: outside 500");
        s.set_live_target(Some(-4_500));
        assert!(g.release_due(&s).is_some(), "eased to within 500");
        assert!(g.stopped() && !g.pending());
        assert!(g.complete(&s, done(&[-9_000]), 1).is_none());

        let (g, stop) = group(&[-4_000], 2);
        stop.store(true, Ordering::Relaxed);
        assert!(g.stopped());
        assert!(
            g.release_due(&s).is_none(),
            "a stopped lease's group is dropped"
        );
        assert!(!g.pending(), "and no longer counts toward the cap");
    }

    #[test]
    fn a_stopped_lease_drops_the_group_on_its_last_rerun() {
        let s = Stage2::new(config(0.1));
        let (g, stop) = group(&[-10], 1);
        stop.store(true, Ordering::Relaxed);
        let cancelled = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Cancelled,
            device_access_time_us: 0,
        };
        assert!(g.complete(&s, cancelled, 1).is_none());
        assert!(!g.pending());
    }

    #[test]
    fn a_fatal_rerun_error_is_delivered_at_once() {
        let s = Stage2::new(config(0.1));
        let (g, _stop) = group(&[-10], 2);
        let fault = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Err(SampleError::DeviceFault("x".into()))),
            device_access_time_us: 0,
        };
        let p = g.complete(&s, fault, 1).expect("fatal goes out at once");
        assert!(matches!(
            p.result.outcome,
            StreamOutcome::Completed(Err(ref e)) if e.is_fatal()
        ));
        let busy = StreamResult {
            job_id: b"j".to_vec(),
            outcome: StreamOutcome::Completed(Err(SampleError::DeviceBusy)),
            device_access_time_us: 0,
        };
        let (g, _stop) = group(&[-10], 2);
        assert!(g.complete(&s, busy, 1).is_none(), "a busy rerun is skipped");
        assert_eq!(
            g.complete(&s, done(&[-11]), 2)
                .and_then(|p| best_of(&p.result)),
            Some(-11)
        );
    }

    #[test]
    fn energies_compare_with_the_instance_offset() {
        let s = Stage2::new(config(0.1));
        s.set_live_target(Some(500));
        let stop = Arc::new(AtomicBool::new(false));
        let g: Group<(), (), ()> = Group::new(
            1_200,
            1_000,
            2,
            stop,
            Payload {
                job: (),
                family: (),
                dest: (),
                result: done(&[200]),
            },
        );
        // Lane best 200 + offset 1000 = 1200 reported: 700 above the target.
        assert!(g.release_due(&s).is_none());
        s.set_live_target(Some(700));
        assert!(g.release_due(&s).is_some());
    }
}
