//! Local build (`BudFox`, 2026-11-14): stage 1, a short-anneal screen on the
//! family lease salts that reach a lane (successive halving after stage 0).
//!
//! Off by default: with `QUIP_OTSUKA_STAGE1_FRAC` unset the sampler never
//! builds a [`Stage1`], and every job runs the stock single anneal launch
//! for launch, with the same seed and the same draws.
//!
//! On, a lane anneals each validated lease salt twice at most:
//!
//! 1. A short anneal: the job's ladder over `FRAC` of its sweeps, with
//!    `READS` reads (default the job's), on a seed derived from the job's
//!    seed ([`SHORT_SEED_SALT`]), so its draws are independent of step 2.
//! 2. When the short run's best energy is at or below the `KEEP` quantile of
//!    the recent short-run bests ([`RollingCutoff`]), the full anneal: a fresh
//!    run with the job's own parameters, exactly the run it gets today. The
//!    short reads that beat the full run's worst reads replace them, so the
//!    read count stays the job's and nothing better is dropped.
//!
//! A salt that is not promoted completes like a stage-0 reject: done for its
//! lease, no reads, nothing sent. The exception: short reads at or below the
//! `SUBMIT_Q` quantile of the recent full-run bests are as deep as a typical
//! full result, so they are returned (and reach the coordinator) instead of
//! being dropped. A non-fatal failure of the full run (a governor yield)
//! returns the short reads; a fatal error and a cancellation propagate.
//!
//! Both windows hold raw energies, which only compare within one problem
//! shape. When the sampler passes the job's shape ([`Stage1::solve_shaped`]),
//! a salt of a new shape (a topology change: the chain moved from 4575 to
//! 4560 nodes on 2026-11-14) empties both windows before it is gated, and a
//! full result of the old shape that finishes afterwards is not recorded.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_STAGE1_FRAC` | unset (off) | Short-run share of the job's sweeps, `0 < FRAC < 1`. Setting it turns stage 1 on. |
//! | `QUIP_OTSUKA_STAGE1_KEEP` | 1 | Fraction of short runs promoted to the full anneal, `0 < KEEP <= 1`. 1 promotes every salt (calibration: every salt pays the short run and logs a short/full pair). |
//! | `QUIP_OTSUKA_STAGE1_READS` | job reads | Reads of the short run, `1..=256`, never more than the job's. |
//! | `QUIP_OTSUKA_STAGE1_SUBMIT_Q` | 0.5 | A screened-out salt still returns its short reads at or below this quantile of the last 256 full-run bests. 0 never returns them. |

use quip_solver_core::{SampleParams, SamplerResult};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock, PoisonError};

/// `XORed` into the job seed for the short run, so its draws differ from the
/// full run's (which keeps the job seed unchanged).
pub const SHORT_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Stage-1 knobs. Present only when stage 1 is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Config {
    /// Short-run share of the job's sweeps, in `(0, 1)`.
    pub frac: f64,
    /// Fraction of short runs promoted, in `(0, 1]`.
    pub keep: f64,
    /// Short-run reads; 0 = the job's reads.
    pub reads: usize,
    /// Quantile of recent full-run bests below which a screened-out salt's
    /// short reads are returned; 0 = never.
    pub submit_q: f64,
}

impl Stage1Config {
    /// Read the knobs through `get`. `None` (off) unless `FRAC` is valid.
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
        let frac = num("QUIP_OTSUKA_STAGE1_FRAC", &|v| v > 0.0 && v < 1.0)?;
        let keep = num("QUIP_OTSUKA_STAGE1_KEEP", &|v| v > 0.0 && v <= 1.0).unwrap_or(1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let reads = num("QUIP_OTSUKA_STAGE1_READS", &|v| {
            (1.0..=256.0).contains(&v) && v.fract() == 0.0
        })
        .map_or(0, |v| v as usize);
        let submit_q =
            num("QUIP_OTSUKA_STAGE1_SUBMIT_Q", &|v| (0.0..=1.0).contains(&v)).unwrap_or(0.5);
        Some(Self {
            frac,
            keep,
            reads,
            submit_q,
        })
    }

    /// The process environment, read once.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        static ENV: OnceLock<Option<Stage1Config>> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }

    /// The short run's parameters: the job's ladder ends and schedule over
    /// `frac` of its sweeps (at least one), the stage-1 read count (at most
    /// the job's), and a derived seed.
    #[must_use]
    pub fn short_params(&self, full: &SampleParams) -> SampleParams {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let sweeps = ((full.num_sweeps as f64 * self.frac).round() as usize).max(1);
        let reads = if self.reads == 0 {
            full.num_reads
        } else {
            self.reads.min(full.num_reads)
        };
        SampleParams {
            num_reads: reads.max(1),
            num_sweeps: sweeps,
            seed: full.seed ^ SHORT_SEED_SALT,
            ..full.clone()
        }
    }
}

/// A rolling quantile over the last `cap` values, one decision per value
/// with no buffering (the stage-0 gate's rule): a value passes when it is at
/// or below the `q` quantile of the history, itself included. The cutoff is
/// recomputed every `refresh` values, and on every value while fewer than
/// `refresh` have been seen.
#[derive(Debug)]
pub struct RollingCutoff {
    q: f64,
    cap: usize,
    refresh: usize,
    history: VecDeque<i64>,
    cutoff: Option<i64>,
    since: usize,
}

impl RollingCutoff {
    /// An empty window keeping the `q` quantile of the last `cap` values.
    #[must_use]
    pub fn new(q: f64, cap: usize, refresh: usize) -> Self {
        Self {
            q: q.clamp(0.0, 1.0),
            cap: cap.max(1),
            refresh: refresh.max(1),
            history: VecDeque::with_capacity(cap.max(1)),
            cutoff: None,
            since: 0,
        }
    }

    /// Record `value` and return whether it is at or below the cutoff.
    /// `q >= 1` admits every value.
    pub fn admit(&mut self, value: i64) -> bool {
        self.push(value);
        self.q >= 1.0 || self.cutoff.is_none_or(|c| value <= c)
    }

    /// Record `value` without a decision.
    pub fn push(&mut self, value: i64) {
        if self.history.len() == self.cap {
            self.history.pop_front();
        }
        self.history.push_back(value);
        self.since += 1;
        if self.history.len() < self.refresh || self.since >= self.refresh {
            self.since = 0;
            self.cutoff = self.quantile();
        }
    }

    /// The current cutoff, `None` before the first value.
    #[must_use]
    pub fn cutoff(&self) -> Option<i64> {
        self.cutoff
    }

    fn quantile(&self) -> Option<i64> {
        let mut sorted: Vec<i64> = self.history.iter().copied().collect();
        sorted.sort_unstable();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let rank = ((sorted.len() as f64 * self.q).ceil() as usize).clamp(1, sorted.len().max(1));
        sorted.get(rank - 1).copied()
    }
}

/// How a stage-1 job ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage1Outcome {
    /// Screened out after the short run.
    Rejected,
    /// Promoted and fully annealed.
    Promoted,
    /// Promoted, but the full run failed without a fatal error or a
    /// cancellation; the short reads are the result.
    FullFailed,
}

/// What one stage-1 job did, for the debug log and the benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1Trace {
    /// Best short-run energy.
    pub short_best: Option<i64>,
    /// Best full-run energy, before the merge; `None` without a full run.
    pub full_best: Option<i64>,
    /// Short reads returned: merged into the full reads, or kept on reject.
    pub short_kept: usize,
    /// How the job ended.
    pub outcome: Stage1Outcome,
}

struct State {
    shorts: RollingCutoff,
    fulls: RollingCutoff,
    decided: u64,
    promoted: u64,
    /// Problem shape (nodes, edges) the windows hold energies of.
    shape: Option<(usize, usize)>,
}

/// Stage-1 state shared by every lane: the promotion gate and the recent
/// full-run bests.
pub struct Stage1 {
    config: Stage1Config,
    state: Mutex<State>,
}

impl Stage1 {
    /// Short-run bests the promotion quantile looks back over.
    pub const HISTORY: usize = 1024;
    /// Promotion cutoff refresh period.
    pub const REFRESH: usize = 32;
    /// Full-run bests the submit bound looks back over.
    pub const FULL_HISTORY: usize = 256;
    const LOG_EVERY: u64 = 256;

    /// Fresh state for `config`.
    #[must_use]
    pub fn new(config: Stage1Config) -> Self {
        Self {
            config,
            state: Mutex::new(State {
                shorts: RollingCutoff::new(config.keep, Self::HISTORY, Self::REFRESH),
                fulls: RollingCutoff::new(config.submit_q, Self::FULL_HISTORY, 1),
                decided: 0,
                promoted: 0,
                shape: None,
            }),
        }
    }

    /// The knobs.
    #[must_use]
    pub const fn config(&self) -> &Stage1Config {
        &self.config
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Run one job through stage 1. `run` anneals the job with the given
    /// parameters: first the short parameters, then, if the salt is
    /// promoted, the job's own `full` parameters unchanged. `recoverable`
    /// tells a full-run error after which the short reads are returned from
    /// one that must propagate (fatal or cancelled).
    ///
    /// # Errors
    /// The short run's error, or an unrecoverable full-run error.
    pub fn solve<E>(
        &self,
        full: &SampleParams,
        run: impl FnMut(&SampleParams) -> Result<Vec<SamplerResult>, E>,
        recoverable: impl Fn(&E) -> bool,
    ) -> Result<(Vec<SamplerResult>, Stage1Trace), E> {
        self.solve_shaped(None, full, run, recoverable)
    }

    /// [`Self::solve`] for a job of `shape` (nodes, edges). A shape other
    /// than the windows' empties them first, so energies of one topology are
    /// never gated against another's; `None` never resets.
    ///
    /// # Errors
    /// The short run's error, or an unrecoverable full-run error.
    pub fn solve_shaped<E>(
        &self,
        shape: Option<(usize, usize)>,
        full: &SampleParams,
        mut run: impl FnMut(&SampleParams) -> Result<Vec<SamplerResult>, E>,
        recoverable: impl Fn(&E) -> bool,
    ) -> Result<(Vec<SamplerResult>, Stage1Trace), E> {
        let short = run(&self.config.short_params(full))?;
        let short_best = short.iter().map(|r| r.energy_milli).min();
        let promote = {
            let mut s = self.lock();
            if let Some(shape) = shape.filter(|k| s.shape != Some(*k)) {
                if let Some((nodes, edges)) = s.shape {
                    tracing::info!(
                        old_nodes = nodes,
                        old_edges = edges,
                        nodes = shape.0,
                        edges = shape.1,
                        "otsuka stage1: problem shape changed, windows reset"
                    );
                }
                s.shorts = RollingCutoff::new(self.config.keep, Self::HISTORY, Self::REFRESH);
                s.fulls = RollingCutoff::new(self.config.submit_q, Self::FULL_HISTORY, 1);
                s.shape = Some(shape);
            }
            let promote = short_best.is_none_or(|b| s.shorts.admit(b));
            s.decided += 1;
            s.promoted += u64::from(promote);
            if s.decided.is_multiple_of(Self::LOG_EVERY) {
                tracing::info!(
                    decided = s.decided,
                    promoted = s.promoted,
                    cutoff = s.shorts.cutoff(),
                    submit_bound = self.submit_bound(&s),
                    "otsuka stage1"
                );
            }
            promote
        };
        if !promote {
            let bound = self.submit_bound(&self.lock());
            let kept: Vec<SamplerResult> = match bound {
                Some(bound) => short
                    .into_iter()
                    .filter(|r| r.energy_milli <= bound)
                    .collect(),
                None => Vec::new(),
            };
            let short_kept = kept.len();
            return Ok((
                kept,
                Stage1Trace {
                    short_best,
                    full_best: None,
                    short_kept,
                    outcome: Stage1Outcome::Rejected,
                },
            ));
        }
        match run(full) {
            Ok(mut reads) => {
                let full_best = reads.iter().map(|r| r.energy_milli).min();
                if let Some(best) = full_best {
                    let mut s = self.lock();
                    if shape.is_none() || s.shape == shape {
                        s.fulls.push(best);
                    }
                }
                let short_kept = merge_better(&mut reads, short);
                Ok((
                    reads,
                    Stage1Trace {
                        short_best,
                        full_best,
                        short_kept,
                        outcome: Stage1Outcome::Promoted,
                    },
                ))
            }
            Err(e) if recoverable(&e) => {
                let short_kept = short.len();
                Ok((
                    short,
                    Stage1Trace {
                        short_best,
                        full_best: None,
                        short_kept,
                        outcome: Stage1Outcome::FullFailed,
                    },
                ))
            }
            Err(e) => Err(e),
        }
    }

    fn submit_bound(&self, s: &State) -> Option<i64> {
        if self.config.submit_q <= 0.0 {
            return None;
        }
        s.fulls.cutoff()
    }
}

/// Replace the worst reads of `full` with the reads of `short` that are
/// strictly better, best first. The read count and the order of the
/// untouched full reads stay. Returns how many short reads went in.
pub fn merge_better(full: &mut [SamplerResult], mut short: Vec<SamplerResult>) -> usize {
    short.sort_by_key(|r| r.energy_milli);
    let mut merged = 0;
    for read in short {
        let Some((worst, _)) = full
            .iter()
            .enumerate()
            .max_by_key(|(i, r)| (r.energy_milli, std::cmp::Reverse(*i)))
        else {
            break;
        };
        if read.energy_milli >= full[worst].energy_milli {
            break;
        }
        full[worst] = read;
        merged += 1;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> Option<Stage1Config> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        Stage1Config::from_lookup(|k| map.get(k).cloned())
    }

    fn read(e: i64) -> SamplerResult {
        SamplerResult {
            spins: vec![1, -1],
            energy_milli: e,
        }
    }

    fn job() -> SampleParams {
        SampleParams {
            num_reads: 64,
            num_sweeps: 131_072,
            sweeps_per_beta: 1,
            beta_range: Some((1.0, 12.0)),
            seed: 0xDEAD_BEEF,
            ..Default::default()
        }
    }

    fn cfg(frac: f64, keep: f64) -> Stage1Config {
        Stage1Config {
            frac,
            keep,
            reads: 0,
            submit_q: 0.5,
        }
    }

    #[test]
    fn unset_or_invalid_frac_is_off() {
        assert_eq!(env(&[]), None);
        assert_eq!(env(&[("QUIP_OTSUKA_STAGE1_KEEP", "0.2")]), None);
        for bad in ["0", "1", "1.5", "-0.1", "x", "NaN"] {
            assert_eq!(env(&[("QUIP_OTSUKA_STAGE1_FRAC", bad)]), None, "{bad}");
        }
    }

    #[test]
    fn knobs_parse_and_bad_values_fall_back() {
        let c = env(&[
            ("QUIP_OTSUKA_STAGE1_FRAC", "0.125"),
            ("QUIP_OTSUKA_STAGE1_KEEP", "0.2"),
            ("QUIP_OTSUKA_STAGE1_READS", "32"),
            ("QUIP_OTSUKA_STAGE1_SUBMIT_Q", "0"),
        ])
        .unwrap();
        assert_eq!(
            c,
            Stage1Config {
                frac: 0.125,
                keep: 0.2,
                reads: 32,
                submit_q: 0.0
            }
        );
        let d = env(&[
            ("QUIP_OTSUKA_STAGE1_FRAC", "0.0625"),
            ("QUIP_OTSUKA_STAGE1_KEEP", "0"),
            ("QUIP_OTSUKA_STAGE1_READS", "300"),
            ("QUIP_OTSUKA_STAGE1_SUBMIT_Q", "2"),
        ])
        .unwrap();
        assert_eq!(
            d,
            Stage1Config {
                frac: 0.0625,
                keep: 1.0,
                reads: 0,
                submit_q: 0.5
            }
        );
    }

    #[test]
    fn short_params_scale_sweeps_cap_reads_and_derive_the_seed() {
        let full = job();
        let s = cfg(0.125, 0.2).short_params(&full);
        assert_eq!(s.num_sweeps, 16_384);
        assert_eq!(s.num_reads, 64);
        assert_eq!(s.seed, full.seed ^ SHORT_SEED_SALT);
        assert_ne!(s.seed, full.seed);
        assert_eq!(s.sweeps_per_beta, full.sweeps_per_beta);
        assert_eq!(s.beta_range, full.beta_range);
        assert_eq!(s.schedule, full.schedule);
        let tiny = SampleParams {
            num_sweeps: 3,
            num_reads: 4,
            ..full
        };
        let c = Stage1Config {
            reads: 32,
            ..cfg(0.1, 1.0)
        };
        assert_eq!(c.short_params(&tiny).num_sweeps, 1, "at least one sweep");
        assert_eq!(
            c.short_params(&tiny).num_reads,
            4,
            "never more reads than the job"
        );
        assert_eq!(c.short_params(&job()).num_reads, 32);
    }

    #[test]
    fn rolling_cutoff_keeps_about_q_of_a_stream_lowest_first() {
        let mut gate = RollingCutoff::new(0.2, Stage1::HISTORY, Stage1::REFRESH);
        let mut x: u64 = 0x1234_5678;
        let (mut kept, mut kept_sum, mut all_sum) = (0_i64, 0_i64, 0_i64);
        let n = 20_000;
        for _ in 0..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = -5_060_000 - i64::try_from(x % 20_000).unwrap();
            all_sum += v;
            if gate.admit(v) {
                kept += 1;
                kept_sum += v;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let frac = kept as f64 / f64::from(n);
        assert!((0.17..=0.23).contains(&frac), "kept {frac}");
        assert!(
            kept_sum / kept < all_sum / i64::from(n) - 5_000,
            "kept the low values"
        );
    }

    #[test]
    fn rolling_cutoff_admits_the_first_value_and_keep_one_admits_all() {
        let mut gate = RollingCutoff::new(0.1, 16, 4);
        assert_eq!(gate.cutoff(), None);
        assert!(gate.admit(5));
        let mut all = RollingCutoff::new(1.0, 16, 4);
        assert!((0..100).all(|v| all.admit(100 - v * 3 + (v % 7) * 11)));
    }

    #[test]
    fn promoted_full_run_gets_the_job_params_unchanged() {
        let stage1 = Stage1::new(cfg(0.125, 1.0));
        let full = job();
        let mut seen: Vec<SampleParams> = Vec::new();
        let (reads, trace) = stage1
            .solve(
                &full,
                |p| -> Result<_, ()> {
                    seen.push(p.clone());
                    Ok(vec![read(-10); p.num_reads])
                },
                |()| true,
            )
            .unwrap();
        assert_eq!(seen.len(), 2);
        let f = &seen[1];
        assert_eq!(
            (
                f.num_reads,
                f.num_sweeps,
                f.sweeps_per_beta,
                f.beta_range,
                f.schedule,
                f.seed
            ),
            (
                full.num_reads,
                full.num_sweeps,
                full.sweeps_per_beta,
                full.beta_range,
                full.schedule,
                full.seed
            ),
            "the full run is today's run, draw for draw"
        );
        assert_eq!(seen[0].num_sweeps, 16_384);
        assert_eq!(reads.len(), 64);
        assert_eq!(trace.outcome, Stage1Outcome::Promoted);
        assert_eq!(trace.short_kept, 0, "equal reads are not swapped in");
    }

    #[test]
    fn reject_completes_with_no_reads_and_skips_the_full_run() {
        let stage1 = Stage1::new(cfg(0.125, 0.5));
        let mut runs = 0;
        // The first short best passes (empty history); a worse one is rejected.
        for (short, promoted) in [(-100, true), (-50, false)] {
            let (reads, trace) = stage1
                .solve(
                    &job(),
                    |p| -> Result<_, ()> {
                        runs += 1;
                        let e = if p.num_sweeps == 131_072 { -200 } else { short };
                        Ok(vec![read(e); p.num_reads])
                    },
                    |()| true,
                )
                .unwrap();
            assert_eq!(trace.outcome == Stage1Outcome::Promoted, promoted);
            if !promoted {
                assert!(reads.is_empty(), "a screened salt completes with no reads");
                assert_eq!(trace.full_best, None);
            }
        }
        assert_eq!(runs, 3, "short + full, then short only");
    }

    #[test]
    fn a_rejected_short_run_as_deep_as_a_typical_full_run_is_kept() {
        // keep 0.01: after the first salt only the lowest short best passes.
        let picky = Stage1::new(Stage1Config {
            submit_q: 1.0,
            ..cfg(0.5, 0.01)
        });
        let run = |short: i64| {
            move |p: &SampleParams| -> Result<_, ()> {
                let e = if p.num_sweeps == 131_072 {
                    -1_000
                } else {
                    short
                };
                Ok(vec![read(e), read(e + 500)])
            }
        };
        let (_, first) = picky.solve(&job(), run(-5_000), |()| true).unwrap();
        assert_eq!(
            first.outcome,
            Stage1Outcome::Promoted,
            "full history now holds -1000"
        );
        for _ in 0..200 {
            picky.solve(&job(), run(-6_000), |()| true).unwrap();
        }
        let (reads, trace) = picky.solve(&job(), run(-900), |()| true).unwrap();
        assert_eq!(trace.outcome, Stage1Outcome::Rejected);
        assert!(
            reads.is_empty(),
            "shallower than every recent full result: dropped"
        );
        let (reads, trace) = picky.solve(&job(), run(-2_000), |()| true).unwrap();
        assert_eq!(trace.outcome, Stage1Outcome::Rejected);
        assert_eq!(
            reads.iter().map(|r| r.energy_milli).collect::<Vec<_>>(),
            vec![-2_000, -1_500],
            "short reads at or below the recent full bests are returned"
        );
        assert_eq!(trace.short_kept, 2);
    }

    #[test]
    fn submit_q_zero_never_returns_screened_reads() {
        let stage1 = Stage1::new(Stage1Config {
            submit_q: 0.0,
            ..cfg(0.5, 0.01)
        });
        let run = |short: i64| {
            move |p: &SampleParams| -> Result<_, ()> {
                Ok(vec![read(if p.num_sweeps == 131_072 {
                    -1_000
                } else {
                    short
                })])
            }
        };
        stage1.solve(&job(), run(-5_000), |()| true).unwrap();
        for _ in 0..100 {
            stage1.solve(&job(), run(-6_000), |()| true).unwrap();
        }
        let (reads, trace) = stage1.solve(&job(), run(-2_000), |()| true).unwrap();
        assert_eq!(trace.outcome, Stage1Outcome::Rejected);
        assert!(reads.is_empty());
    }

    #[test]
    fn full_run_errors_fall_back_or_propagate() {
        let stage1 = Stage1::new(cfg(0.25, 1.0));
        let run = |fatal: bool| {
            move |p: &SampleParams| -> Result<Vec<SamplerResult>, bool> {
                if p.num_sweeps == 131_072 {
                    Err(fatal)
                } else {
                    Ok(vec![read(-7); p.num_reads])
                }
            }
        };
        let (reads, trace) = stage1.solve(&job(), run(false), |fatal| !fatal).unwrap();
        assert_eq!(trace.outcome, Stage1Outcome::FullFailed);
        assert_eq!(reads.len(), 64, "a yielded full run keeps the short reads");
        assert!(stage1.solve(&job(), run(true), |fatal| !fatal).unwrap_err());
        // A short-run error propagates and never starts the full run.
        let mut calls = 0;
        let err = stage1
            .solve(
                &job(),
                |_| -> Result<Vec<SamplerResult>, &str> {
                    calls += 1;
                    Err("cancelled")
                },
                |_| true,
            )
            .unwrap_err();
        assert_eq!((err, calls), ("cancelled", 1));
    }

    #[test]
    fn a_new_problem_shape_resets_both_windows() {
        // keep 0.01: once a deep short best is in the window, a shallow one
        // is rejected; submit_q 1: rejected reads at or below the worst
        // recent full best come back.
        let stage1 = Stage1::new(Stage1Config {
            submit_q: 1.0,
            ..cfg(0.5, 0.01)
        });
        let run = |short: i64, full: i64| {
            move |p: &SampleParams| -> Result<_, ()> {
                Ok(vec![read(if p.num_sweeps == 131_072 {
                    full
                } else {
                    short
                })])
            }
        };
        let old = Some((4575, 41_474));
        let new = Some((4560, 41_205));
        for _ in 0..64 {
            stage1
                .solve_shaped(old, &job(), run(-5_070, -5_080), |()| true)
                .unwrap();
        }
        let (_, t) = stage1
            .solve_shaped(old, &job(), run(-5_000, -5_080), |()| true)
            .unwrap();
        assert_eq!(
            t.outcome,
            Stage1Outcome::Rejected,
            "shallow against the old window"
        );
        // The first salt of the new shape sees empty windows: promoted, and
        // its shallower full best becomes the submit bound.
        let (_, t) = stage1
            .solve_shaped(new, &job(), run(-5_000, -5_010), |()| true)
            .unwrap();
        assert_eq!(
            t.outcome,
            Stage1Outcome::Promoted,
            "new shape starts a new window"
        );
        let (reads, t) = stage1
            .solve_shaped(new, &job(), run(-4_990, -5_010), |()| true)
            .unwrap();
        assert_eq!(t.outcome, Stage1Outcome::Rejected);
        assert!(reads.is_empty(), "-4990 is above every new-shape full best");
        let (reads, _) = stage1
            .solve_shaped(new, &job(), run(-5_012, -5_010), |()| true)
            .unwrap();
        assert_eq!(
            reads.len(),
            1,
            "promoted or kept: -5012 beats the new window"
        );
        // A shape-less solve never resets.
        let (_, t) = stage1
            .solve(&job(), run(-4_000, -4_000), |()| true)
            .unwrap();
        assert_eq!(t.outcome, Stage1Outcome::Rejected);
    }

    #[test]
    fn merge_swaps_only_strictly_better_short_reads_into_the_worst_slots() {
        let mut full = vec![read(-10), read(-30), read(-20), read(-30)];
        let n = merge_better(&mut full, vec![read(-25), read(-5), read(-40), read(-15)]);
        assert_eq!(
            n, 2,
            "-40 and -25 go in; -15 is no better than the new worst"
        );
        let mut e: Vec<i64> = full.iter().map(|r| r.energy_milli).collect();
        assert_eq!(e.len(), 4);
        e.sort_unstable();
        assert_eq!(e, vec![-40, -30, -30, -25]);
        assert_eq!(
            full[1].energy_milli, -30,
            "untouched reads keep their slots"
        );
        let mut same = vec![read(-10)];
        assert_eq!(merge_better(&mut same, vec![read(-10)]), 0);
    }
}
