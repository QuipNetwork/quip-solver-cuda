//! Local build (`BudFox`, 2026-11-14): lease salts generated and screened on a
//! worker pool (`QUIP_OTSUKA_SCREEN_THREADS`).
//!
//! On core's default path every lease salt is generated inline in core's
//! expander, a task on the session's single-threaded runtime (~12 ms per
//! dimer instance), and stage 0 then scores it on the sampler's one job
//! thread. Together they cap the salts a tight `QUIP_OTSUKA_STAGE0_KEEP` can
//! screen per second (~80/s from generation alone), well before the GPU lanes
//! run out of work.
//!
//! With `QUIP_OTSUKA_SCREEN_THREADS` above 1 the sampler declares local
//! generation, so core hands each lease to [`Sampler::sample_lease`] on a
//! lease thread instead. There, `THREADS` workers claim salt indices in
//! turn and, holding one of `THREADS` CPU slots shared by every live lease,
//! generate the canonical instance (`FamilyLease::generate`, the instance
//! the coordinator regenerates), build the job graph exactly as core does,
//! and run the same stage-0 gates the session path uses. Each gate decides
//! under its own lock, so the rolling-quantile rule is the serial one applied
//! in arrival order. A rejected salt is pushed with no reads (counted done,
//! nothing sent: what the session path's empty completion does), a skipped
//! one (`SaltFault`) is skipped, and a kept one goes to the lanes' queue,
//! which holds two salts per lane, so a lane finds the next screened salt
//! waiting when it finishes one. Workers only block when that queue is full.
//!
//! all5 (agent X fix): the slots go to the oldest lease with unclaimed salts
//! first (FIFO); a younger lease waits on one sleeping thread and spawns its
//! workers on its turn, so leases finish in order and `LeaseDone` flows. A
//! kept salt keeps its slot until the lanes' queue takes it, and a full queue
//! parks the worker on a condvar the lanes signal (back-pressure: the screen
//! runs at the lanes' rate). The sampler reports the lane count as its stream
//! width here, which sizes the session's lease credit window.
//!
//! Lanes solve a local salt exactly like a session one (stage 1, polish,
//! row sort) and hand the result back to its lease thread, which adds the
//! instance offset and pushes it once. Accounting: every salt index is
//! claimed by exactly one worker, and core's `LeaseSink` refuses a repeat.
//!
//! Stop: the lease thread polls `LeaseSink::is_stopped` (a round change,
//! the deadline, shutdown) every 10 ms into a flag. Workers stop claiming,
//! a queued salt of a stopped lease is dropped by the lane without a launch,
//! and a running one stops at its next sweep chunk, as on the session path.
//!
//! 1 (the default) does not declare local generation: core's own path,
//! unchanged. A legacy (non-family) lease, which core also hands over, is
//! solved serially on the SA sampler.
//!
//! A line every 60 s (`otsuka screen`) reports salts screened and kept per
//! second, the lanes' queue depth, the share of lane time spent waiting for
//! work, and the mean generate-and-score time per salt.
//!
//! all6 (both off by default; unset = all5):
//! - `QUIP_OTSUKA_POOL` (see [`crate::cluster::pool`]): a salt that passes
//!   stage 0 with a score is offered to a best-first pool instead of the
//!   FIFO queue, and the worker returns to screening at once (no
//!   back-pressure); the lanes take the pool's lowest score. A salt the pool
//!   refuses or later evicts or ages out is released unsolved: its lease
//!   thread pushes it with no reads, exactly as a stage-0 reject.
//! - `QUIP_OTSUKA_FAST_GEN=1`: stage 0 scores the couplings of the cached
//!   dimer generator ([`crate::family_screen::fast_dimer`], ~3 ms instead of
//!   ~12 ms per instance), and only a salt that passes is generated
//!   canonically (the instance the lanes anneal and the coordinator
//!   regenerates). Each such salt, and one screened salt in 64, is compared
//!   with the canonical instance; any difference turns the cached generator
//!   off for that topology and parameter set (logged). A salt the cached generator cannot build goes
//!   the canonical way, which also classifies skips.

use super::{ClusterBackend, FamilyRef, Gate0, LaneItem, OtsukaSampler, TWISTED_DIMER_V1};
use crate::family_screen::fast_dimer::FastDimer;
use quip_solver_core::ising_generators::family::SaltFault;
use quip_solver_core::ising_generators::ice::local_gauges;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    CancelToken, FamilyLease, IsingGraph, Lease, LeaseSink, SampleError, SampleParams, Sampler,
    StreamJob, StreamOutcome, StreamResult, TopologyView,
};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, TryLockError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;

/// How often the lease thread folds `LeaseSink::is_stopped` into the flag.
const STOP_POLL: Duration = Duration::from_millis(10);
/// Screen summary period.
const LOG_EVERY: Duration = Duration::from_mins(1);

/// `QUIP_OTSUKA_SCREEN_THREADS`, `1..=64`, default 1, read once.
pub(crate) fn screen_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let Ok(raw) = std::env::var("QUIP_OTSUKA_SCREEN_THREADS") else {
            return 1;
        };
        let parsed = raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|v| (1..=64).contains(v));
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_SCREEN_THREADS={raw:?} {}",
            if parsed.is_some() {
                "applied"
            } else {
                "ignored"
            }
        );
        parsed.unwrap_or(1)
    })
}

/// `QUIP_OTSUKA_FAST_GEN`, `0` or `1`, default off, read once. Needs the
/// local path (`QUIP_OTSUKA_SCREEN_THREADS` above 1).
pub(crate) fn fast_gen_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let Ok(raw) = std::env::var("QUIP_OTSUKA_FAST_GEN") else {
            return false;
        };
        let parsed = match raw.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        };
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_FAST_GEN={raw:?} {}",
            if parsed.is_some() {
                "applied"
            } else {
                "ignored"
            }
        );
        let on = parsed.unwrap_or(false);
        if on && screen_threads() <= 1 {
            eprintln!(
                "quip-otsuka: QUIP_OTSUKA_FAST_GEN needs QUIP_OTSUKA_SCREEN_THREADS above 1; fast generation off"
            );
            return false;
        }
        if on {
            eprintln!("quip-otsuka: fast generation on (local screen)");
        }
        on
    })
}

/// One salt in this many is also generated canonically and compared with
/// the cached generator, on top of every salt that passes stage 0.
const FAST_CHECK_EVERY: u64 = 64;

/// Local build (all6): the cached dimer generator of one topology and
/// parameter set, with its shared edge list.
pub(crate) struct FastGen {
    params: Vec<u8>,
    order_digest: [u8; 32],
    dimer: FastDimer,
    /// The instance edge order (`(min, max)` ascending), one allocation
    /// shared by every scored salt and kept job graph.
    edges: Arc<[(usize, usize)]>,
    trusted: AtomicBool,
    screened: AtomicU64,
}

impl FastGen {
    fn matches(&self, family: &FamilyLease) -> bool {
        self.params == family.params() && self.order_digest == *family.topology().order_digest()
    }

    /// Whether `instance` is exactly the cached generator's salt.
    fn agrees(&self, instance: &Instance, j: &[i32], tiles: &[[u32; 4]]) -> bool {
        instance.j() == j
            && instance.edges() == self.dimer.edges()
            && instance.h().len() == self.dimer.nodes()
            && instance.h().iter().all(|&h| h == 0)
            && instance.offset_milli() == 0
            && local_gauges(instance).is_some_and(|(meta, _)| meta == tiles)
    }

    fn distrust(&self, index: u64) {
        if self.trusted.swap(false, Ordering::Relaxed) {
            tracing::error!(
                salt_index = index,
                "cached dimer generator disagrees with the canonical one; local screen uses the canonical generator"
            );
        }
    }
}

/// Local build (all6): whether a pooled item's lease has stopped.
pub(crate) fn stale_item(item: &LaneItem) -> bool {
    match &item.dest {
        Dest::Local(reply) => reply.stop.load(Ordering::Relaxed),
        Dest::Session => false,
    }
}

/// Local build (all6): hand a pooled salt back to its lease thread unsolved
/// (refused, evicted or aged out of the pool). The lease thread pushes it
/// with no reads, as a stage-0 reject.
fn release_unsolved(item: LaneItem) {
    if let Dest::Local(reply) = item.dest {
        let result = StreamResult {
            job_id: item.job.job_id,
            outcome: StreamOutcome::Completed(Ok(Vec::new())),
            device_access_time_us: 0,
        };
        drop(reply.tx.send((reply.index, reply.offset, result)));
    }
}

/// What the screen made of one claimed salt. One per salt, consumed at
/// once, so the large `Keep` is not boxed.
#[allow(clippy::large_enum_variant)]
enum Screened {
    /// Generation skipped it (FAM-52) or broke its contract; already logged.
    Skip,
    /// Stage 0 screened it out.
    Reject,
    /// Stage 0 passed it, with its score when the screen scored it.
    Keep {
        instance: Instance,
        edges: Arc<[(usize, usize)]>,
        score: Option<f32>,
    },
}

fn log_fault(fault: SaltFault, index: u64) {
    // Codes 13 and 14 skip quietly (FAM-52); a construction that broke its
    // contract is logged, as core does.
    if let SaltFault::Defect(defect) = fault {
        tracing::error!(
            ?defect,
            salt_index = index,
            "family construction broke its contract; skipping the salt"
        );
    }
}

/// A local salt's result: its index, its instance offset, the lane's result.
type Done = (u64, i64, StreamResult);

/// Where a family result goes.
pub(crate) enum Dest {
    /// Core's session stream, stopped by the job's watermark.
    Session,
    /// A locally generated lease salt, back to its lease thread.
    Local(LocalReply),
}

/// The way back to a local salt's lease thread.
pub(crate) struct LocalReply {
    index: u64,
    offset: i64,
    stop: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<Done>,
}

impl LocalReply {
    /// The instance offset (reported energy = lane energy + offset).
    pub(crate) fn offset(&self) -> i64 {
        self.offset
    }

    /// The lease's stop flag, shared with a stage-2 group.
    pub(crate) fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }
}

impl Dest {
    /// Whether the job's round, lease or session has stopped.
    pub(crate) fn stopped(&self, cancel: &CancelToken, watermark: Option<u64>) -> bool {
        match self {
            Self::Session => cancel.is_cancelled(watermark),
            Self::Local(reply) => reply.stop.load(Ordering::Relaxed),
        }
    }

    /// Send `result`. False only when the session stream is closed; a lease
    /// thread that has gone just drops the result.
    pub(crate) fn send(self, result: StreamResult, out: &Sender<StreamResult>) -> bool {
        match self {
            Self::Session => out.blocking_send(result).is_ok(),
            Self::Local(reply) => {
                // An error means the lease thread has returned (stopped).
                drop(reply.tx.send((reply.index, reply.offset, result)));
                true
            }
        }
    }
}

/// Screen and lane counters for the periodic `otsuka screen` line.
#[derive(Default)]
pub(crate) struct ScreenStats {
    /// Lease salts through stage 0 (on either path).
    pub(crate) screened: AtomicU64,
    /// Of those, kept for a lane.
    pub(crate) kept: AtomicU64,
    queued: AtomicI64,
    /// Signalled when a lane takes an item, so a screen worker blocked on a
    /// full lane queue wakes on room instead of polling every 500 us.
    room: (Mutex<()>, Condvar),
    lane_wait_us: AtomicU64,
    generated: AtomicU64,
    generate_us: AtomicU64,
    last: Mutex<Option<Snapshot>>,
}

#[derive(Clone, Copy)]
struct Snapshot {
    at: Instant,
    screened: u64,
    kept: u64,
    lane_wait_us: u64,
    generated: u64,
    generate_us: u64,
}

impl ScreenStats {
    pub(crate) fn enqueued(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn dequeued(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        // Under the room lock: a worker holds it from its failed `try_send`
        // until it waits, so this wakeup cannot fall in between and be lost.
        let _room = self.room.0.lock().unwrap_or_else(PoisonError::into_inner);
        self.room.1.notify_one();
    }

    pub(crate) fn lane_waited(&self, waited: Duration) {
        let us = u64::try_from(waited.as_micros()).unwrap_or(u64::MAX);
        self.lane_wait_us.fetch_add(us, Ordering::Relaxed);
    }

    fn generated(&self, took: Duration) {
        let us = u64::try_from(took.as_micros()).unwrap_or(u64::MAX);
        self.generate_us.fetch_add(us, Ordering::Relaxed);
        self.generated.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self, at: Instant) -> Snapshot {
        Snapshot {
            at,
            screened: self.screened.load(Ordering::Relaxed),
            kept: self.kept.load(Ordering::Relaxed),
            lane_wait_us: self.lane_wait_us.load(Ordering::Relaxed),
            generated: self.generated.load(Ordering::Relaxed),
            generate_us: self.generate_us.load(Ordering::Relaxed),
        }
    }

    /// Log the `otsuka screen` line when [`LOG_EVERY`] has passed since the
    /// last one. Cheap and never blocks: a caller that finds another one
    /// logging returns.
    #[allow(clippy::cast_precision_loss)] // Rates for a log line.
    pub(crate) fn maybe_log(&self, lanes: usize) {
        let mut last = match self.last.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        let now = Instant::now();
        let Some(prev) = *last else {
            *last = Some(self.snapshot(now));
            return;
        };
        let secs = now.duration_since(prev.at).as_secs_f64();
        if secs < LOG_EVERY.as_secs_f64() {
            return;
        }
        let now_snap = self.snapshot(now);
        let generated = now_snap.generated - prev.generated;
        let generate_ms = if generated == 0 {
            0.0
        } else {
            (now_snap.generate_us - prev.generate_us) as f64 / generated as f64 / 1000.0
        };
        tracing::info!(
            threads = screen_threads(),
            screened_per_s = (now_snap.screened - prev.screened) as f64 / secs,
            kept_per_s = (now_snap.kept - prev.kept) as f64 / secs,
            queue = self.queued.load(Ordering::Relaxed).max(0),
            lane_wait_frac = (now_snap.lane_wait_us - prev.lane_wait_us) as f64
                / 1e6
                / (secs * lanes.max(1) as f64),
            generate_ms,
            "otsuka screen"
        );
        *last = Some(now_snap);
    }
}

/// CPU slots shared by every live lease's screen workers, so two staged
/// leases never run more than `THREADS` generations at once.
///
/// Local build (all5, agent X): the slots go to the OLDEST lease that still
/// has unclaimed salts (FIFO). In all4 every live lease's workers raced for
/// them; under T6 the session kept ~340 leases of 4500 salts live (credit
/// window sized from the SA stream width), none ever finished, the
/// coordinator's salts/s stayed 0, and ~2400 `quip-lease` threads polled
/// this condvar every 10 ms. FIFO lets one lease claim all its salts and
/// finish, so `LeaseDone` flows and the coordinator sizes leases from a real
/// rate. A lease waiting for its turn holds one sleeping thread.
pub(crate) struct ScreenTokens {
    free: Mutex<usize>,
    ready: Condvar,
    /// Leases with unclaimed salts, by arrival sequence.
    order: Mutex<std::collections::BTreeSet<u64>>,
    next_seq: AtomicU64,
}

/// One held slot, returned on drop.
struct Token<'a>(&'a ScreenTokens);

impl Drop for Token<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        // notify_all: the waiter whose turn it is may not be the one woken.
        self.0.ready.notify_all();
    }
}

/// A lease's place in the FIFO, retired on drop.
struct Turn<'a> {
    tokens: &'a ScreenTokens,
    seq: u64,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.tokens.retire(self.seq);
    }
}

/// How long a waiting lease or worker sleeps between stop checks. Wakeups
/// come from the condvar; this only bounds how late a stop is seen.
const TOKEN_POLL: Duration = Duration::from_millis(100);

impl ScreenTokens {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            free: Mutex::new(slots.max(1)),
            ready: Condvar::new(),
            order: Mutex::new(std::collections::BTreeSet::new()),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Register a lease; returns its FIFO sequence number.
    fn enter(&self) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.order
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(seq);
        seq
    }

    /// The lease has no unclaimed salts left (or stopped): the next one may
    /// go. Idempotent.
    fn retire(&self, seq: u64) {
        let removed = self
            .order
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&seq);
        if removed {
            // Under `free`, which every waiter holds from its turn check
            // until it waits, so the wakeup is never lost.
            let _free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
            self.ready.notify_all();
        }
    }

    /// [`Self::enter`] with a guard that retires the lease on drop, so a
    /// panic or an early return never leaves a dead head blocking every
    /// later lease (the sampler outlives the session).
    fn turn(&self) -> Turn<'_> {
        Turn {
            tokens: self,
            seq: self.enter(),
        }
    }

    fn my_turn(&self, seq: u64) -> bool {
        self.order
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .first()
            .is_none_or(|&head| head >= seq)
    }

    /// Block until `seq` is the oldest lease with work, or `stopped`.
    /// Called by the lease thread BEFORE it spawns its workers, so a waiting
    /// lease costs one sleeping thread, not THREADS+1.
    fn wait_turn(&self, seq: u64, stopped: &dyn Fn() -> bool) -> bool {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if stopped() {
                return false;
            }
            if self.my_turn(seq) {
                return true;
            }
            free = self
                .ready
                .wait_timeout(free, TOKEN_POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Take a slot for lease `seq`, or `None` once `stopped`. A lease older
    /// than the head (it retired its turn after claiming its last salt) still
    /// gets slots for the salts it already claimed.
    fn acquire(&self, seq: u64, stopped: &dyn Fn() -> bool) -> Option<Token<'_>> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if stopped() {
                return None;
            }
            if *free > 0 && self.my_turn(seq) {
                *free -= 1;
                return Some(Token(self));
            }
            free = self
                .ready
                .wait_timeout(free, TOKEN_POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

fn seed_base() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    #[allow(clippy::cast_possible_truncation)]
    let base = nanos as u64;
    base ^ 0x9E37_79B9_7F4A_7C15
}

/// A per-salt seed (splitmix64 of the lease base and the index).
fn salt_seed(base: u64, index: u64) -> u64 {
    let mut z = base.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

fn local_job_id(index: u64) -> Vec<u8> {
    let mut id = b"local:".to_vec();
    id.extend_from_slice(&index.to_le_bytes());
    id
}

/// One lease's shared worker state.
struct LeaseWork<'a> {
    lease: &'a Lease,
    family: &'a FamilyLease,
    params: &'a SampleParams,
    sink: &'a LeaseSink,
    lanes: &'a SyncSender<LaneItem>,
    seeds: u64,
    next: AtomicU64,
    outstanding: AtomicU64,
    stop: Arc<AtomicBool>,
    /// The lease's FIFO sequence ([`ScreenTokens::enter`]).
    seq: u64,
}

impl LeaseWork<'_> {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn halt(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl<D: ClusterBackend, S: Sampler> OtsukaSampler<D, S> {
    /// [`Sampler::sample_lease`] for `QUIP_OTSUKA_SCREEN_THREADS` above 1.
    ///
    /// # Errors
    /// A fatal lane error (the session then ends), or `Capacity` for a
    /// family lease this sampler did not declare.
    #[allow(clippy::too_many_lines)] // One lease thread, start to retire.
    pub(crate) fn sample_lease_local(
        &self,
        lease: &Lease,
        topology: &TopologyView,
        params: &SampleParams,
        sink: &LeaseSink,
    ) -> Result<(), SampleError> {
        let Some(family) = lease.family() else {
            return self.sample_legacy_lease(lease, topology, params, sink);
        };
        if family.family_id() != TWISTED_DIMER_V1.family_id
            || family.revision() != TWISTED_DIMER_V1.revision
        {
            tracing::error!(
                family = family.family_id(),
                revision = family.revision(),
                "local lease of a family this sampler does not generate; refused"
            );
            return Err(SampleError::Capacity);
        }
        let Some(lanes) = self.wait_lane_queue(sink) else {
            return Ok(());
        };
        // FIFO turn: a lease that is not the oldest with unclaimed salts
        // waits here, one sleeping thread, instead of spawning its workers.
        // The guard retires the turn however this function ends.
        let turn = self.screen_tokens.turn();
        if !self
            .screen_tokens
            .wait_turn(turn.seq, &|| sink.is_stopped())
        {
            // all7: a lease stopped before its turn (a round change).
            self.note_cut();
            return Ok(());
        }
        let work = LeaseWork {
            lease,
            family,
            params,
            sink,
            lanes: &lanes,
            seeds: seed_base(),
            next: AtomicU64::new(0),
            outstanding: AtomicU64::new(0),
            stop: Arc::new(AtomicBool::new(false)),
            seq: turn.seq,
        };
        let threads = screen_threads().max(1);
        let running = AtomicUsize::new(threads);
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<Done>();
        let mut fatal = None;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let reply = reply_tx.clone();
                let (work, running) = (&work, &running);
                scope.spawn(move || {
                    self.screen_worker(work, &reply);
                    running.fetch_sub(1, Ordering::AcqRel);
                });
            }
            drop(reply_tx);
            loop {
                if sink.is_stopped() {
                    work.halt();
                }
                // Stage 2 never holds a result near the live session target.
                if let Some(stage2) = self.stage2.as_ref() {
                    stage2.set_live_target(sink.target_energy_milli());
                }
                // all7: nor does the deep lane, which also probes against it.
                if let Some(deep) = self.deep.as_ref() {
                    deep.set_live_target(sink.target_energy_milli());
                }
                match reply_rx.recv_timeout(STOP_POLL) {
                    Ok((index, offset, result)) => {
                        work.outstanding.fetch_sub(1, Ordering::AcqRel);
                        match result.outcome {
                            StreamOutcome::Completed(Ok(mut reads)) if !work.stopped() => {
                                // FAM-33: the reported energy includes the
                                // instance offset, as core adds on its path.
                                for read in &mut reads {
                                    read.energy_milli = read.energy_milli.saturating_add(offset);
                                }
                                if sink.push(index, reads).is_err() {
                                    work.halt();
                                }
                            }
                            StreamOutcome::Completed(Err(error)) if error.is_fatal() => {
                                fatal.get_or_insert(error);
                                work.halt();
                            }
                            StreamOutcome::Completed(Err(error)) => {
                                tracing::warn!(%error, "lease salt failed");
                            }
                            _ => {}
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
                if running.load(Ordering::Acquire) == 0
                    && (work.stopped() || work.outstanding.load(Ordering::Acquire) == 0)
                {
                    break;
                }
            }
            // Queued salts of this lease are dropped by the lanes unsolved.
            work.halt();
        });
        // all7: a lease stopped with salts unclaimed or unsolved was cut (a
        // round change): the deep lane's round starts again.
        if sink.is_stopped()
            && (work.next.load(Ordering::Relaxed) < work.lease.salt_count()
                || work.outstanding.load(Ordering::Acquire) > 0)
        {
            self.note_cut();
        }
        drop(turn);
        fatal.map_or(Ok(()), Err)
    }

    /// all7: tell the deep lane a lease was cut (its round restarts).
    fn note_cut(&self) {
        if let Some(deep) = self.deep.as_ref() {
            deep.new_round(Instant::now());
        }
    }

    /// The lanes' queue, which exists while the stream session runs its
    /// lanes; `None` once the lease stops first.
    fn wait_lane_queue(&self, sink: &LeaseSink) -> Option<SyncSender<LaneItem>> {
        loop {
            if sink.is_stopped() {
                return None;
            }
            let queue = self
                .lane_queue
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if queue.is_some() {
                return queue;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One screen worker: claim, generate, screen, and push or queue salts
    /// until the lease runs out or stops.
    fn screen_worker(&self, work: &LeaseWork<'_>, reply: &std::sync::mpsc::Sender<Done>) {
        let lane_count = 1 + self.extra.len();
        let count = work.lease.salt_count();
        let stopped = || work.stopped();
        // all6: the cached generator, when on and it fits this family.
        let fast = self.fast_gen_for(work.family);
        loop {
            if work.stopped() {
                break;
            }
            let index = work.next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                // Every salt claimed: hand the CPU slots to the next lease
                // while this one's anneals finish.
                self.screen_tokens.retire(work.seq);
                break;
            }
            let Some(token) = self.screen_tokens.acquire(work.seq, &stopped) else {
                break;
            };
            let started = Instant::now();
            let screened = fast
                .as_deref()
                .and_then(|fast| self.fast_screen(fast, work, index))
                .unwrap_or_else(|| self.canonical_screen(work, index));
            self.screen_stats.generated(started.elapsed());
            // Back-pressure: a rejected salt frees its slot now, a kept one
            // holds it through `enqueue`. With the lanes' queue full the
            // screen pauses at THREADS parked instances. In all4 every worker
            // parked one (~1.5 MB) and polled, kept salts piled up at ~1/s
            // above what the lanes took (RSS +~170 MB/min, ~20 cores busy)
            // and were dropped unsolved at the next round.
            match screened {
                Screened::Skip => {
                    drop(token);
                    if work.sink.skip(index).is_err() {
                        work.halt();
                    }
                }
                Screened::Reject => {
                    drop(token);
                    if work.sink.push(index, Vec::new()).is_err() {
                        work.halt();
                    }
                }
                Screened::Keep {
                    instance,
                    edges,
                    score,
                } => {
                    // all6: a scored salt goes to the pool and the worker
                    // screens on; the pool decides what the lanes get.
                    if let (Some(pool), Some(score)) = (self.pool.as_ref(), score) {
                        if pool.would_admit(score) {
                            let item = Self::lane_item_for(
                                work,
                                reply,
                                index,
                                instance,
                                edges,
                                Some(score),
                            );
                            work.outstanding.fetch_add(1, Ordering::AcqRel);
                            let spill = pool.offer(item, score, Instant::now(), &stale_item);
                            drop(token);
                            for item in spill.release {
                                release_unsolved(item);
                            }
                            drop(spill.stale);
                        } else {
                            drop(token);
                            pool.refused();
                            if work.sink.push(index, Vec::new()).is_err() {
                                work.halt();
                            }
                        }
                        pool.maybe_log(self.screen_stats.screened.load(Ordering::Relaxed));
                    } else {
                        let item = Self::lane_item_for(work, reply, index, instance, edges, score);
                        work.outstanding.fetch_add(1, Ordering::AcqRel);
                        if !self.enqueue(work.lanes, item, &stopped) {
                            work.outstanding.fetch_sub(1, Ordering::AcqRel);
                            work.halt();
                        }
                        drop(token);
                    }
                }
            }
            self.screen_stats.maybe_log(lane_count);
        }
    }

    /// The lane item of a kept local salt.
    fn lane_item_for(
        work: &LeaseWork<'_>,
        reply: &std::sync::mpsc::Sender<Done>,
        index: u64,
        instance: Instance,
        edges: Arc<[(usize, usize)]>,
        score: Option<f32>,
    ) -> LaneItem {
        let graph = IsingGraph {
            h: instance
                .h()
                .iter()
                .map(|&m| f64::from(m) / 1000.0)
                .collect(),
            j: instance
                .j()
                .iter()
                .map(|&m| f64::from(m) / 1000.0)
                .collect(),
            edges,
        };
        let offset = instance.offset_milli();
        LaneItem {
            job: StreamJob {
                job_id: local_job_id(index),
                graph,
                params: SampleParams {
                    seed: salt_seed(work.seeds, index),
                    ..work.params.clone()
                },
                watermark: None,
            },
            family: FamilyRef::Local {
                instance: Arc::new(instance),
                contract: work.family.contract(),
            },
            stage0: score,
            dest: Dest::Local(LocalReply {
                index,
                offset,
                stop: Arc::clone(&work.stop),
                tx: reply.clone(),
            }),
            rerun: None,
            deep: None,
            ease: None,
        }
    }

    /// The all5 screen of one salt: the canonical instance, then stage 0.
    fn canonical_screen(&self, work: &LeaseWork<'_>, index: u64) -> Screened {
        let instance = match work.family.generate(work.lease.nonce(index)) {
            Ok(instance) => instance,
            Err(fault) => {
                log_fault(fault, index);
                return Screened::Skip;
            }
        };
        // The job graph exactly as core's `family_graph` builds it for f64.
        let edges: Arc<[(usize, usize)]> = instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        match self.stage0_decide(&instance, &edges, true) {
            Gate0::Reject => Screened::Reject,
            Gate0::Pass(score) => Screened::Keep {
                instance,
                edges,
                score,
            },
        }
    }

    /// all6: stage 0 on the cached generator's couplings; the canonical
    /// instance only for a salt that passes (and one in
    /// [`FAST_CHECK_EVERY`], as a check). `None` sends the salt the
    /// canonical way: the cached generator is off or cannot build it.
    fn fast_screen(&self, fast: &FastGen, work: &LeaseWork<'_>, index: u64) -> Option<Screened> {
        if !fast.trusted.load(Ordering::Relaxed) {
            return None;
        }
        let nonce = work.lease.nonce(index);
        let (j, tiles) = fast.dimer.couplings(nonce)?;
        let verdict = self.stage0_decide_raw(fast.dimer.nodes(), &fast.edges, &j, &tiles);
        let check = fast
            .screened
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(FAST_CHECK_EVERY);
        if matches!(verdict, Gate0::Reject) && !check {
            return Some(Screened::Reject);
        }
        let instance = match work.family.generate(nonce) {
            Ok(instance) => instance,
            Err(fault) => {
                // The cached generator built a salt the canonical one skips.
                fast.distrust(index);
                log_fault(fault, index);
                return Some(Screened::Skip);
            }
        };
        let agrees = fast.agrees(&instance, &j, &tiles);
        if !agrees {
            fast.distrust(index);
        }
        Some(match verdict {
            Gate0::Reject => Screened::Reject,
            Gate0::Pass(score) => {
                let edges = if agrees {
                    Arc::clone(&fast.edges)
                } else {
                    instance
                        .edges()
                        .iter()
                        .map(|&(u, v)| (u as usize, v as usize))
                        .collect()
                };
                Screened::Keep {
                    instance,
                    edges,
                    score,
                }
            }
        })
    }

    /// all6: the cached generator for `family`, built on first use for each
    /// topology and parameter set; `None` when fast generation is off or the
    /// family has no cached generator.
    fn fast_gen_for(&self, family: &FamilyLease) -> Option<Arc<FastGen>> {
        let cache = self.fast_gen.as_ref()?;
        let mut cached = cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(fast) = cached.as_ref().filter(|f| f.matches(family)) {
            return Some(Arc::clone(fast));
        }
        let started = Instant::now();
        let Some(dimer) = FastDimer::new(
            family.family_id(),
            family.revision(),
            family.params(),
            family.topology(),
        ) else {
            tracing::warn!("no cached generator for this family lease; local screen uses the canonical generator");
            *cached = None;
            return None;
        };
        let edges: Arc<[(usize, usize)]> = dimer
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        let fast = Arc::new(FastGen {
            params: family.params().to_vec(),
            order_digest: *family.topology().order_digest(),
            dimer,
            edges,
            trusted: AtomicBool::new(true),
            screened: AtomicU64::new(0),
        });
        tracing::info!(
            nodes = fast.dimer.nodes(),
            edges = fast.edges.len(),
            build_ms = started.elapsed().as_millis(),
            "otsuka fast generator built"
        );
        *cached = Some(Arc::clone(&fast));
        Some(fast)
    }

    /// Queue `item` for the lanes, waiting while the queue is full. False
    /// when stopped first or the lanes are gone.
    fn enqueue(
        &self,
        lanes: &SyncSender<LaneItem>,
        mut item: LaneItem,
        stopped: &dyn Fn() -> bool,
    ) -> bool {
        let (room, wake) = &self.screen_stats.room;
        // Held from each `try_send` until the wait releases it, so a lane's
        // `dequeued` wakeup (sent under this lock) cannot be missed.
        let mut guard = room.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            match lanes.try_send(item) {
                Ok(()) => {
                    drop(guard);
                    self.screen_stats.enqueued();
                    return true;
                }
                Err(TrySendError::Full(back)) => {
                    if stopped() {
                        return false;
                    }
                    item = back;
                    // Sleep until a lane takes an item (all4 polled every
                    // 500 us); the timeout only bounds how late a stop is seen.
                    guard = wake
                        .wait_timeout(guard, STOP_POLL)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
                Err(TrySendError::Disconnected(_)) => return false,
            }
        }
    }

    /// A legacy (non-family) lease under local generation: drawn and solved
    /// one salt at a time on the SA sampler. Correct but slow; the chain
    /// binds a family to every live challenge today.
    fn sample_legacy_lease(
        &self,
        lease: &Lease,
        topology: &TopologyView,
        params: &SampleParams,
        sink: &LeaseSink,
    ) -> Result<(), SampleError> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "legacy lease with QUIP_OTSUKA_SCREEN_THREADS above 1: solved serially on the SA sampler"
            );
        }
        let seeds = seed_base();
        let edges: Arc<[(usize, usize)]> = topology.edges.clone().into();
        for index in 0..lease.salt_count() {
            if sink.is_stopped() {
                break;
            }
            let (h, j) = topology
                .draw(lease.nonce(index))
                .map_err(|e| SampleError::DeviceFault(format!("legacy lease draw: {e}")))?;
            let graph = IsingGraph {
                h: h.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
                j: j.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
                edges: Arc::clone(&edges),
            };
            let salt_params = SampleParams {
                seed: salt_seed(seeds, index),
                ..params.clone()
            };
            match self.sa.sample(&graph, &salt_params) {
                Ok(mut reads) => {
                    reads.sort_by_key(|r| r.energy_milli);
                    if sink.push(index, reads).is_err() {
                        break;
                    }
                }
                Err(error) if error.is_fatal() => return Err(error),
                Err(error) => tracing::warn!(%error, "legacy lease salt failed"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salt_seeds_differ_and_are_never_zero() {
        let base = 0x1234_5678_9ABC_DEF0;
        let seeds: std::collections::HashSet<u64> =
            (0..10_000).map(|i| salt_seed(base, i)).collect();
        assert_eq!(seeds.len(), 10_000);
        assert!(!seeds.contains(&0));
    }

    #[test]
    fn tokens_bound_concurrency_and_stop_waiting() {
        let tokens = ScreenTokens::new(2);
        let turn = tokens.turn();
        let a = tokens.acquire(turn.seq, &|| false).unwrap();
        let _b = tokens.acquire(turn.seq, &|| false).unwrap();
        assert!(
            tokens.acquire(turn.seq, &|| true).is_none(),
            "a stop ends the wait"
        );
        drop(a);
        assert!(
            tokens.acquire(turn.seq, &|| false).is_some(),
            "a returned slot is reusable"
        );
    }

    #[test]
    fn tokens_go_to_the_oldest_lease_first() {
        let tokens = ScreenTokens::new(4);
        let first = tokens.turn();
        let second = tokens.turn();
        assert!(tokens.wait_turn(first.seq, &|| false));
        assert!(
            !tokens.wait_turn(second.seq, &|| true),
            "a younger lease waits (here: until stopped)"
        );
        assert!(
            tokens.acquire(second.seq, &|| true).is_none(),
            "no slot for a younger lease while the head has salts"
        );
        // The head claims its last salt and retires; its in-flight claims
        // still get slots, and the next lease's turn begins.
        tokens.retire(first.seq);
        assert!(tokens.acquire(first.seq, &|| false).is_some());
        assert!(tokens.wait_turn(second.seq, &|| false));
        let third = tokens.turn();
        drop(second);
        assert!(
            tokens.wait_turn(third.seq, &|| false),
            "a dropped turn (early return, panic) never blocks later leases"
        );
        drop(first);
    }

    #[test]
    fn a_waiting_lease_wakes_when_the_head_retires() {
        let tokens = Arc::new(ScreenTokens::new(1));
        let head = tokens.turn();
        let next_seq = tokens.enter();
        let waiter = {
            let tokens = Arc::clone(&tokens);
            std::thread::spawn(move || {
                assert!(tokens.wait_turn(next_seq, &|| false));
                Instant::now()
            })
        };
        std::thread::sleep(Duration::from_millis(20));
        let retired = Instant::now();
        drop(head);
        let delay = waiter.join().unwrap().saturating_duration_since(retired);
        assert!(
            delay < Duration::from_millis(60),
            "woken by the retire, not by the {TOKEN_POLL:?} poll: {delay:?}"
        );
        tokens.retire(next_seq);
    }

    #[test]
    fn local_job_ids_carry_the_index() {
        assert_eq!(&local_job_id(7)[..6], b"local:");
        assert_eq!(local_job_id(7)[6..], 7_u64.to_le_bytes());
    }
}
