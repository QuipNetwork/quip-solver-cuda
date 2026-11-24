//! Lease-level sampling: many CPU threads generate salts and rank them by a structural score, and one GPU thread
//! runs the screening pipeline only on the salts that rank best. Every other salt is counted with `screen`, so it
//! costs the coordinator no result and no attempt record.

use super::{ClusterBackend, Fam, OtsukaSampler, ScreenCfg};
use crate::cluster::{PreparedGraph, Weights};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{IsingGraph, Lease, LeaseSink, SampleError, SampleParams, Sampler, SamplerResult, TopologyView};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, TrySendError};
use std::io::Write;
use std::sync::Mutex;
use std::time::Duration;
use crate::cuda_device::CudaDevice;
use crate::nvml_gov::UtilGovernor;
use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};

// Linear model fitted on 300 harness instances (see ceiling/ notes): predicted stage-B energy from triangle-class
// densities (per edge). Lower is better.
pub(crate) const MU: [f64; 9] = [1.768387697e-02, 1.753626664e-02, 2.590771266e-01, 2.592588959e-01, 6.250523286e-02, 6.240551776e-02, 1.603363568e-03, 1.861428717e-01, 3.485132130e-01];
pub(crate) const SD: [f64; 9] = [4.964411289e-04, 4.991529657e-04, 1.749831546e-03, 1.785575513e-03, 8.954755659e-04, 8.489335426e-04, 1.837574141e-04, 1.820795295e-04, 1.794829881e-03];
pub(crate) const W: [f64; 9] = [-2.303337563e-02, 4.506261176e-02, -8.540215007e-01, -4.184292146e-01, 9.857215758e-01, 1.333054746e+00, 2.224419601e-01, 9.623952112e-01, -8.609915459e-02];

/// The coordinator issues many large leases at once; the GPU and the generator pool serve one at a time.
static LEASE_GATE: Mutex<()> = Mutex::new(());

/// Calibration log: one JSON line per screened salt that reached stage B, per forced exploration salt, and one in
/// ten of the rest, so the stage energies and the structural score can be fitted on the live family offline.
static CALIB: Mutex<Option<(std::io::BufWriter<std::fs::File>, u64, u32)>> = Mutex::new(None);
const CALIB_DIR: &str = "/data/calib";
const CALIB_MAX_BYTES: u64 = 250 << 20;

static SCR_BEST: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MAX);
static SCR_N: AtomicU64 = AtomicU64::new(0);

/// Remember the best energy a screened (non-deep) salt reached since the last round change.
#[cfg(test)]
pub(crate) fn ising_graph_for_timing(inst: &Instance) -> IsingGraph {
    ising_graph(inst)
}

fn scr_note(best: i64) {
    SCR_BEST.fetch_min(best, Ordering::Relaxed);
    SCR_N.fetch_add(1, Ordering::Relaxed);
}

/// Take and reset the screening summary (best energy, salt count).
fn scr_take() -> (i64, u64) {
    (SCR_BEST.swap(i64::MAX, Ordering::Relaxed), SCR_N.swap(0, Ordering::Relaxed))
}

fn calib_write(line: &str) {
    let mut g = CALIB.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if g.is_none() {
        let _ = std::fs::create_dir_all(CALIB_DIR);
        if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(format!("{CALIB_DIR}/calib.jsonl")) {
            let size = f.metadata().map_or(0, |m| m.len());
            *g = Some((std::io::BufWriter::new(f), size, 0));
        } else {
            return;
        }
    }
    if let Some((w, bytes, n)) = g.as_mut() {
        if *bytes > CALIB_MAX_BYTES {
            return;
        }
        let _ = w.write_all(line.as_bytes());
        let _ = w.write_all(b"\n");
        *bytes += line.len() as u64 + 1;
        *n += 1;
        if *n % 64 == 0 || line.starts_with("{\"cancelled\"") {
            let _ = w.flush();
        }
    }
}

/// Write the family inputs (parameters and base topology) once per descriptor, so offline tools can regenerate any
/// logged nonce.
fn calib_family_dump(family: &quip_solver_core::FamilyLease) {
    let digest: String = family.descriptor_digest().iter().take(6).map(|b| format!("{b:02x}")).collect();
    let path = format!("{CALIB_DIR}/family-{digest}.json");
    if std::path::Path::new(&path).exists() {
        return;
    }
    let _ = std::fs::create_dir_all(CALIB_DIR);
    let params: String = family.params().iter().map(|b| format!("{b:02x}")).collect();
    let topo = family.topology();
    let body = serde_json::json!({
        "params_hex": params,
        "nodes": topo.node_ids(),
        "edges": topo.edges(),
    });
    let _ = std::fs::write(path, body.to_string());
}

/// Structural score of an instance: lower predicts a lower ground energy.
/// Quick structural scorer: lean coupling generator plus the triangle table. It gives the same score as
/// `structural_score` of the canonical instance, at a fraction of the cost.
pub(crate) struct Lean {
    gen: crate::cluster::lean_gen::LeanGen,
    tri: crate::cluster::lean_gen::Triangles,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Option<crate::cluster::lean_gen::FastScratch>> = const { std::cell::RefCell::new(None) };
}

impl Lean {
    fn score(&self, nonce: &[u8; 32]) -> Option<f64> {
        let gt = self.gen.gpu_tables()?;
        SCRATCH.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().map_or(true, |sc| sc.flags.len() != gt.edges) {
                *slot = Some(crate::cluster::lean_gen::FastScratch::new(gt));
            }
            let sc = slot.as_mut()?;
            if !self.gen.fast_flags(nonce, sc) {
                return None;
            }
            let f = self.tri.features_flags(&sc.flags);
            Some((0..9).map(|i| W[i] * (f[i] - MU[i]) / SD[i]).sum())
        })
    }
}

static LEAN: Mutex<Option<([u8; 32], Option<std::sync::Arc<Lean>>)>> = Mutex::new(None);

/// The quick scorer of this family, built once per descriptor. `None` when the family is not the twisted ice dimer or
/// the parameters are not understood; the canonical path is used then.
fn lean_for(family: &quip_solver_core::FamilyLease) -> Option<std::sync::Arc<Lean>> {
    if std::env::var("QS_LEAN").map_or(false, |v| v == "0") {
        return None;
    }
    let digest: [u8; 32] = *family.descriptor_digest();
    let mut g = LEAN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((d, l)) = g.as_ref() {
        if *d == digest {
            return l.clone();
        }
    }
    let params = family.params();
    let mut tlv = std::collections::BTreeMap::new();
    let mut at = 1usize;
    while at + 6 <= params.len() {
        let tag = u16::from_le_bytes([params[at], params[at + 1]]);
        let len = u32::from_le_bytes([params[at + 2], params[at + 3], params[at + 4], params[at + 5]]) as usize;
        if at + 6 + len > params.len() {
            break;
        }
        let mut value = 0u64;
        for (k, b) in params[at + 6..at + 6 + len].iter().enumerate() {
            value |= u64::from(*b) << (8 * k.min(7));
        }
        tlv.insert(tag, value);
        at += 6 + len;
    }
    let built = (|| {
        let base = family.topology();
        let gen = crate::cluster::lean_gen::LeanGen::new(base, digest, *tlv.get(&1)? as u16, *tlv.get(&2)? as u8, *tlv.get(&3)? as i32, *tlv.get(&4)? as usize)?;
        let tri = crate::cluster::lean_gen::Triangles::new(base.node_ids().len(), &gen.edges);
        let lean = Lean { gen, tri };
        // Self-check against the canonical generator before trusting the quick path.
        for k in 1..=4u8 {
            let mut nonce = [k; 32];
            nonce[31] = k.wrapping_mul(37);
            let Ok(inst) = family.generate(nonce) else { continue };
            let want = structural_score(&inst);
            let same = lean.score(&nonce).map_or(false, |got| (got - want).abs() < 1e-9);
            if !same || lean.gen.couplings(&nonce).as_deref() != Some(inst.j()) || lean.gen.edges != inst.edges() {
                tracing::warn!("quick structural scorer disagrees with the canonical generator; disabled");
                return None;
            }
        }
        Some(std::sync::Arc::new(lean))
    })();
    tracing::info!(ok = built.is_some(), "quick structural scorer");
    *g = Some((digest, built.clone()));
    built
}

pub(crate) fn structural_score(inst: &Instance) -> f64 {
    score_raw(inst.h().len(), inst.edges(), inst.j())
}

pub(crate) fn features_raw(n: usize, edges: &[(u32, u32)], js: &[i32]) -> [f64; 9] {
    let m = edges.len().max(1);
    // Adjacency with the coupling of each neighbour, sorted by neighbour id for merge intersection.
    let mut adj: Vec<Vec<(u32, i32)>> = vec![Vec::new(); n];
    for (&(a, b), &j) in edges.iter().zip(js.iter()) {
        adj[a as usize].push((b, j));
        adj[b as usize].push((a, j));
    }
    for row in &mut adj {
        row.sort_unstable_by_key(|x| x.0);
    }
    let mut cnt = [0u32; 8]; // index = ns * 2 + frustration flag (1 when the sign product is positive)
    let mut weak_pos = 0u32;
    for (&(a, b), &jab) in edges.iter().zip(js.iter()) {
        if jab != 0 && jab.abs() < 500 && jab > 0 {
            weak_pos += 1;
        }
        let (a, b, jab) = if a < b { (a as usize, b, jab) } else { (b as usize, a, jab) };
        // Triangles a < b < c.
        let ra = &adj[a];
        let rb = &adj[b as usize];
        let (mut i, mut k) = (0usize, 0usize);
        while i < ra.len() && k < rb.len() {
            let (na, ja) = ra[i];
            let (nb, jb) = rb[k];
            if na < nb {
                i += 1;
            } else if na > nb {
                k += 1;
            } else {
                if na > b {
                    let ns = usize::from(jab.abs() >= 500) + usize::from(ja.abs() >= 500) + usize::from(jb.abs() >= 500);
                    let nonpos = usize::from(jab <= 0) + usize::from(ja <= 0) + usize::from(jb <= 0);
                    let pos = nonpos % 2 == 0;
                    cnt[ns * 2 + usize::from(pos)] += 1;
                }
                i += 1;
                k += 1;
            }
        }
    }
    let mf = m as f64;
    let mut f = [0f64; 9];
    for i in 0..8 {
        f[i] = f64::from(cnt[i]) / mf;
    }
    f[8] = f64::from(weak_pos) / mf;
    f
}

pub(crate) fn score_raw(n: usize, edges: &[(u32, u32)], js: &[i32]) -> f64 {
    let f = features_raw(n, edges, js);
    (0..9).map(|i| W[i] * (f[i] - MU[i]) / SD[i]).sum()
}

/// Screening state that survives a restart: the stage rank windows, the structural-score window and its pass
/// fraction. Without it every restart re-learns the thresholds from priors, and the slowest stage needs ~40 minutes
/// to fill its window.

// ---- MSA first stage ------------------------------------------------------------------------------------------------
// The in-tree multi-spin-coded annealer (one SM per nonce, `stream_width` different models in flight) screens the salts that
// pass S0. Measured on 100 live instances against the bottom energy of a deep run: msa 128x4096 correlates 0.33 at ~86 nonce/s,
// the cluster annealer's stage A (256x64) only 0.10 at a similar cost. The survivors go on to the cluster stages.

static MSA_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the MSA first stage is used (QS_MSA=0 turns it off; a device that cannot open it turns it off too).
pub(crate) fn msa_mode() -> bool {
    std::env::var("QS_MSA").map_or(false, |v| v == "1") && !MSA_FAILED.load(Ordering::Relaxed)
}

fn msa_dev() -> Option<&'static (CudaDevice, UtilGovernor)> {
    static D: std::sync::OnceLock<Option<(CudaDevice, UtilGovernor)>> = std::sync::OnceLock::new();
    let r = D
        .get_or_init(|| {
            let d = CudaDevice::open_with_nodes(0, crate::KernelKind::Msa, 4600).ok()?;
            let g = UtilGovernor::start(&d.pci_bus_id, 100, false);
            Some((d, g))
        })
        .as_ref();
    if r.is_none() {
        MSA_FAILED.store(true, Ordering::Relaxed);
        tracing::warn!("MSA first stage unavailable; using the cluster stages only");
    }
    r
}

/// MSA energy of the salt the funnel is processing and whether it was let through at random (to measure the
/// MSA→deep correlation without selection bias). Only one MSA worker exists, so plain statics are enough.
static MSA_CUR: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MAX);
static MSA_X: AtomicBool = AtomicBool::new(false);

/// One MSA stream session over a fixed job list; returns `(job id, best energy)` for every job that completed.
fn msa_session(dev: &'static CudaDevice, gov: &'static UtilGovernor, jobs: Vec<(u32, IsingGraph, SampleParams)>, stopped: &dyn Fn() -> bool) -> Vec<(u32, Option<i64>)> {
    let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(64);
    let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(64);
    let token = CancelToken::default();
    let halt = AtomicBool::new(false);
    let mut got = Vec::with_capacity(jobs.len());
    std::thread::scope(|sc| {
        let tok = token.clone();
        sc.spawn(move || crate::streaming::run_stream(dev, crate::KernelKind::Msa, job_rx, res_tx, tok, gov, crate::streaming::StallConfig::default()));
        let halt_r = &halt;
        sc.spawn(move || {
            for (id, graph, params) in jobs {
                if halt_r.load(Ordering::Relaxed) || job_tx.blocking_send(StreamJob { job_id: id.to_le_bytes().to_vec(), graph, params, watermark: Some(1) }).is_err() {
                    break;
                }
            }
        });
        loop {
            match res_rx.try_recv() {
                Ok(r) => {
                    let id = u32::from_le_bytes(r.job_id[..4].try_into().unwrap_or([0; 4]));
                    let e = if let StreamOutcome::Completed(Ok(rs)) = r.outcome { rs.iter().map(|x| x.energy_milli).min() } else { None };
                    got.push((id, e));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    if stopped() {
                        halt.store(true, Ordering::Relaxed);
                        token.cancel_through(1);
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
    });
    got
}

static MSA_WIN: Mutex<VecDeque<i64>> = Mutex::new(VecDeque::new());

/// A salt passes the MSA stage when its energy ranks in the best `1/q` of the rolling window of recent MSA energies.
fn msa_passes(e: i64, q: usize) -> bool {
    let mut w = MSA_WIN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let n = w.len();
    let rank = w.iter().filter(|&&x| x < e).count();
    w.push_back(e);
    while w.len() > 4096 {
        w.pop_front();
    }
    n >= 64 && rank * q < n
}

fn state_path() -> &'static str {
    if msa_mode() { "/data/calib/screen_state_msa2.json" } else { "/data/calib/screen_state.json" }
}

static LAST_SAVE: Mutex<Option<std::time::Instant>> = Mutex::new(None);
static S0_GLOBAL: std::sync::OnceLock<S0> = std::sync::OnceLock::new();

fn s0_global(q0: f64) -> &'static S0 {
    S0_GLOBAL.get_or_init(|| S0::new(q0))
}

/// Restore windows once per process. `nodes` guards against a topology change.
pub(super) fn state_load(nodes: usize, s0: &S0, stages: &Mutex<Vec<VecDeque<i64>>>) {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| {
        let Ok(text) = std::fs::read_to_string(state_path()) else { return };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { return };
        if v["nodes"].as_u64() != Some(nodes as u64) {
            return;
        }
        if let Some(q) = v["s0_q_milli"].as_u64() {
            s0.q_milli.store(q, Ordering::Relaxed);
        }
        if let Some(w) = v["s0_window"].as_array() {
            *s0.window.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                w.iter().filter_map(serde_json::Value::as_f64).collect();
        }
        if let Some(ws) = v["stages"].as_array() {
            let mut g = stages.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if g.is_empty() {
                *g = ws
                    .iter()
                    .map(|w| w.as_array().map_or_else(VecDeque::new, |a| a.iter().filter_map(serde_json::Value::as_i64).collect()))
                    .collect();
            }
        }
        tracing::info!("screening state restored from {}", state_path());
    });
}

/// Log the host-side phase times once a minute: where the device thread spends its time between kernels.
fn prof_report_if_due(salts: u64) {
    static LAST: Mutex<Option<(std::time::Instant, [u64; 9], u64)>> = Mutex::new(None);
    let mut g = LAST.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = std::time::Instant::now();
    let cur: [u64; 9] = std::array::from_fn(|i| crate::cluster::device::PROF_US[i].load(Ordering::Relaxed));
    match g.as_mut() {
        None => *g = Some((now, cur, salts)),
        Some((t, prev, ps)) => {
            if t.elapsed() >= Duration::from_secs(60) {
                let d: Vec<u64> = (0..9).map(|i| cur[i] - prev[i]).collect();
                tracing::info!(
                    salts = salts - *ps,
                    init_ms = d[0] / 1000,
                    launch_sync_ms = d[1] / 1000,
                    post_ms = d[2] / 1000,
                    work_new_ms = d[3] / 1000,
                    run_total_ms = d[4] / 1000,
                    score_ms = d[5] / 1000,
                    pa_host_ms = d[6] / 1000,
                    kernel_wait_ms = d[7] / 1000,
                    "device thread host-side time per minute"
                );
                *g = Some((now, cur, salts));
            }
        }
    }
}

fn state_save_if_due(nodes: usize, s0: &S0, stages: &Mutex<Vec<VecDeque<i64>>>) {
    let mut last = LAST_SAVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if last.is_some_and(|t| t.elapsed() < Duration::from_secs(30)) {
        return;
    }
    *last = Some(std::time::Instant::now());
    drop(last);
    let w: Vec<Vec<i64>> = stages
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|d| d.iter().copied().collect())
        .collect();
    let sw: Vec<f64> = s0.window.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().copied().collect();
    let body = serde_json::json!({
        "nodes": nodes,
        "s0_q_milli": s0.q_milli.load(Ordering::Relaxed),
        "s0_window": sw,
        "stages": w,
    });
    let tmp = format!("{}.tmp", state_path());
    let _ = std::fs::create_dir_all(CALIB_DIR);
    if std::fs::write(&tmp, body.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, state_path());
    }
}

/// A rolling window of structural scores with an adaptive pass fraction `1/q`.
struct S0 {
    window: Mutex<VecDeque<f64>>,
    /// `q` in thousandths: the pass fraction is `1000 / q_milli`.
    q_milli: AtomicU64,
}

/// Upper bound of the S0 pass divisor `q`, in thousandths (QS_S0MAX, default 20). The stronger the screen, the more the picks
/// favour instances that are easy for a short anneal; 20 is the level that was validated against deep results.
fn s0_max_milli() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QS_S0MAX").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(85.0).max(1.0) * 1000.0)
}

impl S0 {
    fn new(q: f64) -> Self {
        Self { window: Mutex::new(VecDeque::new()), q_milli: AtomicU64::new((q * 1000.0) as u64) }
    }
    fn passes(&self, score: f64) -> bool {
        let q = self.q_milli.load(Ordering::Relaxed) as f64 / 1000.0;
        let mut w = self.window.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = w.len();
        let rank = w.iter().filter(|&&s| s < score).count();
        w.push_back(score);
        // A window of 8192 resolves a pass fraction down to 1/4096 with two samples.
        while w.len() > 8192 {
            w.pop_front();
        }
        n < 64 || (rank as f64) * q < n as f64
    }
    /// Multiply the pass divisor `q` (more than 1 tightens the filter, less than 1 loosens it).
    fn scale(&self, f: f64) {
        let q = self.q_milli.load(Ordering::Relaxed) as f64 * f;
        self.q_milli.store(q.clamp(1000.0, s0_max_milli()) as u64, Ordering::Relaxed);
    }
    /// The GPU waited for work: let more salts through. Backlog: tighten.
    fn adapt(&self, starving: bool, backlog: bool) {
        let mut q = self.q_milli.load(Ordering::Relaxed) as f64;
        if starving {
            q *= 0.97;
        } else if backlog {
            q *= 1.03;
        }
        self.q_milli.store(q.clamp(1000.0, s0_max_milli()) as u64, Ordering::Relaxed);
    }
}

fn ising_graph(instance: &Instance) -> IsingGraph {
    IsingGraph {
        h: instance.h().iter().map(|&v| f64::from(v) / 1000.0).collect(),
        j: instance.j().iter().map(|&v| f64::from(v) / 1000.0).collect(),
        edges: instance
            .edges()
            .iter()
            .map(|&(u, v)| (usize::try_from(u).unwrap_or(0), usize::try_from(v).unwrap_or(0)))
            .collect::<Vec<_>>()
            .into(),
    }
}

impl<D: ClusterBackend, S: Sampler> OtsukaSampler<D, S> {
    /// Work one lease: parallel generation and structural ranking on the CPU, the screening pipeline on the GPU.
    pub(super) fn run_lease(&self, lease: &Lease, params: &SampleParams, out: &LeaseSink) -> Result<(), SampleError> {
        let Some(family) = lease.family() else {
            return Err(SampleError::DeviceFault("otsuka lease without family inputs".into()));
        };
        let contract = family.contract();
        calib_family_dump(family);
        // Wait for the gate in short slices so a stale lease (round change) leaves at once.
        let _gate = loop {
            if out.is_stopped() {
                return Ok(());
            }
            match LEASE_GATE.try_lock() {
                Ok(g) => break g,
                Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(5)),
            }
        };
        let lean = lean_for(family);
        let msa_on = msa_mode() && msa_dev().is_some() && msa_mode();
        let gen_threads: usize = std::env::var("QS_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(24usize).max(1);
        let q0: f64 = std::env::var("QS_S0Q").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0);
        let s0 = s0_global(q0);
        state_load(lease.family().map_or(0, |f| f.topology().node_ids().len()), s0, &self.screen);
        let count = lease.salt_count();
        let next = AtomicU64::new(0);
        let queued = std::sync::atomic::AtomicUsize::new(0);
        let (tx, rx) = sync_channel::<(u64, Option<Instance>, IsingGraph, Option<PreparedGraph>, f64, bool)>(8);
        let rx = Mutex::new(rx);
        let reads0 = ScreenCfg::from_env().stages.first().map_or(params.num_reads, |s| s.0).min(params.num_reads).max(1);
        let failure: Mutex<Option<SampleError>> = Mutex::new(None);
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..gen_threads {
                let tx = tx.clone();
                let (s0, next, stop, queued, lean, msa_on) = (s0, &next, &stop, &queued, &lean, msa_on);
                scope.spawn(move || {
                    while !stop.load(Ordering::Relaxed) && !out.is_stopped() {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        let nonce = lease.nonce(i);
                        // One salt in twenty thousand bypasses the structural filter, to measure it on unbiased salts.
                        let explore = (u32::from(nonce[0]) | (u32::from(nonce[1]) << 8)) % 20000 == 0;
                        // Score the salt from its couplings alone; only salts that pass are built as full instances.
                        let quick = lean.as_ref().and_then(|l| l.score(&nonce));
                        if let Some(score) = quick {
                            if !(s0.passes(score) || explore) {
                                if out.screen(i, i64::MAX).is_err() {
                                    return;
                                }
                                continue;
                            }
                        }
                        match family.generate(nonce) {
                            Ok(inst) => {
                                let score = quick.unwrap_or_else(|| structural_score(&inst));
                                if s0.passes(score) || explore {
                                    // Build the device graph here, on a spare core, so the GPU thread never waits for it.
                                    let graph = ising_graph(&inst);
                                    let mut item = if msa_on {
                                        // The MSA stage needs only the graph; survivors are rebuilt from their nonce.
                                        (i, None, graph, None, score, explore)
                                    } else {
                                        let prepared = PreparedGraph::from_instance(&inst, contract, Weights::from_env(), &self.limits, reads0).ok();
                                        (i, Some(inst), graph, prepared, score, explore)
                                    };
                                    loop {
                                        if stop.load(Ordering::Relaxed) || out.is_stopped() {
                                            return;
                                        }
                                        match tx.try_send(item) {
                                            Ok(()) => {
                                                queued.fetch_add(1, Ordering::Relaxed);
                                                break;
                                            }
                                            Err(TrySendError::Full(back)) => {
                                                item = back;
                                                std::thread::sleep(Duration::from_millis(2));
                                            }
                                            Err(TrySendError::Disconnected(_)) => return,
                                        }
                                    }
                                } else if out.screen(i, i64::MAX).is_err() {
                                    return;
                                }
                            }
                            Err(_) => {
                                if out.skip(i).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                });
            }
            drop(tx);
            // GPU workers: each takes one salt at a time through the screening pipeline. Two workers keep the GPU
            // busy while the other is in a host phase (state init, the population-annealing sort, read-back).
            let handle = |i: u64, inst: Instance, graph: IsingGraph, prepared0: Option<PreparedGraph>, score0: f64, explore0: bool| -> bool {
                let fam = Fam { instance: &inst, contract };
                let deep = AtomicBool::new(false);
                let target = out.target_energy_milli();
                state_save_if_due(graph.h.len(), s0, &self.screen);
                static SALTS: AtomicU64 = AtomicU64::new(0);
                prof_report_if_due(SALTS.fetch_add(1, Ordering::Relaxed) + 1);
                let mut stage_e: Vec<i64> = Vec::new();
                let result = self.screened_reads(&graph, params, &fam, &|| out.is_stopped(), &deep, target, prepared0, &mut stage_e);
                let offset = inst.offset_milli();
                match result {
                    Ok(mut reads) => {
                        let _ = crate::cluster::device::take_deep_progress();
                        for r in &mut reads {
                            r.energy_milli = r.energy_milli.saturating_add(offset);
                        }
                        let best = reads.iter().map(|r| r.energy_milli).min().unwrap_or(i64::MAX);
                        let nonce = lease.nonce(i);
                        if explore0 || stage_e.len() >= 2 || nonce[2] % 10 == 0 || MSA_X.load(Ordering::Relaxed) {
                            let hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
                            calib_write(&format!(
                                "{{\"n\":\"{hex}\",\"s0\":{score0:.5},\"ex\":{},\"msa\":{},\"mx\":{},\"st\":{:?},\"deep\":{},\"off\":{}}}",
                                u8::from(explore0),
                                MSA_CUR.load(Ordering::Relaxed),
                                u8::from(MSA_X.load(Ordering::Relaxed)),
                                stage_e,
                                u8::from(deep.load(Ordering::Relaxed)),
                                offset
                            ));
                        }
                        // Only a deep run reports: the dashboard and the coordinator should see the best nonces, not
                        // one record per screened salt.
                        let sent = if deep.load(Ordering::Relaxed) && !reads.is_empty() {
                            out.push(i, reads)
                        } else {
                            scr_note(best);
                            out.screen(i, best)
                        };
                        if sent.is_err() {
                            return false;
                        }
                    }
                    Err(SampleError::DeviceBusy | SampleError::Capacity) => {
                        // A round change surfaces here as DeviceBusy; record how far the cut-off deep run got.
                        if out.is_stopped() {
                            let prog = crate::cluster::device::take_deep_progress();
                            let (scr_best, scr_n) = scr_take();
                            if prog.is_some() || scr_n > 0 {
                                let (done, total, best) = prog.map_or((0, 0, i64::MAX), |(d, t, b)| (d, t, b.saturating_add(offset)));
                                calib_write(&format!(
                                    "{{\"cancelled\":1,\"ts_ms\":{},\"done\":{done},\"total\":{total},\"best_milli\":{best},\"scr_best_milli\":{scr_best},\"scr_n\":{scr_n}}}",
                                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())
                                ));
                            }
                        }
                        if out.skip(i).is_err() {
                            return false;
                        }
                    }
                    Err(e) => {
                        *failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(e);
                        return false;
                    }
                }
                true
            };
            let rebuild = |i: u64| -> Option<Instance> { family.generate(lease.nonce(i)).ok() };
            let worker = || {
            loop {
                if out.is_stopped() || stop.load(Ordering::Relaxed) {
                    break;
                }
                // Keep a short queue: tighten the structural filter while salts pile up, loosen it when the GPU
                // drains the queue or waits.
                let first = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).try_recv();
                let (i, inst, graph, prepared0, score0, explore0) = match first {
                    Ok(item) => item,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        let waited = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).recv_timeout(Duration::from_millis(20));
                        match waited {
                            Ok(item) => item,
                            Err(RecvTimeoutError::Timeout) => {
                                s0.scale(0.85);
                                continue;
                            }
                            Err(RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                };
                let left = queued.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                if left >= 5 {
                    s0.scale(1.04);
                } else if left == 0 {
                    s0.scale(0.96);
                }
                let inst = match inst {
                    Some(x) => x,
                    None => match rebuild(i) {
                        Some(x) => x,
                        None => {
                            if out.skip(i).is_err() {
                                break;
                            }
                            continue;
                        }
                    },
                };
                if !handle(i, inst, graph, prepared0, score0, explore0) {
                    break;
                }
            }
            };
            // MSA first stage: stream a batch of S0-passing salts through the MSA kernel, keep the best 1/QS_MSAQ, and send those
            // through the cluster stages.
            let msa_worker = || {
                let envn = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d).max(1);
                let (k_batch, q_a, reads, sweeps) = (envn("QS_MSAK", 1024), envn("QS_MSAQ", 512), envn("QS_MSAR", 128), envn("QS_MSAS", 4096));
                let Some((dev, gov)) = msa_dev() else { return };
                // The sampler's default annealing range is far from the best one for this family: (1.5, 5.0) gains ~30 energy
                // units at the same cost and raises the correlation with the deep result from 0.33 to 0.50 (QS_MSAB="lo,hi").
                let msa_beta: Option<(f64, f64)> = std::env::var("QS_MSAB").ok().map_or(Some((1.5, 5.0)), |v| {
                    let (a, b) = v.split_once(',')?;
                    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                });
                loop {
                    if out.is_stopped() || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(64);
                    let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(64);
                    let token = CancelToken::default();
                    let meta: Mutex<Vec<(u64, f64, bool)>> = Mutex::new(Vec::new());
                    let mut energies: Vec<Option<i64>> = Vec::new();
                    let mut disconnected = false;
                    std::thread::scope(|sc| {
                        let tok = token.clone();
                        sc.spawn(move || crate::streaming::run_stream(dev, crate::KernelKind::Msa, job_rx, res_tx, tok, gov, crate::streaming::StallConfig::default()));
                        let (meta_ref, rx_ref, discon) = (&meta, &rx, &mut disconnected);
                        let (stop_r, queued_r) = (&stop, &queued);
                        sc.spawn(move || {
                            let mut k = 0usize;
                            while k < k_batch {
                                if out.is_stopped() || stop_r.load(Ordering::Relaxed) {
                                    break;
                                }
                                let got = rx_ref.lock().unwrap_or_else(std::sync::PoisonError::into_inner).try_recv();
                                let (i, _inst, graph, _prep, score0, explore0) = match got {
                                    Ok(item) => item,
                                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                                        let waited = rx_ref.lock().unwrap_or_else(std::sync::PoisonError::into_inner).recv_timeout(Duration::from_millis(20));
                                        match waited {
                                            Ok(item) => item,
                                            Err(RecvTimeoutError::Timeout) => {
                                                s0.scale(0.85);
                                                continue;
                                            }
                                            Err(RecvTimeoutError::Disconnected) => {
                                                *discon = true;
                                                break;
                                            }
                                        }
                                    }
                                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                        *discon = true;
                                        break;
                                    }
                                };
                                let left = queued_r.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                                if left >= 5 {
                                    s0.scale(1.04);
                                } else if left == 0 {
                                    s0.scale(0.96);
                                }
                                meta_ref.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((i, score0, explore0));
                                let p = SampleParams { num_reads: reads, num_sweeps: sweeps, sweeps_per_beta: 1, seed: 11 + i, beta_range: msa_beta, ..SampleParams::default() };
                                if job_tx.blocking_send(StreamJob { job_id: (k as u32).to_le_bytes().to_vec(), graph, params: p, watermark: Some(1) }).is_err() {
                                    break;
                                }
                                k += 1;
                            }
                            drop(job_tx);
                        });
                        loop {
                            match res_rx.try_recv() {
                                Ok(r) => {
                                    let k = u32::from_le_bytes(r.job_id[..4].try_into().unwrap_or([0; 4])) as usize;
                                    if energies.len() <= k {
                                        energies.resize(k + 1, None);
                                    }
                                    if let StreamOutcome::Completed(Ok(rs)) = r.outcome {
                                        energies[k] = rs.iter().map(|x| x.energy_milli).min();
                                    }
                                }
                                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                    if out.is_stopped() {
                                        token.cancel_through(1);
                                    }
                                    std::thread::sleep(Duration::from_millis(2));
                                }
                                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                            }
                        }
                    });
                    let meta = meta.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
                    energies.resize(meta.len(), None);
                    if out.is_stopped() {
                        break;
                    }
                    // Repeated MSA: one run tells little about an instance (two runs of the same salt correlate 0.36), but the
                    // mean of several runs is a judge as good as stage B at a fraction of its cost (measured: mean of 8 runs
                    // correlates 0.54 with the deep result, stage B 0.50, at 0.1 s against 0.9 s). So the batch is narrowed in
                    // passes: the best 1/p1 by one run get k2 more runs, the best 1/p2 by their mean get k3 more, and the best
                    // 1/p3 by the mean of all runs go on to the cluster stages. QS_MSAREP=0 restores the single-run rule.
                    let mut surv = vec![false; meta.len()];
                    let mut mean_e: Vec<i64> = energies.iter().map(|e| e.unwrap_or(i64::MAX)).collect();
                    if envn("QS_MSAREP", 1) == 0 || std::env::var("QS_MSAREP").as_deref() == Ok("0") {
                        for k in 0..meta.len() {
                            if let Some(e) = energies[k] {
                                surv[k] = msa_passes(e, q_a);
                            }
                        }
                    } else {
                        let (p1, p2, p3, k2, k3) = (envn("QS_MSAP1", 8), envn("QS_MSAP2", 8), envn("QS_MSAP3", 4), envn("QS_MSAK2", 3), envn("QS_MSAK3", 12));
                        let mut sum: Vec<i64> = energies.iter().map(|e| e.unwrap_or(0)).collect();
                        let mut cnt: Vec<i64> = energies.iter().map(|e| i64::from(e.is_some())).collect();
                        let mut alive: Vec<usize> = (0..meta.len()).filter(|&k| energies[k].is_some()).collect();
                        alive.sort_by_key(|&k| sum[k]);
                        alive.truncate(alive.len() / p1);
                        for (pass, &(reps, q)) in [(k2, p2), (k3, p3)].iter().enumerate() {
                            if alive.is_empty() || out.is_stopped() {
                                alive.clear();
                                break;
                            }
                            // Rebuild the survivors' graphs on a few threads (the batch kept none of them).
                            let graphs: Vec<Option<IsingGraph>> = {
                                let slots: Vec<Mutex<Option<IsingGraph>>> = alive.iter().map(|_| Mutex::new(None)).collect();
                                let next = std::sync::atomic::AtomicUsize::new(0);
                                std::thread::scope(|rs| {
                                    for _ in 0..8usize.min(alive.len()) {
                                        rs.spawn(|| loop {
                                            let s = next.fetch_add(1, Ordering::Relaxed);
                                            if s >= alive.len() {
                                                break;
                                            }
                                            if let Some(inst) = rebuild(meta[alive[s]].0) {
                                                *slots[s].lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ising_graph(&inst));
                                            }
                                        });
                                    }
                                });
                                slots.into_iter().map(|m| m.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner)).collect()
                            };
                            let mut jobs: Vec<(u32, IsingGraph, SampleParams)> = Vec::with_capacity(alive.len() * reps);
                            for (slot, g) in graphs.into_iter().enumerate() {
                                let Some(g) = g else { continue };
                                let i = meta[alive[slot]].0;
                                for r in 0..reps {
                                    let seed = (11 + i).wrapping_add(((pass as u64 + 1) * 1000 + r as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                                    let p = SampleParams { num_reads: reads, num_sweeps: sweeps, sweeps_per_beta: 1, seed, beta_range: msa_beta, ..SampleParams::default() };
                                    jobs.push(((slot as u32) | ((r as u32) << 16), g.clone(), p));
                                }
                            }
                            for (id, e) in msa_session(dev, gov, jobs, &|| out.is_stopped()) {
                                if let Some(e) = e {
                                    let k = alive[(id & 0xffff) as usize];
                                    sum[k] += e;
                                    cnt[k] += 1;
                                }
                            }
                            if out.is_stopped() {
                                alive.clear();
                                break;
                            }
                            alive.sort_by_key(|&k| sum[k] / cnt[k].max(1));
                            alive.truncate((alive.len() / q).max(1));
                        }
                        for &k in &alive {
                            surv[k] = true;
                            mean_e[k] = sum[k] / cnt[k].max(1);
                        }
                    }
                    if out.is_stopped() {
                        break;
                    }
                    let mut order: Vec<usize> = (0..meta.len()).filter(|&k| surv[k]).collect();
                    order.sort_by_key(|&k| mean_e[k]);
                    // A small random sample of rejected salts goes through the funnel as well; their MSA energy and
                    // later stage energies give the unbiased MSA→B/deep correlation on live data (QS_MSAX, 0 = off).
                    let x_every = std::env::var("QS_MSAX").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(2048);
                    let mut xp = vec![false; meta.len()];
                    if x_every > 0 {
                        for k in 0..meta.len() {
                            if !surv[k] && energies[k].is_some() && (meta[k].0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) % x_every == 0 {
                                xp[k] = true;
                                order.push(k);
                            }
                        }
                    }
                    let n_surv = order.len();
                    let mut ok = true;
                    for (k, &(i, score0, explore0)) in meta.iter().enumerate() {
                        if surv[k] || xp[k] {
                            continue;
                        }
                        let sent = match energies[k] {
                            Some(e) => {
                                scr_note(e);
                                out.screen(i, e)
                            }
                            None => out.skip(i),
                        };
                        if sent.is_err() {
                            ok = false;
                            break;
                        }
                        let _ = (score0, explore0);
                    }
                    if !ok {
                        break;
                    }
                    // Survivors, best first, through the cluster stages.
                    for &k in order.iter().take(n_surv) {
                        let (i, score0, explore0) = meta[k];
                        if out.is_stopped() {
                            break;
                        }
                        let Some(inst) = rebuild(i) else {
                            if out.skip(i).is_err() {
                                ok = false;
                                break;
                            }
                            continue;
                        };
                        let graph = ising_graph(&inst);
                        MSA_CUR.store(mean_e[k], Ordering::Relaxed);
                        MSA_X.store(xp[k], Ordering::Relaxed);
                        let cont = handle(i, inst, graph, None, score0, explore0);
                        MSA_X.store(false, Ordering::Relaxed);
                        if !cont {
                            ok = false;
                            break;
                        }
                    }
                    if !ok || disconnected {
                        break;
                    }
                }
            };
            let workers: usize = std::env::var("QS_GPUW").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).clamp(1, 4);
            std::thread::scope(|gpu| {
                if msa_on {
                    msa_worker();
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
                for _ in 1..workers {
                    gpu.spawn(|| {
                        worker();
                        stop.store(true, Ordering::Relaxed);
                    });
                }
                worker();
                stop.store(true, Ordering::Relaxed);
            });
        });
        match failure.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[allow(dead_code)]
fn _unused(_: SamplerResult, _: &TopologyView) {}

#[cfg(test)]
mod tests {
    use super::score_raw;

    /// Compare against the Python reference on dumped instances: `S0_DIR` holds i{idx}.txt, `S0_REF` lines "idx score".
    #[test]
    #[ignore = "needs dumped instances"]
    fn matches_python_reference() {
        let (Ok(dir), Ok(reff)) = (std::env::var("S0_DIR"), std::env::var("S0_REF")) else { return };
        let mut worst = 0f64;
        for line in std::fs::read_to_string(reff).unwrap().lines() {
            let mut it = line.split_whitespace();
            let idx: usize = it.next().unwrap().parse().unwrap();
            let want: f64 = it.next().unwrap().parse().unwrap();
            let text = std::fs::read_to_string(format!("{dir}/i{idx}.txt")).unwrap();
            let mut lines = text.lines();
            let n: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
            let m: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
            let mut edges = Vec::new();
            let mut js = Vec::new();
            for _ in 0..m {
                let mut p = lines.next().unwrap().split_whitespace();
                edges.push((p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap()));
                js.push(p.next().unwrap().parse().unwrap());
            }
            let got = score_raw(n, &edges, &js);
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 1e-6, "max abs diff {worst}");
    }
}
