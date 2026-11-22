//! `quip-gpu-feeder v2.0`: 100% GPU-Native Pre-Screener Server (`QCF1` protocol).
//!
//! Each `OP_SUBMIT_CHUNK` (512 salts) is processed end-to-end on the RTX 3080Ti GPU:
//!   1. Host computes ONLY 5 ChaCha8 keys/salt (`80 KB` for 512 salts, `<0.3 ms`).
//!   2. GPU `gpu_generate_dimer_batch`: generates all 512 contracted graphs directly in VRAM.
//!   3. GPU `bm4_batch_sdp_smem_fp16` (optional pre-filter, e.g. 35 steps @ 15% keep)
//!      -> GPU `bm8_batch_sdp_smem_fp16` (60 steps @ 2% keep)
//!      -> GPU `bm8_batch_sdp_fp32_indexed` (200 steps exact FP32 + `u_vec` on ~10 survivors).
//!   4. CPU worker pool runs exact `Micro-BP + 8x8 Jacobi + BmGpuSeed` ONLY on ~0.36% (~2 salts/chunk!).

mod engine;
mod gpu_bm8;

use engine::{FastDimer, FastGate};
use gpu_bm8::{GpuArena, GpuSharedRuntime};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const MAGIC: &[u8; 4] = b"QCF1";
const OKAY: &[u8; 4] = b"OKAY";
const OP_SET_TOPO: u8 = 1;
const OP_SUBMIT_CHUNK: u8 = 2;
const OP_CANCEL_LEASE: u8 = 3;
const OP_PING: u8 = 4;
const OP_CHUNK_RESULT: u8 = 10;
const OP_PONG: u8 = 11;

#[derive(Clone, Copy)]
struct ServerCfg {
    gen_1tps_workers: usize,
    bm2_pre_steps: i32,
    bm2_pre_keep: f64,
    bm4_pre_steps: i32,
    bm4_pre_keep: f64,
    bm60_steps: i32,
    bm60_keep: f64,
    bm30_keep: f64,
    stage_e_keep: f64,
    super_stage_e: bool,
    deep_reps: usize,
    deep_iters: usize,
    gw_trials: usize,
    run_unseeded: bool,
    bp_weight: f64,
    bp_seed_weight: f64,
}

struct HitOut {
    index: u64,
    score: f32,
    failed_fast: bool,
    u_orig: Vec<[i8; 8]>,
    evecs: [[f32; 8]; 8],
}

struct TopoGates {
    bm2: FastGate,
    bm4: FastGate,
    bm60: FastGate,
    bm30: FastGate,
    stage_e: FastGate,
}

impl TopoGates {
    fn new(cfg: ServerCfg) -> Self {
        let bm4_rel_keep = if cfg.bm2_pre_steps > 0 {
            (cfg.bm4_pre_keep / cfg.bm2_pre_keep.max(0.05)).min(1.0)
        } else {
            cfg.bm4_pre_keep
        };
        Self {
            bm2: FastGate::new(cfg.bm2_pre_keep, 2048, 128, 64),
            bm4: FastGate::new(bm4_rel_keep, 2048, 128, 64),
            bm60: FastGate::new(
                if cfg.bm4_pre_steps > 0 {
                    (cfg.bm60_keep / cfg.bm4_pre_keep.max(0.01)).min(1.0)
                } else {
                    cfg.bm60_keep
                },
                2048,
                128,
                64,
            ),
            bm30: FastGate::new(cfg.bm30_keep, 512, 32, 32),
            stage_e: FastGate::new(cfg.stage_e_keep, 512, 32, 16),
        }
    }
}

struct ChunkJob {
    client: Arc<ClientCtx>,
    topo: Arc<FastDimer>,
    gates: Arc<TopoGates>,
    lease_id: u64,
    chunk_id: u64,
    start_idx: u64,
    count: u32,
    pool_cutoff: f32,
    nonces: Vec<[u8; 32]>,
    keys_flat: Vec<u32>,
    seeds: Vec<u64>,
}

struct BpTask {
    client: Arc<ClientCtx>,
    topo: Arc<FastDimer>,
    gates: Arc<TopoGates>,
    lease_id: u64,
    chunk_id: u64,
    start_idx: u64,
    count: u32,
    pool_cutoff: f32,
    failed_indices: Vec<u64>,
    candidates: Vec<(u64, [u8; 32], u64, f32, Vec<[f32; 8]>)>, // (idx, nonce, seed, bm_deep, u_vec)
}

struct ClientCtx {
    writer: Mutex<BufWriter<TcpStream>>,
    closed: AtomicBool,
    cancelled_leases: Mutex<HashSet<u64>>,
}

impl ClientCtx {
    fn is_cancelled(&self, lease_id: u64) -> bool {
        self.closed.load(Ordering::Relaxed)
            || self.cancelled_leases.lock().unwrap().contains(&lease_id)
    }

    fn send_chunk_result(
        &self,
        lease_id: u64,
        chunk_id: u64,
        start_idx: u64,
        count: u32,
        hits: &[HitOut],
    ) {
        if self.closed.load(Ordering::Relaxed) {
            return;
        }
        let mut w = self.writer.lock().unwrap();
        let res = (|| -> io::Result<()> {
            w.write_all(&[OP_CHUNK_RESULT])?;
            w.write_all(&lease_id.to_le_bytes())?;
            w.write_all(&chunk_id.to_le_bytes())?;
            w.write_all(&start_idx.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
            w.write_all(&(hits.len() as u32).to_le_bytes())?;
            for h in hits {
                w.write_all(&h.index.to_le_bytes())?;
                w.write_all(&h.score.to_le_bytes())?;
                w.write_all(&[u8::from(h.failed_fast)])?;
                w.write_all(&(h.u_orig.len() as u32).to_le_bytes())?;
                if !h.u_orig.is_empty() {
                    let mut raw = Vec::with_capacity(h.u_orig.len() * 8 + 64 * 4);
                    for row in &h.u_orig {
                        for &b in row {
                            raw.push(b as u8);
                        }
                    }
                    for row in &h.evecs {
                        for &f in row {
                            raw.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    w.write_all(&raw)?;
                }
            }
            w.flush()
        })();
        if res.is_err() {
            self.closed.store(true, Ordering::Relaxed);
        }
    }

    fn send_pong(&self) {
        if self.closed.load(Ordering::Relaxed) {
            return;
        }
        let mut w = self.writer.lock().unwrap();
        if w.write_all(&[OP_PONG]).and_then(|_| w.flush()).is_err() {
            self.closed.store(true, Ordering::Relaxed);
        }
    }
}

struct SchedulerInner {
    queues: HashMap<u64, VecDeque<ChunkJob>>,
    order: VecDeque<u64>,
    active_clients: usize,
}

struct Scheduler {
    inner: Mutex<SchedulerInner>,
    cv: Condvar,
}

impl Scheduler {
    fn new() -> Self {
        Self {
            inner: Mutex::new(SchedulerInner {
                queues: HashMap::new(),
                order: VecDeque::new(),
                active_clients: 0,
            }),
            cv: Condvar::new(),
        }
    }
    fn register_client(&self, cid: u64) {
        let mut st = self.inner.lock().unwrap();
        st.queues.insert(cid, VecDeque::new());
        st.order.push_back(cid);
        st.active_clients += 1;
    }
    fn unregister_client(&self, cid: u64) {
        let mut st = self.inner.lock().unwrap();
        st.queues.remove(&cid);
        st.order.retain(|&x| x != cid);
        st.active_clients = st.active_clients.saturating_sub(1);
    }
    fn push_chunk(&self, cid: u64, job: ChunkJob) {
        let mut st = self.inner.lock().unwrap();
        if let Some(q) = st.queues.get_mut(&cid) {
            q.push_back(job);
            self.cv.notify_one();
        }
    }
    fn pop_fair_coalesced(&self, max_salts: usize) -> Vec<ChunkJob> {
        let mut st = self.inner.lock().unwrap();
        let first = 'outer: loop {
            let len = st.order.len();
            for _ in 0..len {
                if let Some(cid) = st.order.pop_front() {
                    st.order.push_back(cid);
                    if let Some(q) = st.queues.get_mut(&cid) {
                        if let Some(job) = q.pop_front() {
                            break 'outer job;
                        }
                    }
                }
            }
            st = self.cv.wait(st).unwrap();
        };
        let mut out = vec![first];
        let mut total = out[0].nonces.len();
        let key = (out[0].topo.order_digest, out[0].topo.descriptor_digest);
        let deadline = Instant::now() + Duration::from_millis(2);
        while total < max_salts {
            // Try grabbing more chunks with the same topology across clients round-robin
            let mut grabbed = false;
            let len = st.order.len();
            for _ in 0..len {
                if let Some(cid) = st.order.pop_front() {
                    st.order.push_back(cid);
                    if let Some(q) = st.queues.get_mut(&cid) {
                        if let Some(front) = q.front() {
                            let fk = (front.topo.order_digest, front.topo.descriptor_digest);
                            if fk == key && total + front.nonces.len() <= max_salts {
                                let j = q.pop_front().unwrap();
                                total += j.nonces.len();
                                out.push(j);
                                grabbed = true;
                                if total >= max_salts {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if grabbed {
                continue;
            }
            let rem = deadline.saturating_duration_since(Instant::now());
            if rem.is_zero() {
                break;
            }
            let (guard, timeout) = self.cv.wait_timeout(st, rem).unwrap();
            st = guard;
            if timeout.timed_out() {
                break;
            }
        }
        out
    }
    fn active_clients(&self) -> usize {
        self.inner.lock().unwrap().active_clients
    }
}

struct GlobalStats {
    screened: AtomicU64,
    s0b_pass: AtomicU64,
    s0c_pass: AtomicU64,
    hits_sent: AtomicU64,
    seeds_sent: AtomicU64,
    gen_us: AtomicU64,
    bm4_us: AtomicU64,
    bm8_us: AtomicU64,
    w30_us: AtomicU64,
    bp_us: AtomicU64,
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> { let mut b=[0u8;1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> { let mut b=[0u8;2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> { let mut b=[0u8;8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }

fn gpu_worker_loop(
    sched: Arc<Scheduler>,
    rt: Arc<GpuSharedRuntime>,
    mut arena: GpuArena,
    bp_tx: mpsc::SyncSender<BpTask>,
    cfg: ServerCfg,
    stats: Arc<GlobalStats>,
) {
    let mut keys_buf: Vec<u32> = Vec::with_capacity(arena.max_batch * 40);
    let mut seeds_buf: Vec<u64> = Vec::with_capacity(arena.max_batch);
    let mut slot_owner: Vec<(usize, usize)> = Vec::with_capacity(arena.max_batch); // (job_idx, pos_in_job)

    loop {
        let jobs = sched.pop_fair_coalesced(arena.max_batch);
        let active_jobs: Vec<ChunkJob> = jobs
            .into_iter()
            .filter(|j| !j.client.is_cancelled(j.lease_id))
            .collect();
        if active_jobs.is_empty() {
            continue;
        }
        let topo = Arc::clone(&active_jobs[0].topo);
        let gates = Arc::clone(&active_jobs[0].gates);
        let Ok(gpu_topo) = rt.get_or_upload_topo(&arena.stream, &topo) else {
            continue;
        };
        let weak_f = topo.weak as f32;

        keys_buf.clear();
        seeds_buf.clear();
        slot_owner.clear();
        for (ji, j) in active_jobs.iter().enumerate() {
            let avail = arena.max_batch.saturating_sub(seeds_buf.len());
            let take = j.nonces.len().min(avail);
            keys_buf.extend_from_slice(&j.keys_flat[..take * 40]);
            seeds_buf.extend_from_slice(&j.seeds[..take]);
            for pos in 0..take {
                slot_owner.push((ji, pos));
            }
        }
        let bsz = seeds_buf.len();
        if bsz == 0 {
            continue;
        }

        let mut per_job_failed: Vec<Vec<u64>> = vec![Vec::new(); active_jobs.len()];
        let mut per_job_cands: Vec<Vec<(u64, [u8; 32], u64, f32, Vec<[f32; 8]>)>> =
            vec![Vec::new(); active_jobs.len()];

        let t0 = Instant::now();
        let csr_ready = cfg.gen_1tps_workers > 0;
        let Ok(n_per_slot) = (if csr_ready {
            arena.generate_batch_1tps(&gpu_topo, &keys_buf, &seeds_buf, cfg.gen_1tps_workers)
        } else {
            arena.generate_batch(&gpu_topo, &keys_buf, &seeds_buf)
        }) else {
            continue;
        };
        stats.gen_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        for (s, &nv) in n_per_slot.iter().enumerate() {
            if nv <= 0 {
                let (ji, pos) = slot_owner[s];
                per_job_failed[ji].push(active_jobs[ji].start_idx + pos as u64);
            }
        }

        let mut s0b_slots: Vec<i32> = Vec::new();
        if cfg.bm4_pre_steps > 0 {
            let t1 = Instant::now();
            let bm8_slots = if cfg.bm2_pre_steps > 0 {
                let Ok(bm2_scores) = arena.run_bm2_all(bsz, csr_ready, weak_f, cfg.bm2_pre_steps, 0.25) else {
                    continue;
                };
                let mut bm2_valid = Vec::with_capacity(bsz);
                for (s, &sc2) in bm2_scores.iter().enumerate() {
                    if n_per_slot[s] > 0 && sc2 < 1e8 { bm2_valid.push(sc2); }
                }
                gates.bm2.observe_batch(&bm2_valid);
                let mut bm4_slots = Vec::new();
                for (s, &sc2) in bm2_scores.iter().enumerate() {
                    if n_per_slot[s] > 0 && sc2 < 1e8 && gates.bm2.check_fast(sc2) {
                        bm4_slots.push(s as i32);
                    }
                }
                let mut out8 = Vec::new();
                if !bm4_slots.is_empty() {
                    let Ok(bm4_scores) = arena.run_bm4_smem(Some(&bm4_slots), bsz, true, weak_f, cfg.bm4_pre_steps, 0.22) else {
                        continue;
                    };
                    gates.bm4.observe_batch(&bm4_scores);
                    for (&slot, &sc4) in bm4_slots.iter().zip(&bm4_scores) {
                        if gates.bm4.check_fast(sc4) { out8.push(slot); }
                    }
                }
                out8
            } else {
                let Ok(bm4_scores) = arena.run_bm4_smem(None, bsz, csr_ready, weak_f, cfg.bm4_pre_steps, 0.22) else {
                    continue;
                };
                let mut bm4_valid = Vec::with_capacity(bsz);
                for (s, &sc4) in bm4_scores.iter().enumerate() {
                    if n_per_slot[s] > 0 && sc4 < 1e8 { bm4_valid.push(sc4); }
                }
                gates.bm4.observe_batch(&bm4_valid);
                let mut out8 = Vec::new();
                for (s, &sc4) in bm4_scores.iter().enumerate() {
                    if n_per_slot[s] > 0 && sc4 < 1e8 && gates.bm4.check_fast(sc4) {
                        out8.push(s as i32);
                    }
                }
                out8
            };
            stats.bm4_us.fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);

            if !bm8_slots.is_empty() {
                let t2 = Instant::now();
                let Ok(bm8_scores) = arena.run_bm8_smem(
                    Some(&bm8_slots),
                    bsz,
                    true,
                    weak_f,
                    cfg.bm60_steps,
                    0.20,
                ) else {
                    continue;
                };
                stats.bm8_us.fetch_add(t2.elapsed().as_micros() as u64, Ordering::Relaxed);
                gates.bm60.observe_batch(&bm8_scores);
                for (&slot, &sc8) in bm8_slots.iter().zip(&bm8_scores) {
                    if gates.bm60.check_fast(sc8) {
                        s0b_slots.push(slot);
                    }
                }
            }
        } else {
            let t2 = Instant::now();
            let Ok(bm8_scores) = arena.run_bm8_smem(
                None,
                bsz,
                false,
                weak_f,
                cfg.bm60_steps,
                0.20,
            ) else {
                continue;
            };
            stats.bm8_us.fetch_add(t2.elapsed().as_micros() as u64, Ordering::Relaxed);
            let mut bm8_valid = Vec::with_capacity(bsz);
            for (s, &sc8) in bm8_scores.iter().enumerate() {
                if n_per_slot[s] > 0 && sc8 < 1e8 {
                    bm8_valid.push(sc8);
                }
            }
            gates.bm60.observe_batch(&bm8_valid);
            for (s, &sc8) in bm8_scores.iter().enumerate() {
                if n_per_slot[s] > 0 && sc8 < 1e8 && gates.bm60.check_fast(sc8) {
                    s0b_slots.push(s as i32);
                }
            }
        }

        if !s0b_slots.is_empty() {
            stats.s0b_pass.fetch_add(s0b_slots.len() as u64, Ordering::Relaxed);
            let t3 = Instant::now();
            let Ok(w30_surv) = arena.run_w30_fp32_filtered(&s0b_slots, &n_per_slot, weak_f, |scores| {
                gates.bm30.observe_batch(scores);
                let mut keep = Vec::new();
                for (j, &e) in scores.iter().enumerate() {
                    if gates.bm30.check_fast(e) {
                        keep.push(j);
                    }
                }
                keep
            }) else {
                continue;
            };
            stats.w30_us.fetch_add(t3.elapsed().as_micros() as u64, Ordering::Relaxed);

            for (slot, e, u_vec) in w30_surv {
                let (ji, pos) = slot_owner[slot as usize];
                let j = &active_jobs[ji];
                per_job_cands[ji].push((
                    j.start_idx + pos as u64,
                    j.nonces[pos],
                    j.seeds[pos],
                    e,
                    u_vec,
                ));
            }
        }

        stats.screened.fetch_add(bsz as u64, Ordering::Relaxed);

        for (ji, job) in active_jobs.into_iter().enumerate() {
            if job.client.is_cancelled(job.lease_id) {
                continue;
            }
            let failed_indices = std::mem::take(&mut per_job_failed[ji]);
            let candidates = std::mem::take(&mut per_job_cands[ji]);
            if candidates.is_empty() && failed_indices.is_empty() {
                job.client.send_chunk_result(
                    job.lease_id,
                    job.chunk_id,
                    job.start_idx,
                    job.count,
                    &[],
                );
            } else {
                let _ = bp_tx.send(BpTask {
                    client: job.client,
                    topo: job.topo,
                    gates: job.gates,
                    lease_id: job.lease_id,
                    chunk_id: job.chunk_id,
                    start_idx: job.start_idx,
                    count: job.count,
                    pool_cutoff: job.pool_cutoff,
                    failed_indices,
                    candidates,
                });
            }
        }
    }
}

fn bp_worker_loop(
    rx: Arc<Mutex<mpsc::Receiver<BpTask>>>,
    cfg: ServerCfg,
    stats: Arc<GlobalStats>,
) {
    let mut cached_key: ([u8; 32], [u8; 32]) = ([0; 32], [0; 32]);
    let mut scratch_opt: Option<engine::WorkerScratch> = None;

    loop {
        let task = {
            let guard = rx.lock().unwrap();
            match guard.recv() {
                Ok(t) => t,
                Err(_) => break,
            }
        };
        if task.client.is_cancelled(task.lease_id) {
            continue;
        }
        let key = (task.topo.order_digest, task.topo.descriptor_digest);
        if scratch_opt.is_none() || cached_key != key {
            scratch_opt = Some(task.topo.new_scratch());
            cached_key = key;
        }
        let sc = scratch_opt.as_mut().unwrap();
        let mut hits = Vec::with_capacity(task.failed_indices.len() + task.candidates.len());
        for idx in task.failed_indices {
            hits.push(HitOut {
                index: idx,
                score: 0.0,
                failed_fast: true,
                u_orig: Vec::new(),
                evecs: [[0.0; 8]; 8],
            });
        }
        let t_bp = Instant::now();
        for (idx, nonce, seed, bm_deep, u_vec) in task.candidates {
            if task.client.is_cancelled(task.lease_id) {
                break;
            }
            stats.s0c_pass.fetch_add(1, Ordering::Relaxed);
            if !task.topo.generate_into_scratch(nonce, sc) {
                continue;
            }
            sc.set_u8_from_gpu(&u_vec);
            let (score, u_orig, evecs) = if cfg.super_stage_e {
                sc.stage_e_super(seed, bm_deep as f64, cfg.deep_reps, cfg.deep_iters, cfg.gw_trials)
            } else {
                sc.stage_d_and_seed(
                    seed,
                    cfg.run_unseeded,
                    cfg.bp_weight,
                    cfg.bp_seed_weight,
                    bm_deep as f64,
                )
            };
            if !score.is_finite() {
                continue;
            }
            if cfg.stage_e_keep < 0.999 {
                task.gates.stage_e.observe_batch(&[score]);
                if !task.gates.stage_e.check_fast(score) {
                    continue;
                }
            }
            let cutoff = task.pool_cutoff;
            if cutoff.is_finite() && score > cutoff + 4.0 {
                continue;
            }
            let send_seed =
                task.gates.bm30.is_warm() && (!cutoff.is_finite() || score <= cutoff + 1.5);
            if send_seed {
                stats.seeds_sent.fetch_add(1, Ordering::Relaxed);
            }
            hits.push(HitOut {
                index: idx,
                score,
                failed_fast: false,
                u_orig: if send_seed { u_orig } else { Vec::new() },
                evecs,
            });
        }
        stats.bp_us.fetch_add(t_bp.elapsed().as_micros() as u64, Ordering::Relaxed);
        if !task.client.is_cancelled(task.lease_id) {
            if !hits.is_empty() {
                stats.hits_sent.fetch_add(hits.len() as u64, Ordering::Relaxed);
            }
            task.client.send_chunk_result(
                task.lease_id,
                task.chunk_id,
                task.start_idx,
                task.count,
                &hits,
            );
        }
    }
}

fn handle_client(
    stream: TcpStream,
    cid: u64,
    token: String,
    cfg: ServerCfg,
    sched: Arc<Scheduler>,
    topo_cache: Arc<Mutex<HashMap<([u8; 32], [u8; 32]), (Arc<FastDimer>, Arc<TopoGates>)>>>,
) {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "unknown".into());
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(120)));
    let Ok(w_stream) = stream.try_clone() else { return };
    let mut reader = BufReader::with_capacity(256 * 1024, stream);
    let mut writer = BufWriter::with_capacity(256 * 1024, w_stream);

    let mut magic = [0u8; 4];
    if reader.read_exact(&mut magic).is_err() || &magic != MAGIC { return; }
    let Ok(tlen) = read_u16(&mut reader) else { return };
    if tlen > 256 { return; }
    let mut tbuf = vec![0u8; tlen as usize];
    if reader.read_exact(&mut tbuf).is_err() || tbuf != token.as_bytes() {
        eprintln!("[client {cid} {peer}] auth rejected");
        return;
    }
    let Ok(nlen) = read_u16(&mut reader) else { return };
    let mut nbuf = vec![0u8; (nlen as usize).min(128)];
    if reader.read_exact(&mut nbuf).is_err() { return; }
    let name = String::from_utf8_lossy(&nbuf).to_string();
    if writer.write_all(OKAY).and_then(|_| writer.flush()).is_err() { return; }

    eprintln!("[client {cid} {peer}] connected name={name:?}");
    let client = Arc::new(ClientCtx {
        writer: Mutex::new(writer),
        closed: AtomicBool::new(false),
        cancelled_leases: Mutex::new(HashSet::new()),
    });
    sched.register_client(cid);
    let mut current_topo: Option<(Arc<FastDimer>, Arc<TopoGates>)> = None;

    loop {
        let op = match read_u8(&mut reader) { Ok(v) => v, Err(_) => break };
        match op {
            OP_SET_TOPO => {
                let mut od = [0u8; 32];
                let mut dd = [0u8; 32];
                if reader.read_exact(&mut od).is_err() || reader.read_exact(&mut dd).is_err() { break; }
                let Ok(plen) = read_u32(&mut reader) else { break };
                if plen > 1_000_000 { break; }
                let mut params = vec![0u8; plen as usize];
                if reader.read_exact(&mut params).is_err() { break; }
                let Ok(nlen) = read_u32(&mut reader) else { break };
                if nlen > 100_000 { break; }
                let mut raw_nodes = vec![0u8; nlen as usize * 4];
                if reader.read_exact(&mut raw_nodes).is_err() { break; }
                let node_ids: Vec<u32> = raw_nodes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let Ok(elen) = read_u32(&mut reader) else { break };
                if elen > 500_000 { break; }
                let mut raw_edges = vec![0u8; elen as usize * 8];
                if reader.read_exact(&mut raw_edges).is_err() { break; }
                let edges: Vec<(u32, u32)> = raw_edges
                    .chunks_exact(8)
                    .map(|c| (u32::from_le_bytes([c[0], c[1], c[2], c[3]]), u32::from_le_bytes([c[4], c[5], c[6], c[7]])))
                    .collect();

                let key = (od, dd);
                let cached = topo_cache.lock().unwrap().get(&key).cloned();
                let res = if let Some(pair) = cached {
                    Ok(pair)
                } else {
                    match FastDimer::from_raw(node_ids, &edges, &params, od) {
                        Ok(built) => {
                            let arc = Arc::new(built);
                            let gates = Arc::new(TopoGates::new(cfg));
                            eprintln!(
                                "[client {cid} {name}] built FastDimer VERIFIED! nodes={} edges={} pairs={} weak={}",
                                arc.nodes, arc.edges.len(), arc.pairs, arc.weak
                            );
                            topo_cache.lock().unwrap().insert(key, (Arc::clone(&arc), Arc::clone(&gates)));
                            Ok((arc, gates))
                        }
                        Err(e) => Err(e),
                    }
                };
                if let Ok(pair) = res { current_topo = Some(pair); }
            }
            OP_SUBMIT_CHUNK => {
                let Ok(lease_id) = read_u64(&mut reader) else { break };
                let Ok(chunk_id) = read_u64(&mut reader) else { break };
                let Ok(start_idx) = read_u64(&mut reader) else { break };
                let Ok(count) = read_u32(&mut reader) else { break };
                let Ok(pool_cutoff) = read_f32(&mut reader) else { break };
                if count == 0 || count > 8192 { break; }
                let mut raw = vec![0u8; count as usize * 32];
                if reader.read_exact(&mut raw).is_err() { break; }
                let Some((topo, gates)) = current_topo.as_ref().cloned() else { continue };

                let mut nonces = Vec::with_capacity(count as usize);
                let mut keys_flat = Vec::with_capacity(count as usize * 40);
                let mut seeds = Vec::with_capacity(count as usize);
                for c in raw.chunks_exact(32) {
                    let mut n = [0u8; 32];
                    n.copy_from_slice(c);
                    let kw = topo.compute_stream_keys(&n).unwrap_or([0u32; 40]);
                    keys_flat.extend_from_slice(&kw);
                    seeds.push(u64::from_le_bytes(n[0..8].try_into().unwrap()));
                    nonces.push(n);
                }
                sched.push_chunk(
                    cid,
                    ChunkJob {
                        client: Arc::clone(&client),
                        topo,
                        gates,
                        lease_id,
                        chunk_id,
                        start_idx,
                        count,
                        pool_cutoff,
                        nonces,
                        keys_flat,
                        seeds,
                    },
                );
            }
            OP_CANCEL_LEASE => {
                let Ok(lease_id) = read_u64(&mut reader) else { break };
                let mut set = client.cancelled_leases.lock().unwrap();
                if set.len() > 256 { set.clear(); }
                set.insert(lease_id);
            }
            OP_PING => client.send_pong(),
            _ => break,
        }
    }

    client.closed.store(true, Ordering::Relaxed);
    sched.unregister_client(cid);
    eprintln!("[client {cid} {peer}] disconnected name={name:?}");
}

fn load_topo_bin(topo_path: &str) -> Arc<FastDimer> {
    let mut f = std::fs::File::open(topo_path).expect("open topo_path");
    let mut od = [0u8; 32];
    let mut dd = [0u8; 32];
    f.read_exact(&mut od).unwrap();
    f.read_exact(&mut dd).unwrap();
    let plen = read_u32(&mut f).unwrap() as usize;
    let mut params = vec![0u8; plen];
    f.read_exact(&mut params).unwrap();
    let nlen = read_u32(&mut f).unwrap() as usize;
    let mut raw_nodes = vec![0u8; nlen * 4];
    f.read_exact(&mut raw_nodes).unwrap();
    let node_ids: Vec<u32> = raw_nodes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let elen = read_u32(&mut f).unwrap() as usize;
    let mut raw_edges = vec![0u8; elen * 8];
    f.read_exact(&mut raw_edges).unwrap();
    let edges: Vec<(u32, u32)> = raw_edges
        .chunks_exact(8)
        .map(|c| (u32::from_le_bytes([c[0], c[1], c[2], c[3]]), u32::from_le_bytes([c[4], c[5], c[6], c[7]])))
        .collect();
    Arc::new(FastDimer::from_raw(node_ids, &edges, &params, od).expect("FastDimer"))
}

fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va > 0.0 && vb > 0.0 { cov / (va * vb).sqrt() } else { 0.0 }
}

fn run_verify_and_bench(topo_path: &str, rt: Arc<GpuSharedRuntime>, batch: usize, n_salts: usize) {
    let fd = load_topo_bin(topo_path);
    let mut arena = GpuArena::new(Arc::clone(&rt), batch.max(512), 128).expect("GpuArena");
    let gpu_topo = rt.get_or_upload_topo(&arena.stream, &fd).expect("upload topo");
    let weak_f = fd.weak as f32;

    eprintln!("=== [1/4] BYTE-EXACT VERIFICATION: GPU FastDimer + SMEM-FP16 BM4/BM8 + FP32 BM8_w30 (64 salts) ===");
    let v_n = 64usize;
    let mut nonces = Vec::with_capacity(v_n);
    let mut keys = Vec::with_capacity(v_n * 40);
    let mut seeds = Vec::with_capacity(v_n);
    for i in 0..v_n as u64 {
        let mut nc = [0u8; 32];
        let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1337);
        nc[0..8].copy_from_slice(&x.to_le_bytes());
        keys.extend_from_slice(&fd.compute_stream_keys(&nc).unwrap());
        seeds.push(u64::from_le_bytes(nc[0..8].try_into().unwrap()));
        nonces.push(nc);
    }

    let _n_gpu = arena.generate_batch(&gpu_topo, &keys, &seeds).expect("gen");
    let mut sc = fd.new_scratch();
    let mut cpu_bm4 = Vec::new();
    let mut cpu_bm8 = Vec::new();
    let mut cpu_w30 = Vec::new();
    for i in 0..v_n {
        let cpu_g = fd.generate_gpu_input(nonces[i], seeds[i], &mut sc).expect("cpu gen");
        let gpu_g = arena.readback_generated_slot(i).expect("readback");
        assert_eq!(cpu_g.n_super, gpu_g.n_super, "n_super mismatch at {i}");
        assert_eq!(cpu_g.n_tiles, gpu_g.n_tiles, "n_tiles mismatch at {i}");
        assert!((cpu_g.base_const - gpu_g.base_const).abs() < 1e-4, "base_const mismatch at {i}");
        assert_eq!(cpu_g.tiles, gpu_g.tiles, "tiles mismatch at {i}");
        assert_eq!(cpu_g.gauges, gpu_g.gauges, "gauges mismatch at {i}");
        assert_eq!(cpu_g.weak_off, gpu_g.weak_off, "weak_off mismatch at {i}");
        // Parallel SMEM weak CSR scatter preserves exact per-row multiset of (nb | sign)
        let mut cpu_w = cpu_g.weak_nb_packed.clone();
        let mut gpu_w = gpu_g.weak_nb_packed.clone();
        for u in 0..cpu_g.n_super as usize {
            let s = cpu_g.weak_off[u] as usize;
            let e = cpu_g.weak_off[u + 1] as usize;
            cpu_w[s..e].sort_unstable();
            gpu_w[s..e].sort_unstable();
        }
        assert_eq!(cpu_w, gpu_w, "weak_nb_packed multiset mismatch at {i}");
        assert!(fd.generate_into_scratch(nonces[i], &mut sc));
        cpu_bm4.push(sc.bm4_sdp(seeds[i], 35, 0.20, 1.00, 2.0, 0.22));
        cpu_bm8.push(sc.bm8_sdp(seeds[i], 60, 0.20, 1.00, 2.0, 0.20));
        cpu_w30.push(sc.bm8_sdp(seeds[i].wrapping_add(29), 200, 0.20, 1.00, 3.0, 0.15));
    }
    eprintln!("  [PASS] GPU FastDimer 100.000% BYTE-FOR-BYTE IDENTICAL to CPU across all {v_n} salts!");

    let n_gpu_1tps = arena.generate_batch_1tps(&gpu_topo, &keys, &seeds, 512).expect("gen_1tps");
    for i in 0..v_n {
        let cpu_g = fd.generate_gpu_input(nonces[i], seeds[i], &mut sc).expect("cpu gen");
        let gpu_g = arena.readback_generated_slot(i).expect("readback 1tps");
        assert_eq!(cpu_g.n_super, gpu_g.n_super, "1tps n_super mismatch at {i}");
        assert_eq!(cpu_g.n_tiles, gpu_g.n_tiles, "1tps n_tiles mismatch at {i}");
        assert!((cpu_g.base_const - gpu_g.base_const).abs() < 1e-4, "1tps base_const mismatch at {i}");
        assert_eq!(cpu_g.tiles, gpu_g.tiles, "1tps tiles mismatch at {i}");
        assert_eq!(cpu_g.gauges, gpu_g.gauges, "1tps gauges mismatch at {i}");
        assert_eq!(cpu_g.weak_off, gpu_g.weak_off, "1tps weak_off mismatch at {i}");
        let mut cpu_w = cpu_g.weak_nb_packed.clone();
        let mut gpu_w = gpu_g.weak_nb_packed.clone();
        for u in 0..cpu_g.n_super as usize {
            let s = cpu_g.weak_off[u] as usize;
            let e = cpu_g.weak_off[u + 1] as usize;
            cpu_w[s..e].sort_unstable();
            gpu_w[s..e].sort_unstable();
        }
        assert_eq!(cpu_w, gpu_w, "1tps weak_nb_packed mismatch at {i}");
    }
    eprintln!("  [PASS] GPU v5.0 1-Thread-Per-Salt FastDimer 100.000% BYTE-FOR-BYTE IDENTICAL across all {v_n} salts!");

    let gpu_bm2: Vec<f64> = arena.run_bm2_all(v_n, true, weak_f, 20, 0.25).unwrap().into_iter().map(|x| x as f64).collect();
    let gpu_bm4: Vec<f64> = arena.run_bm4_smem(None, v_n, true, weak_f, 35, 0.22).unwrap().into_iter().map(|x| x as f64).collect();
    let gpu_bm8: Vec<f64> = arena.run_bm8_smem(None, v_n, true, weak_f, 60, 0.20).unwrap().into_iter().map(|x| x as f64).collect();
    let slots_all: Vec<i32> = (0..v_n as i32).collect();
    let gpu_w30_res = arena.run_w30_fp32(&slots_all, &n_gpu_1tps, weak_f).unwrap();
    let gpu_w30: Vec<f64> = gpu_w30_res.iter().map(|(e, _)| *e as f64).collect();

    let max_d4 = gpu_bm4.iter().zip(&cpu_bm4).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let max_d8 = gpu_bm8.iter().zip(&cpu_bm8).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let max_dw = gpu_w30.iter().zip(&cpu_w30).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    eprintln!("  SMEM-FP16 BM2_s20 vs BM8_w30:  r={:.6} (vs BM4_s35 r={:.6})", pearson_r(&gpu_bm2, &gpu_w30), pearson_r(&gpu_bm2, &gpu_bm4));
    eprintln!("  SMEM-FP16 BM4_s35 vs CPU FP32: r={:.6}, max_diff={:.4} (using 1tps CSR!)", pearson_r(&gpu_bm4, &cpu_bm4), max_d4);
    eprintln!("  SMEM-FP16 BM8_s60 vs CPU FP32: r={:.6}, max_diff={:.4}", pearson_r(&gpu_bm8, &cpu_bm8), max_d8);
    eprintln!("  FP32      BM8_w30 vs CPU FP32: r={:.6}, max_diff={:.6}", pearson_r(&gpu_w30, &cpu_w30), max_dw);
    assert!(pearson_r(&gpu_bm8, &cpu_bm8) > 0.995 && max_dw < 0.01);

    // Prepare bench batch keys
    let mut b_keys = Vec::with_capacity(batch * 40);
    let mut b_seeds = Vec::with_capacity(batch);
    let mut b_nonces = Vec::with_capacity(batch);
    for i in 0..batch as u64 {
        let mut nc = [0u8; 32];
        let x = i.wrapping_mul(0x517C_C1B7_2722_0A95).wrapping_add(99);
        nc[0..8].copy_from_slice(&x.to_le_bytes());
        b_keys.extend_from_slice(&fd.compute_stream_keys(&nc).unwrap());
        b_seeds.push(u64::from_le_bytes(nc[0..8].try_into().unwrap()));
        b_nonces.push(nc);
    }
    let rounds = (n_salts / batch).max(6);
    let tot = rounds * batch;

    eprintln!("=== [2/5] PURE GPU FastDimer GENERATION: v4.1 vs v5.0 1-Thread-Per-Salt ({tot} salts, batch={batch}) ===");
    let t_gen = Instant::now();
    for _ in 0..rounds {
        let _ = arena.generate_batch(&gpu_topo, &b_keys, &b_seeds).unwrap();
    }
    let dt_gen = t_gen.elapsed().as_secs_f64();
    eprintln!("  v4.1 (3-blk/SM 128t):    {:8.1} salts/s ({:.2}s)", tot as f64 / dt_gen, dt_gen);

    let mut best_w = 4096usize;
    let mut best_w_rate = 0.0f64;
    for nw in [2048usize, 4096, 6144, 8192] {
        if nw > batch { continue; }
        let t1 = Instant::now();
        for _ in 0..rounds {
            let _ = arena.generate_batch_1tps(&gpu_topo, &b_keys, &b_seeds, nw).unwrap();
        }
        let dt1 = t1.elapsed().as_secs_f64();
        let r1 = tot as f64 / dt1;
        eprintln!("  v5.1 SoA 1tps (W={nw:4}):  {:8.1} salts/s ({:.2}s, {:.2}x vs v4.1)", r1, dt1, r1 / (tot as f64 / dt_gen));
        if r1 > best_w_rate { best_w_rate = r1; best_w = nw; }
    }

    // Also measure pure CPU FastDimer throughput on 64 EPYC threads!
    {
        let t_cpu = Instant::now();
        let cpu_n = 8192usize;
        let n_thr = 64usize;
        let per = cpu_n / n_thr;
        std::thread::scope(|s| {
            for t in 0..n_thr {
                let fd_r = &fd;
                s.spawn(move || {
                    let mut sc = fd_r.new_scratch();
                    for i in 0..per {
                        let mut nc = [0u8; 32];
                        let x = ((t * per + i) as u64).wrapping_mul(0x517C_C1B7_2722_0A95).wrapping_add(7);
                        nc[0..8].copy_from_slice(&x.to_le_bytes());
                        let _ = fd_r.generate_into_scratch(nc, &mut sc);
                    }
                });
            }
        });
        let dt_c = t_cpu.elapsed().as_secs_f64();
        eprintln!("  CPU 64-thr FastDimer:    {:8.1} salts/s ({cpu_n} salts in {:.2}s)", cpu_n as f64 / dt_c, dt_c);
    }

    eprintln!("=== [3/5] PURE SMEM-FP16 BM2_s20 / BM4_s35 / BM8_s60 PEAK ({tot} salts, batch={batch}) ===");
    let t_b2 = Instant::now();
    for _ in 0..rounds {
        let _ = arena.run_bm2_all(batch, true, weak_f, 20, 0.25).unwrap();
    }
    let dt_b2 = t_b2.elapsed().as_secs_f64();
    eprintln!("  SMEM-FP16 BM2_s20 ONLY:  {:8.1} salts/s ({:.2}s)", tot as f64 / dt_b2, dt_b2);

    let t_b4 = Instant::now();
    for _ in 0..rounds {
        let _ = arena.run_bm4_smem(None, batch, true, weak_f, 35, 0.22).unwrap();
    }
    let dt_b4 = t_b4.elapsed().as_secs_f64();
    eprintln!("  SMEM-FP16 BM4_s35 ONLY:  {:8.1} salts/s ({:.2}s, skip_csr=1)", tot as f64 / dt_b4, dt_b4);

    let t_b8 = Instant::now();
    for _ in 0..rounds {
        let _ = arena.run_bm8_smem(None, batch, true, weak_f, 60, 0.20).unwrap();
    }
    let dt_b8 = t_b8.elapsed().as_secs_f64();
    eprintln!("  SMEM-FP16 BM8_s60 ONLY:  {:8.1} salts/s ({:.2}s)", tot as f64 / dt_b8, dt_b8);

    drop(arena);
    eprintln!("=== [4/5] 3-STREAM END-TO-END: v4.1 vs v5.0(1tps, W={best_w}) vs v5.0+BM2(20@45%) ===");
    for (label, use_1tps, use_bm2) in [
        ("v4.1 baseline (3-str)       ", false, false),
        ("v5.0 1tps exact (3-str)     ", true, false),
        ("v5.0 1tps + BM2@45% (3-str) ", true, true),
    ] {
        let n_st = 3usize;
        let mut arenas: Vec<GpuArena> = (0..n_st)
            .map(|_| GpuArena::new(Arc::clone(&rt), batch, 128).expect("arena"))
            .collect();
        let b_keys_arc = Arc::new(b_keys.clone());
        let b_seeds_arc = Arc::new(b_seeds.clone());
        let g2m = Arc::new(FastGate::new(0.45, 2048, 128, 64));
        let g4m = Arc::new(FastGate::new(if use_bm2 { 0.16 / 0.45 } else { 0.16 }, 2048, 128, 64));
        let g60m = Arc::new(FastGate::new(0.012 / 0.16, 2048, 128, 64));
        let rounds_each = (rounds / n_st).max(3);
        let t_m = Instant::now();
        std::thread::scope(|s| {
            for mut ar in arenas.drain(..) {
                let topo_g = Arc::clone(&gpu_topo);
                let bk = Arc::clone(&b_keys_arc);
                let bs = Arc::clone(&b_seeds_arc);
                let g2m = Arc::clone(&g2m);
                let g4m = Arc::clone(&g4m);
                let g60m = Arc::clone(&g60m);
                s.spawn(move || {
                    for _ in 0..rounds_each {
                        let n_arr = if use_1tps {
                            ar.generate_batch_1tps(&topo_g, &bk, &bs, best_w).unwrap()
                        } else {
                            ar.generate_batch(&topo_g, &bk, &bs).unwrap()
                        };
                        let s4 = if use_bm2 {
                            let sc2 = ar.run_bm2_all(batch, use_1tps, weak_f, 20, 0.25).unwrap();
                            g2m.observe_batch(&sc2);
                            let mut k2 = Vec::new();
                            for (idx, &v) in sc2.iter().enumerate() {
                                if n_arr[idx] > 0 && g2m.check_fast(v) { k2.push(idx as i32); }
                            }
                            let mut k4 = Vec::new();
                            if !k2.is_empty() {
                                let sc4 = ar.run_bm4_smem(Some(&k2), batch, true, weak_f, 30, 0.22).unwrap();
                                g4m.observe_batch(&sc4);
                                for (&sl, &v) in k2.iter().zip(&sc4) {
                                    if g4m.check_fast(v) { k4.push(sl); }
                                }
                            }
                            k4
                        } else {
                            let sc4 = ar.run_bm4_smem(None, batch, use_1tps, weak_f, 35, 0.22).unwrap();
                            g4m.observe_batch(&sc4);
                            let mut k4 = Vec::new();
                            for (idx, &v) in sc4.iter().enumerate() {
                                if n_arr[idx] > 0 && g4m.check_fast(v) { k4.push(idx as i32); }
                            }
                            k4
                        };
                        if !s4.is_empty() {
                            let sc8 = ar.run_bm8_smem(Some(&s4), batch, true, weak_f, 60, 0.20).unwrap();
                            g60m.observe_batch(&sc8);
                            let mut s0b = Vec::new();
                            for (&slot, &v) in s4.iter().zip(&sc8) {
                                if g60m.check_fast(v) { s0b.push(slot); }
                            }
                            if !s0b.is_empty() {
                                let _ = ar.run_w30_fp32_filtered(&s0b, &n_arr, weak_f, |es| {
                                    let mut keep = Vec::new();
                                    if !es.is_empty() { keep.push(0); }
                                    keep
                                }).unwrap();
                            }
                        }
                    }
                });
            }
        });
        let dt_m = t_m.elapsed().as_secs_f64();
        let tot_m = n_st * rounds_each * batch;
        eprintln!("  {label}: {:8.1} salts/s ({tot_m} salts in {:.2}s)", tot_m as f64 / dt_m, dt_m);
    }
}

fn hex_to_nonce(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 { return None; }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = (b[2*i] as char).to_digit(16)?;
        let lo = (b[2*i+1] as char).to_digit(16)?;
        out[i] = ((hi << 4) | lo) as u8;
    }
    Some(out)
}

fn compute_auc(scores: &[(f64, bool)]) -> f64 {
    // Lower score = better (more positive)!
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for &(s, is_pos) in scores {
        if is_pos { pos.push(s); } else { neg.push(s); }
    }
    if pos.is_empty() || neg.is_empty() { return 0.5; }
    let mut wins = 0.0;
    for &p in &pos {
        for &n in &neg {
            if p < n { wins += 1.0; } else if p == n { wins += 0.5; }
        }
    }
    wins / ((pos.len() * neg.len()) as f64)
}

fn recall_at_keep(scores: &[(f64, bool)], keep: f64) -> (usize, usize) {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cut = ((v.len() as f64) * keep).ceil() as usize;
    let tot_pos = v.iter().filter(|x| x.1).count();
    let hit_pos = v[..cut.min(v.len())].iter().filter(|x| x.1).count();
    (hit_pos, tot_pos)
}

fn run_eval_real(topo_path: &str, tsv_path: &str, rt: Arc<GpuSharedRuntime>, threads: usize) {
    let fd = load_topo_bin(topo_path);
    let raw = std::fs::read_to_string(tsv_path).expect("read tsv");
    let mut rows: Vec<([u8; 32], String, i64, i64)> = Vec::new();
    let mut probe_neg = 0usize;
    let mut p0_neg = 0usize;
    for line in raw.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 { continue; }
        let Some(nc) = hex_to_nonce(parts[0]) else { continue; };
        let st = parts[1].to_string();
        let p0: i64 = parts[2].parse().unwrap_or(0);
        let fb: i64 = parts[3].parse().unwrap_or(0);
        // Keep ALL deep (<= -5088), plus up to 800 probe-killed + 800 probe0-killed for fast balanced eval!
        if fb <= -5088_000 {
            rows.push((nc, st, p0, fb));
        } else if st == "probe0" && p0_neg < 300 {
            rows.push((nc, st, p0, fb));
            p0_neg += 1;
        } else if st != "probe0" && probe_neg < 300 {
            rows.push((nc, st, p0, fb));
            probe_neg += 1;
        }
    }
    let n_rows = rows.len();
    let n_deep88 = rows.iter().filter(|r| r.3 <= -5088_000).count();
    let n_deep90 = rows.iter().filter(|r| r.3 <= -5090_000).count();
    let n_deep93 = rows.iter().filter(|r| r.3 <= -5093_000).count();
    eprintln!("=== REAL LIVE NONCE EVAL: n={n_rows} (<= -5088: {n_deep88}, <= -5090: {n_deep90}, <= -5093: {n_deep93}) ===");

    // Also measure BM2/BM4 distribution on 2048 random background nonces so we know exact population quantile!
    let mut arena = GpuArena::new(Arc::clone(&rt), 1024, 1024).expect("arena");
    let gpu_topo = rt.get_or_upload_topo(&arena.stream, &fd).expect("topo");
    let weak_f = fd.weak as f32;

    let mut bg_keys = Vec::with_capacity(2048 * 40);
    let mut bg_seeds = Vec::with_capacity(2048);
    for i in 0..2048u64 {
        let mut nc = [0u8; 32];
        let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(424242);
        nc[0..8].copy_from_slice(&x.to_le_bytes());
        bg_keys.extend_from_slice(&fd.compute_stream_keys(&nc).unwrap());
        bg_seeds.push(u64::from_le_bytes(nc[0..8].try_into().unwrap()));
    }
    let mut bg_bm2 = Vec::new();
    let mut bg_bm4 = Vec::new();
    for ch in 0..2 {
        let k = &bg_keys[ch * 1024 * 40..(ch + 1) * 1024 * 40];
        let s = &bg_seeds[ch * 1024..(ch + 1) * 1024];
        let _ = arena.generate_batch_1tps(&gpu_topo, k, s, 512).unwrap();
        bg_bm2.extend(arena.run_bm2_all(1024, true, weak_f, 20, 0.25).unwrap());
        bg_bm4.extend(arena.run_bm4_smem(None, 1024, true, weak_f, 35, 0.22).unwrap());
    }
    bg_bm2.sort_by(f32::total_cmp);
    bg_bm4.sort_by(f32::total_cmp);

    // Run GPU BM2, BM4, BM8, BM8_w30 on all `n_rows` real nonces!
    let mut gpu_bm2 = Vec::with_capacity(n_rows);
    let mut gpu_bm4 = Vec::with_capacity(n_rows);
    let mut gpu_w30_e = Vec::with_capacity(n_rows);
    let mut gpu_w30_u = Vec::with_capacity(n_rows);
    for chunk in rows.chunks(512) {
        let k = chunk.len();
        let mut ks = Vec::with_capacity(k * 40);
        let mut ss = Vec::with_capacity(k);
        for r in chunk {
            ks.extend_from_slice(&fd.compute_stream_keys(&r.0).unwrap());
            ss.push(u64::from_le_bytes(r.0[0..8].try_into().unwrap()));
        }
        let n_arr = arena.generate_batch_1tps(&gpu_topo, &ks, &ss, 512).unwrap();
        gpu_bm2.extend(arena.run_bm2_all(k, true, weak_f, 20, 0.25).unwrap());
        gpu_bm4.extend(arena.run_bm4_smem(None, k, true, weak_f, 35, 0.22).unwrap());
        let slots: Vec<i32> = (0..k as i32).collect();
        for (e, u) in arena.run_w30_fp32(&slots, &n_arr, weak_f).unwrap() {
            gpu_w30_e.push(e as f64);
            gpu_w30_u.push(u);
        }
    }

    // Check BM2 population recall on real deep nonces!
    for pct in [0.30, 0.40, 0.45, 0.50, 0.60] {
        let cut = bg_bm2[((bg_bm2.len() as f64) * pct) as usize];
        let mut h88 = 0; let mut h90 = 0; let mut h93 = 0;
        for (i, r) in rows.iter().enumerate() {
            if gpu_bm2[i] <= cut {
                if r.3 <= -5088_000 { h88 += 1; }
                if r.3 <= -5090_000 { h90 += 1; }
                if r.3 <= -5093_000 { h93 += 1; }
            }
        }
        eprintln!("  BM2(20) @ top {:2.0}% pop (cut={:.2}): recall <=-5088={}/{} ({:.1}%), <=-5090={}/{} ({:.1}%), <=-5093={}/{}",
            pct*100.0, cut, h88, n_deep88, 100.0*h88 as f64/n_deep88.max(1) as f64,
            h90, n_deep90, 100.0*h90 as f64/n_deep90.max(1) as f64, h93, n_deep93);
    }

    // Parallel CPU Stage-0d (V3Fast & V3) + Stage-0e (GW48 + Deep Polished K4-BP 12x240) across EPYC cores!
    let t_cpu = Instant::now();
    let next_idx = AtomicU64::new(0);
    // Output tuple: (bm_w30, v3fast, v3, gw_best, gw_m5, dbp_best, dbp_mean, dbp_top3, fb)
    let out = Mutex::new(vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0i64); n_rows]);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            let fd_ref = &fd;
            let rows_ref = &rows;
            let w30_e_ref = &gpu_w30_e;
            let w30_u_ref = &gpu_w30_u;
            let idx_ref = &next_idx;
            let out_ref = &out;
            s.spawn(move || {
                let mut sc = fd_ref.new_scratch();
                loop {
                    let i = idx_ref.fetch_add(1, Ordering::Relaxed) as usize;
                    if i >= n_rows { break; }
                    let nc = rows_ref[i].0;
                    let seed = u64::from_le_bytes(nc[0..8].try_into().unwrap());
                    if !fd_ref.generate_into_scratch(nc, &mut sc) { continue; }
                    sc.set_u8_from_gpu(&w30_u_ref[i]);
                    let w30 = w30_e_ref[i];
                    let (v3f, _, _) = sc.stage_d_and_seed(seed, false, 0.15, 0.20, w30);
                    let (v3, _, _)  = sc.stage_d_and_seed(seed, true, 0.15, 0.20, w30);
                    let (se_fix4, _, _) = sc.stage_e_super(seed, w30, 4, 160, 20);
                    // Also measure old buggy 4x160 (unscaled ramps=[90,130,170,200], top3) via deep_bp_polished!
                    let (_, _, old_dbp_t3) = sc.deep_bp_polished(seed, 4, 160);
                    let (_, gw_m5, _, _, _) = sc.gw_round_and_polish(seed, 20);
                    let se_buggy4 = ((v3 as f64 + 0.30 * old_dbp_t3 + 0.15 * gw_m5) * (1.20 / 1.80)) as f64;
                    let se_v3_gw = ((v3 as f64 + 0.20 * gw_m5) * (1.20 / 1.55)) as f64;
                    out_ref.lock().unwrap()[i] = (w30, v3f as f64, v3 as f64, se_buggy4, se_fix4 as f64, se_v3_gw, old_dbp_t3, gw_m5, rows_ref[i].3);
                }
            });
        }
    });
    let dt_cpu = t_cpu.elapsed().as_secs_f64();
    let res = out.into_inner().unwrap();
    eprintln!("  CPU Stage-0d+0e(GW48 + DeepBP 8x220+1flip) on {n_rows} nonces: {:.2}s across {threads} threads", dt_cpu);

    let report = |name: &str, f: &dyn Fn(&(f64,f64,f64,f64,f64,f64,f64,f64,i64)) -> f64| {
        let s88: Vec<(f64, bool)> = res.iter().map(|r| (f(r), r.8 <= -5088_000)).collect();
        let s90: Vec<(f64, bool)> = res.iter().map(|r| (f(r), r.8 <= -5090_000)).collect();
        let s93: Vec<(f64, bool)> = res.iter().map(|r| (f(r), r.8 <= -5093_000)).collect();
        let (h90_33, t90) = recall_at_keep(&s90, 0.333);
        let (h88_33, t88) = recall_at_keep(&s88, 0.333);
        let (h93_33, t93) = recall_at_keep(&s93, 0.333);
        let (h90_50, _) = recall_at_keep(&s90, 0.50);
        eprintln!(
            "  {:28} | AUC90={:.4} AUC88={:.4} | Top33%(3:1 cut): <=-5090 {}/{} ({:5.1}%) <=-5088 {}/{} ({:5.1}%) <=-5093 {}/{} | Top50%: {}/{}",
            name, compute_auc(&s90), compute_auc(&s88),
            h90_33, t90, 100.0*h90_33 as f64/t90.max(1) as f64,
            h88_33, t88, 100.0*h88_33 as f64/t88.max(1) as f64,
            h93_33, t93, h90_50, t90
        );
    };
    eprintln!("=== LIVE CONFIG ABLATION (REAL NONCES): v3fast vs v3 vs v5.0 Super-Stage-0e ===");
    report("1. v3fast (v4.1 baseline)",    &|r| r.1);
    report("2. v3 (8x120 BP only)",         &|r| r.2);
    report("3. v5.0 ref (deep_bp + GW20)",  &|r| r.3);
    report("4. v5.0 super (4x160 + GW20)",  &|r| r.4);
    report("5. v3 + 0.20*GW20 (no DeepBP)", &|r| r.5);
    let mut vals_buggy: Vec<f64> = res.iter().map(|r| r.3).collect();
    let mut vals_fix4: Vec<f64> = res.iter().map(|r| r.4).collect();
    vals_buggy.sort_by(f64::total_cmp);
    vals_fix4.sort_by(f64::total_cmp);
    eprintln!("  Spread check: BUGGY4 p05={:.2} p50={:.2} p95={:.2} (p95-p05={:.2})", vals_buggy[vals_buggy.len()*5/100], vals_buggy[vals_buggy.len()/2], vals_buggy[vals_buggy.len()*95/100], vals_buggy[vals_buggy.len()*95/100]-vals_buggy[vals_buggy.len()*5/100]);
    eprintln!("  Spread check: FIXED4 p05={:.2} p50={:.2} p95={:.2} (p95-p05={:.2})", vals_fix4[vals_fix4.len()*5/100], vals_fix4[vals_fix4.len()/2], vals_fix4[vals_fix4.len()*95/100], vals_fix4[vals_fix4.len()*95/100]-vals_fix4[vals_fix4.len()*5/100]);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get_arg = |flag: &str, def: &str| -> String {
        for i in 0..args.len().saturating_sub(1) {
            if args[i] == flag { return args[i + 1].clone(); }
        }
        def.to_string()
    };

    let device: usize = get_arg("--device", "0").parse().unwrap_or(0);
    let streams: usize = get_arg("--streams", "2").parse().unwrap_or(2);
    let batch: usize = get_arg("--batch", "512").parse().unwrap_or(512);
    let bp_threads: usize = get_arg("--bp-threads", "8").parse().unwrap_or(8);

    let rt = GpuSharedRuntime::init(device).expect("init GPU runtime");

    let bench_path = get_arg("--bench", "");
    if !bench_path.is_empty() {
        let n_salts: usize = get_arg("--n-salts", "20000").parse().unwrap_or(20000);
        run_verify_and_bench(&bench_path, rt, batch, n_salts);
        return;
    }

    let eval_topo = get_arg("--eval-topo", "");
    let eval_tsv = get_arg("--eval-tsv", "");
    if !eval_topo.is_empty() && !eval_tsv.is_empty() {
        run_eval_real(&eval_topo, &eval_tsv, rt, bp_threads);
        return;
    }

    let bench_se = get_arg("--bench-stage-e", "");
    if !bench_se.is_empty() {
        let fd = load_topo_bin(&bench_se);
        let mut arena = GpuArena::new(Arc::clone(&rt), 128, 128).expect("arena");
        let gpu_topo = rt.get_or_upload_topo(&arena.stream, &fd).expect("topo");
        let n_test = 128usize;
        let mut nonces = Vec::with_capacity(n_test);
        let mut keys = Vec::with_capacity(n_test * 40);
        let mut seeds = Vec::with_capacity(n_test);
        for i in 0..n_test as u64 {
            let mut nc = [0u8; 32];
            let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(777);
            nc[0..8].copy_from_slice(&x.to_le_bytes());
            keys.extend_from_slice(&fd.compute_stream_keys(&nc).unwrap());
            seeds.push(u64::from_le_bytes(nc[0..8].try_into().unwrap()));
            nonces.push(nc);
        }
        let n_arr = arena.generate_batch(&gpu_topo, &keys, &seeds).unwrap();
        let _ = arena.run_bm4_all(n_test, fd.weak as f32, 35, 0.22).unwrap();
        let slots: Vec<i32> = (0..n_test as i32).collect();
        let w30 = arena.run_w30_fp32(&slots, &n_arr, fd.weak as f32).unwrap();
        for (dr, di, gw) in [(6usize, 200usize, 32usize), (4, 200, 24), (8, 220, 48)] {
            let t0 = Instant::now();
            let idx = AtomicU64::new(0);
            std::thread::scope(|s| {
                for _ in 0..bp_threads {
                    let fd_r = &fd;
                    let n_r = &nonces;
                    let s_r = &seeds;
                    let w_r = &w30;
                    let i_r = &idx;
                    s.spawn(move || {
                        let mut sc = fd_r.new_scratch();
                        loop {
                            let k = i_r.fetch_add(1, Ordering::Relaxed) as usize;
                            if k >= n_test { break; }
                            if !fd_r.generate_into_scratch(n_r[k], &mut sc) { continue; }
                            sc.set_u8_from_gpu(&w_r[k].1);
                            let _ = sc.stage_e_super(s_r[k], w_r[k].0 as f64, dr, di, gw);
                        }
                    });
                }
            });
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("  stage_e_super(deep={dr}x{di}, gw={gw}) on {bp_threads} thr: {:.1} salts/s ({:.0} ms/salt 1-thr equiv)",
                n_test as f64 / dt, dt * 1000.0 * (bp_threads as f64) / (n_test as f64));
        }
        return;
    }

    let listen = get_arg("--listen", "127.0.0.1:19900");
    let token = get_arg("--token", "change-me");
    let gen_1tps_workers: usize = get_arg("--gen-1tps", "0").parse().unwrap_or(0);
    let bm2_pre_steps: i32 = get_arg("--bm2-pre-steps", "0").parse().unwrap_or(0);
    let bm2_pre_keep: f64 = get_arg("--bm2-pre-keep", "0.45").parse().unwrap_or(0.45);
    let bm4_pre_steps: i32 = get_arg("--bm4-pre-steps", "35").parse().unwrap_or(35);
    let bm4_pre_keep: f64 = get_arg("--bm4-pre-keep", "0.15").parse().unwrap_or(0.15);
    let bm60_steps: i32 = get_arg("--bm60-steps", "60").parse().unwrap_or(60);
    let bm60_keep: f64 = get_arg("--bm60-keep", "0.020").parse().unwrap_or(0.020);
    let bm30_keep: f64 = get_arg("--bm30-keep", "0.18").parse().unwrap_or(0.18);
    let stage_e_keep: f64 = get_arg("--stage-e-keep", "1.0").parse().unwrap_or(1.0);
    let deep_reps: usize = get_arg("--deep-reps", "6").parse().unwrap_or(6);
    let deep_iters: usize = get_arg("--deep-iters", "200").parse().unwrap_or(200);
    let gw_trials: usize = get_arg("--gw-trials", "32").parse().unwrap_or(32);
    let mode = get_arg("--mode", "v3fast");
    let super_stage_e = mode.eq_ignore_ascii_case("super");
    let run_unseeded = mode.eq_ignore_ascii_case("v3");

    let cfg = ServerCfg {
        gen_1tps_workers,
        bm2_pre_steps,
        bm2_pre_keep,
        bm4_pre_steps,
        bm4_pre_keep,
        bm60_steps,
        bm60_keep,
        bm30_keep,
        stage_e_keep,
        super_stage_e,
        deep_reps,
        deep_iters,
        gw_trials,
        run_unseeded,
        bp_weight: 0.15,
        bp_seed_weight: 0.20,
    };

    let listener = TcpListener::bind(&listen).expect("bind TCP listener");
    eprintln!(
        "quip-gpu-feeder v5.0 listening on {listen} (gpu={device}, streams={streams}, batch={batch}, bp_threads={bp_threads}, mode={mode}, stage_e_keep={stage_e_keep}, deep={deep_reps}x{deep_iters}+gw{gw_trials}, bm4={bm4_pre_steps}@{bm4_pre_keep}, bm60={bm60_steps}@{bm60_keep}, bm30_keep={bm30_keep})"
    );

    let sched = Arc::new(Scheduler::new());
    let stats = Arc::new(GlobalStats {
        screened: AtomicU64::new(0),
        s0b_pass: AtomicU64::new(0),
        s0c_pass: AtomicU64::new(0),
        hits_sent: AtomicU64::new(0),
        seeds_sent: AtomicU64::new(0),
        gen_us: AtomicU64::new(0),
        bm4_us: AtomicU64::new(0),
        bm8_us: AtomicU64::new(0),
        w30_us: AtomicU64::new(0),
        bp_us: AtomicU64::new(0),
    });
    let topo_cache = Arc::new(Mutex::new(HashMap::new()));
    let (bp_tx, bp_rx) = mpsc::sync_channel::<BpTask>(256);
    let bp_rx = Arc::new(Mutex::new(bp_rx));

    for i in 0..bp_threads {
        let rx = Arc::clone(&bp_rx);
        let st = Arc::clone(&stats);
        std::thread::Builder::new()
            .name(format!("bp-worker-{i}"))
            .spawn(move || bp_worker_loop(rx, cfg, st))
            .expect("spawn bp worker");
    }

    for i in 0..streams.max(1) {
        let arena = GpuArena::new(Arc::clone(&rt), batch, 256).expect("GpuArena");
        let s = Arc::clone(&sched);
        let r = Arc::clone(&rt);
        let tx = bp_tx.clone();
        let st = Arc::clone(&stats);
        std::thread::Builder::new()
            .name(format!("gpu-stream-{i}"))
            .spawn(move || gpu_worker_loop(s, r, arena, tx, cfg, st))
            .expect("spawn gpu worker");
    }

    {
        let s = Arc::clone(&sched);
        let st = Arc::clone(&stats);
        std::thread::spawn(move || {
            let mut prev = (Instant::now(), 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
            loop {
                std::thread::sleep(Duration::from_secs(15));
                let now = Instant::now();
                let dt = now.duration_since(prev.0).as_secs_f64().max(0.001);
                let sc = st.screened.load(Ordering::Relaxed);
                let b60 = st.s0b_pass.load(Ordering::Relaxed);
                let b30 = st.s0c_pass.load(Ordering::Relaxed);
                let hits = st.hits_sent.load(Ordering::Relaxed);
                let seeds = st.seeds_sent.load(Ordering::Relaxed);
                let g = st.gen_us.load(Ordering::Relaxed);
                let m4 = st.bm4_us.load(Ordering::Relaxed);
                let m8 = st.bm8_us.load(Ordering::Relaxed);
                let w3 = st.w30_us.load(Ordering::Relaxed);
                let bp = st.bp_us.load(Ordering::Relaxed);
                let dsc = sc - prev.1;
                if dsc > 0 || s.active_clients() > 0 {
                    let denom = dt * 1e6;
                    eprintln!(
                        "GPU_FEEDER_STATS clients={} streams={} salts_per_s={:.1} s0b_pct={:.2} s0c_pct={:.2} hits_per_s={:.2} seeds_per_s={:.2} t_gen={:.2} t_bm4={:.2} t_bm8={:.2} t_w30={:.2} t_bp={:.2}",
                        s.active_clients(),
                        streams,
                        dsc as f64 / dt,
                        100.0 * (b60 - prev.2) as f64 / dsc.max(1) as f64,
                        100.0 * (b30 - prev.3) as f64 / dsc.max(1) as f64,
                        (hits - prev.4) as f64 / dt,
                        (seeds - prev.5) as f64 / dt,
                        (g - prev.6) as f64 / denom,
                        (m4 - prev.7) as f64 / denom,
                        (m8 - prev.8) as f64 / denom,
                        (w3 - prev.9) as f64 / denom,
                        (bp - prev.10) as f64 / denom,
                    );
                }
                prev = (now, sc, b60, b30, hits, seeds, g, m4, m8, w3, bp);
            }
        });
    }

    let next_cid = AtomicU64::new(1);
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let cid = next_cid.fetch_add(1, Ordering::Relaxed);
                let tok = token.clone();
                let s = Arc::clone(&sched);
                let tc = Arc::clone(&topo_cache);
                std::thread::spawn(move || handle_client(stream, cid, tok, cfg, s, tc));
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}
