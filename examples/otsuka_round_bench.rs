//! Best twisted-ice energy one CUDA device reaches in a fixed-length round (manual bench).
//!
//! Generates canonical `quip/twisted-ice-dimer` salts on the Aglais topology and solves them with
//! the `quip-cuda-otsuka` cluster path, in one of three modes:
//!
//! - `plain`: every salt in order at `160 x 8,192`, the population-anneal defaults.
//! - `screen`: CPU threads score every salt with `screen_lease::score`; the GPU takes the best eight
//!   unsolved salts, solves each at `160 x 8,192`, then gives the best of them one long anneal
//!   (`256 x 262,144`). This models the opt-in lease screen with one lease holder.
//! - `stock`: every salt in order with the settings before the population anneal: no population
//!   step, one sweep per launch, full cluster weights, no weak scale, and the generic beta range.
//!   Run it with `QUIP_OTSUKA_RESAMPLES=0 QUIP_OTSUKA_CHUNK=1 QUIP_OTSUKA_BETA=stock`.
//!
//! Each round prints one JSON line. Energies include the instance offset, as the chain scores them.
//! A solve that the round deadline cuts off does not count.

use clap::{Parser, ValueEnum};
use quip_solver_conformance::families::export_instances;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{IsingGraph, SampleParams};
use quip_solver_cuda::capacity::DeviceLimits;
use quip_solver_cuda::cluster::device::ClusterDevice;
use quip_solver_cuda::cluster::sampler::ClusterWork;
use quip_solver_cuda::cluster::screen_lease::{score, DEFAULT_WEIGHTS};
use quip_solver_cuda::cluster::{PreparedGraph, Weights};
use quip_solver_cuda::cuda_device::CudaDevice;
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum, Debug)]
enum Mode {
    Plain,
    Screen,
    Stock,
}

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    /// Rounds to run.
    #[arg(long, default_value_t = 1)]
    rounds: u64,
    /// Seconds per round.
    #[arg(long, default_value_t = 600)]
    seconds: u64,
    /// First round number; seeds the salts so modes can share rounds.
    #[arg(long, default_value_t = 0)]
    first_round: u64,
    /// Generator threads in screen mode.
    #[arg(long, default_value_t = 14)]
    threads: usize,
    /// Salts screened before the GPU starts in screen mode.
    #[arg(long, default_value_t = 10_000)]
    warm: u64,
    /// `weak_milli` of the recipe (the Aglais chain runs 45).
    #[arg(long, default_value_t = 45)]
    weak_milli: u16,
    /// `pairs` of the recipe (the Aglais chain runs 1350).
    #[arg(long, default_value_t = 1350)]
    pairs: u16,
    /// Topology spec JSON (`nodes`, `edges`); default the Aglais test fixture.
    #[arg(long)]
    topology: Option<std::path::PathBuf>,
    /// Print every finished solve to stderr.
    #[arg(long)]
    verbose: bool,
}

#[derive(Deserialize)]
struct Fixture {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
}

/// Canonical twisted-dimer parameters on Zephyr z(12, 4), laid out as the registered recipes.
fn params(weak_milli: u16, pairs: u16) -> Vec<u8> {
    let [w0, w1] = weak_milli.to_le_bytes();
    let [p0, p1] = pairs.to_le_bytes();
    vec![
        0x01, 1, 0, 2, 0, 0, 0, 12, 0, 2, 0, 1, 0, 0, 0, 4, 3, 0, 2, 0, 0, 0, w0, w1, 4, 0, 4, 0,
        0, 0, p0, p1, 0x00, 0x00,
    ]
}

fn nonce(round: u64, salt: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut x = round.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    for chunk in out.chunks_mut(8) {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    out
}

struct Ctx {
    verbose: bool,
    base: BaseTopology,
    params: Vec<u8>,
    limits: DeviceLimits,
    weights: Weights,
    weak: f64,
}

impl Ctx {
    fn instance(&self, round: u64, salt: u64) -> Option<Instance> {
        export_instances(
            &TWISTED_DIMER_V1,
            &self.params,
            &self.base,
            &[nonce(round, salt)],
        )
        .ok()?
        .pop()?
        .ok()
    }

    /// The lowest energy (offset included) of one solve, or `None` when the deadline stopped it.
    fn solve(
        &self,
        inst: &Instance,
        reads: usize,
        sweeps: usize,
        seed: u64,
        deadline: Instant,
    ) -> Result<Option<i64>> {
        let edges = inst
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        let h = inst.h().iter().map(|&x| f64::from(x) / 1000.0).collect();
        let j = inst.j().iter().map(|&x| f64::from(x) / 1000.0).collect();
        let graph = IsingGraph::new(h, j, edges);
        let params = SampleParams {
            num_reads: reads,
            num_sweeps: sweeps,
            seed,
            ..SampleParams::default()
        };
        let prepared = PreparedGraph::from_instance_scaled(
            inst,
            &TWISTED_DIMER_V1,
            self.weights,
            &self.limits,
            reads,
            self.weak,
        )?;
        let work = ClusterWork::new(&graph, &params)?;
        let late = || Instant::now() >= deadline;
        let start = Instant::now();
        match ClusterDevice::open(0, prepared)?.run_work_population(&work, &late, &late) {
            Ok(run) => {
                let best = run.energies.iter().min().map(|e| e + inst.offset_milli());
                if self.verbose {
                    eprintln!(
                        "solve reads={reads} sweeps={sweeps} best={best:?} s={:.2}",
                        start.elapsed().as_secs_f64()
                    );
                }
                Ok(best)
            }
            Err(_) if late() => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

struct Round {
    best: i64,
    solved: u64,
    long: u64,
}

impl Round {
    fn take(&mut self, energy: Option<i64>) {
        if let Some(e) = energy {
            self.best = self.best.min(e);
            self.solved += 1;
        }
    }
}

fn sequential(ctx: &Ctx, round: u64, deadline: Instant) -> Result<Round> {
    let mut out = Round {
        best: i64::MAX,
        solved: 0,
        long: 0,
    };
    let mut salt = 0;
    while Instant::now() < deadline {
        salt += 1;
        let Some(inst) = ctx.instance(round, salt) else {
            continue;
        };
        out.take(ctx.solve(&inst, 160, 8192, salt, deadline)?);
    }
    Ok(out)
}

/// Screen mode: generator threads keep the best `KEEP` unsolved salts; the GPU drains them.
fn screened(ctx: &Ctx, args: &Args, round: u64, deadline: Instant) -> Result<(Round, u64)> {
    const KEEP: usize = 48;
    let pool: Mutex<Vec<(i64, u64, Arc<Instance>)>> = Mutex::new(Vec::new());
    let next = AtomicU64::new(0);
    let screened = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let mut out = Round {
        best: i64::MAX,
        solved: 0,
        long: 0,
    };
    std::thread::scope(|scope| -> Result<()> {
        for _ in 0..args.threads {
            scope.spawn(|| {
                while !done.load(Ordering::Relaxed) && Instant::now() < deadline {
                    let salt = next.fetch_add(1, Ordering::Relaxed);
                    let Some(inst) = ctx.instance(round, salt) else {
                        continue;
                    };
                    #[expect(clippy::cast_possible_truncation, reason = "scores are small")]
                    let s = (score(&inst, &DEFAULT_WEIGHTS).0 * 1000.0) as i64;
                    screened.fetch_add(1, Ordering::Relaxed);
                    let mut p = pool.lock().unwrap();
                    if p.len() < KEEP {
                        p.push((s, salt, Arc::new(inst)));
                    } else if let Some(worst) = (0..p.len()).max_by_key(|&i| p[i].0) {
                        if s < p[worst].0 {
                            p[worst] = (s, salt, Arc::new(inst));
                        }
                    }
                }
            });
        }
        let result = (|| -> Result<()> {
            while screened.load(Ordering::Relaxed) < args.warm && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
            while Instant::now() < deadline {
                let batch: Vec<_> = {
                    let mut p = pool.lock().unwrap();
                    p.sort_by_key(|c| c.0);
                    let n = 8.min(p.len());
                    p.drain(..n).collect()
                };
                if batch.is_empty() {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                let mut probed = Vec::new();
                for (_, salt, inst) in &batch {
                    if let Some(e) = ctx.solve(inst, 160, 8192, *salt, deadline)? {
                        out.take(Some(e));
                        probed.push((e, *salt, Arc::clone(inst)));
                    }
                }
                if let Some((_, salt, inst)) = probed.into_iter().min_by_key(|c| c.0) {
                    let long = ctx.solve(&inst, 256, 262_144, salt ^ 0x5eed, deadline)?;
                    if long.is_some() {
                        out.long += 1;
                    }
                    out.take(long);
                }
            }
            Ok(())
        })();
        done.store(true, Ordering::Relaxed);
        result
    })?;
    Ok((out, screened.load(Ordering::Relaxed)))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fixture: Fixture = match &args.topology {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
        None => serde_json::from_str(include_str!("../tests/fixtures/aglais.spec.json"))?,
    };
    let stock = args.mode == Mode::Stock;
    let ctx = Ctx {
        verbose: args.verbose,
        base: BaseTopology::new(fixture.nodes, &fixture.edges)
            .map_err(|e| format!("topology: {e:?}"))?,
        params: params(args.weak_milli, args.pairs),
        limits: CudaDevice::open(0)?.limits,
        weights: if stock {
            Weights::FULL
        } else {
            Weights {
                pair_milli: 250,
                tile_milli: 250,
            }
        },
        // The miner's default weak scale; `QUIP_OTSUKA_WEAK_SCALE` overrides it, as in the miner.
        weak: if stock {
            1.0
        } else {
            std::env::var("QUIP_OTSUKA_WEAK_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.5)
        },
    };
    if let Some(inst) = ctx.instance(args.first_round, 1) {
        eprintln!(
            "instance: nodes={} edges={} offset_milli={}",
            inst.node_ids().len(),
            inst.edges().len(),
            inst.offset_milli()
        );
    }
    for round in args.first_round..args.first_round + args.rounds {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(args.seconds);
        let (r, screened) = match args.mode {
            Mode::Plain | Mode::Stock => (sequential(&ctx, round, deadline)?, 0),
            Mode::Screen => screened(&ctx, &args, round, deadline)?,
        };
        #[expect(clippy::cast_precision_loss, reason = "milli energies are small")]
        let best = (r.best != i64::MAX).then(|| r.best as f64 / 1000.0);
        println!(
            "{}",
            json!({
                "mode": format!("{:?}", args.mode).to_lowercase(),
                "round": round,
                "seconds": args.seconds,
                "best": best,
                "solved": r.solved,
                "long_anneals": r.long,
                "screened": screened,
                "recipe": format!("w{}-p{}", args.weak_milli, args.pairs),
                "device": std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            })
        );
    }
    Ok(())
}
