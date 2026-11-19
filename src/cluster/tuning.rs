//! Local build (`BudFox`, 2026-11-13): opt-in Otsuka knobs.
//!
//! Every field defaults to the upstream behaviour, so an unset environment
//! runs the stock kernel path launch for launch. The sweep and read budgets
//! live in the binary (`QUIP_OTSUKA_SWEEP_MULT`, `QUIP_OTSUKA_READS`), next to
//! the adapt envelope they scale; `QUIP_BETA_RANGE` reaches the ladder through
//! `streaming::build_beta_schedule`.
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `QUIP_OTSUKA_CHUNK` | 1 | Sweeps per launch. Above 1 the kernel computes its own per-sweep geometric beta from the ladder ends, so a launch spans rungs and the host synchronizes once per chunk instead of once per sweep. Final descent then runs as one launch. |
//! | `QUIP_OTSUKA_SINGLE_EVERY` | 1 | Colored Metropolis sweep after every N cluster sweeps. 0 disables. |
//! | `QUIP_OTSUKA_ROUNDS` | 2 | Random component groups decided in parallel per cluster sweep. |
//! | `QUIP_OTSUKA_PAIR_MILLI` | 1000 | Share of each pair coupling in the cluster draw; the rest joins the weak layer. The CPU Otsuka family default is 250. |
//! | `QUIP_OTSUKA_CONCURRENCY` | 1 | Family jobs annealed at once, each on its own CUDA stream. |
//! | `QUIP_OTSUKA_PA` | 0 | 1 = population annealing over the job's reads (implies chunked launches). |
//! | `QUIP_OTSUKA_PA_RESAMPLES` | 128 | Resampling checkpoints over the schedule. |
//! | `QUIP_OTSUKA_FAST` | 0 | 1 = kernel variant with 16-bit union-find parents (7 shared bytes per node, three blocks per SM) and select-based tile picks (no local memory). Same draws and results. |
//! | `QUIP_OTSUKA_SWEEP_V2` | 0 | 1 = restructured sweep kernel (fewer barriers, cheaper per-phase work). Same draws and results as the stock kernel. |
//! | `QUIP_OTSUKA_SWEEP_V3` | 0 | 1 = the V2 sweep with both ELLs repacked into int4 chunks (a row's chunks and degree load together: one L2 round trip per node) and spin-sign-tagged roots for the weak-field accumulate (one shared load per edge). Implies V2; same draws and results. |
//! | `QUIP_OTSUKA_SWEEP_V4` | 0 | 1 = the V3 sweep with a load-balanced union pass (flat per-lane find/link loop, both finds one hop per step, a block-wide queue for the last edges), a batched compress pass, and packed tile/pair records for the draws (one load per item, table tile picks). Implies V3; same draws and results. |
//! | `QUIP_OTSUKA_PA_DEVICE` | 0 | 1 = with `QUIP_OTSUKA_PA=1`, run every checkpoint (energies, lineage bests, systematic resampling, parent gather, Houdayer moves) as device kernels (`kernels/cluster_pa.cu`) on the lane's stream. The host only keeps two segments queued, so frequent resampling costs no synchronization. |
//! | `QUIP_OTSUKA_SWEEP_V5` | 0 | 1 = the V4 sweep with 16-bit ELL entries (neighbor << 2 \| coupling code, four couplings per ELL in the header) for the Metropolis and accumulate gathers: a width-20 row is three 16-byte loads instead of five. Graphs with more than four distinct couplings in an ELL keep V4's layout. Implies V4; same draws and results. |
//! | `QUIP_OTSUKA_SWEEP_V6` | 0 | 1 = the V5 sweep with each ELL row split by coupling magnitude (strong entries in chunk 0, weak ones after, one multiply per class per row). Metropolis entries are byte offsets into the block's spin window (s, a -s copy in shared memory, a zero byte for padding), so an entry costs an address, a shared load and an add; accumulate entries carry the coupling sign, and tagged parents carry bit 30 so one xor and shift give the sign of J * s. Needs exactly two coupling magnitudes per ELL and at most eight strong entries per row (else V5's layout for that graph), and `n + 4` more shared bytes per block. Implies V5; same draws and results. |
//! | `QUIP_OTSUKA_UNION_SV` | 0 | 1 = the V4 sweep with a Shiloach-Vishkin-style union pass: red.min hooks of every edge, pointer jumping, `SV_ROUNDS` (NVRTC define, 1) rounds of root hooks, then CAS unions of the edges still apart. Roots stay each component's smallest node. Implies V4; same draws and results. |
//! | `QUIP_OTSUKA_HOUDAYER_BETA` | 3.0 | From this beta each checkpoint pairs the replicas for one Houdayer move each. 0 disables. |
//! | `QUIP_OTSUKA_TILE_MILLI` | 1000 | Share of each tile coupling in the cluster draw, `1..=1000`; the rest joins the weak layer, so the model energy is unchanged. A cluster-move parameter only: every reported energy is the instance's own. The CPU Otsuka research solver uses 250. |
//! | `QUIP_OTSUKA_HOUDAYER_EVERY` | 0 (off) | With `QUIP_OTSUKA_PA=1` and `QUIP_OTSUKA_PA_DEVICE=1`: inside each resampling segment, one pairing-only Houdayer move per replica pair every K sweeps once beta reaches `QUIP_OTSUKA_HOUDAYER_BETA` (no resampling), then lineage-best tracking. Other paths ignore it. |
//! | `QUIP_OTSUKA_STAGE0_KEEP` | 1 (off) | Below 1: score each validated twisted-dimer job with the stage-0 lean model (`family_screen::stage0`) before it reaches a lane, and solve only about this fraction, the lowest scores. A screened salt completes with no reads: it counts as done for its lease and sends nothing. |
//! | `QUIP_OTSUKA_STAGE0_MODEL` | 1 | Stage-0 model when the screen is on. 1 = agentA's lean triangle/4-cycle model (`family_screen::stage0`); 2 = agent H's model over the short cycles of the pair-contracted multigraph, fitted on GPU Otsuka labels (`family_screen::stage0v2`, ~15 ms per salt on one core); 3 = agent P's cascade: model 1 (~2 ms) passes the lowest `QUIP_OTSUKA_STAGE0_PRE` of salts, and model 2 ranks only those, so the pair keeps about `QUIP_OTSUKA_STAGE0_KEEP` overall at ~4x less CPU than model 2. On the spec-120 topology model 1's lowest 20% hold every salt of model 2's lowest 0.5% and 98.5% of its lowest 1%, so model 3 ranks like model 2 and affords a tighter keep. |
//! | `QUIP_OTSUKA_STAGE0_PRE` | 0.2 | Model 3 only: the fraction model 1 passes to model 2. A keep at or above it leaves model 1 alone at the keep. |
//! | `QUIP_OTSUKA_SCREEN_THREADS` | 1 | Above 1: generate and stage-0 screen lease salts on this many workers in the sampler (core's local lease generation) instead of core's serial expander, feeding the lanes from a queue of two salts per lane. 1 keeps core's path exactly. Read in `cluster::sampler::local`, which documents the protocol. |
//! | `QUIP_OTSUKA_WEAK_SCALE` | 1 (off) | Twisted-dimer jobs: the anneal sweeps read every weak coupling (`|J| < 500` milli) multiplied by this scale, `0.25..=8`, rounded to whole milli (`cluster::prepare::AnnealEll`). Final descent, the population checkpoints (energies, resampling, lineage bests), the CPU polish, and every reported energy use the instance's exact couplings. Idea from QuipNetwork/quip-solver-cuda#2. |
//! | `QUIP_OTSUKA_BETA_KNEE` | unset (off) | Local build (all11): a knee in the chunked/PA anneal's geometric beta schedule (`kernels/cluster.cu::beta_at`). The first `QUIP_OTSUKA_KNEE_FRAC` of the sweeps go geometrically from the ladder's hot end to this beta, the rest from it to the cold end. Valid only strictly between the ladder ends (checked against `QUIP_BETA_RANGE` when set, and per job); otherwise the job runs the plain schedule. The host-side schedule (population checkpoints, the Houdayer gate, the deep lane's probes) follows the same formula. The stock one-sweep path ignores it. |
//! | `QUIP_OTSUKA_KNEE_FRAC` | 0.8 with a knee | Fraction of the sweeps before the knee, `0 < FRAC < 1`. Ignored without `QUIP_OTSUKA_BETA_KNEE`. |
//! | `QUIP_OTSUKA_CPU_POLISH` | 0 (off) | Host CPU polish of lease reads on a dedicated thread pool after the GPU returns them (1 = single-spin descent, 2 = plus a low-temperature cluster continuation of the top reads). Its knobs (`QUIP_OTSUKA_POLISH_*`) are documented in `cluster::polish`. |

use std::sync::OnceLock;

/// Opt-in Otsuka knobs. `Default` is the upstream behaviour.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct OtsukaTuning {
    /// Sweeps per launch; 0 or 1 keeps the stock one-sweep launches.
    pub chunk: usize,
    /// Metropolis sweep period, 0 disables.
    pub single_every: usize,
    /// Parallel decision groups per cluster sweep, at least 1.
    pub rounds: usize,
    /// Pair weight in the cluster draw, `0..=1000` milli.
    pub pair_milli: i32,
    /// Family jobs in flight at once, at least 1.
    pub concurrency: usize,
    /// Population annealing over the reads.
    pub population: bool,
    /// Resampling checkpoints over the schedule, at least 1.
    pub resamples: usize,
    /// Beta from which checkpoints take Houdayer moves; 0 disables.
    pub houdayer_beta: f32,
    /// Compile the kernel with 16-bit union-find parents and select-based
    /// tile picks (`QUIP_OTSUKA_FAST`): three blocks per SM instead of two.
    pub fast: bool,
    /// Restructured sweep kernel (`QUIP_OTSUKA_SWEEP_V2`): fewer barriers and
    /// cheaper phases, same draws and results as the stock kernel.
    pub sweep_v2: bool,
    /// V2 sweep with int4-packed ELL gathers (`QUIP_OTSUKA_SWEEP_V3`); implies
    /// V2, same draws and results.
    pub sweep_v3: bool,
    /// V3 sweep with the balanced union pass, batched compress and packed
    /// draw records (`QUIP_OTSUKA_SWEEP_V4`); implies V3, same draws and results.
    pub sweep_v4: bool,
    /// V4 sweep with 16-bit ELL entries for the Metropolis and accumulate
    /// gathers (`QUIP_OTSUKA_SWEEP_V5`); implies V4, same draws and results.
    pub sweep_v5: bool,
    /// V5 sweep with magnitude-split 16-bit gathers (`QUIP_OTSUKA_SWEEP_V6`);
    /// implies V5, same draws and results.
    pub sweep_v6: bool,
    /// Shiloach-Vishkin-style union pass (`QUIP_OTSUKA_UNION_SV`); implies V4,
    /// same draws and results.
    pub union_sv: bool,
    /// Run population checkpoints on the device (`QUIP_OTSUKA_PA_DEVICE`):
    /// no host synchronization or state copies per checkpoint.
    pub device_pa: bool,
    /// Fraction of family jobs the stage-0 screen passes; 1 is off.
    pub stage0_keep: f64,
    /// Tile weight in the cluster draw, `1..=1000` milli
    /// (`QUIP_OTSUKA_TILE_MILLI`). 1000 is the stock full-weight draw.
    pub tile_milli: i32,
    /// Device-PA Houdayer period inside resampling segments, in sweeps
    /// (`QUIP_OTSUKA_HOUDAYER_EVERY`); 0 keeps checkpoint-only moves.
    pub houdayer_every: usize,
    /// Stage-0 model (`QUIP_OTSUKA_STAGE0_MODEL`): 1 = lean, 2 = contracted
    /// cycles, 3 = model 1 prefilter then model 2.
    pub stage0_model: u8,
    /// Model 3's prefilter fraction (`QUIP_OTSUKA_STAGE0_PRE`).
    pub stage0_pre: f64,
    /// Scale on the weak couplings in the anneal sweeps
    /// (`QUIP_OTSUKA_WEAK_SCALE`); 1 is off.
    pub weak_scale: f64,
    /// Local build (all11): knee beta of the chunked/PA schedule
    /// (`QUIP_OTSUKA_BETA_KNEE`); 0 is off.
    pub beta_knee: f32,
    /// Fraction of the sweeps before the knee (`QUIP_OTSUKA_KNEE_FRAC`); 0
    /// when the knee is off.
    pub knee_frac: f32,
}

/// all11: the knee fraction when `QUIP_OTSUKA_KNEE_FRAC` is unset or invalid.
pub const DEFAULT_KNEE_FRAC: f32 = 0.8;

impl Default for OtsukaTuning {
    fn default() -> Self {
        Self {
            chunk: 1,
            single_every: 1,
            rounds: 2,
            pair_milli: 1000,
            concurrency: 1,
            population: false,
            resamples: 128,
            houdayer_beta: 3.0,
            fast: false,
            sweep_v2: false,
            sweep_v3: false,
            sweep_v4: false,
            sweep_v5: false,
            sweep_v6: false,
            union_sv: false,
            device_pa: false,
            stage0_keep: 1.0,
            tile_milli: 1000,
            houdayer_every: 0,
            stage0_model: 1,
            stage0_pre: 0.2,
            weak_scale: 1.0,
            beta_knee: 0.0,
            knee_frac: 0.0,
        }
    }
}

impl OtsukaTuning {
    /// True when the launch path is the stock one-sweep-per-launch path.
    #[must_use]
    pub fn stock_launches(&self) -> bool {
        self.chunk <= 1 && !self.population
    }

    /// The cluster-draw weights of these knobs. Only the draw changes: the
    /// prepared graph moves each softened share into the weak layer, so the
    /// model energy, and every reported energy, stay the instance's own.
    #[must_use]
    pub fn weights(&self) -> super::Weights {
        super::Weights {
            pair_milli: self.pair_milli,
            tile_milli: self.tile_milli,
        }
    }

    /// Read the knobs through `get`, logging each applied or ignored value.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let mut t = Self::default();
        let num = |name: &str, lo: f64, hi: f64| -> Option<f64> {
            let raw = get(name)?;
            let parsed = raw
                .trim()
                .parse::<f64>()
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
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            if let Some(v) = num("QUIP_OTSUKA_CHUNK", 1.0, 1_048_576.0) {
                t.chunk = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_SINGLE_EVERY", 0.0, 1_048_576.0) {
                t.single_every = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_ROUNDS", 1.0, 64.0) {
                t.rounds = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_PAIR_MILLI", 0.0, 1000.0) {
                t.pair_milli = v as i32;
            }
            if let Some(v) = num("QUIP_OTSUKA_CONCURRENCY", 1.0, 64.0) {
                t.concurrency = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_PA", 0.0, 1.0) {
                t.population = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_PA_RESAMPLES", 1.0, 1_048_576.0) {
                t.resamples = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_HOUDAYER_BETA", 0.0, 1.0e6) {
                t.houdayer_beta = v as f32;
            }
            if let Some(v) = num("QUIP_OTSUKA_FAST", 0.0, 1.0) {
                t.fast = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_SWEEP_V2", 0.0, 1.0) {
                t.sweep_v2 = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_SWEEP_V3", 0.0, 1.0) {
                t.sweep_v3 = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_SWEEP_V4", 0.0, 1.0) {
                t.sweep_v4 = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_SWEEP_V5", 0.0, 1.0) {
                t.sweep_v5 = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_SWEEP_V6", 0.0, 1.0) {
                t.sweep_v6 = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_UNION_SV", 0.0, 1.0) {
                t.union_sv = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_PA_DEVICE", 0.0, 1.0) {
                t.device_pa = v >= 0.5;
            }
            if let Some(v) = num("QUIP_OTSUKA_STAGE0_KEEP", 1.0e-6, 1.0) {
                t.stage0_keep = v;
            }
            if let Some(v) = num("QUIP_OTSUKA_TILE_MILLI", 1.0, 1000.0) {
                t.tile_milli = v as i32;
            }
            if let Some(v) = num("QUIP_OTSUKA_HOUDAYER_EVERY", 0.0, 1_048_576.0) {
                t.houdayer_every = v as usize;
            }
            if let Some(v) = num("QUIP_OTSUKA_STAGE0_MODEL", 1.0, 3.0).filter(|v| v.fract() == 0.0)
            {
                t.stage0_model = v as u8;
            }
            if let Some(v) = num("QUIP_OTSUKA_STAGE0_PRE", 1.0e-6, 1.0) {
                t.stage0_pre = v;
            }
            if let Some(v) = num("QUIP_OTSUKA_WEAK_SCALE", 0.25, 8.0) {
                t.weak_scale = v;
            }
        }
        let (knee, frac) = knee_from(&get);
        t.beta_knee = knee;
        t.knee_frac = frac;
        if t.houdayer_every > 0 && !(t.population && t.device_pa) {
            eprintln!(
                "quip-otsuka: QUIP_OTSUKA_HOUDAYER_EVERY needs QUIP_OTSUKA_PA=1 and \
                 QUIP_OTSUKA_PA_DEVICE=1; checkpoint-only Houdayer moves"
            );
        }
        t
    }

    /// all11: the `(beta_knee, knee_frac)` pair for a job whose ladder runs
    /// from `hot` to `cold`. `(0, 0)` (the plain geometric schedule, every
    /// launch as in all10) unless the knee is on and strictly inside the
    /// ladder; a knee outside a job's ladder is logged once.
    #[must_use]
    pub fn knee_for(&self, hot: f32, cold: f32) -> (f32, f32) {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        if self.beta_knee <= 0.0 || !(self.knee_frac > 0.0 && self.knee_frac < 1.0) {
            return (0.0, 0.0);
        }
        if hot < self.beta_knee && self.beta_knee < cold {
            return (self.beta_knee, self.knee_frac);
        }
        LOGGED.call_once(|| {
            eprintln!(
                "quip-otsuka: beta knee={} outside a job's ladder [{hot}, {cold}]; \
                 such jobs run the plain geometric schedule",
                self.beta_knee
            );
        });
        (0.0, 0.0)
    }

    /// The process environment, read once.
    #[must_use]
    pub fn from_env() -> Self {
        static ENV: OnceLock<OtsukaTuning> = OnceLock::new();
        *ENV.get_or_init(|| Self::from_lookup(|name| std::env::var(name).ok()))
    }
}

/// all11: parse `QUIP_OTSUKA_BETA_KNEE` / `QUIP_OTSUKA_KNEE_FRAC`, logging
/// each value. Returns `(0, 0)` when the knee is off or invalid.
#[allow(clippy::cast_possible_truncation)]
fn knee_from(get: &impl Fn(&str) -> Option<String>) -> (f32, f32) {
    let raw_frac = get("QUIP_OTSUKA_KNEE_FRAC");
    let Some(raw) = get("QUIP_OTSUKA_BETA_KNEE") else {
        if let Some(f) = raw_frac {
            eprintln!(
                "quip-otsuka: QUIP_OTSUKA_KNEE_FRAC={f:?} ignored (no QUIP_OTSUKA_BETA_KNEE)"
            );
        }
        return (0.0, 0.0);
    };
    // The ladder ends, when `QUIP_BETA_RANGE` names them (else checked per job).
    let range = get("QUIP_BETA_RANGE").and_then(|v| {
        let (a, b) = v.split_once(',')?;
        let hot = a.trim().parse::<f64>().ok()?;
        let cold = b.trim().parse::<f64>().ok()?;
        (hot.is_finite() && cold.is_finite() && hot > 0.0 && cold >= hot).then_some((hot, cold))
    });
    let knee = raw
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0 && f64::from(*v as f32) > 0.0)
        .filter(|v| range.is_none_or(|(hot, cold)| hot < *v && *v < cold));
    let Some(knee) = knee else {
        eprintln!(
            "quip-otsuka: QUIP_OTSUKA_BETA_KNEE={raw:?} ignored (needs beta_min < knee < beta_max{})",
            range.map_or(String::new(), |(h, c)| format!(", range {h},{c}"))
        );
        if let Some(f) = raw_frac {
            eprintln!("quip-otsuka: QUIP_OTSUKA_KNEE_FRAC={f:?} ignored (knee off)");
        }
        return (0.0, 0.0);
    };
    let frac = match raw_frac {
        None => DEFAULT_KNEE_FRAC,
        Some(f) => match f
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| *v > 0.0 && *v < 1.0)
        {
            Some(v) if (v as f32) > 0.0 && (v as f32) < 1.0 => v as f32,
            _ => {
                eprintln!(
                    "quip-otsuka: QUIP_OTSUKA_KNEE_FRAC={f:?} ignored (0 < FRAC < 1); \
                     frac={DEFAULT_KNEE_FRAC}"
                );
                DEFAULT_KNEE_FRAC
            }
        },
    };
    let knee = knee as f32;
    eprintln!("quip-otsuka: beta knee={knee} frac={frac} applied");
    (knee, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn unset_environment_is_stock() {
        let t = OtsukaTuning::from_lookup(|_| None);
        assert_eq!(t, OtsukaTuning::default());
        assert!(t.stock_launches());
        assert_eq!(t.tile_milli, 1000, "stock full tile weight");
        assert_eq!(t.houdayer_every, 0, "stock checkpoint-only Houdayer");
        assert_eq!(
            t.weights(),
            crate::cluster::Weights::FULL,
            "stock draw weights"
        );
    }

    #[test]
    fn knobs_parse_and_bad_values_are_ignored() {
        let env: HashMap<&str, &str> = [
            ("QUIP_OTSUKA_CHUNK", "512"),
            ("QUIP_OTSUKA_PAIR_MILLI", "250"),
            ("QUIP_OTSUKA_ROUNDS", "0"),
            ("QUIP_OTSUKA_PA", "1"),
            ("QUIP_OTSUKA_CONCURRENCY", "x"),
            ("QUIP_OTSUKA_SWEEP_V2", "1"),
            ("QUIP_OTSUKA_SWEEP_V3", "1"),
            ("QUIP_OTSUKA_SWEEP_V4", "1"),
            ("QUIP_OTSUKA_PA_DEVICE", "1"),
            ("QUIP_OTSUKA_STAGE0_KEEP", "0.1"),
            ("QUIP_OTSUKA_TILE_MILLI", "250"),
            ("QUIP_OTSUKA_HOUDAYER_EVERY", "8"),
        ]
        .into_iter()
        .collect();
        let t = OtsukaTuning::from_lookup(|k| env.get(k).map(|v| (*v).to_string()));
        assert_eq!(t.chunk, 512);
        assert_eq!(t.pair_milli, 250);
        assert_eq!(t.rounds, 2);
        assert!(t.population);
        assert_eq!(t.concurrency, 1);
        assert!(t.sweep_v2);
        assert!(t.sweep_v3);
        assert!(t.sweep_v4);
        assert!(t.device_pa);
        assert!((t.stage0_keep - 0.1).abs() < 1e-12);
        assert_eq!(t.tile_milli, 250);
        assert_eq!(t.houdayer_every, 8);
        assert!(!t.stock_launches());
        // A zero tile weight would leave tiles out of the draw: refused, stock kept.
        let zero =
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_TILE_MILLI").then(|| "0".into()));
        assert_eq!(zero.tile_milli, 1000);
        let big =
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_TILE_MILLI").then(|| "1001".into()));
        assert_eq!(big.tile_milli, 1000);
        let off =
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_STAGE0_KEEP").then(|| "0".into()));
        assert!(
            (off.stage0_keep - 1.0).abs() < f64::EPSILON,
            "0 is ignored, so the screen stays off"
        );
        assert_eq!(t.stage0_model, 1, "model 1 by default");
    }

    /// Local build (all10): the V6 decode and SV union knobs default off and
    /// parse as 0/1 flags.
    #[test]
    fn sweep_v6_and_union_sv_knobs() {
        let t = OtsukaTuning::from_lookup(|_| None);
        assert!(!t.sweep_v6 && !t.union_sv, "off by default");
        let on = OtsukaTuning::from_lookup(|k| {
            matches!(k, "QUIP_OTSUKA_SWEEP_V6" | "QUIP_OTSUKA_UNION_SV").then(|| "1".into())
        });
        assert!(on.sweep_v6 && on.union_sv);
        let bad = OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_SWEEP_V6").then(|| "2".into()));
        assert!(!bad.sweep_v6, "out of range is ignored");
    }

    #[test]
    fn stage0_model_knob() {
        let model = |v: &str| {
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_STAGE0_MODEL").then(|| v.to_string()))
                .stage0_model
        };
        assert_eq!(model("3"), 3);
        assert_eq!(model("2"), 2);
        assert_eq!(model("1"), 1);
        for bad in ["0", "4", "1.5", "x", ""] {
            assert_eq!(model(bad), 1, "{bad:?} is ignored");
        }
    }

    #[test]
    fn weak_scale_knob() {
        let ws = |v: &str| {
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_WEAK_SCALE").then(|| v.to_string()))
                .weak_scale
        };
        assert!(
            (OtsukaTuning::default().weak_scale - 1.0).abs() < f64::EPSILON,
            "off by default"
        );
        assert!((ws("1.5") - 1.5).abs() < 1e-12);
        assert!((ws("2") - 2.0).abs() < 1e-12);
        for bad in ["0", "0.1", "9", "-1", "x", ""] {
            assert!((ws(bad) - 1.0).abs() < f64::EPSILON, "{bad:?} is ignored");
        }
    }

    /// Local build (all11): the beta knee knobs.
    #[test]
    fn beta_knee_knobs() {
        let knee = |pairs: &[(&str, &str)]| {
            let env: HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            let t = OtsukaTuning::from_lookup(|k| env.get(k).cloned());
            (t.beta_knee, t.knee_frac)
        };
        // Unset: off, and the rest of the tuning is untouched.
        assert_eq!(knee(&[]), (0.0, 0.0));
        assert_eq!(OtsukaTuning::default().beta_knee.to_bits(), 0);
        assert_eq!(OtsukaTuning::default().knee_frac.to_bits(), 0);
        // Valid, with and without the frac, with and without QUIP_BETA_RANGE.
        assert_eq!(
            knee(&[
                ("QUIP_OTSUKA_BETA_KNEE", "4"),
                ("QUIP_OTSUKA_KNEE_FRAC", "0.873")
            ]),
            (4.0, 0.873)
        );
        assert_eq!(
            knee(&[("QUIP_OTSUKA_BETA_KNEE", "4")]),
            (4.0, DEFAULT_KNEE_FRAC)
        );
        assert_eq!(
            knee(&[
                ("QUIP_OTSUKA_BETA_KNEE", " 3.5 "),
                ("QUIP_OTSUKA_KNEE_FRAC", "0.5"),
                ("QUIP_BETA_RANGE", "1,12"),
            ]),
            (3.5, 0.5)
        );
        // Out of range: not strictly inside the ladder, not positive, not a number.
        for bad in ["1", "12", "0.5", "13", "0", "-4", "nan", "inf", "x", ""] {
            assert_eq!(
                knee(&[
                    ("QUIP_OTSUKA_BETA_KNEE", bad),
                    ("QUIP_OTSUKA_KNEE_FRAC", "0.873"),
                    ("QUIP_BETA_RANGE", "1,12"),
                ]),
                (0.0, 0.0),
                "knee {bad:?} is ignored"
            );
        }
        for bad in ["0", "-4", "nan", "inf", "x", "1e-60"] {
            assert_eq!(
                knee(&[("QUIP_OTSUKA_BETA_KNEE", bad)]),
                (0.0, 0.0),
                "{bad:?}"
            );
        }
        // A bad frac is ignored: the default fraction applies.
        for bad in ["0", "1", "1.5", "-0.1", "x", "", "nan", "0.99999999999"] {
            assert_eq!(
                knee(&[
                    ("QUIP_OTSUKA_BETA_KNEE", "4"),
                    ("QUIP_OTSUKA_KNEE_FRAC", bad)
                ]),
                (4.0, DEFAULT_KNEE_FRAC),
                "frac {bad:?}"
            );
        }
        // A frac alone does nothing.
        assert_eq!(knee(&[("QUIP_OTSUKA_KNEE_FRAC", "0.873")]), (0.0, 0.0));
        // The knee touches no other knob.
        let with =
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_BETA_KNEE").then(|| "4".into()));
        assert_eq!(
            OtsukaTuning {
                beta_knee: 0.0,
                knee_frac: 0.0,
                ..with
            },
            OtsukaTuning::default()
        );
    }

    #[test]
    fn stage0_pre_knob() {
        let pre = |v: &str| {
            OtsukaTuning::from_lookup(|k| (k == "QUIP_OTSUKA_STAGE0_PRE").then(|| v.to_string()))
                .stage0_pre
        };
        assert!(
            (OtsukaTuning::default().stage0_pre - 0.2).abs() < 1e-12,
            "0.2 by default"
        );
        assert!((pre("0.3") - 0.3).abs() < 1e-12);
        assert!((pre("1") - 1.0).abs() < 1e-12);
        for bad in ["0", "1.5", "-0.1", "x", ""] {
            assert!((pre(bad) - 0.2).abs() < 1e-12, "{bad:?} is ignored");
        }
    }
}
