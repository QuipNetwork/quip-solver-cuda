//! Local build (agent O, 2026-11-14): CPU-only checks of the host polish
//! (`QUIP_OTSUKA_CPU_POLISH`). No CUDA device is opened.

use quip_solver_core::ising_generators::descriptor::{admit, params_digest};
use quip_solver_core::ising_generators::family::{generate_instance, BaseTopology};
use quip_solver_core::ising_generators::ice::{RECIPES, TWISTED_DIMER_V1};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::registry::prepare_witness_codec;
use quip_solver_core::SamplerResult;
use quip_solver_cuda::cluster::polish::{polish_reads, PolishConfig, PolishGraph};
use quip_solver_cuda::cluster::Weights;
use std::collections::HashMap;

/// Live weights: the lane's tile and pair draw (`QUIP_OTSUKA_TILE_MILLI`,
/// `QUIP_OTSUKA_PAIR_MILLI` = 250).
const LIVE: Weights = Weights {
    pair_milli: 250,
    tile_milli: 250,
};

struct Live {
    base: BaseTopology,
    params: &'static [u8],
}

fn live() -> Live {
    let blob = include_bytes!("fixtures/aglais-z12t4-spec120.bin.gz");
    let topo = ising_generators::blob::decode(blob)
        .unwrap()
        .topology()
        .unwrap();
    let base = BaseTopology::new(topo.nodes.clone(), &topo.edges).unwrap();
    let params = RECIPES
        .iter()
        .find(|r| r.alias == "x-w62-p1200")
        .unwrap()
        .params;
    Live { base, params }
}

impl Live {
    fn instance(&self, nonce: [u8; 32]) -> (Instance, Vec<u8>) {
        let digest = params_digest(self.params);
        let contracts = [TWISTED_DIMER_V1];
        let admitted = admit(
            &contracts,
            TWISTED_DIMER_V1.family_id.as_bytes(),
            1,
            self.params,
            &digest,
        )
        .unwrap();
        let bounds = admitted.size_bounds(self.base.stats()).unwrap();
        let instance = generate_instance(&admitted, &self.base, &bounds, nonce).unwrap();
        (instance, digest.to_vec())
    }

    fn decode(&self, instance: &Instance, witness: &[u8]) -> Vec<i8> {
        let digest = params_digest(self.params);
        let contracts = [TWISTED_DIMER_V1];
        let admitted = admit(
            &contracts,
            TWISTED_DIMER_V1.family_id.as_bytes(),
            1,
            self.params,
            &digest,
        )
        .unwrap();
        prepare_witness_codec(&admitted, instance)
            .unwrap()
            .decode(witness)
            .unwrap()
    }
}

/// The instance energy, computed independently of the polish code.
fn exact(instance: &Instance, s: &[i8]) -> i64 {
    let mut e = instance.offset_milli();
    for (k, &(u, v)) in instance.edges().iter().enumerate() {
        e += i64::from(instance.j()[k]) * i64::from(s[u as usize]) * i64::from(s[v as usize]);
    }
    for (i, &h) in instance.h().iter().enumerate() {
        e += i64::from(h) * i64::from(s[i]);
    }
    e
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn random_reads(instance: &Instance, count: usize, seed: u64) -> Vec<SamplerResult> {
    let mut x = seed | 1;
    (0..count)
        .map(|_| {
            let spins: Vec<i8> = (0..instance.node_ids().len())
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if x & 1 == 1 {
                        1
                    } else {
                        -1
                    }
                })
                .collect();
            SamplerResult {
                energy_milli: exact(instance, &spins),
                spins,
            }
        })
        .collect()
}

fn config(level: u8, topk: usize, sweeps: usize) -> PolishConfig {
    PolishConfig {
        level,
        topk,
        sweeps,
        ..PolishConfig::default()
    }
}

#[test]
fn polish_is_off_by_default_and_knobs_parse() {
    assert_eq!(PolishConfig::from_lookup(|_| None), None);
    let get = |env: &'static [(&'static str, &'static str)]| {
        let map: HashMap<&str, &str> = env.iter().copied().collect();
        PolishConfig::from_lookup(move |k| map.get(k).map(|v| (*v).to_string()))
    };
    assert_eq!(get(&[("QUIP_OTSUKA_CPU_POLISH", "0")]), None);
    assert_eq!(
        get(&[("QUIP_OTSUKA_CPU_POLISH", "3")]),
        None,
        "unknown level is off"
    );
    let on = get(&[("QUIP_OTSUKA_CPU_POLISH", "2")]).unwrap();
    assert_eq!(on, PolishConfig::default());
    let tuned = get(&[
        ("QUIP_OTSUKA_CPU_POLISH", "1"),
        ("QUIP_OTSUKA_POLISH_THREADS", "4"),
        ("QUIP_OTSUKA_POLISH_TOPK", "8"),
        ("QUIP_OTSUKA_POLISH_SWEEPS", "500"),
        ("QUIP_OTSUKA_POLISH_BETA", "16"),
        ("QUIP_OTSUKA_POLISH_CUTOFF", "-5070000"),
        ("QUIP_OTSUKA_POLISH_QUEUE", "x"),
    ])
    .unwrap();
    assert_eq!(tuned.level, 1);
    assert_eq!(tuned.threads, 4);
    assert_eq!(tuned.topk, 8);
    assert_eq!(tuned.sweeps, 500);
    assert!((tuned.beta - 16.0).abs() < 1e-12);
    assert_eq!(tuned.cutoff, Some(-5_070_000));
    assert_eq!(tuned.queue, 4, "a bad queue falls back to the thread count");
    assert!(tuned.wants(-5_070_000) && !tuned.wants(-5_069_999));
    assert!(on.wants(i64::MAX), "no cutoff polishes every job");
    let bad = get(&[
        ("QUIP_OTSUKA_CPU_POLISH", "2"),
        ("QUIP_OTSUKA_POLISH_THREADS", "0"),
        ("QUIP_OTSUKA_POLISH_BETA", "-1"),
        ("QUIP_OTSUKA_POLISH_TOPK", "1.5"),
    ])
    .unwrap();
    assert_eq!(bad, PolishConfig::default(), "bad values keep the defaults");
}

/// Random reads on a live dimer1200 instance: every replaced read is
/// strictly lower, every reported energy is the exact energy of its spins,
/// the read count and spin alphabet hold, and a fixed seed replays.
#[test]
fn polish_reports_true_energies_and_never_raises_one() {
    let live = live();
    let (instance, _) = live.instance([0x42; 32]);
    let graph = PolishGraph::from_instance(&instance, &TWISTED_DIMER_V1, LIVE).unwrap();
    assert_eq!(graph.num_nodes(), instance.node_ids().len());
    let score = |s: &[i8]| exact(&instance, s);
    for level in [1, 2] {
        let input = random_reads(&instance, 6, 7);
        let mut reads = input.clone();
        let cfg = config(level, 3, 40);
        let out = polish_reads(&graph, &mut reads, &cfg, 99, &score, &|| false).unwrap();
        assert_eq!(reads.len(), input.len());
        assert_eq!(out.mismatched, 0);
        // Every random read descends; level 2 also counts the continued reads it lowered.
        assert!(
            out.improved >= 6 && (level == 2 || out.improved == 6),
            "random states always descend"
        );
        for (before, after) in input.iter().zip(&reads) {
            assert_eq!(after.spins.len(), before.spins.len());
            assert!(after.spins.iter().all(|&x| x == 1 || x == -1));
            assert_eq!(after.energy_milli, exact(&instance, &after.spins));
            assert_eq!(
                after.energy_milli,
                graph.energy(&after.spins) + instance.offset_milli()
            );
            assert!(after.energy_milli < before.energy_milli);
        }
        assert_eq!(
            out.after,
            reads.iter().map(|r| r.energy_milli).min().unwrap()
        );
        assert!(out.after < out.before);
        let mut again = input.clone();
        let replay = polish_reads(&graph, &mut again, &cfg, 99, &score, &|| false).unwrap();
        assert_eq!(replay, out, "a fixed seed replays");
        assert!(again.iter().zip(&reads).all(|(a, b)| a.spins == b.spins));
    }
}

/// Our own deep on-chain proofs (GPU states at -5079..-5081 units): level 1
/// reproduces the measured single-flip gains exactly, level 2 never raises
/// any read, and every returned energy is exact.
#[test]
fn polish_lowers_our_deep_chain_states_exactly() {
    let live = live();
    let text = include_str!("fixtures/polish_chain_witnesses.txt");
    // Level 1 gains measured on these proofs by the research probe (agent O).
    let level1 = [0_i64, 744, 620];
    let mut checked = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let nonce: [u8; 32] = unhex(f[0]).try_into().unwrap();
        let reported: i64 = f[1].parse().unwrap();
        let (instance, _) = live.instance(nonce);
        let spins = live.decode(&instance, &unhex(f[2]));
        assert_eq!(
            exact(&instance, &spins),
            reported,
            "the witness decodes to the proof energy"
        );
        let graph = PolishGraph::from_instance(&instance, &TWISTED_DIMER_V1, LIVE).unwrap();
        let score = |s: &[i8]| exact(&instance, s);
        let read = SamplerResult {
            spins,
            energy_milli: reported,
        };
        let mut l1 = vec![read.clone()];
        let o1 = polish_reads(&graph, &mut l1, &config(1, 1, 0), 1, &score, &|| false).unwrap();
        let mut l2 = vec![read.clone()];
        let o2 = polish_reads(&graph, &mut l2, &config(2, 1, 300), 1, &score, &|| false).unwrap();
        for (o, r) in [(o1, &l1[0]), (o2, &l2[0])] {
            assert_eq!(o.mismatched, 0);
            assert_eq!(r.energy_milli, exact(&instance, &r.spins));
            assert!(r.energy_milli <= reported);
        }
        assert!(l2[0].energy_milli <= l1[0].energy_milli);
        assert_eq!(
            reported - l1[0].energy_milli,
            level1[checked],
            "single-flip gain"
        );
        eprintln!(
            "chain state {reported}: level 1 {:+}, level 2 {:+}",
            reported - l1[0].energy_milli,
            reported - l2[0].energy_milli
        );
        checked += 1;
    }
    assert_eq!(checked, 3);
}

#[test]
fn a_stop_abandons_the_polish() {
    let live = live();
    let (instance, _) = live.instance([0x43; 32]);
    let graph = PolishGraph::from_instance(&instance, &TWISTED_DIMER_V1, LIVE).unwrap();
    let score = |s: &[i8]| exact(&instance, s);
    let mut reads = random_reads(&instance, 2, 3);
    assert!(polish_reads(&graph, &mut reads, &config(2, 2, 100), 1, &score, &|| true).is_none());
    // A stop during the continuation (after the descent pass) also abandons it.
    let calls = std::cell::Cell::new(0_u32);
    let stop = || {
        calls.set(calls.get() + 1);
        calls.get() > 3
    };
    let mut reads = random_reads(&instance, 2, 3);
    assert!(polish_reads(&graph, &mut reads, &config(2, 2, 1000), 1, &score, &stop).is_none());
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            exact(&instance, &r.spins),
            "partial polish stays exact"
        );
    }
}

/// A score that disagrees with the polish model (here: not the instance
/// energy) leaves every read unchanged instead of reporting a wrong energy.
#[test]
fn a_score_mismatch_keeps_the_read() {
    let live = live();
    let (instance, _) = live.instance([0x44; 32]);
    let graph = PolishGraph::from_instance(&instance, &TWISTED_DIMER_V1, LIVE).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let wrong = |s: &[i8]| exact(&instance, s) + s.iter().filter(|&&x| x > 0).count() as i64;
    let input: Vec<SamplerResult> = random_reads(&instance, 3, 5)
        .into_iter()
        .map(|mut r| {
            r.energy_milli = wrong(&r.spins);
            r
        })
        .collect();
    let mut reads = input.clone();
    let out = polish_reads(&graph, &mut reads, &config(1, 1, 0), 1, &wrong, &|| false).unwrap();
    // A descent that changes the number of up spins changes the bogus offset,
    // so it is refused; any accepted read still matches its score.
    assert_eq!(out.improved + out.mismatched, 3);
    assert!(out.mismatched >= 1);
    for (a, b) in input.iter().zip(&reads) {
        assert_eq!(b.energy_milli, wrong(&b.spins));
        if b.spins == a.spins {
            assert_eq!(b.energy_milli, a.energy_milli);
        }
    }
}

/// Timing of the default level-2 polish on one deep chain state copied into
/// 64 reads (the live read count). Ignored: `cargo test --release --test
/// cluster_polish -- --ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn polish_default_timing() {
    let live = live();
    let line = include_str!("fixtures/polish_chain_witnesses.txt")
        .lines()
        .find(|l| !l.starts_with('#'))
        .unwrap();
    let f: Vec<&str> = line.split_whitespace().collect();
    let (instance, _) = live.instance(unhex(f[0]).try_into().unwrap());
    let spins = live.decode(&instance, &unhex(f[2]));
    let started = std::time::Instant::now();
    let graph = PolishGraph::from_instance(&instance, &TWISTED_DIMER_V1, LIVE).unwrap();
    let built = started.elapsed();
    let score = |s: &[i8]| exact(&instance, s);
    let energy = exact(&instance, &spins);
    let mut reads = vec![
        SamplerResult {
            spins,
            energy_milli: energy
        };
        64
    ];
    let started = std::time::Instant::now();
    let out = polish_reads(
        &graph,
        &mut reads,
        &PolishConfig::default(),
        5,
        &score,
        &|| false,
    )
    .unwrap();
    eprintln!(
        "graph {:.1} ms, polish {:.1} ms, {} -> {} ({} reads improved)",
        built.as_secs_f64() * 1e3,
        started.elapsed().as_secs_f64() * 1e3,
        out.before,
        out.after,
        out.improved
    );
}
