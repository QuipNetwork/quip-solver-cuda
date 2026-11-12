//! Export the descent parity inputs and record the research results as the
//! committed fixture. `tools/msa_descent_reference.py` drives both steps.
//!
//! `export DIR` writes, for each fixture nonce, the regenerated instance and
//! its start reads under `DIR/<nonce>/`. `record DIR` reads the research
//! outputs that the tool wrote there and writes
//! `tests/fixtures/msa_descent_parity.json`.

#[path = "../tests/support/charged_ice.rs"]
#[allow(
    dead_code,
    reason = "the support module serves the parity tests too; this tool uses a subset"
)]
mod charged_ice;

use std::path::{Path, PathBuf};

use charged_ice::{Fixture, Record, NONCES, READS};

fn spins_bytes(reads: &[Vec<i8>]) -> Vec<u8> {
    reads
        .iter()
        .flat_map(|s| s.iter().map(|&x| x.to_le_bytes()[0]))
        .collect()
}

fn read_spins(path: &Path, n: usize) -> Vec<Vec<i8>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % n, 0, "{}: not whole reads", path.display());
    bytes
        .chunks(n)
        .map(|c| c.iter().map(|&b| i8::from_le_bytes([b])).collect())
        .collect()
}

fn export(dir: &Path) {
    for (k, hex_nonce) in NONCES.iter().enumerate() {
        let instance = charged_ice::instance(charged_ice::nonce(hex_nonce));
        let n = instance.h().len();
        let out = dir.join(hex_nonce);
        std::fs::create_dir_all(&out).expect("output directory");
        let model = serde_json::json!({
            "h_milli": instance.h(),
            "edges": instance.edges().iter().map(|&(u, v)| [u, v]).collect::<Vec<_>>(),
            "j_milli": instance.j(),
            "tetrahedra": charged_ice::metadata_tiles(&instance),
            "loop_seed": charged_ice::loop_seed(hex_nonce),
            "reads": READS,
        });
        std::fs::write(out.join("model.json"), model.to_string()).expect("model.json");
        let reads = charged_ice::reads(n, charged_ice::read_seed(k), READS);
        std::fs::write(out.join("reads.bin"), spins_bytes(&reads)).expect("reads.bin");
        println!(
            "exported {hex_nonce}: {n} nodes, {} edges",
            instance.edges().len()
        );
    }
}

fn record(dir: &Path) {
    let mut records = Vec::new();
    for (k, hex_nonce) in NONCES.iter().enumerate() {
        let instance = charged_ice::instance(charged_ice::nonce(hex_nonce));
        let n = instance.h().len();
        let at = dir.join(hex_nonce);
        let greedy = read_spins(&at.join("greedy.bin"), n);
        let loop_state = read_spins(&at.join("loop.bin"), n);
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(at.join("result.json")).expect("result.json"))
                .expect("result JSON");
        let ising = quip_solver_cuda::descent::job_ising(&charged_ice::graph(&instance))
            .expect("an exact model");
        let reads = charged_ice::reads(n, charged_ice::read_seed(k), READS);
        let raw: Vec<i64> = reads.iter().map(|s| ising.energy(s)).collect();
        let energies: Vec<i64> = greedy.iter().map(|s| ising.energy(s)).collect();
        let reported: Vec<i64> =
            serde_json::from_value(result["greedy_energies"].clone()).expect("greedy energies");
        assert_eq!(energies, reported, "{hex_nonce}: research greedy energies");
        let loop_energy = ising.energy(&loop_state[0]);
        assert_eq!(
            Some(loop_energy),
            result["loop_energy"].as_i64(),
            "{hex_nonce}: research loop energy"
        );
        records.push(Record {
            nonce: (*hex_nonce).to_owned(),
            topology_hash: charged_ice::hex(instance.topology_order_digest()),
            integrity_hash: charged_ice::hex(&instance.digest()),
            read_seed: charged_ice::read_seed(k),
            reads: READS,
            loop_seed: charged_ice::loop_seed(hex_nonce),
            tiles: charged_ice::metadata_tiles(&instance).len(),
            raw_best: raw.iter().copied().min().expect("reads"),
            greedy_energies_blake3: charged_ice::energies_hash(&energies),
            greedy_spins_blake3: charged_ice::spins_hash(&greedy),
            greedy_best: energies.iter().copied().min().expect("reads"),
            loop_from: usize::try_from(result["loop_from"].as_u64().expect("loop_from"))
                .expect("index"),
            loop_spins_blake3: charged_ice::spins_hash(&loop_state),
            loop_energy,
        });
    }
    let fixture = Fixture {
        description: vec![
            "Research descent results on regenerated quip/charged-ice@1 ice-charge instances on z(12, 4).".into(),
            "Made by tools/msa_descent_reference.py, which runs new-pow/gpu/scripts/descent.py and libgreedy.so.".into(),
            "Start reads: SplitMix64 from read_seed, low bit 0 is +1, reads in order.".into(),
            "Greedy: natural moves on every read. Loop: nat_loop on the lowest raw read, zero budget 0, seed loop_seed.".into(),
            "Hashes: BLAKE3, energies as little-endian i64, spins as one byte each.".into(),
        ],
        records,
    };
    let text = serde_json::to_string_pretty(&fixture).expect("fixture JSON") + "\n";
    std::fs::write(charged_ice::FIXTURE, text).expect("fixture file");
    println!("wrote {}", charged_ice::FIXTURE);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, step, dir] if step == "export" => export(&PathBuf::from(dir)),
        [_, step, dir] if step == "record" => record(&PathBuf::from(dir)),
        _ => {
            eprintln!("usage: msa_descent_reference export|record DIR");
            std::process::exit(2);
        }
    }
}
