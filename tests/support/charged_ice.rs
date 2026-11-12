//! Canonical `quip/charged-ice@1` instances of the `ice-charge` recipe,
//! regenerated from their nonces, and deterministic start reads.

use quip_solver_conformance::families::export_instances;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::{zephyr_lattice, CHARGED_ICE_V1, ICE_CHARGE_RECIPE};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::IsingGraph;

/// Decode a 64-digit hex nonce.
pub fn nonce(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64, "a nonce has 64 hex digits");
    let mut out = [0_u8; 32];
    for (k, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).expect("hex nonce");
    }
    out
}

/// Lowercase hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        write!(out, "{b:02x}").expect("a String accepts writes");
        out
    })
}

/// The canonical instance of `nonce` on the full Zephyr z(12, 4) lattice.
pub fn instance(nonce: [u8; 32]) -> Instance {
    let (nodes, edges) = zephyr_lattice(12, 4).expect("the z(12, 4) lattice");
    let topology = BaseTopology::new(nodes, &edges).expect("a topology");
    export_instances(
        &CHARGED_ICE_V1,
        ICE_CHARGE_RECIPE.params,
        &topology,
        &[nonce],
    )
    .expect("the recipe admits z(12, 4)")
    .remove(0)
    .expect("the nonce gives an instance")
}

/// The job graph of an instance, as the session decodes an i32 milli job.
pub fn graph(instance: &Instance) -> IsingGraph {
    let units = |v: &i32| f64::from(*v) / 1000.0;
    IsingGraph::new(
        instance.h().iter().map(units).collect(),
        instance.j().iter().map(units).collect(),
        instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    )
}

/// `SplitMix64`, the start-read generator of the fixtures.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// The next output.
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// `count` start reads of `n` spins: spin `i` of a read is `+1` when the low
/// bit of the next `SplitMix64` output is 0, else `-1`, reads in order.
pub fn reads(n: usize, seed: u64, count: usize) -> Vec<Vec<i8>> {
    let mut rng = SplitMix64(seed);
    (0..count)
        .map(|_| {
            (0..n)
                .map(|_| if rng.next() & 1 == 0 { 1 } else { -1 })
                .collect()
        })
        .collect()
}

/// The committed parity records (`tests/fixtures/msa_descent_parity.json`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fixture {
    /// How the records were made.
    pub description: Vec<String>,
    /// One record per regenerated instance.
    pub records: Vec<Record>,
}

/// One regenerated instance, its inputs, and the research results on it.
/// Hashes are lowercase hex BLAKE3: energies over little-endian `i64`
/// values in read order, spins over one byte per spin (`0x01` for +1,
/// `0xff` for -1), reads in order.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Record {
    /// The generator nonce, 64 hex digits.
    pub nonce: String,
    /// The FAM-19 topology order digest of the instance.
    pub topology_hash: String,
    /// The FAM-18 instance digest.
    pub integrity_hash: String,
    /// The `SplitMix64` seed of the start reads.
    pub read_seed: u64,
    /// The number of start reads.
    pub reads: usize,
    /// The worm seed: the first 12 hex digits of the nonce.
    pub loop_seed: u64,
    /// The kind 1 tile count.
    pub tiles: usize,
    /// The lowest start-read energy.
    pub raw_best: i64,
    /// Hash of the greedy energies.
    pub greedy_energies_blake3: String,
    /// Hash of the greedy states.
    pub greedy_spins_blake3: String,
    /// The lowest greedy energy.
    pub greedy_best: i64,
    /// The read the loop started from.
    pub loop_from: usize,
    /// Hash of the loop state.
    pub loop_spins_blake3: String,
    /// The loop energy.
    pub loop_energy: i64,
}

/// The fixture path.
pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/msa_descent_parity.json"
);

/// The generator nonces of the fixtures: the first two `ice-charge` rows of
/// the research targets. The archived models of these rows are not
/// charged-ice@1 instances, so only the nonce bytes are shared.
pub const NONCES: [&str; 2] = [
    "06fef0b7a59d82ae777cf83a30d9e9d834bca1be8a3e71c58dbee636ca0cc2d0",
    "3693ad03d06a4f1b9d2755fa1c011deafd8364b03b7cc5b6ee94613f161021df",
];

/// The start reads per instance.
pub const READS: usize = 64;

/// The worm seed of a nonce, as the research driver derives it
/// (`campaign_greedy.py:260`).
pub fn loop_seed(hex_nonce: &str) -> u64 {
    u64::from_str_radix(&hex_nonce[..12], 16).expect("hex nonce")
}

/// The read seed of fixture `k`.
pub fn read_seed(k: usize) -> u64 {
    0x00c0_7000 + k as u64
}

/// BLAKE3 of energies, little-endian `i64` in order.
pub fn energies_hash(energies: &[i64]) -> String {
    let mut h = blake3::Hasher::new();
    for e in energies {
        h.update(&e.to_le_bytes());
    }
    hex(h.finalize().as_bytes())
}

/// BLAKE3 of states, one byte per spin, reads in order.
pub fn spins_hash(reads: &[Vec<i8>]) -> String {
    let mut h = blake3::Hasher::new();
    for s in reads {
        let bytes: Vec<u8> = s.iter().map(|&x| x.to_le_bytes()[0]).collect();
        h.update(&bytes);
    }
    hex(h.finalize().as_bytes())
}

/// The kind 1 tiles of an instance.
pub fn metadata_tiles(instance: &Instance) -> Vec<[u32; 4]> {
    quip_solver_cuda::descent::tiles::instance_tiles(&graph(instance), instance)
        .expect("a charged-ice instance has tiles")
}
