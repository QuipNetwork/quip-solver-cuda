//! The lease screen's structural score on canonical twisted-ice instances (CPU only).

use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::{
    family::BaseTopology, ice::TWISTED_DIMER_V1, instance::Instance,
};
use quip_solver_cuda::cluster::screen_lease::{score, DEFAULT_WEIGHTS};

fn instances() -> Vec<Instance> {
    let reference = reference_revision(TWISTED_DIMER_V1.family_id, 1).unwrap();
    let stored = quip_solver_conformance::topology(reference.topology).unwrap();
    let base = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
    let nonces: Vec<[u8; 32]> = (1..=24u8).map(|b| [b; 32]).collect();
    reference
        .params
        .iter()
        .flat_map(|p| {
            export_instances(reference.contract, p, &base, &nonces)
                .into_iter()
                .flatten()
                .flatten()
        })
        .collect()
}

const UNITS: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[test]
fn the_score_is_a_weighted_sum_of_four_counts() {
    let instances = instances();
    assert!(instances.len() >= 2, "the reference recipes admit salts");
    for inst in &instances {
        let nodes = f64::from(u32::try_from(inst.node_ids().len()).unwrap());
        let parts: Vec<f64> = UNITS.iter().map(|w| score(inst, w).0).collect();
        assert!(parts.iter().all(|x| x.is_finite()));
        // Free nodes and tile-to-tile pairs are counts of this instance's nodes.
        assert!((0.0..=nodes).contains(&parts[0]) && parts[0].fract().abs() < 1e-9);
        assert!((0.0..=nodes / 2.0).contains(&parts[1]) && parts[1].fract().abs() < 1e-9);
        let (total, all_up) = score(inst, &DEFAULT_WEIGHTS);
        let expected: f64 = parts.iter().zip(DEFAULT_WEIGHTS).map(|(p, w)| p * w).sum();
        assert!((total - expected).abs() < 1e-6, "{total} vs {expected}");
        // A screened-out salt reports the energy of its all-up state.
        let sum: i64 = inst.h().iter().chain(inst.j()).map(|&w| i64::from(w)).sum();
        assert_eq!(all_up, sum);
    }
}

#[test]
fn the_score_is_deterministic_and_separates_salts() {
    let instances = instances();
    let bits = || -> Vec<u64> {
        instances
            .iter()
            .map(|i| score(i, &DEFAULT_WEIGHTS).0.to_bits())
            .collect()
    };
    assert_eq!(bits(), bits());
    let scores: Vec<f64> = instances
        .iter()
        .map(|i| score(i, &DEFAULT_WEIGHTS).0)
        .collect();
    let low = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let high = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(high > low, "every salt scored {low}");
}
