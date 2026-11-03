//! Models past one million couplers reach SA and Gibbs through the identity
//! the binaries advertise. `quip_solver_core::driver::solve` applies the same
//! node and edge caps as `--solve` before the sampler sees the model.

use quip_solver_core::driver::solve;
use quip_solver_cuda::capacity::{advertised_couplers, advertised_nodes};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::{cuda_gibbs_identity, cuda_sa_identity, CudaSampler, KernelKind};
use serial_test::serial;

/// complete:1415 has 1,000,405 couplers, past the old fixed limit of 1,000,000.
const NODES: usize = 1415;

fn complete_binary_problem() -> Vec<u8> {
    let mut edges = Vec::with_capacity(NODES * (NODES - 1) / 2);
    for u in 0..NODES {
        for v in u + 1..NODES {
            edges.push((u, v));
        }
    }
    assert_eq!(edges.len(), 1_000_405);
    serde_json::to_vec(&serde_json::json!({
        "h": vec![0.0; NODES],
        "j": vec![-1.0; edges.len()],
        "edges": edges,
        "num_reads": 8,
        "num_sweeps": 8,
        "sweeps_per_beta": 1,
        "beta_range": null,
        "seed": 7,
    }))
    .unwrap()
}

fn solves_past_one_million_couplers(kernel: KernelKind) {
    let mut device = CudaDevice::open_with_nodes(0, kernel, usize::MAX).unwrap();
    let nodes = advertised_nodes(kernel, usize::MAX, Some(&device.limits));
    let edges = advertised_couplers(kernel, &device.limits);
    device.max_nodes = nodes;
    let identity = match kernel {
        KernelKind::Sa => cuda_sa_identity(nodes, edges),
        KernelKind::Gibbs => cuda_gibbs_identity(nodes, edges),
        KernelKind::Msa => unreachable!("msa advertises msa_max_couplers"),
    };
    assert!(
        identity.max_edges > 1_000_405,
        "{kernel:?}: {}",
        identity.max_edges
    );
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    let sampler = CudaSampler::new(device, gov, kernel);
    let output = solve(&sampler, &identity, &complete_binary_problem())
        .unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
    let solutions: Vec<serde_json::Value> = serde_json::from_slice(&output).unwrap();
    assert_eq!(solutions.len(), 8, "{kernel:?}");
    for solution in &solutions {
        assert_eq!(solution["spins"].as_array().unwrap().len(), NODES);
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn sa_solves_a_model_past_one_million_couplers() {
    solves_past_one_million_couplers(KernelKind::Sa);
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn gibbs_solves_a_model_past_one_million_couplers() {
    solves_past_one_million_couplers(KernelKind::Gibbs);
}
