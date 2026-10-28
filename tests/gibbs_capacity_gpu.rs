//! Gibbs shared-prefix parity and device-memory admission.

use cudarc::driver::CudaContext;
use quip_solver_cuda::capacity::{instances, max_model, Connectivity, Precision};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{bench_one, sample_one};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams, SamplerResult};

fn circulant(n: usize) -> IsingGraph {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for i in 0..n {
        for d in 1..=10 {
            edges.push((i, (i + d) % n));
            j.push(if (i * 7 + d) % 3 == 0 { -1.0 } else { 1.0 });
        }
    }
    let h = (0..n)
        .map(|i| f64::from(i32::try_from(i % 3).unwrap()) - 1.0)
        .collect();
    IsingGraph::new(h, j, edges)
}

fn params() -> SampleParams {
    SampleParams {
        num_reads: 256,
        num_sweeps: 64,
        sweeps_per_beta: 1,
        seed: 42,
        ..SampleParams::default()
    }
}

#[allow(clippy::float_cmp)] // Binary fixture energies are exactly representable.
fn check_energies(graph: &IsingGraph, reads: &[SamplerResult]) {
    assert_eq!(reads.len(), 256);
    for r in reads {
        assert_eq!(r.spins.len(), graph.num_nodes());
        assert!(r.spins.iter().all(|&s| s == -1 || s == 1));
        let energy = graph
            .h
            .iter()
            .zip(&r.spins)
            .map(|(h, s)| h * f64::from(*s))
            .sum::<f64>()
            + graph
                .edges
                .iter()
                .zip(&graph.j)
                .map(|(&(u, v), j)| j * f64::from(r.spins[u]) * f64::from(r.spins[v]))
                .sum::<f64>();
        #[allow(clippy::cast_precision_loss)]
        // Small integral fixture energies are exactly representable.
        let device_energy = r.energy_milli as f64;
        assert_eq!(device_energy, energy * 1000.0);
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_spill_matches_all_shared_at_every_cap() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 4000).unwrap();
    let graph = circulant(4000);
    for metropolis in [false, true] {
        device.gibbs_metropolis = metropolis;
        device.gibbs_force_spill = false;
        device.gibbs_shared_spins_cap = None;
        let baseline = sample_one(&device, &graph, &params(), KernelKind::Gibbs).unwrap();
        check_energies(&graph, &baseline);
        device.gibbs_force_spill = true;
        for cap in [Some(0), Some(1_000), Some(3_999), None] {
            device.gibbs_shared_spins_cap = cap;
            let reads = sample_one(&device, &graph, &params(), KernelKind::Gibbs).unwrap();
            check_energies(&graph, &reads);
            for (a, b) in baseline.iter().zip(&reads) {
                assert_eq!(a.spins, b.spins, "metropolis={metropolis} cap={cap:?}");
                assert_eq!(a.energy_milli, b.energy_milli);
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_threshold_selects_instantiation_and_runs() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 4801).unwrap();
    for (n, variant) in [(4800, "gibbs"), (4801, "gibbs-spill")] {
        let graph = circulant(n);
        let reads = sample_one(&device, &graph, &params(), KernelKind::Gibbs).unwrap();
        check_energies(&graph, &reads);
        assert_eq!(
            quip_solver_cuda::streaming::last_gibbs_variant(&device),
            Some(variant)
        );
    }
    let device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 1000).unwrap();
    assert!(matches!(
        sample_one(&device, &circulant(1001), &params(), KernelKind::Gibbs),
        Err(SampleError::GraphTooLarge { limit: 1000, .. })
    ));
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_model_past_limit_refused_before_allocation() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, usize::MAX).unwrap();
    let mut limits = device.limits;
    limits.free_bytes = 64 << 20;
    device = device.with_limits_for_test(limits);
    let model = max_model(
        KernelKind::Gibbs,
        Connectivity::Degree(20),
        Precision::Binary,
        256,
        &limits,
    );
    assert!(model.nodes > 4800);
    let graph = circulant(model.nodes);
    sample_one(&device, &graph, &params(), KernelKind::Gibbs).unwrap();
    bench_one(&device, &graph, &params(), KernelKind::Gibbs, None).unwrap();
    let graph = circulant(model.nodes + 1);
    let ctx = CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let (before, _) = ctx.mem_get_info().unwrap();
    device.max_nodes = 4800;
    for error in [
        sample_one(&device, &graph, &params(), KernelKind::Gibbs).unwrap_err(),
        bench_one(&device, &graph, &params(), KernelKind::Gibbs, None).unwrap_err(),
    ] {
        let SampleError::ModelExceedsDeviceMemory {
            required_bytes,
            available_bytes,
            instances: count,
            ..
        } = error
        else {
            panic!("expected memory refusal, got {error:?}");
        };
        assert!(required_bytes > available_bytes);
        assert_eq!(available_bytes, limits.usable_bytes());
        assert_eq!(count, instances(KernelKind::Gibbs, device.max_sms));
    }
    ctx.synchronize().unwrap();
    let (after, _) = ctx.mem_get_info().unwrap();
    eprintln!(
        "Gibbs boundary={} before={before} after={after}",
        model.nodes
    );
    assert!(before.abs_diff(after) <= 2 << 20);
}
