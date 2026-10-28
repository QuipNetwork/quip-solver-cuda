//! SA global-state parity and device-memory admission.

use cudarc::driver::CudaContext;
use quip_solver_cuda::capacity::{max_model, Connectivity, Precision};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{bench_one, sample_one};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};

fn circulant(n: usize, wide: bool) -> IsingGraph {
    let edges: Vec<_> = (0..n)
        .flat_map(|i| (1..=10).map(move |d| (i, (i + d) % n)))
        .collect();
    let j = (0..edges.len())
        .map(|i| {
            if i % 2 == 0 {
                if wide {
                    1.001
                } else {
                    1.0
                }
            } else {
                -1.0
            }
        })
        .collect();
    IsingGraph::new(vec![0.0; n], j, edges)
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

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_global_matches_local_samples() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 5000).unwrap();
    for wide in [false, true] {
        let graph = circulant(2000, wide);
        device.sa_force_global = false;
        let local = sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap();
        device.sa_force_global = true;
        let global = sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap();
        assert_eq!(local.len(), 256);
        for (a, b) in local.iter().zip(&global) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
        assert_eq!(local.len(), global.len());
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_threshold_selects_instantiation_and_runs() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 5001).unwrap();
    let params = SampleParams {
        num_sweeps: 8192,
        sweeps_per_beta: 128,
        ..params()
    };
    for n in [5000, 5001] {
        let mut graph = circulant(n, false);
        graph.j.fill(-1.0);
        let reads = sample_one(&device, &graph, &params, KernelKind::Sa).unwrap();
        assert_eq!(reads.len(), 256);
        // Each of the N * 10 edges contributes at least -1. Aligned spins attain it.
        let ground_milli = -i64::try_from(n).unwrap() * 10 * 1000;
        let mut reached_ground = false;
        for r in reads {
            assert_eq!(r.spins.len(), n);
            assert!(r.spins.iter().all(|&spin| spin == -1 || spin == 1));
            let host_energy: i64 = graph
                .edges
                .iter()
                .map(|&(u, v)| -i64::from(r.spins[u]) * i64::from(r.spins[v]) * 1000)
                .sum();
            assert_eq!(r.energy_milli, host_energy);
            if host_energy == ground_milli {
                assert!(r.spins.iter().all(|&spin| spin == r.spins[0]));
                reached_ground = true;
            }
        }
        assert!(
            reached_ground,
            "N={n} did not reach the ferromagnetic ground state"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_model_past_limit_refused_before_allocation() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Sa, usize::MAX).unwrap();
    // Keep the boundary fixture fast and leave room for another GPU workload.
    let mut limits = device.limits;
    limits.free_bytes = 512 << 20;
    device = device.with_limits_for_test(limits);
    for (wide, precision) in [(false, Precision::Binary), (true, Precision::Int32)] {
        device.max_nodes = usize::MAX;
        let model = max_model(
            KernelKind::Sa,
            Connectivity::Degree(20),
            precision,
            256,
            &limits,
        );
        assert!(model.nodes > 5000);
        let graph = circulant(model.nodes, wide);
        sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap();
        bench_one(&device, &graph, &params(), KernelKind::Sa, None).unwrap();
        let graph = circulant(model.nodes + 1, wide);
        let ctx = CudaContext::new(0).unwrap();
        ctx.synchronize().unwrap();
        let (before, _) = ctx.mem_get_info().unwrap();
        // Memory refusal must take precedence over an optional lower node cap.
        device.max_nodes = 5000;
        for error in [
            sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap_err(),
            bench_one(&device, &graph, &params(), KernelKind::Sa, None).unwrap_err(),
        ] {
            let SampleError::ModelExceedsDeviceMemory {
                required_bytes,
                available_bytes,
                instances,
                ..
            } = error
            else {
                panic!("expected memory refusal, got {error:?}");
            };
            assert!(required_bytes > available_bytes);
            assert_eq!(available_bytes, limits.usable_bytes());
            assert_eq!(instances, device.max_sms);
        }
        ctx.synchronize().unwrap();
        let (after, _) = ctx.mem_get_info().unwrap();
        eprintln!(
            "SA {precision:?} boundary={} before={before} after={after}",
            model.nodes
        );
        assert!(
            before.abs_diff(after) <= 2 << 20,
            "before={before} after={after}"
        );
    }
}
