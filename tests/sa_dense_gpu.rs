//! Dense SA sample parity, planted ground states, and memory admission.

use cudarc::driver::CudaContext;
use quip_solver_cuda::capacity::{max_model, Connectivity, Precision, Storage};
use quip_solver_cuda::coefficients::{self, CoefficientPath};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{last_storage, sample_one, sample_one_real_forced};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn complete(n: usize, precision: Precision, planted: bool) -> IsingGraph {
    let mut seed = 42;
    let spins: Vec<_> = (0..n)
        .map(|_| {
            if random(&mut seed) & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let mut edges = Vec::with_capacity(n * (n - 1) / 2);
    let mut j = Vec::with_capacity(edges.capacity());
    for u in 0..n {
        for v in u + 1..n {
            let draw = random(&mut seed);
            let magnitude = match precision {
                Precision::Binary => 1.0,
                Precision::Int8 => f64::from(u32::try_from(draw % 127 + 1).unwrap()),
                Precision::Int32 => f64::from(u32::try_from(draw % 1000 + 1).unwrap()) / 1000.0,
                Precision::Float32 => {
                    0.5 + f64::from(u32::try_from(draw >> 32).unwrap()) / 4_294_967_296.0
                }
            };
            let sign = if planted {
                -spins[u] * spins[v]
            } else if draw & 1 == 0 {
                -1.0
            } else {
                1.0
            };
            edges.push((u, v));
            j.push(sign * magnitude);
        }
    }
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
fn sa_dense_matches_sparse_integer_samples() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 600).unwrap();
    // The small graph also exercises the i8-delta dense instantiation.
    for n in [32, 200] {
        for precision in [Precision::Binary, Precision::Int8, Precision::Int32] {
            let graph = complete(n, precision, false);
            let sparse =
                sample_one_real_forced(&device, &graph, &params(), Storage::Sparse).unwrap();
            assert_eq!(last_storage(&device), Some(Storage::Sparse));
            let dense = sample_one_real_forced(&device, &graph, &params(), Storage::Dense).unwrap();
            assert_eq!(last_storage(&device), Some(Storage::Dense));
            assert_eq!(dense.len(), 256);
            assert_eq!(sparse.len(), dense.len());
            for (a, b) in sparse.iter().zip(&dense) {
                assert_eq!(a.spins, b.spins, "N={n} {precision:?}");
                assert_eq!(a.energy.to_bits(), b.energy.to_bits());
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_dense_planted_ground_state_every_precision() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 600).unwrap();
    let params = SampleParams {
        num_sweeps: 1024,
        sweeps_per_beta: 16,
        ..params()
    };
    for precision in [
        Precision::Binary,
        Precision::Int8,
        Precision::Int32,
        Precision::Float32,
    ] {
        let graph = complete(600, precision, true);
        if precision != Precision::Float32 {
            let plan = coefficients::plan(KernelKind::Sa, &graph).unwrap();
            assert_eq!(
                plan.path,
                if precision == Precision::Int32 {
                    CoefficientPath::Wide
                } else {
                    CoefficientPath::WeightedInt8
                }
            );
        }
        let reads = sample_one_real_forced(&device, &graph, &params, Storage::Dense).unwrap();
        assert_eq!(last_storage(&device), Some(Storage::Dense));
        assert_eq!(reads.len(), 256);
        if precision == Precision::Float32 {
            let ground = -graph.j.iter().map(|j| j.abs()).sum::<f64>();
            assert!(
                reads
                    .iter()
                    .any(|r| (r.energy - ground).abs() <= 1e-6 * ground.abs()),
                "{precision:?}: ground={ground}"
            );
        } else {
            let ground = -graph
                .j
                .iter()
                .map(|&j| coefficients::to_milli(j).unwrap().abs())
                .sum::<i64>();
            assert!(
                reads.iter().any(|r| quip_protocol::scoring::energy_milli(
                    &r.spins,
                    &graph.h,
                    &graph.j,
                    &graph.edges
                ) == ground),
                "{precision:?}: ground={ground}"
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_dense_model_past_limit_refused() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, usize::MAX).unwrap();
    // Exercise the boundary without consuming the other workload's memory.
    let limits = quip_solver_cuda::capacity::DeviceLimits {
        free_bytes: 512 << 20,
        ..device.limits
    };
    let device = device.with_limits_for_test(limits);
    let model = max_model(
        KernelKind::Sa,
        Connectivity::Complete,
        Precision::Binary,
        32,
        &limits,
    );
    assert!(model.nodes > 200);
    let params = SampleParams {
        num_sweeps: 8,
        num_reads: 32,
        ..params()
    };
    let graph = complete(model.nodes, Precision::Binary, true);
    let reads = sample_one(&device, &graph, &params, KernelKind::Sa).unwrap();
    assert_eq!(last_storage(&device), Some(Storage::Dense));
    assert_eq!(reads.len(), 32);
    let graph = complete(model.nodes + 1, Precision::Binary, true);
    let ctx = CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let (before, _) = ctx.mem_get_info().unwrap();
    let error = sample_one(&device, &graph, &params, KernelKind::Sa).unwrap_err();
    let SampleError::ModelExceedsDeviceMemory {
        required_bytes,
        available_bytes,
        instances,
        ..
    } = error
    else {
        panic!("unexpected error: {error}")
    };
    assert!(required_bytes > available_bytes);
    assert_eq!(instances, device.max_sms);
    ctx.synchronize().unwrap();
    let (after, _) = ctx.mem_get_info().unwrap();
    assert!(before.abs_diff(after) <= 2 << 20);
    eprintln!(
        "SA dense boundary={} before={before} after={after}",
        model.nodes
    );
}
