//! Dense SA sample parity, planted ground states, and memory admission.

use cudarc::driver::CudaContext;
use quip_solver_cuda::capacity::{max_model, Connectivity, Precision, Storage};
use quip_solver_cuda::coefficients::{self, CoefficientPath};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{
    last_storage, sample_one, sample_one_real, sample_one_real_forced,
};
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
fn sa_repeated_couplers_match_sparse_samples() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 2).unwrap();
    for edges in [vec![(0, 1), (0, 1)], vec![(0, 1), (1, 0)]] {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0; 2], edges);
        let sparse = sample_one_real_forced(&device, &graph, &params(), Storage::Sparse).unwrap();
        let automatic = sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap();
        assert_eq!(last_storage(&device), Some(Storage::Sparse));
        let real = sample_one_real(&device, &graph, &params()).unwrap();
        assert_eq!(last_storage(&device), Some(Storage::Sparse));
        assert_eq!(automatic.len(), 256);
        assert_eq!(automatic.len(), sparse.len());
        assert_eq!(real.len(), sparse.len());
        for ((exact, real), sparse) in automatic.iter().zip(&real).zip(&sparse) {
            assert_eq!(exact.spins, sparse.spins);
            assert_eq!(real.spins, sparse.spins);
            assert_eq!(real.energy.to_bits(), sparse.energy.to_bits());
            assert_eq!(
                exact.energy_milli,
                quip_protocol::scoring::energy_milli(
                    &sparse.spins,
                    &graph.h,
                    &graph.j,
                    &graph.edges,
                )
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_repeated_couplers_refuse_forced_dense() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 2).unwrap();
    for edges in [vec![(0, 1), (0, 1)], vec![(0, 1), (1, 0)]] {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0; 2], edges);
        let error = sample_one_real_forced(&device, &graph, &params(), Storage::Dense).unwrap_err();
        assert!(matches!(error, SampleError::Unsupported(_)), "{error}");
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_repeated_couplers_use_sparse_capacity_check() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, 2).unwrap();
    let limits = quip_solver_cuda::capacity::DeviceLimits {
        free_bytes: 64 << 20,
        ..device.limits
    };
    let shape = quip_solver_cuda::capacity::ModelShape {
        nodes: 2,
        couplers: 2,
        max_degree: 2,
    };
    let dense = quip_solver_cuda::capacity::sa_footprint(
        shape,
        Precision::Binary,
        Storage::Dense,
        256,
        64,
        &limits,
    );
    let sparse = quip_solver_cuda::capacity::sa_footprint(
        shape,
        Precision::Binary,
        Storage::Sparse,
        256,
        64,
        &limits,
    );
    assert!(dense.total() <= limits.usable_bytes());
    assert!(sparse.total() > limits.usable_bytes());
    let device = device.with_limits_for_test(limits);
    for edges in [vec![(0, 1), (0, 1)], vec![(0, 1), (1, 0)]] {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0; 2], edges);
        for error in [
            sample_one(&device, &graph, &params(), KernelKind::Sa).unwrap_err(),
            sample_one_real(&device, &graph, &params()).unwrap_err(),
            sample_one_real_forced(&device, &graph, &params(), Storage::Sparse).unwrap_err(),
        ] {
            let SampleError::ModelExceedsDeviceMemory {
                required_bytes,
                available_bytes,
                ..
            } = error
            else {
                panic!("expected sparse memory refusal, got {error}");
            };
            assert_eq!(required_bytes, sparse.total());
            assert_eq!(available_bytes, limits.usable_bytes());
        }
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

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_dense_planted_ground_state_every_precision() {
    sa_dense_planted_ground_state_every_precision_case(
        600,
        &SampleParams {
            num_sweeps: 1024,
            sweeps_per_beta: 16,
            ..params()
        },
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_dense_planted_ground_state_ci() {
    sa_dense_planted_ground_state_every_precision_case(
        96,
        &SampleParams {
            num_reads: 32,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            ..params()
        },
    );
}

fn sa_dense_planted_ground_state_every_precision_case(nodes: usize, params: &SampleParams) {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Sa, nodes).unwrap();
    for precision in [
        Precision::Binary,
        Precision::Int8,
        Precision::Int32,
        Precision::Float32,
    ] {
        let graph = complete(nodes, precision, true);
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
        let reads = sample_one_real_forced(&device, &graph, params, Storage::Dense).unwrap();
        assert_eq!(last_storage(&device), Some(Storage::Dense));
        assert_eq!(reads.len(), params.num_reads);
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
