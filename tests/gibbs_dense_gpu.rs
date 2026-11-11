//! Dense Gibbs sample parity, planted ground states, and memory admission.

use cudarc::driver::CudaContext;
use quip_solver_cuda::capacity::{max_model, Connectivity, Precision, Storage};
use quip_solver_cuda::coefficients::{self, CoefficientPath};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{
    last_gibbs_variant, last_storage, sample_one, sample_one_real, sample_one_real_forced,
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

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_planted_ground_state_every_precision() {
    gibbs_dense_planted_ground_state_every_precision_case(600, &params());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_planted_ground_state_ci() {
    gibbs_dense_planted_ground_state_every_precision_case(
        96,
        &SampleParams {
            num_reads: 32,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            ..params()
        },
    );
}

fn gibbs_dense_planted_ground_state_every_precision_case(nodes: usize, params: &SampleParams) {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, nodes).unwrap();
    for metropolis in [false, true] {
        device.gibbs_metropolis = metropolis;
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let graph = complete(nodes, precision, true);
            if precision != Precision::Float32 {
                let plan = coefficients::plan(KernelKind::Gibbs, &graph).unwrap();
                assert_eq!(
                    plan.path,
                    if precision == Precision::Int32 {
                        CoefficientPath::Wide
                    } else {
                        CoefficientPath::Int8
                    }
                );
            }
            let reads = sample_one_real_forced(&device, &graph, params, Storage::Dense).unwrap();
            assert_eq!(last_storage(&device), Some(Storage::Dense));
            assert_eq!(
                last_gibbs_variant(&device),
                Some(match precision {
                    Precision::Binary | Precision::Int8 => "gibbs-dense",
                    Precision::Int32 => "gibbs-i32-dense",
                    Precision::Float32 => "gibbs-f32-dense",
                })
            );
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
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_model_past_limit_refused() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, usize::MAX).unwrap();
    // Exercise the boundary without consuming the other workload's memory.
    let limits = quip_solver_cuda::capacity::DeviceLimits {
        free_bytes: 512 << 20,
        ..device.limits
    };
    let device = device.with_limits_for_test(limits);
    let model = max_model(
        KernelKind::Gibbs,
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
    let reads = sample_one(&device, &graph, &params, KernelKind::Gibbs).unwrap();
    assert_eq!(last_storage(&device), Some(Storage::Dense));
    assert_eq!(reads.len(), 32);
    let graph = complete(model.nodes + 1, Precision::Binary, true);
    let ctx = CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let (before, _) = ctx.mem_get_info().unwrap();
    let error = sample_one(&device, &graph, &params, KernelKind::Gibbs).unwrap_err();
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
    assert_eq!(instances, device.max_sms / 4);
    ctx.synchronize().unwrap();
    let (after, _) = ctx.mem_get_info().unwrap();
    assert!(before.abs_diff(after) <= 2 << 20);
    eprintln!(
        "Gibbs dense boundary={} before={before} after={after}",
        model.nodes
    );
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_spill_matches_shared_every_precision() {
    gibbs_dense_spill_matches_shared_every_precision_case(600, &params());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_spill_matches_shared_ci() {
    gibbs_dense_spill_matches_shared_every_precision_case(
        48,
        &SampleParams {
            num_reads: 8,
            num_sweeps: 4,
            ..params()
        },
    );
}

fn gibbs_dense_spill_matches_shared_every_precision_case(nodes: usize, params: &SampleParams) {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, nodes).unwrap();
    for metropolis in [false, true] {
        device.gibbs_metropolis = metropolis;
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let graph = complete(nodes, precision, false);
            device.gibbs_shared_spins_cap = None;
            let baseline = sample_one_real_forced(&device, &graph, params, Storage::Dense).unwrap();
            assert_eq!(baseline.len(), params.num_reads);
            for cap in [0, nodes / 2, nodes - 1] {
                device.gibbs_shared_spins_cap = Some(cap);
                let reads =
                    sample_one_real_forced(&device, &graph, params, Storage::Dense).unwrap();
                assert_eq!(reads.len(), baseline.len());
                for (expected, actual) in baseline.iter().zip(&reads) {
                    assert_eq!(
                        expected.spins, actual.spins,
                        "{precision:?} metropolis={metropolis} cap={cap}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_i32_reduction_keeps_low_bits() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 3).unwrap();
    // The first two fields pin +1 and -1. The last node's exact field is
    // 16_777_217 - 16_777_216 = 1. A float reduction loses that low bit.
    let graph = IsingGraph::new(
        vec![-40000.001, 40000.001, 0.0],
        vec![16777.217, 16777.216],
        vec![(0, 2), (1, 2)],
    );
    assert_eq!(
        coefficients::plan(KernelKind::Gibbs, &graph).unwrap().path,
        CoefficientPath::Wide
    );
    let p = SampleParams {
        num_sweeps: 2,
        beta_range: Some((1e6, 1e6)),
        ..params()
    };
    let reads = sample_one_real_forced(&device, &graph, &p, Storage::Dense).unwrap();
    assert_eq!(reads.len(), 256);
    assert!(reads.iter().all(|read| read.spins == [1, -1, -1]));
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_dense_real_large_bond_stays_finite() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 2).unwrap();
    let graph = IsingGraph::new(vec![0.0; 2], vec![1e38], vec![(0, 1)]);
    let p = SampleParams {
        num_sweeps: 2,
        beta_range: Some((2e38, 2e38)),
        ..params()
    };
    for metropolis in [false, true] {
        device.gibbs_metropolis = metropolis;
        let reads = sample_one_real_forced(&device, &graph, &p, Storage::Dense).unwrap();
        assert_eq!(reads.len(), 256);
        assert!(reads.iter().all(|read| read.spins[0] != read.spins[1]));
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_repeated_couplers_keep_sparse_storage() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 2).unwrap();
    for edges in [vec![(0, 1), (0, 1)], vec![(0, 1), (1, 0)]] {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0; 2], edges);
        let sparse = sample_one_real_forced(&device, &graph, &params(), Storage::Sparse).unwrap();
        let automatic = sample_one_real(&device, &graph, &params()).unwrap();
        assert_eq!(last_storage(&device), Some(Storage::Sparse));
        assert_eq!(sparse.len(), 256);
        assert_eq!(automatic.len(), sparse.len());
        for (expected, actual) in sparse.iter().zip(&automatic) {
            assert_eq!(expected.spins, actual.spins);
        }
        let error = sample_one_real_forced(&device, &graph, &params(), Storage::Dense).unwrap_err();
        assert!(matches!(error, SampleError::Unsupported(_)));
    }
}
