//! Cluster launch contract and tiny exhaustive device checks.

use quip_solver_cuda::capacity::DeviceLimits;
use quip_solver_cuda::cluster::device::{ClusterDevice, ClusterSchedule};
use quip_solver_cuda::cluster::{Model, Motifs, PreparedGraph, Weights};
use serial_test::serial;

#[path = "support/cluster_reference.rs"]
mod reference;

fn limits() -> DeviceLimits {
    DeviceLimits {
        shared_bytes_per_block: 48 << 10,
        shared_bytes_per_block_optin: 100 << 10,
        free_bytes: 1 << 30,
        sm_count: 48,
        threads_per_sm: 1536,
    }
}

// The CU-1 corner-sharing fixture, with nontrivial gauges and residual edges.
fn graph(weights: Weights) -> PreparedGraph {
    let tiles = [[0, 1, 2, 3], [3, 4, 5, 6]];
    let gauges = [[1, -1, 1, 1], [-1, 1, 1, -1]];
    let mut edges = vec![(0, 4), (5, 7), (1, 5), (2, 6), (1, 4), (6, 7)];
    let mut j = vec![-1000, -1000, 300, -200, 700, 500];
    for (tile, gauge) in tiles.iter().zip(gauges) {
        for a in 0..4 {
            for b in a + 1..4 {
                edges.push((tile[a], tile[b]));
                j.push(1000 * i32::from(gauge[a] * gauge[b]));
            }
        }
    }
    PreparedGraph::new(
        &Model {
            h_milli: &[0; 8],
            edges: &edges,
            j_milli: &j,
        },
        &Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &[(0, 4), (5, 7)],
        },
        weights,
        &limits(),
        32,
    )
    .unwrap()
}

#[test]
fn packed_move_deltas_match_original_energy() {
    let prepared = graph(Weights {
        tile_milli: 250,
        pair_milli: 250,
    });
    let layout = prepared.layout();
    for bits in 0..256_u32 {
        let spins: Vec<i8> = (0..8)
            .map(|i| if bits >> i & 1 == 0 { 1 } else { -1 })
            .collect();
        let device = prepared.permute(&spins).unwrap();
        for u in 0..8 {
            let mut field = 0_i64;
            for e in 0..usize::try_from(layout.all_deg[u]).unwrap() {
                let bytes = layout.all_ell[e * 8 + u].to_le_bytes();
                let v = usize::from(u16::from_le_bytes([bytes[2], bytes[3]]));
                let coupling = i64::from(i16::from_le_bytes([bytes[0], bytes[1]]));
                field += coupling * i64::from(device[v]);
            }
            let mut moved = spins.clone();
            moved[prepared.device_to_original()[u]] *= -1;
            assert_eq!(
                prepared.energy(&moved).unwrap() - prepared.energy(&spins).unwrap(),
                -2 * i64::from(device[u]) * field
            );
        }
        // A residual component flip counts only edges crossing its boundary.
        let component = [true, false, true, true, false, false, true, false];
        let mut acc = 0_i64;
        let mut moved = device.clone();
        for u in 0..8 {
            if !component[u] {
                continue;
            }
            moved[u] *= -1;
            for e in 0..usize::try_from(layout.weak_deg[u]).unwrap() {
                let bytes = layout.weak_ell[e * 8 + u].to_le_bytes();
                let v = usize::from(u16::from_le_bytes([bytes[2], bytes[3]]));
                if !component[v] {
                    acc += i64::from(i16::from_le_bytes([bytes[0], bytes[1]]))
                        * i64::from(device[u] * device[v]);
                }
            }
        }
        assert_eq!(
            prepared
                .weak_energy(&prepared.unpermute(&moved).unwrap())
                .unwrap()
                - prepared.weak_energy(&spins).unwrap(),
            -2 * acc
        );
    }
}

#[test]
fn research_schedule_defaults_and_validation() {
    let q = ClusterSchedule::default();
    assert_eq!(
        (q.sweeps, q.single_every, q.rounds, q.chunk),
        (65536, 1, 2, 2048)
    );
    assert_eq!(
        (q.beta_min, q.beta_knee, q.knee_frac, q.beta_max),
        (1.0, 0.0, 0.0, 12.0)
    );
    q.validate().unwrap();
    for bad in [
        ClusterSchedule {
            sweeps: 0,
            ..q.clone()
        },
        ClusterSchedule {
            rounds: 0,
            ..q.clone()
        },
        ClusterSchedule {
            chunk: 0,
            ..q.clone()
        },
        ClusterSchedule {
            beta_min: f32::NAN,
            ..q.clone()
        },
        ClusterSchedule {
            beta_max: f32::INFINITY,
            ..q.clone()
        },
        ClusterSchedule {
            sweeps: usize::MAX,
            ..q.clone()
        },
        ClusterSchedule {
            single_every: usize::MAX,
            ..q.clone()
        },
        ClusterSchedule {
            rounds: usize::MAX,
            ..q.clone()
        },
        ClusterSchedule {
            chunk: usize::MAX,
            ..q.clone()
        },
        ClusterSchedule {
            knee_frac: 1.0,
            ..q.clone()
        },
        ClusterSchedule { beta_min: 0.0, ..q },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn tiny_anneals_reach_exhaustive_ground_and_rescore() {
    for weights in [
        Weights::FULL,
        Weights {
            tile_milli: 250,
            pair_milli: 250,
        },
        Weights {
            tile_milli: 1,
            pair_milli: 0,
        },
    ] {
        let p = graph(weights);
        let ground = (0..256_u32)
            .map(|bits| {
                let s: Vec<i8> = (0..8)
                    .map(|i| if bits >> i & 1 == 0 { 1 } else { -1 })
                    .collect();
                p.energy(&s).unwrap()
            })
            .min()
            .unwrap();
        let mut device = ClusterDevice::open(0, p).unwrap();
        let q = ClusterSchedule {
            sweeps: 32,
            chunk: 7,
            ..Default::default()
        };
        let run = device.run(&q, 32, 1).unwrap();
        assert_eq!(run.spins.len(), 32);
        assert_eq!(run.energies.len(), 32);
        let p = graph(weights);
        for (s, &energy) in run.spins.iter().zip(&run.energies) {
            assert_eq!(s.len(), 8);
            assert!(s.iter().all(|&v| v == 1 || v == -1));
            assert_eq!(p.energy(s).unwrap(), energy);
        }
        assert_eq!(run.energies.iter().copied().min(), Some(ground));
        let again = device.run(&q, 32, 1).unwrap();
        assert_eq!(run.spins, again.spins);
        let unchunked = ClusterSchedule {
            chunk: 32,
            ..q.clone()
        };
        assert_eq!(run.spins, device.run(&unchunked, 32, 1).unwrap().spins);
        assert!(device.run(&q, 0, 1).is_err());
        assert!(device.run(&q, usize::MAX, 1).is_err());
    }
}

#[derive(Clone)]
struct TraceWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for TraceWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn assert_cluster_stays_lazy(kind: quip_solver_cuda::KernelKind) {
    let bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let writer = TraceWriter(std::sync::Arc::clone(&bytes));
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let device =
            quip_solver_cuda::cuda_device::CudaDevice::open_with_nodes(0, kind, 8).unwrap();
        drop(device);
    });
    let trace = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    assert!(
        trace.contains("cuda kernel sa:"),
        "kernel-load tracing must be active: {trace}"
    );
    assert!(
        !trace.contains("cuda kernel cluster:"),
        "ordinary open loaded the cluster kernel: {trace}"
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn sa_open_keeps_cluster_lazy() {
    assert_cluster_stays_lazy(quip_solver_cuda::KernelKind::Sa);
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn msa_open_keeps_cluster_lazy() {
    assert_cluster_stays_lazy(quip_solver_cuda::KernelKind::Msa);
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn gibbs_open_keeps_cluster_lazy() {
    assert_cluster_stays_lazy(quip_solver_cuda::KernelKind::Gibbs);
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn optin_shared_memory_with_knee_and_no_periodic_single_spin() {
    // 49,525 dynamic bytes: above the default 48 KiB, odd spin count,
    // and more than ten iterations of the 512-thread node loops.
    let mut edges = vec![(4, 5500), (3, 5500)];
    let mut couplings = vec![-1000, 300];
    for first in 0..4 {
        for second in first + 1..4 {
            edges.push((first, second));
            couplings.push(1000);
        }
    }
    let prepared = PreparedGraph::new(
        &Model {
            h_milli: &vec![0; 5501],
            edges: &edges,
            j_milli: &couplings,
        },
        &Motifs {
            tiles: &[[0, 1, 2, 3]],
            gauges: &[[1; 4]],
            pairs: &[(4, 5500)],
        },
        Weights {
            tile_milli: 250,
            pair_milli: 250,
        },
        &limits(),
        4,
    )
    .unwrap();
    assert!(prepared.dynamic_shared_bytes() > 48 << 10);
    let schedule = ClusterSchedule {
        sweeps: 5,
        chunk: 2,
        single_every: 0,
        rounds: 3,
        beta_knee: 3.0,
        knee_frac: 0.4,
        ..Default::default()
    };
    let mut device = ClusterDevice::open(0, prepared).unwrap();
    let run = device.run(&schedule, 4, u64::MAX).unwrap();
    assert_eq!(run.spins.len(), 4);
    assert_eq!(run.energies.len(), 4);
    for (spins, &energy) in run.spins.iter().zip(&run.energies) {
        assert_eq!(spins.len(), 5501);
        assert!(spins.iter().all(|&v| v == 1 || v == -1));
        let direct: i64 = edges
            .iter()
            .zip(&couplings)
            .map(|(&(u, v), &coupling)| {
                i64::from(coupling)
                    * i64::from(
                        spins[usize::try_from(u).unwrap()] * spins[usize::try_from(v).unwrap()],
                    )
            })
            .sum();
        assert_eq!(direct, energy);
        // Final descent still runs when periodic single-spin sweeps are off.
        for node in [0, 1, 2, 3, 4, 5500] {
            let mut moved = spins.clone();
            moved[node] *= -1;
            let moved_energy: i64 = edges
                .iter()
                .zip(&couplings)
                .map(|(&(u, v), &coupling)| {
                    i64::from(coupling)
                        * i64::from(
                            moved[usize::try_from(u).unwrap()] * moved[usize::try_from(v).unwrap()],
                        )
                })
                .sum();
            assert!(moved_energy >= energy);
        }
    }
}

// Missing annealing, wrong residual signs, lost RNG, and omitted tail sweeps
// must disagree before final descent can hide their effects.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn pre_descent_sweeps_match_research_reference() {
    for weights in [
        Weights::FULL,
        Weights {
            tile_milli: 250,
            pair_milli: 250,
        },
        Weights {
            tile_milli: 1,
            pair_milli: 0,
        },
    ] {
        let p = graph(weights);
        let mut device = ClusterDevice::open(0, graph(weights)).unwrap();
        for (sweeps, single_every, knee) in [(1, 0, false), (2, 1, false), (7, 2, true)] {
            let q = ClusterSchedule {
                sweeps,
                single_every,
                chunk: 3,
                rounds: 3,
                beta_min: 0.3,
                beta_max: 4.0,
                beta_knee: if knee { 1.2 } else { 0.0 },
                knee_frac: if knee { 0.4 } else { 0.0 },
            };
            for seed in [1, u64::MAX] {
                let (expected, rng) = reference::run(&p, &q, 4, seed);
                let (actual, actual_rng) = device.run_before_descent_for_test(&q, 4, seed).unwrap();
                assert_eq!(
                    actual.spins, expected,
                    "spins: {q:?}, {weights:?}, seed {seed}"
                );
                assert_eq!(actual_rng, rng, "RNG: {q:?}, {weights:?}, seed {seed}");
                assert_eq!(actual.energies.len(), 4);
                for (s, &e) in expected.iter().zip(&actual.energies) {
                    assert_eq!(e, p.energy(s).unwrap());
                }
                let unchunked = ClusterSchedule {
                    chunk: sweeps,
                    ..q.clone()
                };
                let (whole, whole_rng) = device
                    .run_before_descent_for_test(&unchunked, 4, seed)
                    .unwrap();
                assert_eq!(whole.spins, expected);
                assert_eq!(whole_rng, rng);
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn union_find_overlapping_tiles_and_links_matches_host_components() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    // Corner-sharing tiles form two long chains. Links connect those chains,
    // with cycles and duplicate bonds to contend on shared roots. Node 1100
    // stays isolated. The 1100-node component crosses warps and block strides.
    let n = 1101_usize;
    let mut edges = Vec::new();
    for offset in [0, 550] {
        for first in (offset..offset + 549).step_by(3) {
            let tile = [first, first + 1, first + 2, first + 3];
            for a in 0..4 {
                for b in a + 1..4 {
                    edges.push((tile[a], tile[b]));
                }
            }
        }
        edges.push((offset + 546, offset + 549));
    }
    edges.extend([(0, 550), (31, 590), (500, 1099), (0, 550)]);
    let expected = reference::components(n, &edges);
    assert_eq!(expected.iter().filter(|&&r| r == 0).count(), 1100);
    assert_eq!(expected[1100], 1100);
    // Append only a test entry point; compile the real production unite and
    // find_root unchanged. A separate host traversal supplies the oracle.
    let source = format!(
        "{}\n{}",
        include_str!("../kernels/cluster.cu"),
        r#"
extern "C" __global__ void test_components(int n, int m, const int2* edges, int* roots) {
    extern __shared__ int parent[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) parent[i] = i;
    __syncthreads();
    for (int e = threadIdx.x; e < m; e += blockDim.x) unite(parent, edges[e].x, edges[e].y);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) roots[i] = find_root(parent, i);
}
"#
    );
    let ctx = CudaContext::new(0).unwrap();
    let (major, minor) = ctx.compute_capability().unwrap();
    let ptx = compile_ptx_with_opts(
        source,
        CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        },
    )
    .unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let function = module.load_function("test_components").unwrap();
    let stream = ctx.default_stream();
    let flat: Vec<i32> = edges
        .iter()
        .flat_map(|&(a, b)| [i32::try_from(a).unwrap(), i32::try_from(b).unwrap()])
        .collect();
    let device_edges = stream.clone_htod(&flat).unwrap();
    let mut roots = stream.alloc_zeros::<i32>(n).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: u32::try_from(n * 4).unwrap(),
    };
    let mut launch = stream.launch_builder(&function);
    let nodes = i32::try_from(n).unwrap();
    let bonds = i32::try_from(edges.len()).unwrap();
    launch
        .arg(&nodes)
        .arg(&bonds)
        .arg(&device_edges)
        .arg(&mut roots);
    // SAFETY: Edges are in range, output and shared parent have n entries,
    // and the argument types match the appended test kernel.
    unsafe { launch.launch(config) }.unwrap();
    let actual = stream.clone_dtoh(&roots).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        actual,
        expected
            .iter()
            .map(|&r| i32::try_from(r).unwrap())
            .collect::<Vec<_>>()
    );
}
