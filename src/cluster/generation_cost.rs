//! Opt-in CU-4 measurement. Run one shape per process on physical GPU 1.

use super::*;
use crate::capacity::DeviceLimits;
use crate::cluster::sampler::ClusterWork;
use cudarc::driver::sys::CUevent_flags;
use quip_solver_core::ising_generators::descriptor::{
    admit, params_digest, AdvertisedLimits, SizeBounds,
};
use quip_solver_core::ising_generators::family::{
    admit_topology, generate_instance, BaseTopology, SaltFault,
};
use quip_solver_core::ising_generators::ice::{zephyr_lattice, TWISTED_DIMER_V1};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::random::NonceDeriver;
use quip_solver_core::quip_protocol::lease::{Generator, LeaseSpec};
use quip_solver_core::{IsingGraph, SampleParams};
use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

const HOST_RUNS_DEFAULT: usize = 20;
const KERNEL_RUNS: usize = 5;

#[allow(dead_code)] // Reuse the real coordinator rather than copying session behavior.
#[path = "../../tests/common/lease.rs"]
mod session;

#[allow(clippy::too_many_lines)] // One timed session and its untimed result validation.
async fn throughput(base: &BaseTopology, params: &[u8], mode: &str) -> serde_json::Value {
    use quip_solver_core::quip_proto::v1::{
        coord_msg, EdgeList, FamilyDescriptor, GeneratorAlgorithm, Topology,
    };
    use std::time::Duration;

    assert!(["serial", "core-prefetch"].contains(&mode));
    let binary = std::env::var("CU4_BINARY").expect("set CU4_BINARY to the release miner path");
    let mut session = session::Session::start(&binary, 1, "").await;
    let hash = [0x73; 32];
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: hash.to_vec(),
            nodes: base.node_ids().to_vec(),
            edges: Some(EdgeList {
                u: base.edges().iter().map(|&(u, _)| u).collect(),
                v: base.edges().iter().map(|&(_, v)| v).collect(),
            }),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }))
        .await;
    session
        .send(coord_msg::Msg::SetTarget(session::target(i64::MAX, 256)))
        .await;
    let request = |id: &[u8], start, count| {
        let mut request = session::lease(id, 1, hash, start, count);
        let generator = request.generator.as_mut().unwrap();
        generator.algorithm = GeneratorAlgorithm::FamilyChacha8V1 as i32;
        generator.family = Some(FamilyDescriptor {
            family_id: TWISTED_DIMER_V1.family_id.into(),
            revision: 1,
            params: params.to_vec(),
            params_digest: params_digest(params).to_vec(),
        });
        generator.topology_order_digest = base.order_digest().to_vec();
        // Match the original cost measurement's nonce inputs.
        generator.last_proof_block_hash = vec![1; 32];
        generator.miner_account = vec![2; 32];
        generator.base_salt = vec![3; 32];
        request
    };
    session
        .send(coord_msg::Msg::Job(request(b"warmup", 999, 1)))
        .await;
    session
        .until("warmup", Duration::from_secs(10), |log| {
            log.done_count(b"warmup") == 1 && log.refunds() == 1
        })
        .await;
    assert!(session.log.rejects.is_empty());
    let gpu_start = telemetry();
    let start = Instant::now();
    let leases = if mode == "serial" { 50 } else { 1 };
    for index in 0..leases {
        let id = format!("timed-{index}");
        let (salt_start, count) = if mode == "serial" {
            (index, 1)
        } else {
            (0, 50)
        };
        session
            .send(coord_msg::Msg::Job(request(
                id.as_bytes(),
                salt_start,
                count,
            )))
            .await;
        session
            .until("50-job throughput", Duration::from_secs(15), |log| {
                log.done_count(id.as_bytes()) == 1
                    && log.refunds() == u32::try_from(index).unwrap() + 2
            })
            .await;
        assert!(start.elapsed() < Duration::from_secs(40));
    }
    let elapsed = start.elapsed().as_secs_f64();
    let gpu_end = telemetry();
    assert!(session.log.rejects.is_empty());
    assert!(session.log.fatal.is_none());
    assert_eq!(session.log.results.len(), 51);
    let spec =
        LeaseSpec::from_proto(request(b"expected", 0, 50).generator.as_ref().unwrap()).unwrap();
    let digest = params_digest(params);
    let contracts = [TWISTED_DIMER_V1];
    let admitted = admit(
        &contracts,
        TWISTED_DIMER_V1.family_id.as_bytes(),
        1,
        params,
        &digest,
    )
    .unwrap();
    let bounds = admit_topology(
        &admitted,
        base,
        base.order_digest(),
        &AdvertisedLimits {
            max_nodes: 0,
            max_edges: 0,
            max_degree: 0,
        },
    )
    .unwrap();
    let mut energies = Vec::new();
    let mut sampler_time_us = 0_u64;
    for (index, result) in session.log.results.iter().skip(1).enumerate() {
        let index = u64::try_from(index).unwrap();
        assert_eq!(result.salt, spec.salt(index).unwrap());
        assert_eq!(result.nonce, spec.nonce(index).unwrap());
        assert_eq!(result.solutions.len(), 64);
        let meta = result.meta.as_ref().unwrap();
        assert_eq!((meta.reads, meta.sweeps), (64, 256));
        sampler_time_us += meta.device_access_time_us;
        let instance =
            generate(params, &digest, base, &bounds, spec.nonce(index).unwrap()).unwrap();
        let edges: Vec<_> = instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        for solution in &result.solutions {
            let spins = quip_solver_core::quip_protocol::wire::decode_spins_packed(
                &solution.spins,
                instance.node_ids().len(),
            )
            .unwrap();
            assert_eq!(
                solution.energy_milli,
                quip_solver_core::quip_protocol::scoring::energy_from_milli(
                    &spins,
                    instance.h(),
                    instance.j(),
                    &edges
                )
            );
        }
        energies.push(
            result
                .solutions
                .iter()
                .map(|s| s.energy_milli)
                .collect::<Vec<_>>(),
        );
    }
    let ctx = CudaContext::new(0).unwrap();
    let width = limits(&ctx).sm_count;
    assert_eq!(session.shutdown(2000).await, 0);
    json!({"mode": mode, "jobs": 50, "elapsed_s": elapsed,
        "jobs_per_s": 50.0 / elapsed, "reads": 64, "sweeps": 256,
        "stream_width": width, "lease_prefetch_depth": width * 2,
        "sampler_time_s": Duration::from_micros(sampler_time_us).as_secs_f64(),
        "gpu_start": gpu_start, "gpu_end": gpu_end, "energies_milli": energies})
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[mid - 1].midpoint(values[mid])
    } else {
        values[mid]
    }
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn command(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_owned(),
        _ => String::new(),
    }
}

fn telemetry() -> String {
    command(
        "nvidia-smi",
        &[
            "-i", "1", "--query-gpu=name,uuid,pci.bus_id,driver_version,temperature.gpu,clocks.sm,pstate,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    )
}

fn parameters(m: u16, t: u8, weak: u16, pairs: u32) -> Vec<u8> {
    let mut bytes = vec![1];
    for (tag, value) in [
        (1_u16, m.to_le_bytes().to_vec()),
        (2, vec![t]),
        (3, weak.to_le_bytes().to_vec()),
        (4, pairs.to_le_bytes().to_vec()),
    ] {
        bytes.extend(tag.to_le_bytes());
        bytes.extend(u32::try_from(value.len()).unwrap().to_le_bytes());
        bytes.extend(value);
    }
    bytes
}

// FamilyLease has no public constructor. This is its exact generate body:
// re-admit the saved descriptor, then call the canonical generate_instance.
fn generate(
    params: &[u8],
    digest: &[u8; 32],
    base: &BaseTopology,
    bounds: &SizeBounds,
    nonce: [u8; 32],
) -> Result<Instance, SaltFault> {
    let admitted = admit(
        &[TWISTED_DIMER_V1],
        TWISTED_DIMER_V1.family_id.as_bytes(),
        1,
        params,
        digest,
    )
    .unwrap();
    generate_instance(&admitted, base, bounds, nonce)
}

fn limits(ctx: &CudaContext) -> DeviceLimits {
    DeviceLimits {
        shared_bytes_per_block: attribute(
            ctx,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )
        .unwrap(),
        shared_bytes_per_block_optin: attribute(
            ctx,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .unwrap(),
        free_bytes: ctx.mem_get_info().unwrap().0,
        sm_count: attribute(
            ctx,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap(),
        threads_per_sm: attribute(
            ctx,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )
        .unwrap(),
    }
}

fn ising_graph(instance: &Instance) -> IsingGraph {
    IsingGraph {
        h: instance
            .h()
            .iter()
            .map(|&v| f64::from(v) / 1000.0)
            .collect(),
        j: instance
            .j()
            .iter()
            .map(|&v| f64::from(v) / 1000.0)
            .collect(),
        edges: instance
            .edges()
            .iter()
            .map(|&(u, v)| (usize::try_from(u).unwrap(), usize::try_from(v).unwrap()))
            .collect::<Vec<_>>()
            .into(),
    }
}

fn topology(m: u16, t: u8) -> BaseTopology {
    if m != 0 || t != 0 {
        let (nodes, edges) = zephyr_lattice(m, t).unwrap();
        return BaseTopology::new(nodes, &edges).unwrap();
    }
    // Adjacency-mode endpoint: disjoint K4 motifs connected in a ring.
    // Every node has degree four, with one disjoint dimer per tile.
    let n = TWISTED_DIMER_V1.limits.max_nodes();
    let mut edges = Vec::new();
    for tile in (0..n).step_by(4) {
        for a in 0..4 {
            for b in a + 1..4 {
                edges.push((tile + a, tile + b));
            }
        }
        edges.push((tile + 1, (tile + 4) % n));
    }
    BaseTopology::new((0..n).collect(), &edges).unwrap()
}

// Measure the five actual upload buffers that carry coefficients. ELL words
// also contain neighbor indices, so counting their entire transfer is an
// optimistic upper bound on the bytes device generation could eliminate.
fn coefficients(device: &mut ClusterDevice) -> f64 {
    let stream = &device.stream;
    let start = stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let d = device.graph.layout();
    start.record(stream).unwrap();
    stream
        .memcpy_htod(&d.weak_ell, &mut device.weak_ell)
        .unwrap();
    stream.memcpy_htod(&d.all_ell, &mut device.all_ell).unwrap();
    stream.memcpy_htod(&d.h, &mut device.h).unwrap();
    stream.memcpy_htod(&d.tile_w, &mut device.tile_w).unwrap();
    stream.memcpy_htod(&d.link_w, &mut device.link_w).unwrap();
    end.record(stream).unwrap();
    f64::from(start.elapsed_ms(&end).unwrap())
}

// Use the production per-sweep callback and launch arguments, with events
// immediately around each launch. Exclude host gaps from the kernel sum.
fn kernel_ms(device: &ClusterDevice, work: &ClusterWork) -> (f64, Vec<i8>) {
    let (spin_count, rng_count, blocks) = launch_shape(&device.graph, work.reads).unwrap();
    let (spins, rng) = initial_state(spin_count, rng_count, work.seed);
    let mut spins = device.stream.clone_htod(&spins).unwrap();
    let mut rng = device.stream.clone_htod(&rng).unwrap();
    device.stream.synchronize().unwrap();
    let ctx = device.stream.context();
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: u32::try_from(device.graph.dynamic_shared_bytes()).unwrap(),
    };
    let params = ClusterSchedule {
        sweeps: work.betas.len() * work.sweeps_per_beta,
        chunk: 1,
        ..ClusterSchedule::default()
    }
    .params()
    .unwrap();
    let wall = Instant::now();
    let mut total = 0.0;
    work.for_each_launch(
        &|| wall.elapsed().as_secs() >= 40,
        |sweep, beta, descent| {
            start.record(&device.stream)?;
            device.launch(
                Params {
                    beta_min: beta,
                    beta_max: beta,
                    ..params
                },
                i32::try_from(sweep).unwrap(),
                i32::from(!descent),
                if descent { 2 } else { 0 },
                config,
                &mut spins,
                &mut rng,
            )?;
            end.record(&device.stream)?;
            device.stream.synchronize()?;
            total += f64::from(start.elapsed_ms(&end)?);
            Ok(())
        },
    )
    .unwrap();
    let spins = device.stream.clone_dtoh(&spins).unwrap();
    device.stream.synchronize().unwrap();
    (total, spins)
}

#[test]
#[ignore = "CU-4 measurement: requires CUDA_VISIBLE_DEVICES=1 and CU4_SHAPE=m,t,weak,pairs"]
#[allow(clippy::too_many_lines)] // One sequential measurement keeps GPU phases serialized.
fn measure_generation_cost() {
    let shape = std::env::var("CU4_SHAPE").expect("set CU4_SHAPE=m,t,weak,pairs");
    let fields: Vec<_> = shape.split(',').collect();
    assert_eq!(fields.len(), 4);
    let (m, t, weak, pairs) = (
        fields[0].parse().unwrap(),
        fields[1].parse().unwrap(),
        fields[2].parse().unwrap(),
        fields[3].parse().unwrap(),
    );
    let output =
        std::env::var("CU4_OUTPUT").expect("set CU4_OUTPUT to a JSONL path inside this checkout");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root.join(output);
    let parent = output.parent().unwrap().canonicalize().unwrap();
    assert!(
        parent.starts_with(root.canonicalize().unwrap()) || parent.starts_with("/home/vanthoi/lam-viec-tam"),
        "CU4_OUTPUT must be inside the checkout or the tmpfs working directory"
    );
    let start_telemetry = telemetry();
    if let Ok(mode) = std::env::var("CU4_THROUGHPUT") {
        let base = topology(m, t);
        let params = parameters(m, t, weak, pairs);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut row = runtime.block_on(throughput(&base, &params, &mode));
        row["shape"] = json!(shape);
        row["commit"] = json!(command("git", &["rev-parse", "HEAD"]));
        row["date_utc"] = json!(command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]));
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output)
            .unwrap();
        serde_json::to_writer(&mut file, &row).unwrap();
        writeln!(file).unwrap();
        println!("CU4 {row}");
        return;
    }
    let kernel = ClusterKernel::load(0).unwrap(); // Visible ordinal 0 is physical GPU 1.
    let ctx = kernel.stream.context();
    let limits = limits(ctx);
    let (base, params) = if let Ok(fp) = std::env::var("CU4_FAMILY") {
        // The live family: its saved topology and descriptor parameters (a subset of the full lattice).
        let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fp).unwrap()).unwrap();
        let hexb = |x: &str| -> Vec<u8> { (0..x.len() / 2).map(|i| u8::from_str_radix(&x[2 * i..2 * i + 2], 16).unwrap()).collect() };
        let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
        (BaseTopology::new(nodes.clone(), &edges).unwrap(), hexb(fam["params_hex"].as_str().unwrap()))
    } else {
        (topology(m, t), parameters(m, t, weak, pairs))
    };
    let file_nonces: Vec<[u8; 32]> = std::env::var("CU4_NONCE_FILE").ok().map_or_else(Vec::new, |f| {
        std::fs::read_to_string(f).unwrap().lines().filter(|l| l.len() == 64).map(|l| {
            let mut n = [0u8; 32];
            for (i, b) in n.iter_mut().enumerate() {
                *b = u8::from_str_radix(&l[2 * i..2 * i + 2], 16).unwrap();
            }
            n
        }).collect()
    });
    let digest = params_digest(&params);
    let contracts = [TWISTED_DIMER_V1];
    let admitted = admit(
        &contracts,
        TWISTED_DIMER_V1.family_id.as_bytes(),
        1,
        &params,
        &digest,
    )
    .unwrap();
    let bounds = admit_topology(
        &admitted,
        &base,
        base.order_digest(),
        &AdvertisedLimits {
            max_nodes: 0,
            max_edges: 0,
            max_degree: 0,
        },
    );
    let bounds = match bounds {
        Ok(bounds) => bounds,
        Err(error) => {
            let row = json!({"shape": shape, "nodes": base.node_ids().len(), "edges": base.edges().len(),
                "family": TWISTED_DIMER_V1.family_id, "revision": 1, "admission_rejection": error.value(),
                "commit": command("git", &["rev-parse", "HEAD"]), "gpu_start": start_telemetry,
                "date_utc": command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])});
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(output)
                .unwrap();
            serde_json::to_writer(&mut file, &row).unwrap();
            writeln!(file).unwrap();
            println!("CU4 {row}");
            return;
        }
    };
    let lease = LeaseSpec::new(
        Generator::FamilyChacha8V1,
        [1; 32],
        [2; 32],
        [3; 32],
        0,
        1000,
    )
    .unwrap();
    let nonces = NonceDeriver::new(lease.last_proof_block_hash, lease.miner_account);
    let budget = std::env::var("CU4_BUDGET").unwrap_or_else(|_| "mining-easy".into());
    let mut sample_params = SampleParams::default();
    if budget == "custom" {
        sample_params.num_reads = std::env::var("CU4_READS").unwrap().parse().unwrap();
        sample_params.num_sweeps = std::env::var("CU4_SWEEPS").unwrap().parse().unwrap();
    }
    if let Ok(b) = std::env::var("CU4_BETA") {
        let v: Vec<f64> = b.split(',').map(|x| x.parse().unwrap()).collect();
        sample_params.beta_range = Some((v[0], v[1]));
    }
    if let Ok(b) = std::env::var("CU4_SEED") {
        sample_params.seed = b.parse().unwrap();
    }
    if let Ok(b) = std::env::var("CU4_SPB") {
        sample_params.sweeps_per_beta = b.parse().unwrap();
    }
    if budget == "custom" {
    } else if budget != "api-default" {
        let target = match budget.as_str() {
            "mining-easy" => i64::MAX,
            "mining-hard" => i64::MIN,
            _ => panic!("CU4_BUDGET must be api-default, mining-easy, mining-hard, or custom"),
        };
        let identity = crate::cluster::sampler::cuda_otsuka_identity(&limits, usize::MAX);
        let adapted = quip_solver_core::adapt::adapt_params(
            target,
            1,
            base.node_ids().len(),
            base.edges().len(),
            &[0],
            &identity.adapt,
        );
        sample_params.num_reads = usize::try_from(adapted.num_reads).unwrap();
        sample_params.num_sweeps = usize::try_from(adapted.num_sweeps).unwrap();
    }
    let mut instances = Vec::new();
    let mut instance_nonces = Vec::new();
    let mut generation = Vec::new();
    let mut skipped = Vec::new();
    // Untimed generator warmup. The same lease nonces are used on every rerun.
    let _warmup = generate(
        &params,
        &digest,
        &base,
        &bounds,
        nonces.derive(lease.salt(0).unwrap()),
    );
    let host_runs: usize = std::env::var("CU4_HOST").ok().map_or(HOST_RUNS_DEFAULT, |v| v.parse().unwrap());
    for index in 0..100_000 {
        if !file_nonces.is_empty() && index as usize >= file_nonces.len() {
            break;
        }
        let nonce = if file_nonces.is_empty() { nonces.derive(lease.salt(index).unwrap()) } else { file_nonces[index as usize] };
        let start = Instant::now();
        let instance = generate(&params, &digest, &base, &bounds, nonce);
        let ms = milliseconds(start);
        match instance {
            Ok(instance) => {
                instances.push(instance);
                instance_nonces.push(nonce);
                generation.push(ms);
            }
            Err(SaltFault::Skip(code)) => {
                skipped.push(json!({"index": index, "code": code.value(), "ms": ms}));
            }
            Err(error) => panic!("canonical generator defect: {error:?}"),
        }
        if instances.len() == host_runs {
            break;
        }
    }
    assert!(instances.len() == host_runs || !file_nonces.is_empty(), "not enough valid jobs");
    let prepare = |instance: &Instance| {
        PreparedGraph::from_instance(
            instance,
            &TWISTED_DIMER_V1,
            crate::cluster::Weights::FULL,
            &limits,
            sample_params.num_reads,
        )
    };
    let _warmup = prepare(&instances[0]);
    let mut preparation = Vec::new();
    let mut prepared = Vec::new();
    let mut rejection = None;
    for instance in &instances {
        let start = Instant::now();
        let result = prepare(instance);
        preparation.push(milliseconds(start));
        match result {
            Ok(graph) => prepared.push(graph),
            Err(error) => rejection = Some(error.to_string()),
        }
    }
    let obstructions: Vec<serde_json::Value> = prepared
        .iter()
        .map(|g| {
            let l = g.layout();
            let tiles: Vec<[u32; 4]> = l.tiles.chunks(4).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32]).collect();
            let gauges: Vec<[i8; 4]> = l.gauges.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
            match quip_solver_core::ising_generators::ice::gauge_obstructions(&tiles, &gauges, g.original_graph().num_nodes()) {
                Ok(o) => json!([tiles.len(), l.pairs.len() / 2, o.components, o.cycle_rank, o.frustrated_cycles]),
                Err(_) => json!(null),
            }
        })
        .collect();
    eprintln!("OBS {}", serde_json::Value::Array(obstructions));
    if let Ok(dir) = std::env::var("CU4_SCAN") {
        let chunk: usize = std::env::var("CU4_CHUNK").ok().map_or(1, |v| v.parse().unwrap());
        for (idx, inst) in instances.iter().enumerate() {
            let g = ising_graph(inst);
            let pg = prepare(inst).unwrap();
            let og = pg.original_graph();
            let mut out = String::new();
            out.push_str(&format!("N {}\nE {}\n", g.h.len(), g.edges.len()));
            for (k, &(u, v)) in g.edges.iter().enumerate() {
                out.push_str(&format!("{} {} {}\n", u, v, (g.j[k] * 1000.0).round() as i64));
            }
            out.push_str("T\n");
            for t in og.tile_nodes() {
                out.push_str(&format!("{} {} {} {}\n", t[0], t[1], t[2], t[3]));
            }
            out.push_str("P\n");
            for (a, b) in og.pair_list() {
                out.push_str(&format!("{a} {b}\n"));
            }
            if std::env::var("CU4_SCAN_NOFILES").is_err() {
                std::fs::write(format!("{dir}/i{idx}.txt"), out).unwrap();
            }
            if std::env::var("CU4_CORR").is_ok() {
                // Cheap predictors of the bottom energy on the same instances: in-tree MSA (one SM per nonce) at two budgets and
                // the cluster annealer's stage A. One line per instance: nonce, msa(128x4096), msa(128x1024), otsuka A(256x64).
                let dev_msa = crate::cuda_device::CudaDevice::open_with_nodes(0, crate::KernelKind::Msa, 4600).unwrap();
                let g2 = ising_graph(inst);
                let mut out = Vec::new();
                for &(reads, sweeps) in &[(128usize, 4096usize), (128, 1024)] {
                    let p = SampleParams { num_reads: reads, num_sweeps: sweeps, sweeps_per_beta: 1, seed: 11, ..SampleParams::default() };
                    let (rs, _) = crate::streaming::bench_one(&dev_msa, &g2, &p, crate::KernelKind::Msa, None).unwrap();
                    out.push(rs.iter().map(|r| r.energy_milli).min().unwrap_or(0) as f64 * 0.001);
                }
                let mut p = sample_params.clone();
                p.num_reads = 256;
                p.num_sweeps = 64;
                p.sweeps_per_beta = 1;
                let pgx = PreparedGraph::from_instance(inst, &TWISTED_DIMER_V1, crate::cluster::Weights::FULL, &limits, 256).unwrap();
                let mut dx = ClusterDevice::open_with_kernel(pgx, &kernel).unwrap();
                let wk = ClusterWork::new(&g2, &p).unwrap();
                let rn = dx.run_work_with_chunk_for_test(&wk, 1).unwrap();
                out.push(rn.energies.iter().copied().min().unwrap_or(0) as f64 * 0.001);
                let nh: String = instance_nonces[idx].iter().map(|b| format!("{b:02x}")).collect();
                eprintln!("CORR {} {} {:.3} {:.3} {:.3}", idx, nh, out[0], out[1], out[2]);
                continue;
            }
            if std::env::var("CU4_RACE").is_ok() {
                // Same live instance, several solvers and budgets: best energy and time (kernel time for the in-tree SA, MSA,
                // Gibbs kernels; wall time of a run for the cluster annealer).
                let dev = crate::cuda_device::CudaDevice::open(0).unwrap();
                let g2 = ising_graph(inst);
                let dev_msa = crate::cuda_device::CudaDevice::open_with_nodes(0, crate::KernelKind::Msa, 4600).unwrap();
                for (kind, name) in [(crate::KernelKind::Msa, "msa"), (crate::KernelKind::Sa, "sa"), (crate::KernelKind::Gibbs, "gibbs")] {
                    let configs: &[(usize, usize)] = if name == "msa" {
                        &[(64, 64), (64, 256), (128, 256), (128, 1024), (128, 4096), (128, 16384), (128, 65536)]
                    } else {
                        &[(64, 64), (64, 256)]
                    };
                    for &(reads, sweeps) in configs {
                        let p = SampleParams { num_reads: reads, num_sweeps: sweeps, sweeps_per_beta: 1, seed: 11, ..SampleParams::default() };
                        let device = if name == "msa" { &dev_msa } else { &dev };
                        match crate::streaming::bench_one(device, &g2, &p, kind, None) {
                            Ok((rs, tm)) => {
                                let best = rs.iter().map(|r| r.energy_milli).min().unwrap_or(0);
                                eprintln!("RACE {idx} {name} {reads} {sweeps} {:.3} {:.3}", best as f64 * 0.001, tm.kernel_ns as f64 * 1e-6);
                            }
                            Err(e) => eprintln!("RACE {idx} {name} {reads} {sweeps} ERR {e:?}"),
                        }
                    }
                }
                for &(reads, sweeps) in &[(256usize, 64usize), (256, 256), (256, 1024), (256, 4096)] {
                    let mut p = sample_params.clone();
                    p.num_reads = reads;
                    p.num_sweeps = sweeps;
                    let spb = (sweeps / 64).max(1);
                    p.sweeps_per_beta = spb;
                    let pgx = PreparedGraph::from_instance(inst, &TWISTED_DIMER_V1, crate::cluster::Weights::FULL, &limits, reads).unwrap();
                    let mut dx = ClusterDevice::open_with_kernel(pgx, &kernel).unwrap();
                    let wk = ClusterWork::new(&g2, &p).unwrap();
                    let t0 = Instant::now();
                    let rn = dx.run_work_with_chunk_for_test(&wk, spb).unwrap();
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    let best = rn.energies.iter().copied().min().unwrap_or(0);
                    eprintln!("RACE {idx} otsuka {reads} {sweeps} {:.3} {:.3}", best as f64 * 0.001, ms);
                }
                continue;
            }
            let mut dev = ClusterDevice::open_with_kernel(pg, &kernel).unwrap();
            let work = ClusterWork::new(&g, &sample_params).unwrap();
            let run = dev.run_work_with_chunk_for_test(&work, chunk).unwrap();
            let best = run.energies.iter().copied().min().unwrap_or(0);
            let nh: String = instance_nonces[idx].iter().map(|b| format!("{b:02x}")).collect();
            let f = crate::cluster::sampler::lease_run::features_raw(inst.h().len(), inst.edges(), inst.j());
            eprintln!("SCAN {} {:.3} {} {:?}", idx, best as f64 * 0.001, nh, f);
        }
        return;
    }
    let mut row = json!({
        "date_utc": command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "commit": command("git", &["rev-parse", "HEAD"]),
        "source_diff": command("git", &["diff", "--stat"]),
        "gpu_start": start_telemetry, "family": TWISTED_DIMER_V1.family_id, "revision": 1,
        "shape": shape, "nodes": base.node_ids().len(), "edges": base.edges().len(),
        "budget": budget,
        "reads": sample_params.num_reads, "sweeps": sample_params.num_sweeps, "sweeps_per_beta": sample_params.sweeps_per_beta,
        "host_runs": host_runs, "generation_ms": median(&generation), "preparation_ms": median(&preparation),
        "generation_samples_ms": generation, "preparation_samples_ms": preparation,
        "skipped": skipped, "preparation_rejection": rejection,
        "shared_budget_bytes": limits.shared_bytes_per_block_optin,
    });
    if rejection.is_none() {
        let gpu_phase = Instant::now();
        assert_eq!(prepared.len(), host_runs);
        // Compilation, allocation initialization, and the first solve are warm.
        let mut device =
            ClusterDevice::open_with_kernel(prepare(&instances[0]).unwrap(), &kernel).unwrap();
        let instance = &instances[0];
        let graph = ising_graph(instance);
        let work = ClusterWork::new(&graph, &sample_params).unwrap();
        let warmup = Instant::now();
        if std::env::var("CU4_FAST").is_err() {
            device
                .run_work(&work, &|| warmup.elapsed().as_secs() >= 40, &|| false)
                .unwrap();
        }
        let warmup_ms = milliseconds(warmup);
        let mut upload_ms = Vec::new();
        let mut upload_wall_ms = Vec::new();
        let mut coefficient_ms = Vec::new();
        let mut coefficient_wall_ms = Vec::new();
        for graph in prepared {
            kernel.stream.synchronize().unwrap();
            let start_event = ctx
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let end_event = ctx
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let start = Instant::now();
            start_event.record(&kernel.stream).unwrap();
            let next = ClusterDevice::open_with_kernel(graph, &kernel).unwrap();
            end_event.record(&kernel.stream).unwrap();
            let ms = f64::from(start_event.elapsed_ms(&end_event).unwrap());
            upload_wall_ms.push(milliseconds(start));
            upload_ms.push(ms);
            device = next;
            let start = Instant::now();
            coefficient_ms.push(coefficients(&mut device));
            coefficient_wall_ms.push(milliseconds(start));
        }
        let graph = ising_graph(instances.last().unwrap());
        let work = ClusterWork::new(&graph, &sample_params).unwrap();
        let mut kernels = Vec::new();
        let mut kernel_wall_ms = Vec::new();
        let mut jobs = Vec::new();
        let mut results = Vec::new();
        let mut totals = Vec::new();
        for _ in 0..std::env::var("CU4_RUNS").ok().map_or(KERNEL_RUNS, |v| v.parse().unwrap()) {
            let start = Instant::now();
            let fast = std::env::var("CU4_FAST").is_ok();
            let (ms, event_spins) = if fast { (0.0, Vec::new()) } else { kernel_ms(&device, &work) };
            kernels.push(ms);
            kernel_wall_ms.push(milliseconds(start));
            let expected: Vec<_> = event_spins
                .chunks_exact(device.graph.original_graph().num_nodes())
                .map(|spins| device.graph.unpermute(spins).unwrap())
                .collect();
            // An independent end-to-end job includes canonical expansion,
            // float conversion, CU-1, allocation/upload, schedule construction,
            // production launches, download, and the sampler's final rescore.
            let start = Instant::now();
            let nonce = instance_nonces[std::env::var("CU4_INST").ok().map_or(instance_nonces.len() - 1, |v| v.parse().unwrap())];
            let instance = generate(&params, &digest, &base, &bounds, nonce).unwrap();
            let graph = ising_graph(&instance);
            let prepared = prepare(&instance).unwrap();
            let work = ClusterWork::new(&graph, &sample_params).unwrap();
            let mut complete = ClusterDevice::open_with_kernel(prepared, &kernel).unwrap();
            let run_start = Instant::now();
            let run = complete
                .run_work_with_chunk_for_test(&work, std::env::var("CU4_CHUNK").ok().map_or(1, |c| c.parse().unwrap()))
                .unwrap();
            if std::env::var("CU4_PW").is_ok() {
                use crate::pairworm::{PairWormPolish, PwWork};
                let og = complete.graph.original_graph();
                let (tiles, pairs) = (og.tile_nodes(), og.pair_list());
                let hm: Vec<i64> = graph.h.iter().map(|c| (c * 1000.0).round() as i64).collect();
                let em: Vec<(u32, u32)> = graph.edges.iter().map(|&(u, v)| (u as u32, v as u32)).collect();
                let jm: Vec<i64> = graph.j.iter().map(|c| (c * 1000.0).round() as i64).collect();
                let before = run.energies.iter().copied().min().unwrap_or(0);
                let mut idx: Vec<usize> = (0..run.spins.len()).collect();
                idx.sort_by_key(|&i| run.energies[i]);
                let reads: Vec<Vec<i8>> = idx.iter().take(16).map(|&i| run.spins[i].clone()).collect();
                match PairWormPolish::new(&hm, &em, &jm, &tiles, &pairs) {
                    None => eprintln!("PW setup failed"),
                    Some(pol) => {
                        for (label, work) in [
                            ("budget", PwWork::Budget),
                            ("fixed16x300", PwWork::Fixed { reads: 16, anneal_cycles: 300 }),
                            ("fixed16x3000", PwWork::Fixed { reads: 16, anneal_cycles: 3000 }),
                        ] {
                            let t = Instant::now();
                            match pol.polish(&reads, 7, work) {
                                Some((_, e)) => eprintln!(
                                    "PW {label} before={:.3} after={:.3} gain={:.3} t={:.2}s",
                                    before as f64 * 0.001,
                                    e as f64 * 0.001,
                                    (e - before) as f64 * 0.001,
                                    t.elapsed().as_secs_f64()
                                ),
                                None => eprintln!("PW {label} failed"),
                            }
                        }
                    }
                }
            }
            if let Ok(path) = std::env::var("CU4_DUMP") {
                // text dump for offline prototypes: n, edges+j (milli), tiles, pairs, best 16 states (original order)
                let og = complete.graph.original_graph();
                let mut idx: Vec<usize> = (0..run.spins.len()).collect();
                idx.sort_by_key(|&i| run.energies[i]);
                let mut out = String::new();
                out.push_str(&format!("N {}\n", graph.h.len()));
                out.push_str(&format!("E {}\n", graph.edges.len()));
                for (k, &(u, v)) in graph.edges.iter().enumerate() {
                    out.push_str(&format!("{} {} {}\n", u, v, (graph.j[k] * 1000.0).round() as i64));
                }
                out.push_str("T\n");
                for t in og.tile_nodes() {
                    out.push_str(&format!("{} {} {} {}\n", t[0], t[1], t[2], t[3]));
                }
                out.push_str("P\n");
                for (a, b) in og.pair_list() {
                    out.push_str(&format!("{a} {b}\n"));
                }
                out.push_str("S\n");
                for &i in idx.iter().take(16) {
                    let st: String = run.spins[i].iter().map(|&x| if x > 0 { '1' } else { '0' }).collect();
                    out.push_str(&format!("{} {}\n", run.energies[i], st));
                }
                std::fs::write(&path, out).unwrap();
            }
            jobs.push(milliseconds(run_start));
            for spins in &run.spins {
                let energy = quip_solver_core::quip_protocol::scoring::energy_milli(
                    spins,
                    &graph.h,
                    &graph.j,
                    &graph.edges,
                );
                std::hint::black_box(energy);
            }
            totals.push(milliseconds(start));
            assert_eq!(run.spins.len(), sample_params.num_reads);
            let _ = &expected;
            results.push(run.energies);
        }
        let total = median(&totals);
        let saving = median(&generation) + median(&coefficient_ms);
        row["device"] = json!({
            "transfer_ms": median(&upload_ms), "upload_wall_ms": median(&upload_wall_ms),
            "coefficient_transfer_ms": median(&coefficient_ms),
            "coefficient_wall_ms": median(&coefficient_wall_ms),
            "kernel_ms": median(&kernels), "run_wall_ms": median(&jobs), "kernel_runs": KERNEL_RUNS,
            "total_ms": total,
            "frontend_percent": 100.0 * (median(&generation) + median(&preparation) + median(&upload_ms)) / total,
            "saving_upper_ms": saving, "saving_upper_percent": 100.0 * saving / total,
            "threshold_met": saving > 50.0 || saving / total > 0.1,
            "warmup_ms": warmup_ms, "transfer_samples_ms": upload_ms,
            "upload_wall_samples_ms": upload_wall_ms, "coefficient_samples_ms": coefficient_ms,
            "kernel_samples_ms": kernels, "kernel_wall_samples_ms": kernel_wall_ms,
            "run_wall_samples_ms": jobs, "energies_milli": results,
            "total_samples_ms": totals, "gpu_phase_ms": milliseconds(gpu_phase),
        });
    }
    row["gpu_end"] = json!(telemetry());
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)
        .unwrap();
    serde_json::to_writer(&mut file, &row).unwrap();
    writeln!(file).unwrap();
    println!("CU4 {row}");
}


/// Regenerate the instance of every calibration row from its nonce and write its structural features.
/// `CALIB_FAMILY` is /data/calib/family-<digest>.json, `CALIB_ROWS` the jsonl, `CALIB_OUT` the output jsonl.
#[test]
#[ignore = "offline calibration tool"]
fn calib_features() {
    let (Ok(fam), Ok(rows), Ok(out)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS"), std::env::var("CALIB_OUT")) else {
        return;
    };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    // The dump holds dense positions into `nodes`; BaseTopology::new takes native node ids.
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let bounds = admit_topology(&admitted, &base, base.order_digest(), &AdvertisedLimits { max_nodes: 0, max_edges: 0, max_degree: 0 }).unwrap();
    let lines: Vec<String> = std::fs::read_to_string(rows).unwrap().lines().map(str::to_owned).collect();
    let mut outf = std::fs::File::create(out).unwrap();
    for line in lines {
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let nonce_hex = v["n"].as_str().unwrap();
        let nb = hex(nonce_hex);
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&nb);
        let Ok(inst) = generate_instance(&admitted, &base, &bounds, nonce) else { continue };
        let f = crate::cluster::sampler::lease_run::features_raw(inst.h().len(), inst.edges(), inst.j());
        // Spectral proxy: the most negative eigenvalue of the weak-layer coupling matrix by shifted power iteration.
        let nn = inst.h().len();
        let weak: Vec<(usize, usize, f64)> = inst.edges().iter().zip(inst.j().iter()).filter(|(_, &j)| (j as f64).abs() <= 100.0).map(|(e, &j)| (e.0 as usize, e.1 as usize, j as f64)).collect();
        let mut rowsum = vec![0f64; nn];
        for &(a, b, j) in &weak {
            rowsum[a] += j.abs();
            rowsum[b] += j.abs();
        }
        let shift = rowsum.iter().copied().fold(0f64, f64::max);
        let mut x: Vec<f64> = (0..nn).map(|i| (((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f64 / 16_777_216.0) - 0.5).collect();
        let mut lam = [0f64; 3];
        for it in 1..=300usize {
            // y = (shift*I - J_w) x ; the dominant eigenvalue of that matrix is shift - lambda_min(J_w)
            let mut y: Vec<f64> = x.iter().map(|v| shift * v).collect();
            for &(a, b, j) in &weak {
                y[a] -= j * x[b];
                y[b] -= j * x[a];
            }
            let nx = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let rq = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / x.iter().map(|v| v * v).sum::<f64>();
            x = y.iter().map(|v| v / nx).collect();
            if it == 50 { lam[0] = shift - rq; }
            if it == 150 { lam[1] = shift - rq; }
            if it == 300 { lam[2] = shift - rq; }
        }
        let row = json!({"n": nonce_hex, "f": f, "nodes": inst.h().len(), "edges": inst.edges().len(), "st": v["st"], "deep": v["deep"], "ex": v["ex"], "s0": v["s0"], "lw": lam});
        writeln!(outf, "{row}").unwrap();
    }
}

/// Write real chain instances to text files for offline CPU solver studies (no GPU).
/// `CALIB_FAMILY`, `CALIB_ROWS` (one nonce hex per line), `DUMP_DIR`, `DUMP_N` (default 8).
#[test]
#[ignore = "offline dump tool"]
fn dump_instances() {
    let (Ok(fam), Ok(rows), Ok(dir)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS"), std::env::var("DUMP_DIR")) else {
        return;
    };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let bounds = admit_topology(&admitted, &base, base.order_digest(), &AdvertisedLimits { max_nodes: 0, max_edges: 0, max_degree: 0 }).unwrap();
    let n_dump: usize = std::env::var("DUMP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    std::fs::create_dir_all(&dir).unwrap();
    for (k, line) in std::fs::read_to_string(rows).unwrap().lines().take(n_dump).enumerate() {
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&hex(line.trim()));
        let Ok(inst) = generate_instance(&admitted, &base, &bounds, nonce) else { continue };
        let mut out = String::new();
        out.push_str(&format!("N {} E {} OFFSET_MILLI {}\n", inst.h().len(), inst.edges().len(), inst.offset_milli()));
        out.push_str(&format!("H {}\n", inst.h().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")));
        for (e, j) in inst.edges().iter().zip(inst.j().iter()) {
            out.push_str(&format!("{} {} {}\n", e.0, e.1, j));
        }
        std::fs::write(format!("{dir}/inst_{k}.txt"), out).unwrap();
    }
}

/// Time the pieces of per-salt CPU work: generate_instance, the S0 features, the device-graph build.
#[test]
#[ignore = "offline timing tool"]
fn gen_timing() {
    let (Ok(fam), Ok(rows)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS")) else {
        return;
    };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let bounds = admit_topology(&admitted, &base, base.order_digest(), &AdvertisedLimits { max_nodes: 0, max_edges: 0, max_degree: 0 }).unwrap();
    let lines: Vec<String> = std::fs::read_to_string(rows).unwrap().lines().take(300).map(str::to_owned).collect();
    let (mut t_gen, mut t_feat, mut t_graph, mut n) = (0f64, 0f64, 0f64, 0usize);
    for line in lines {
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let nb = hex(v["n"].as_str().unwrap());
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&nb);
        let t0 = std::time::Instant::now();
        let Ok(inst) = generate_instance(&admitted, &base, &bounds, nonce) else { continue };
        t_gen += t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let f = crate::cluster::sampler::lease_run::features_raw(inst.h().len(), inst.edges(), inst.j());
        t_feat += t1.elapsed().as_secs_f64();
        std::hint::black_box(&f);
        let t2 = std::time::Instant::now();
        let g = crate::cluster::sampler::lease_run::ising_graph_for_timing(&inst);
        t_graph += t2.elapsed().as_secs_f64();
        std::hint::black_box(&g);
        n += 1;
    }
    eprintln!("GEN_TIMING n={n} generate_instance {:.2} ms, features {:.2} ms, ising_graph {:.2} ms", t_gen / n as f64 * 1e3, t_feat / n as f64 * 1e3, t_graph / n as f64 * 1e3);
}

/// The lean generator must give the canonical `j`, edge for edge, and be much faster.
#[test]
#[ignore = "offline check"]
fn lean_matches_canonical() {
    let (Ok(fam), Ok(rows)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS")) else {
        return;
    };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let bounds = admit_topology(&admitted, &base, base.order_digest(), &AdvertisedLimits { max_nodes: 0, max_edges: 0, max_degree: 0 }).unwrap();
    // Parameters are tag(u16) len(u32) value triples; tags 1 and 2 are the Zephyr shape, 3 the weak magnitude, 4 the pair count.
    let mut tlv = std::collections::BTreeMap::new();
    let mut at = 1usize; // the first byte is the parameter-set version
    while at + 6 <= params.len() {
        let tag = u16::from_le_bytes([params[at], params[at + 1]]);
        let len = u32::from_le_bytes([params[at + 2], params[at + 3], params[at + 4], params[at + 5]]) as usize;
        let mut value = 0u64;
        for (k, b) in params[at + 6..at + 6 + len].iter().enumerate() {
            value |= u64::from(*b) << (8 * k);
        }
        tlv.insert(tag, value);
        at += 6 + len;
    }
    eprintln!("PARAMS {tlv:?}");
    let lean = crate::cluster::lean_gen::LeanGen::new(&base, admitted.descriptor().digest(), tlv[&1] as u16, tlv[&2] as u8, tlv[&3] as i32, tlv[&4] as usize).expect("lean init");
    let tri = crate::cluster::lean_gen::Triangles::new(base.node_ids().len(), &lean.edges);
    eprintln!("TRIANGLES {}", tri.len());
    let (mut ok, mut bad, mut t_lean, mut t_can, mut t_tri, mut t_old, mut feat_bad) = (0usize, 0usize, 0f64, 0f64, 0f64, 0f64, 0usize);
    for line in std::fs::read_to_string(rows).unwrap().lines().take(300) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&hex(v["n"].as_str().unwrap()));
        let t0 = Instant::now();
        let j = lean.couplings(&nonce);
        t_lean += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let inst = generate_instance(&admitted, &base, &bounds, nonce);
        t_can += t1.elapsed().as_secs_f64();
        if bad == 0 && ok == 0 {
            match (&j, &inst) {
                (Some(j), Ok(inst)) => {
                    let diff: Vec<usize> = (0..j.len()).filter(|&k| j[k] != inst.j()[k]).collect();
                    eprintln!("DBG edges_equal={} len {} {} ndiff={}", lean.edges == inst.edges(), j.len(), inst.j().len(), diff.len());
                    let mut kinds = std::collections::BTreeMap::new();
                    for &k in diff.iter().take(4000) {
                        *kinds.entry((j[k], inst.j()[k])).or_insert(0usize) += 1;
                    }
                    eprintln!("DBG diff kinds (lean, canonical): {kinds:?}");
                    let count = |v: &[i32], x: i32| v.iter().filter(|&&a| a == x).count();
                    eprintln!("DBG lean  +1000 {} -1000 {} +62 {} -62 {}", count(j, 1000), count(j, -1000), count(j, 62), count(j, -62));
                    eprintln!("DBG canon +1000 {} -1000 {} +62 {} -62 {}", count(inst.j(), 1000), count(inst.j(), -1000), count(inst.j(), 62), count(inst.j(), -62));
                }
                _ => eprintln!("DBG one side failed: lean_some={} canon_ok={}", j.is_some(), inst.is_ok()),
            }
        }
        if let (Some(j), Ok(inst)) = (&j, &inst) {
            let t2 = Instant::now();
            let f_tri = tri.features(j);
            t_tri += t2.elapsed().as_secs_f64();
            let t3 = Instant::now();
            let f_old = crate::cluster::sampler::lease_run::features_raw(inst.h().len(), inst.edges(), inst.j());
            t_old += t3.elapsed().as_secs_f64();
            if f_tri != f_old {
                feat_bad += 1;
            }
        }
        match (j, inst) {
            (Some(j), Ok(inst)) if j == inst.j() && lean.edges == inst.edges() => ok += 1,
            _ => bad += 1,
        }
    }
    eprintln!("FEAT feat_mismatch={feat_bad} triangle-table {:.2} ms, old {:.2} ms", t_tri / 300.0 * 1e3, t_old / 300.0 * 1e3);
    eprintln!("LEAN_CHECK identical={ok} different={bad} lean {:.2} ms canonical {:.2} ms", t_lean / 300.0 * 1e3, t_can / 300.0 * 1e3);
    assert_eq!(bad, 0);
}

/// How much bank-conflict cost the colour-class reorder removes (the topology is fixed, so one order serves every instance).
#[test]
#[ignore = "offline layout study"]
fn layout_study() {
    let Ok(fam) = std::env::var("CALIB_FAMILY") else { return };
    let moves: u64 = std::env::var("LAYOUT_MOVES").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let mut sorted: Vec<(u32, u32)> = base.edges().iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
    sorted.sort_unstable();
    let n = base.node_ids().len();
    let e: Vec<(usize, usize)> = sorted.iter().map(|&(u, v)| (u as usize, v as usize)).collect();
    let color = crate::cluster::prepare::color_classes(n, &e);
    let prob = crate::cluster::layout_opt::Problem::new(n, &e, &color);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by_key(|&u| color[u]);
    let (c0, ideal) = prob.total(&perm);
    eprintln!("LAYOUT colors {} nodes {} cost {} ideal {} ratio {:.3}", color.iter().max().unwrap() + 1, n, c0, ideal, c0 as f64 / ideal as f64);
    let t = Instant::now();
    let opt = prob.optimize(perm, moves, 12345);
    let (c1, _) = prob.total(&opt);
    eprintln!("LAYOUT after {moves} moves: cost {} ratio {:.3} ({:.1} s)", c1, c1 as f64 / ideal as f64, t.elapsed().as_secs_f64());
}

/// Sizes of the fixed structure that the GPU generator needs.
#[test]
#[ignore = "offline structure dump"]
fn gen_structure_sizes() {
    let Ok(fam) = std::env::var("CALIB_FAMILY") else { return };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let lattice = quip_solver_core::ising_generators::ice::Lattice::Zephyr { m: 12, t: 4 };
    let cands = quip_solver_core::ising_generators::ice::candidates(lattice, &base).unwrap();
    let sides = quip_solver_core::ising_generators::ice::sides(lattice, &base, &cands).unwrap();
    let free = cands.spines.iter().filter(|s| s.is_none()).count();
    let spine = cands.spines.len() - free;
    let mut adj = vec![0usize; sides.left.len()];
    for sp in cands.spines.iter().flatten() {
        let [e, f] = sp;
        let l = sides.left.binary_search(e).or_else(|_| sides.left.binary_search(f)).unwrap();
        adj[l] += 1;
    }
    eprintln!("STRUCT candidates {} free {} spine_tiles {} left {} right {} max_adj_left {} total_edges {} nodes {}", cands.tiles.len(), free, spine, sides.left.len(), sides.right.len(), adj.iter().max().unwrap(), base.edges().len(), base.node_ids().len());
}

/// The GPU generator must give the same S0 score as the canonical instance (GPU test: stop the miner first).
#[test]
#[ignore = "GPU test"]
fn gpu_gen_matches_cpu() {
    let (Ok(fam), Ok(rows)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS")) else { return };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let lean = crate::cluster::lean_gen::LeanGen::new(&base, admitted.descriptor().digest(), 12, 4, 45, 1350).expect("lean init");
    let tri = crate::cluster::lean_gen::Triangles::new(base.node_ids().len(), &lean.edges);
    let gt = lean.gpu_tables().expect("tables");
    eprintln!("GPUGEN tables: left {} right {} cands {} edges {} sel_expected {}", gt.left, gt.right, gt.cands, gt.edges, gt.sel_expected);
    let mut gpu = crate::cluster::gen_gpu::GenGpu::new(0, gt, &tri).expect("gpu init");
    let n: usize = std::env::var("GEN_N").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let mut nonces = Vec::new();
    for line in std::fs::read_to_string(rows).unwrap().lines().cycle().take(n) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&hex(v["n"].as_str().unwrap()));
        nonces.push(nonce);
    }
    let keys: Vec<u32> = nonces.iter().flat_map(|nn| lean.key_words(nn)).collect();
    let t0 = Instant::now();
    let out = gpu.score(&keys).expect("gpu score");
    let dt = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let out = if n >= 4096 { out } else { gpu.score(&keys).expect("gpu score") };
    let dt2 = t1.elapsed().as_secs_f64();
    let (mut ok, mut bad, mut worst) = (0usize, 0usize, 0f64);
    for (nn, (sc, st)) in nonces.iter().zip(&out).take(256) {
        let j = lean.couplings(nn).unwrap();
        let f = tri.features(&j);
        let w = &crate::cluster::sampler::lease_run::W;
        let mu = &crate::cluster::sampler::lease_run::MU;
        let sd = &crate::cluster::sampler::lease_run::SD;
        let cpu: f64 = (0..9).map(|i| w[i] * (f[i] - mu[i]) / sd[i]).sum();
        if *st == 0 && (cpu - sc).abs() < 1e-6 {
            ok += 1;
        } else {
            bad += 1;
            if bad <= 3 {
                eprintln!("GPUGEN mismatch: gpu {sc} status {st} cpu {cpu}");
            }
        }
        worst = worst.max((cpu - sc).abs());
    }
    let cyc = gpu.phase_cycles();
    eprintln!("GPUGEN phase cycles per instance (x1000): {:?}", cyc.iter().map(|c| c / (2 * n as u64) / 1000).collect::<Vec<_>>());
    eprintln!("GPUGEN n={n} identical={ok} different={bad} worst_abs_diff={worst:.3e} first {dt:.3} s second {dt2:.3} s => {:.0} nonce/s", n as f64 / dt2);
    assert_eq!(bad, 0);
}

/// The allocation-free path must give the canonical features, and scale across threads.
#[test]
#[ignore = "offline check"]
fn fast_flags_matches_and_scales() {
    let (Ok(fam), Ok(rows)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS")) else { return };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hex = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hex(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let lean = crate::cluster::lean_gen::LeanGen::new(&base, admitted.descriptor().digest(), 12, 4, 45, 1350).expect("lean init");
    let tri = crate::cluster::lean_gen::Triangles::new(base.node_ids().len(), &lean.edges);
    let nonces: Vec<[u8; 32]> = std::fs::read_to_string(rows).unwrap().lines().take(300).map(|l| {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        let mut n = [0u8; 32];
        n.copy_from_slice(&hex(v["n"].as_str().unwrap()));
        n
    }).collect();
    let mut sc = crate::cluster::lean_gen::FastScratch::new(lean.gpu_tables().unwrap());
    let (mut ok, mut bad) = (0, 0);
    for n in &nonces {
        let old = tri.features(&lean.couplings(n).unwrap());
        let good = lean.fast_flags(n, &mut sc);
        if good && tri.features_flags(&sc.flags) == old { ok += 1 } else { bad += 1 }
    }
    eprintln!("FAST identical={ok} different={bad}");
    for threads in [1usize, 8, 24] {
        let t = Instant::now();
        let per = 150usize;
        std::thread::scope(|s| {
            for k in 0..threads {
                let (lean, tri, nonces) = (&lean, &tri, &nonces);
                s.spawn(move || {
                    let mut sc = crate::cluster::lean_gen::FastScratch::new(lean.gpu_tables().unwrap());
                    for i in 0..per {
                        let n = &nonces[(k * 7 + i) % nonces.len()];
                        if lean.fast_flags(n, &mut sc) { std::hint::black_box(tri.features_flags(&sc.flags)); }
                    }
                });
            }
        });
        let dt = t.elapsed().as_secs_f64();
        eprintln!("FAST threads {threads}: {:.0} salt/s total ({:.2} ms per salt per thread)", (threads * per) as f64 / dt, dt / per as f64 * 1e3);
    }
    let t = Instant::now();
    let per = 150usize;
    std::thread::scope(|s| {
        for k in 0..24 {
            let (lean, tri, nonces) = (&lean, &tri, &nonces);
            s.spawn(move || {
                for i in 0..per {
                    let n = &nonces[(k * 7 + i) % nonces.len()];
                    if let Some(j) = lean.couplings(n) { std::hint::black_box(tri.features(&j)); }
                }
            });
        }
    });
    let dt = t.elapsed().as_secs_f64();
    eprintln!("OLD(lean, allocating) threads 24: {:.0} salt/s total", (24 * per) as f64 / dt);
}

/// Throughput of the in-tree MSA streaming driver on live chain 120 instances: many different models in flight at once.
#[test]
#[ignore = "GPU test"]
fn msa_stream_throughput() {
    use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
    let (Ok(fam), Ok(rows)) = (std::env::var("CALIB_FAMILY"), std::env::var("CALIB_ROWS")) else { return };
    let fam: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fam).unwrap()).unwrap();
    let hexb = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() };
    let params = hexb(fam["params_hex"].as_str().unwrap());
    let nodes: Vec<u32> = fam["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let edges: Vec<(u32, u32)> = fam["edges"].as_array().unwrap().iter().map(|e| (nodes[e[0].as_u64().unwrap() as usize], nodes[e[1].as_u64().unwrap() as usize])).collect();
    let base = BaseTopology::new(nodes.clone(), &edges).unwrap();
    let digest = params_digest(&params);
    let admitted = admit(&[TWISTED_DIMER_V1], TWISTED_DIMER_V1.family_id.as_bytes(), 1, &params, &digest).unwrap();
    let bounds = admit_topology(&admitted, &base, base.order_digest(), &AdvertisedLimits { max_nodes: 0, max_edges: 0, max_degree: 0 }).unwrap();
    let n_jobs: usize = std::env::var("MSA_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let mut graphs = Vec::new();
    for line in std::fs::read_to_string(rows).unwrap().lines().take(n_jobs) {
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&hexb(line.trim()));
        if let Ok(inst) = generate_instance(&admitted, &base, &bounds, nonce) {
            graphs.push(crate::cluster::sampler::lease_run::ising_graph_for_timing(&inst));
        }
    }
    eprintln!("MSASTREAM instances ready: {}", graphs.len());
    let device = crate::cuda_device::CudaDevice::open_with_nodes(0, crate::KernelKind::Msa, 4600).unwrap();
    eprintln!("MSASTREAM stream width (nonces in flight): {}", crate::streaming::stream_width(&device, crate::KernelKind::Msa));
    let cfgs: Vec<(usize, usize)> = std::env::var("MSA_CFG").ok().map_or_else(
        || vec![(128usize, 1024usize), (128, 4096)],
        |v| v.split(',').filter_map(|c| c.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).collect(),
    );
    // MSA_BETA="lo,hi;lo,hi;default": the annealing range for every configuration (None = the sampler default).
    let betas: Vec<Option<(f64, f64)>> = std::env::var("MSA_BETA").ok().map_or_else(
        || vec![None],
        |v| v.split(';').map(|c| c.split_once(',').and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?)))).collect(),
    );
    // MSA_SEEDS="11,1011,...": repeat every configuration with these base seeds (repeatability of the MSA judge).
    let seeds: Vec<u64> = std::env::var("MSA_SEEDS").ok().map_or_else(|| vec![11], |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    for &seed0 in &seeds {
    for &beta in &betas {
    for &(reads, sweeps) in &cfgs {
        let gov = crate::nvml_gov::UtilGovernor::start(&device.pci_bus_id, 100, false);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(64);
        let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(64);
        let t0 = Instant::now();
        let mut energies = vec![f64::NAN; graphs.len()];
        std::thread::scope(|sc| {
            let dev = &device;
            let g = &gov;
            sc.spawn(move || crate::streaming::run_stream(dev, crate::KernelKind::Msa, job_rx, res_tx, CancelToken::default(), g, crate::streaming::StallConfig::default()));
            let graphs_ref = &graphs;
            sc.spawn(move || {
                for (i, gr) in graphs_ref.iter().enumerate() {
                    let p = SampleParams { num_reads: reads, num_sweeps: sweeps, sweeps_per_beta: 1, seed: seed0 + i as u64, beta_range: beta, ..SampleParams::default() };
                    job_tx.blocking_send(StreamJob { job_id: (i as u32).to_le_bytes().to_vec(), graph: gr.clone(), params: p, watermark: None }).unwrap();
                }
            });
            let mut got = 0usize;
            while got < graphs.len() {
                let Some(r) = res_rx.blocking_recv() else { break };
                let i = u32::from_le_bytes(r.job_id[..4].try_into().unwrap()) as usize;
                if let StreamOutcome::Completed(Ok(rs)) = r.outcome {
                    energies[i] = rs.iter().map(|x| x.energy_milli).min().unwrap_or(0) as f64 * 0.001;
                }
                got += 1;
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        let ok = energies.iter().filter(|e| e.is_finite()).count();
        let mean = energies.iter().filter(|e| e.is_finite()).sum::<f64>() / ok.max(1) as f64;
        eprintln!("MSASTREAM beta={beta:?} {reads}x{sweeps}: {ok}/{} jobs in {dt:.1} s => {:.1} nonce/s ; nang luong tb {mean:.2}", graphs.len(), ok as f64 / dt);
        let line: String = energies.iter().map(|e| format!("{e:.3}")).collect::<Vec<_>>().join(",");
        eprintln!("MSAENERGIES seed={seed0} beta={beta:?} {reads}x{sweeps} {line}");
    }
    }
    }
}
