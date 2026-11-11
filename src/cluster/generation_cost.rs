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

const HOST_RUNS: usize = 20;
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
    let out = Command::new(program).args(args).output().unwrap();
    assert!(out.status.success(), "{program} failed");
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
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
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok("1"));
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
    assert!(parent.starts_with(root.canonicalize().unwrap()));
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
    let base = topology(m, t);
    let params = parameters(m, t, weak, pairs);
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
    if budget != "api-default" {
        let target = match budget.as_str() {
            "mining-easy" => i64::MAX,
            "mining-hard" => i64::MIN,
            _ => panic!("CU4_BUDGET must be api-default, mining-easy, or mining-hard"),
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
    for index in 0..1000 {
        let nonce = nonces.derive(lease.salt(index).unwrap());
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
        if instances.len() == HOST_RUNS {
            break;
        }
    }
    assert_eq!(instances.len(), HOST_RUNS, "not enough valid jobs");
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
    let mut row = json!({
        "date_utc": command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "commit": command("git", &["rev-parse", "HEAD"]),
        "source_diff": command("git", &["diff", "--stat"]),
        "gpu_start": start_telemetry, "family": TWISTED_DIMER_V1.family_id, "revision": 1,
        "shape": shape, "nodes": base.node_ids().len(), "edges": base.edges().len(),
        "budget": budget,
        "reads": sample_params.num_reads, "sweeps": sample_params.num_sweeps, "sweeps_per_beta": sample_params.sweeps_per_beta,
        "host_runs": HOST_RUNS, "generation_ms": median(&generation), "preparation_ms": median(&preparation),
        "generation_samples_ms": generation, "preparation_samples_ms": preparation,
        "skipped": skipped, "preparation_rejection": rejection,
        "shared_budget_bytes": limits.shared_bytes_per_block_optin,
    });
    if rejection.is_none() {
        let gpu_phase = Instant::now();
        assert_eq!(prepared.len(), HOST_RUNS);
        // Compilation, allocation initialization, and the first solve are warm.
        let mut device =
            ClusterDevice::open_with_kernel(prepare(&instances[0]).unwrap(), &kernel).unwrap();
        let instance = &instances[0];
        let graph = ising_graph(instance);
        let work = ClusterWork::new(&graph, &sample_params).unwrap();
        let warmup = Instant::now();
        device
            .run_work(&work, &|| warmup.elapsed().as_secs() >= 40, &|| false)
            .unwrap();
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
        for _ in 0..KERNEL_RUNS {
            let start = Instant::now();
            let (ms, event_spins) = kernel_ms(&device, &work);
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
            let nonce = *instance_nonces.last().unwrap();
            let instance = generate(&params, &digest, &base, &bounds, nonce).unwrap();
            let graph = ising_graph(&instance);
            let prepared = prepare(&instance).unwrap();
            let work = ClusterWork::new(&graph, &sample_params).unwrap();
            let mut complete = ClusterDevice::open_with_kernel(prepared, &kernel).unwrap();
            let run_start = Instant::now();
            let run = complete
                .run_work(&work, &|| start.elapsed().as_secs() >= 40, &|| false)
                .unwrap();
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
            assert_eq!(
                run.spins, expected,
                "event instrumentation changed the solve"
            );
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
