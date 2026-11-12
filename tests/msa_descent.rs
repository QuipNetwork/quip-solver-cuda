//! The `msa-nat-loop/1` descent of `quip-cuda-msa-descent`: parity with the
//! research descent on regenerated `quip/charged-ice@1` instances, tile
//! recovery, family admission, and the exact rescore of the wrapper.
//!
//! The fixture holds no model or state data. Each test regenerates its
//! instance from the nonce, checks the topology and instance digests, and
//! compares hashes of the results with the research results that
//! `tools/msa_descent_reference.py` recorded.

#[path = "support/charged_ice.rs"]
#[allow(
    dead_code,
    reason = "the support module serves the reference tool too; this file uses a subset"
)]
mod charged_ice;
#[path = "support/descent_session.rs"]
mod descent_session;

use std::sync::Mutex;

use charged_ice::{Fixture, Record};
use quip_protocol::scoring::energy_milli;
use quip_solver_core::beta::default_ising_beta_range;
use quip_solver_core::families::Declaration;
use quip_solver_core::ising_generators::descriptor::params_digest;
use quip_solver_core::ising_generators::ice::{CHARGED_ICE_V1, ICE_CHARGE_RECIPE};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamOutcome,
};
use quip_solver_cuda::descent::loop_descent::TileCounts;
use quip_solver_cuda::descent::tiles::recover_tiles;
use quip_solver_cuda::descent::{
    descend, family_tiles, job_ising, MsaDescentSampler, Plan, TileSource, MSA_DESCENT_SOLVER_ID,
};

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(charged_ice::FIXTURE).expect("the parity fixture");
    serde_json::from_str(&text).expect("fixture JSON")
}

/// Regenerate the instance of a record and check both digests first.
fn regenerate(record: &Record) -> Instance {
    let instance = charged_ice::instance(charged_ice::nonce(&record.nonce));
    assert_eq!(
        charged_ice::hex(instance.topology_order_digest()),
        record.topology_hash,
        "{}: topology order digest",
        record.nonce
    );
    assert_eq!(
        charged_ice::hex(&instance.digest()),
        record.integrity_hash,
        "{}: instance digest",
        record.nonce
    );
    instance
}

fn start_reads(record: &Record, instance: &Instance) -> Vec<Vec<i8>> {
    charged_ice::reads(instance.h().len(), record.read_seed, record.reads)
}

#[test]
fn the_fixture_covers_two_nonces_of_64_reads() {
    let fixture = fixture();
    let nonces: Vec<&str> = fixture.records.iter().map(|r| r.nonce.as_str()).collect();
    assert_eq!(nonces, charged_ice::NONCES);
    assert!(fixture
        .records
        .iter()
        .all(|r| r.reads == charged_ice::READS));
}

#[test]
fn greedy_matches_the_research_descent_on_every_read() {
    for record in fixture().records {
        let instance = regenerate(&record);
        let graph = charged_ice::graph(&instance);
        let plan = Plan::new(&graph, Some(charged_ice::metadata_tiles(&instance))).unwrap();
        let reads = start_reads(&record, &instance);
        let raw_best = reads.iter().map(|s| plan.ising().energy(s)).min();
        assert_eq!(raw_best, Some(record.raw_best), "{}", record.nonce);
        let descended: Vec<Vec<i8>> = reads
            .into_iter()
            .map(|mut s| {
                plan.greedy().descend(&mut s, None).unwrap();
                s
            })
            .collect();
        let energies: Vec<i64> = descended.iter().map(|s| plan.ising().energy(s)).collect();
        assert_eq!(
            charged_ice::energies_hash(&energies),
            record.greedy_energies_blake3,
            "{}: greedy energies",
            record.nonce
        );
        assert_eq!(
            charged_ice::spins_hash(&descended),
            record.greedy_spins_blake3,
            "{}: greedy states",
            record.nonce
        );
        assert_eq!(energies.iter().copied().min(), Some(record.greedy_best));
    }
}

#[test]
fn the_policy_matches_the_research_greedy_and_loop() {
    for record in fixture().records {
        let instance = regenerate(&record);
        let graph = charged_ice::graph(&instance);
        let plan = Plan::new(&graph, Some(charged_ice::metadata_tiles(&instance))).unwrap();
        assert_eq!(plan.source(), TileSource::Metadata);
        let reads = start_reads(&record, &instance);
        // Two threads: the result must not depend on the split.
        let out = descend(&plan, &reads, record.loop_seed, 2, None).unwrap();
        assert_eq!(
            charged_ice::energies_hash(&out.greedy_energies),
            record.greedy_energies_blake3,
            "{}: greedy energies",
            record.nonce
        );
        let loop_state = out.loop_state.expect("the instance has tiles");
        assert_eq!(out.report.loop_from, Some(record.loop_from));
        assert_eq!(out.report.loop_best, Some(record.loop_energy));
        assert_eq!(
            charged_ice::spins_hash(std::slice::from_ref(&loop_state)),
            record.loop_spins_blake3,
            "{}: loop state",
            record.nonce
        );
        let best = record.loop_energy.min(record.greedy_best);
        assert_eq!(out.report.final_best, best);
        assert_eq!(out.report.raw_best, record.raw_best);
        let returned: Vec<i64> = out.reads.iter().map(|s| plan.ising().energy(s)).collect();
        assert_eq!(returned.iter().copied().min(), Some(best));
        assert_eq!(out.reads.len(), record.reads);
    }
}

#[test]
fn tiles_recovered_from_the_couplings_equal_the_generator_tiles() {
    for record in fixture().records {
        let instance = regenerate(&record);
        let graph = charged_ice::graph(&instance);
        let metadata = charged_ice::metadata_tiles(&instance);
        assert_eq!(metadata.len(), record.tiles);
        let recovered = recover_tiles(&job_ising(&graph).unwrap());
        assert_eq!(recovered, metadata, "{}", record.nonce);
        let plan = Plan::new(&graph, None).unwrap();
        assert_eq!(plan.source(), TileSource::Recovered);
        assert_eq!(plan.tiles().tiles(), metadata.as_slice());
        assert_eq!(
            plan.tile_counts(),
            TileCounts {
                tiles: 2116,
                dropped: 0,
                spins_over_two: 0
            }
        );
        assert_eq!(plan.tiles().tiled().len(), 4416);
        assert_eq!(plan.moves().tile_subsets, 10_580);
    }
}

#[test]
fn the_declaration_accepts_only_the_ice_charge_recipe() {
    let table = MsaDescentSampler::<FixedReads>::declaration()
        .validate()
        .expect("a valid declaration");
    assert!(table.generated().is_empty());
    let accepted = table.accepted();
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].family_id, CHARGED_ICE_V1.family_id);
    assert_eq!(accepted[0].revisions, vec![CHARGED_ICE_V1.revision]);
    assert_eq!(
        accepted[0].params_digests,
        vec![params_digest(ICE_CHARGE_RECIPE.params).to_vec()]
    );
    let declaration: Declaration = MsaDescentSampler::<FixedReads>::declaration();
    assert_eq!(declaration.solver_id, MSA_DESCENT_SOLVER_ID);
}

#[test]
fn a_recipe_job_is_admitted_with_its_tiles_and_other_params_never_reach_the_sampler() {
    let record = &fixture().records[0];
    let instance = regenerate(record);
    // The same instance under other valid parameters: 64 charged tiles per
    // 256 instead of 128.
    let mut other = ICE_CHARGE_RECIPE.params.to_vec();
    let last = other.len() - 2;
    assert_eq!(other[last], 0x80);
    other[last] = 0x40;
    let contexts = descent_session::contexts(&[
        (ICE_CHARGE_RECIPE.params.to_vec(), instance.clone()),
        (other, instance.clone()),
    ]);
    assert_eq!(contexts.len(), 1, "only the recipe job reaches the sampler");
    let context = &contexts[0];
    assert_eq!(context.params(), ICE_CHARGE_RECIPE.params);
    let graph = charged_ice::graph(&instance);
    let tiles = family_tiles(&graph, Some(context))
        .unwrap()
        .expect("the recipe job uses its metadata tiles");
    assert_eq!(tiles, recover_tiles(&job_ising(&graph).unwrap()));
    // A job graph that differs from the instance is declined.
    let mut changed = graph.clone();
    changed.h[0] += 0.001;
    assert!(matches!(
        family_tiles(&changed, Some(context)),
        Err(SampleError::Declined(_))
    ));
    assert_eq!(family_tiles(&graph, None).unwrap(), None);
}

#[test]
fn the_default_beta_range_of_a_regenerated_instance() {
    let record = &fixture().records[0];
    let graph = charged_ice::graph(&regenerate(record));
    let (hot, cold) = default_ising_beta_range(&graph);
    let n = graph.h.len();
    let mut field_sum = graph.h.iter().map(|h| h.abs()).collect::<Vec<f64>>();
    let mut min_coupling = vec![f64::INFINITY; n];
    for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
        field_sum[u] += j.abs();
        field_sum[v] += j.abs();
        min_coupling[u] = min_coupling[u].min(j.abs());
        min_coupling[v] = min_coupling[v].min(j.abs());
    }
    let max_field = field_sum.iter().copied().fold(0.0, f64::max);
    let weakest = min_coupling.iter().copied().fold(f64::INFINITY, f64::min);
    // The archived models (4,575 spins, largest field sum 2.934, smallest
    // coupling 0.031) give (0.11812324140421693, 210.21826070480182). A
    // regenerated instance has 4,800 spins, so only the hot end can match.
    assert_eq!(n, 4800);
    assert!(
        (weakest - 0.031).abs() < 1e-12,
        "smallest coupling {weakest}"
    );
    let expected_hot = std::f64::consts::LN_2 / (2.0 * max_field);
    assert!(
        (hot - expected_hot).abs() < 1e-12,
        "hot {hot}, field sum {max_field}"
    );
    if (max_field - 2.934).abs() < 1e-12 {
        assert!((hot - 0.118_123_241_404_216_93).abs() < 1e-15);
    }
    let expected_cold = (4800.0_f64 / 0.01).ln() / (2.0 * 0.031);
    assert!((cold - expected_cold).abs() < 1e-9, "cold {cold}");
    eprintln!("default beta range ({hot}, {cold}), largest field sum {max_field}");
}

/// A kernel stand-in that returns fixed reads, scored on the job graph.
struct FixedReads(Mutex<Vec<Vec<i8>>>);

impl Sampler for FixedReads {
    fn sample(
        &self,
        graph: &IsingGraph,
        _: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .iter()
            .map(|s| SamplerResult {
                spins: s.clone(),
                energy_milli: energy_milli(s, &graph.h, &graph.j, &graph.edges),
            })
            .collect())
    }
}

fn check_rescored(graph: &IsingGraph, reads: &[SamplerResult], raw: &[Vec<i8>]) {
    assert_eq!(reads.len(), raw.len());
    for read in reads {
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    let raw_best = raw
        .iter()
        .map(|s| energy_milli(s, &graph.h, &graph.j, &graph.edges))
        .min();
    let best = reads.iter().map(|r| r.energy_milli).min();
    assert!(best <= raw_best, "descended best above the raw best");
}

#[test]
fn the_wrapper_returns_exact_energies_never_above_the_raw_best() {
    let record = &fixture().records[1];
    let instance = regenerate(record);
    let graph = charged_ice::graph(&instance);
    let raw = start_reads(record, &instance)[..8].to_vec();
    let sampler = MsaDescentSampler::new(FixedReads(Mutex::new(raw.clone())));
    let params = SampleParams {
        num_reads: raw.len(),
        seed: record.loop_seed,
        ..SampleParams::default()
    };
    let direct = sampler.sample(&graph, &params).unwrap();
    check_rescored(&graph, &direct, &raw);
    let plan = Plan::new(&graph, None).unwrap();
    let expected = descend(&plan, &raw, record.loop_seed, 1, None).unwrap();
    let spins: Vec<Vec<i8>> = direct.iter().map(|r| r.spins.clone()).collect();
    assert_eq!(spins, expected.reads);

    // The stream path gives the same reads.
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(2);
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(2);
    job_tx
        .blocking_send(StreamJob {
            job_id: vec![7],
            graph,
            params,
            watermark: None,
        })
        .unwrap();
    drop(job_tx);
    sampler.sample_stream(job_rx, out_tx, CancelToken::default());
    let result = out_rx.blocking_recv().expect("one result");
    assert_eq!(result.job_id, vec![7]);
    let StreamOutcome::Completed(Ok(streamed)) = result.outcome else {
        panic!("the stream job completes");
    };
    assert_eq!(streamed, direct);
    assert!(out_rx.blocking_recv().is_none());
}

#[test]
fn a_cancelled_stream_job_is_reported_cancelled() {
    let record = &fixture().records[0];
    let instance = regenerate(record);
    let graph = charged_ice::graph(&instance);
    let raw = start_reads(record, &instance)[..2].to_vec();
    let sampler = MsaDescentSampler::new(FixedReads(Mutex::new(raw)));
    let cancel = CancelToken::default();
    cancel.cancel_through(5);
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(1);
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(1);
    job_tx
        .blocking_send(StreamJob {
            job_id: vec![1],
            graph,
            params: SampleParams::default(),
            watermark: Some(3),
        })
        .unwrap();
    drop(job_tx);
    sampler.sample_stream(job_rx, out_tx, cancel);
    let result = out_rx.blocking_recv().expect("one result");
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
}

/// One regenerated instance, 64 reads, 2,048 sweeps through the wrapper
/// stream on the MSA kernel.
#[test]
#[ignore = "requires a CUDA GPU"]
fn the_msa_stream_descends_a_regenerated_instance() {
    use quip_solver_cuda::capacity::{max_reads, msa_max_nodes, MSA_DEFAULT_NODES};
    use quip_solver_cuda::cuda_device::CudaDevice;
    use quip_solver_cuda::nvml_gov::UtilGovernor;
    use quip_solver_cuda::streaming::StallConfig;
    use quip_solver_cuda::{KernelKind, MsaSampler};

    let record = &fixture().records[0];
    let instance = regenerate(record);
    let graph = charged_ice::graph(&instance);
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES).unwrap();
    let reads = max_reads(KernelKind::Msa, &device.limits);
    device.max_nodes = msa_max_nodes(&device.limits, reads);
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    let sampler = MsaDescentSampler::new(MsaSampler::new(device, gov, StallConfig::default()));
    let params = SampleParams {
        num_reads: 64,
        num_sweeps: 2048,
        seed: record.loop_seed,
        ..SampleParams::default()
    };
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(2);
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(2);
    job_tx
        .blocking_send(StreamJob {
            job_id: vec![1],
            graph: graph.clone(),
            params,
            watermark: None,
        })
        .unwrap();
    drop(job_tx);
    let started = std::time::Instant::now();
    sampler.sample_stream(job_rx, out_tx, CancelToken::default());
    let result = out_rx.blocking_recv().expect("one result");
    let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
        panic!("the job completes");
    };
    assert_eq!(reads.len(), 64);
    for read in &reads {
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    // Every returned state is a greedy local minimum.
    let plan = Plan::new(&graph, None).unwrap();
    for read in &reads {
        let mut s = read.spins.clone();
        assert_eq!(plan.greedy().descend(&mut s, None).unwrap(), 0);
    }
    eprintln!(
        "wall {:?}, kernel {} us, best {:?}",
        started.elapsed(),
        result.device_access_time_us,
        reads.iter().map(|r| r.energy_milli).min()
    );
}
