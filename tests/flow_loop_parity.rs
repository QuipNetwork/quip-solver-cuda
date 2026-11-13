//! Direct-check oracle: quip-solver-cpu ec84be8a31f24c722c83e4701efb1bf877a5a2f7,
//! `tests/fixtures/flow_loops_golden.json` and `src/families/flow.rs:454-480`.

use quip_protocol::scoring::energy_milli;
use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::flow::FLOW_V1;
use quip_solver_core::ising_generators::instance::{Instance, InstanceParts};
use quip_solver_core::ising_generators::polariton::POLARITON_V1;
use quip_solver_core::ising_generators::square::{Boundary, SquareLattice};
use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};
use quip_solver_cuda::families::flow_loops::{
    orient, parse_charges, parse_incidence, parse_link_costs, score, FlowError, FlowLoopsSampler,
    FlowProblem, LinkCosts, LinkTask, LoopError, LoopLayout, Outcome, Score, TaskError,
};
use serde_json::Value;
use serial_test::serial;
use std::fmt::Write;

#[path = "support/flow_session.rs"]
mod flow_session;

fn topology(name: &str) -> BaseTopology {
    let t = quip_solver_conformance::topology(name).unwrap();
    BaseTopology::new(t.nodes, &t.edges).unwrap()
}

fn instance(contract: &Contract, name: &str, nonce: u8) -> (Vec<u8>, Instance) {
    let reference = reference_revision(contract.family_id, contract.revision).unwrap();
    for &params in reference.params {
        if let Ok(built) = export_instances(contract, params, &topology(name), &[[nonce; 32]]) {
            if let Some(i) = built.into_iter().find_map(Result::ok) {
                return (params.to_vec(), i);
            }
        }
    }
    panic!("no instance for {} on {name}", contract.family_id)
}

fn spins(text: &str) -> Vec<i8> {
    text.bytes()
        .map(|b| if b == b'+' { 1 } else { -1 })
        .collect()
}

fn graph(i: &Instance) -> IsingGraph {
    IsingGraph::new(
        i.h().iter().map(|&c| f64::from(c) / 1000.0).collect(),
        i.j().iter().map(|&c| f64::from(c) / 1000.0).collect(),
        i.edges()
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect(),
    )
}

#[test]
fn golden_direct_checks_and_permutation_round_trips() {
    // Copied unchanged from the CPU source commit named in the module docs.
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/flow_loops_golden.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let contract = if case["family_id"] == FLOW_V1.family_id {
            FLOW_V1
        } else {
            POLARITON_V1
        };
        let hex = case["params_hex"].as_str().unwrap();
        let params: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|p| u8::from_str_radix(&hex[p..p + 2], 16).unwrap())
            .collect();
        let nonce = u8::try_from(case["nonce_byte"].as_u64().unwrap()).unwrap();
        let i = export_instances(
            &contract,
            &params,
            &topology(case["topology"].as_str().unwrap()),
            &[[nonce; 32]],
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();
        let mut digest = String::new();
        for b in i.digest() {
            write!(&mut digest, "{b:02x}").unwrap();
        }
        assert_eq!(digest, case["instance_digest"]);
        let problem = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        let task = problem.task();
        for read in case["reads"].as_array().unwrap() {
            let links = spins(read["result_links"].as_str().unwrap());
            let nodes = spins(read["result_nodes"].as_str().unwrap());
            assert_eq!(task.node_spins(&links).unwrap(), nodes);
            assert_eq!(task.link_spins(&nodes).unwrap(), links);
            let got = score(task, &links).unwrap();
            assert_eq!(got.gap_squares, read["gap_squares"].as_u64().unwrap());
            assert_eq!(got.objective, read["objective"].as_u64().unwrap());
            assert!(task.check(&links).unwrap().feasible());
            assert_eq!(
                problem.energy(&nodes).unwrap() + i.offset_milli(),
                read["energy_milli"].as_i64().unwrap()
            );
        }
    }
}

#[test]
fn decoding_and_admission_are_host_only() {
    assert!(parse_charges(&[], 2).is_err());
    assert!(parse_incidence(&[0; 8], 1, 2).is_err());
    assert!(parse_incidence(&[0; 12], 2, 2).is_err());
    assert!(parse_link_costs(&[0; 13], 1).is_err());
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    assert!(LinkTask::new(lattice, vec![0; 12], vec![Some(0); 9], None).is_err());
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, i) = instance(&contract, "../topologies/routing-small.bin.gz", 0);
        let p = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        assert!(matches!(
            p.admit(&[], 10, usize::MAX),
            Err(LoopError::Empty)
        ));
        assert!(matches!(
            p.admit(&[vec![0; 12]], 10, usize::MAX),
            Err(LoopError::Start { index: 0, .. })
        ));
        assert!(matches!(
            p.admit(&[vec![1; 12]], 0, usize::MAX),
            Err(LoopError::Parameter(_))
        ));
        assert!(matches!(
            p.admit(&[vec![1; 12]], 10, 0),
            Err(LoopError::Memory { .. })
        ));
        assert!(p.admit(&[vec![1; 12]], 10, usize::MAX).unwrap().total() > 0);
        assert!(FlowProblem::from_instance(&contract, &[], &i).is_err());
        let parts = InstanceParts {
            descriptor_digest: *i.descriptor_digest(),
            topology_order_digest: *i.topology_order_digest(),
            nonce: i.nonce().copied(),
            node_ids: i.node_ids().to_vec(),
            edges: i.edges().to_vec(),
            h: i.h().to_vec(),
            j: i.j().to_vec(),
            offset_milli: i.offset_milli(),
            metadata: vec![0, 0],
        };
        let bad = Instance::new(parts, &contract.limits, contract.metadata_kinds, None).unwrap();
        assert!(matches!(
            FlowProblem::from_instance(&contract, &params, &bad),
            Err(LoopError::Task(TaskError::Metadata))
        ));
    }
    let sampler = FlowLoopsSampler::new(0);
    assert!(matches!(
        sampler.sample(
            &IsingGraph::new(vec![0.0], vec![], vec![]),
            &SampleParams::default()
        ),
        Err(SampleError::Declined(_))
    ));
}

#[test]
fn all_small_states_match_original_model_and_exact_baseline() {
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, i) = instance(&contract, "../topologies/routing-small.bin.gz", 1);
        let problem = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        let task = problem.task();
        let g = graph(&i);
        let mut best = Score {
            gap_squares: u64::MAX,
            objective: u64::MAX,
        };
        for bits in 0..4096 {
            let links: Vec<i8> = (0..12)
                .map(|e| if bits & (1 << e) == 0 { -1 } else { 1 })
                .collect();
            let nodes = task.node_spins(&links).unwrap();
            let s = score(task, &links).unwrap();
            best = best.min(s);
            let energy = problem.energy(&nodes).unwrap();
            assert_eq!(energy, energy_milli(&nodes, &g.h, &g.j, &g.edges));
            assert_eq!(energy + i.offset_milli(), problem.score_energy(s).unwrap());
            assert_eq!(
                task.check(&links)
                    .unwrap()
                    .charges
                    .iter()
                    .map(|&q| i32::from(q))
                    .sum::<i32>(),
                0
            );
        }
        let Outcome::Optimal(o) = orient(task, &mut || false).unwrap() else {
            panic!("feasible fixture")
        };
        assert_eq!(score(task, &o.link_spins).unwrap(), best);
        assert!(matches!(
            orient(task, &mut || true),
            Err(FlowError::Cancelled)
        ));
    }
}

#[test]
fn signed_parallel_links_and_bad_labels_keep_the_cpu_contract() {
    let lattice = SquareLattice::new(2, Boundary::Periodic).unwrap();
    let task = LinkTask::new(
        lattice,
        (0..8).rev().collect(),
        vec![Some(0); 4],
        Some(LinkCosts {
            targets: vec![1; 8],
            costs: vec![1; 8],
        }),
    )
    .unwrap();
    assert!(task.check(&[0; 8]).is_none());
    assert!(task.check(&[1; 7]).is_none());
    assert_eq!(score(&task, &[1; 8]).unwrap(), Score::default());
    let layout = LoopLayout::new(&task);
    assert!(layout.cycles().chunks_exact(5).any(|row| row[0] == 2));
    for bits in 0..256 {
        let start: Vec<i8> = (0..8)
            .map(|e| if bits & (1 << e) == 0 { -1 } else { 1 })
            .collect();
        for row in layout.cycles().chunks_exact(5) {
            let count = usize::try_from(row[0]).unwrap();
            let sense: Vec<i8> = row[1..=count]
                .iter()
                .map(|&e| start[(e.unsigned_abs() - 1) as usize] * if e > 0 { 1 } else { -1 })
                .collect();
            if sense.iter().all(|&s| s == sense[0]) {
                let mut after = start.clone();
                for &e in &row[1..=count] {
                    after[(e.unsigned_abs() - 1) as usize] *= -1;
                }
                assert_eq!(
                    task.check(&start).unwrap().charges,
                    task.check(&after).unwrap().charges
                );
            }
        }
    }
    assert_eq!(
        task.link_spins(&task.node_spins(&[-1; 8]).unwrap())
            .unwrap(),
        [-1; 8]
    );
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let task = LinkTask::new(lattice, (0..12).collect(), vec![Some(1); 9], None).unwrap();
    assert!(matches!(
        orient(&task, &mut || false).unwrap(),
        Outcome::Infeasible(_)
    ));
}

#[test]
fn footprints_saturate_instead_of_wrapping() {
    use quip_solver_cuda::capacity::flow_loop_footprint;
    let a = flow_loop_footprint(12, 9, 4, 1);
    let b = flow_loop_footprint(12, 9, 4, 2);
    assert_eq!(b.total() - a.total(), a.per_instance_bytes);
    assert_eq!(
        flow_loop_footprint(usize::MAX, usize::MAX, usize::MAX, usize::MAX).total(),
        usize::MAX
    );
}

#[test]
fn capabilities_declare_the_two_revisions_first_without_a_device() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-flow-loops"))
        .arg("--capabilities")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let caps: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cuda");
    assert_eq!(caps["algorithm"], "flow-loops");
    assert_eq!(caps["solverId"], "quip-cuda-flow-loops/1");
    // The two declared families come first in each list. The solver-core
    // default session path adds every other registered family to
    // `generatedFamilies`.
    let registered = quip_solver_core::ising_generators::registry::CONTRACTS.len();
    for (key, len) in [("generatedFamilies", registered), ("acceptedFamilies", 2)] {
        let entries = caps[key].as_array().unwrap();
        assert_eq!(entries.len(), len, "{key}");
        for (entry, id) in entries
            .iter()
            .zip([FLOW_V1.family_id, POLARITON_V1.family_id])
        {
            assert_eq!(entry["familyId"], id);
            assert_eq!(entry["revisions"], serde_json::json!([1]));
            assert!(entry.get("paramsDigests").is_none());
        }
    }
    assert!(caps["features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x == "initial-spins"));
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn gpu_states_scores_optimum_and_determinism() {
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, i) = instance(&contract, "../topologies/routing-small.bin.gz", 1);
        let problem = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        let starts: Vec<Vec<i8>> = (0..16)
            .map(|k| {
                (0..12)
                    .map(|e| {
                        if (k * 173 + 21) & (1 << e) == 0 {
                            -1
                        } else {
                            1
                        }
                    })
                    .collect()
            })
            .collect();
        let run = problem.run(0, &starts, 10).unwrap();
        let again = problem.run(0, &starts, 10).unwrap();
        assert_eq!(run.spins, again.spins);
        assert_eq!(run.scores, again.scores);
        assert_eq!(run.spins.len(), starts.len());
        for (k, nodes) in run.spins.iter().enumerate() {
            let links = problem.task().link_spins(nodes).unwrap();
            let start = problem.task().link_spins(&starts[k]).unwrap();
            assert!(problem.task().check(&links).unwrap().feasible());
            assert_eq!(run.scores[k], score(problem.task(), &links).unwrap());
            assert!(run.scores[k] <= score(problem.task(), &start).unwrap());
            assert_eq!(run.energies[k], problem.energy(nodes).unwrap());
        }
        let optimum = (0..4096)
            .map(|b| {
                let links: Vec<i8> = (0..12)
                    .map(|e| if b & (1 << e) == 0 { -1 } else { 1 })
                    .collect();
                score(problem.task(), &links).unwrap()
            })
            .min()
            .unwrap();
        assert_eq!(*run.scores.iter().min().unwrap(), optimum);
        if contract.family_id == POLARITON_V1.family_id {
            // Distinct feasible P states have tied zero scores. A strict
            // chain leaves them unchanged, which makes parent order observable.
            let mut feasible: Vec<Vec<i8>> = (0..4096)
                .filter_map(|b| {
                    let links: Vec<i8> = (0..12)
                        .map(|e| if b & (1 << e) == 0 { -1 } else { 1 })
                        .collect();
                    (score(problem.task(), &links).unwrap().gap_squares == 0)
                        .then(|| problem.task().node_spins(&links).unwrap())
                })
                .take(4)
                .collect();
            assert!(feasible.len() >= 2);
            feasible.reverse();
            assert_eq!(problem.run(0, &feasible, 10).unwrap().spins, feasible);
        }
        eprintln!(
            "{}: launch {:?}, transfer {:?}, validation {:?}",
            contract.family_id, run.launch_time, run.transfer_time, run.validation_time
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn gpu_permuted_golden_starts_and_cold_baseline() {
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, i) = instance(&contract, "../topologies/polariton-l3-permuted.bin.gz", 0);
        let p = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        let Outcome::Optimal(o) = orient(p.task(), &mut || false).unwrap() else {
            panic!("feasible")
        };
        let baseline = p.task().node_spins(&o.link_spins).unwrap();
        let starts = vec![
            baseline.clone(),
            vec![1; baseline.len()],
            vec![-1; baseline.len()],
            baseline.clone(),
        ];
        let run = p.run(0, &starts, 10).unwrap();
        assert_eq!(run.spins[0], baseline);
        assert_eq!(run.spins[3], baseline);
        for (k, s) in run.spins.iter().enumerate() {
            assert!(p
                .task()
                .check(&p.task().link_spins(s).unwrap())
                .unwrap()
                .feasible());
            assert_eq!(run.energies[k], p.energy(s).unwrap());
            assert!(
                run.scores[k]
                    <= score(p.task(), &p.task().link_spins(&starts[k]).unwrap()).unwrap()
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn gpu_large_instances_strides_and_determinism() {
    use quip_solver_core::ising_generators::ice::topology::zephyr_lattice;

    let (nodes, edges) = zephyr_lattice(10, 4).unwrap();
    let topology = BaseTopology::new(nodes, &edges).unwrap();
    for contract in [FLOW_V1, POLARITON_V1] {
        let mut params = reference_revision(contract.family_id, contract.revision)
            .unwrap()
            .params[0]
            .to_vec();
        // Canonical TLV fields: Zephyr m=10, side=18. Keep the family-specific field 4.
        params[7..9].copy_from_slice(&10_u16.to_le_bytes());
        params[22..24].copy_from_slice(&18_u16.to_le_bytes());
        let i = export_instances(&contract, &params, &topology, &[[1; 32]])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        let problem = FlowProblem::from_instance(&contract, &params, &i).unwrap();
        let model = graph(&i);
        let n = problem.task().positions().len();
        assert_eq!(n, 684);
        assert_eq!(problem.task().labels().len(), 361);
        assert_eq!(LoopLayout::new(problem.task()).cycles().len() / 5, 324);
        let starts: Vec<Vec<i8>> = (0..16_u64)
            .map(|k| {
                let mut state = k + 1;
                (0..n)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if state & 1 == 0 {
                            -1
                        } else {
                            1
                        }
                    })
                    .collect()
            })
            .collect();
        // Two passes cross the shared pass-reset boundary as well as proposal boundaries.
        let run = problem.run(0, &starts, 2).unwrap();
        let again = problem.run(0, &starts, 2).unwrap();
        assert_eq!(run.spins, again.spins);
        assert_eq!(run.scores, again.scores);
        assert_eq!(run.energies, again.energies);
        for result in [&run, &again] {
            assert_eq!(result.spins.len(), starts.len());
            assert_eq!(result.scores.len(), starts.len());
            assert_eq!(result.energies.len(), starts.len());
            for (k, nodes) in result.spins.iter().enumerate() {
                let links = problem.task().link_spins(nodes).unwrap();
                let check = problem.task().check(&links).unwrap();
                assert_eq!(check.charges.iter().map(|&q| i32::from(q)).sum::<i32>(), 0);
                let direct = score(problem.task(), &links).unwrap();
                let start = score(
                    problem.task(),
                    &problem.task().link_spins(&starts[k]).unwrap(),
                )
                .unwrap();
                assert_eq!(result.scores[k], direct);
                assert!(direct <= start);
                assert_eq!(
                    result.energies[k],
                    energy_milli(nodes, &model.h, &model.j, &model.edges)
                );
                assert_eq!(result.energies[k], problem.energy(nodes).unwrap());
            }
        }
        eprintln!(
            "{} large: launch {:?}, transfer {:?}, validation {:?}",
            contract.family_id, run.launch_time, run.transfer_time, run.validation_time
        );
    }
}

#[test]
fn sampler_declines_mismatched_graphs_before_opening_a_device() {
    for (family, _) in flow_session::contexts() {
        let model = graph(family.instance());
        for part in 0..6 {
            let mut bad = model.clone();
            match part {
                0 => {
                    bad.h.pop();
                }
                1 => bad.h[0] += 0.001,
                2 => {
                    bad.j.pop();
                }
                3 => bad.j[0] += 0.001,
                4 => bad.edges = bad.edges[1..].into(),
                5 => {
                    std::sync::Arc::make_mut(&mut bad.edges)[0] = (0, 0);
                }
                _ => unreachable!(),
            }
            assert!(matches!(FlowLoopsSampler::new(usize::MAX).sample_family(
                &bad, &SampleParams::default(), None, family),
                Err(SampleError::Declined(message)) if message == "family instance differs from job graph"));
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial]
fn gpu_sampler_cold_count_warm_order_and_fixed_policy() {
    let sampler = FlowLoopsSampler::new(0);
    for (family, template) in flow_session::contexts() {
        let model = graph(family.instance());
        let p = FlowProblem::from_instance(family.contract(), family.params(), family.instance())
            .unwrap();
        let Outcome::Optimal(o) = orient(p.task(), &mut || false).unwrap() else {
            panic!("feasible")
        };
        let baseline = p.task().node_spins(&o.link_spins).unwrap();
        for reads in [5, 0] {
            let params = SampleParams {
                num_reads: reads,
                ..Default::default()
            };
            let cold = sampler
                .sample_family(&model, &params, None, family)
                .unwrap();
            assert_eq!(cold.len(), reads.max(1));
            for result in cold {
                assert_eq!(result.spins, baseline);
                assert!(p
                    .task()
                    .check(&p.task().link_spins(&result.spins).unwrap())
                    .unwrap()
                    .feasible());
                assert_eq!(
                    result.energy_milli,
                    energy_milli(&result.spins, &model.h, &model.j, &model.edges)
                );
            }
        }
        let mut warm = template.clone();
        warm.spins = if family.family_id() == POLARITON_V1.family_id {
            (0..4096)
                .filter_map(|bits| {
                    let links: Vec<i8> = (0..12)
                        .map(|e| if bits & (1 << e) == 0 { -1 } else { 1 })
                        .collect();
                    p.task()
                        .check(&links)
                        .unwrap()
                        .feasible()
                        .then(|| p.task().node_spins(&links).unwrap())
                })
                .take(4)
                .collect()
        } else {
            vec![
                baseline.clone(),
                vec![1; baseline.len()],
                vec![-1; baseline.len()],
            ]
        };
        warm.spins.reverse();
        let params = SampleParams {
            num_reads: 5,
            num_sweeps: 1,
            seed: 7,
            ..Default::default()
        };
        let run = sampler
            .sample_family(&model, &params, Some(&warm), family)
            .unwrap();
        let changed = SampleParams {
            seed: u64::MAX,
            num_sweeps: 1000,
            ..params
        };
        assert_eq!(
            run,
            sampler
                .sample_family(&model, &changed, Some(&warm), family)
                .unwrap()
        );
        let expected = p.run(0, &warm.spins, 10).unwrap();
        assert_eq!(run.len(), warm.spins.len());
        if family.family_id() == POLARITON_V1.family_id {
            assert_eq!(warm.spins.len(), 4);
            assert!(warm.spins.windows(2).all(|pair| pair[0] != pair[1]));
            assert_eq!(
                run.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
                warm.spins
            );
        }
        for (k, result) in run.iter().enumerate() {
            assert_eq!(result.spins, expected.spins[k]);
            assert_eq!(result.energy_milli, expected.energies[k]);
            assert!(p
                .task()
                .check(&p.task().link_spins(&result.spins).unwrap())
                .unwrap()
                .feasible());
            assert_eq!(
                result.energy_milli,
                energy_milli(&result.spins, &model.h, &model.j, &model.edges)
            );
        }
    }
}
