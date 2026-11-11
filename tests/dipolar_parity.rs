//! D-static parity with the independent family oracle.
// Fixtures and direct checks ported from CPU-13, ec84be8a31f24c722c83e4701efb1bf877a5a2f7,
// tests/dipolar_search.rs:22-235, 273-617.

use quip_solver_conformance::families::export_instances;
use quip_solver_core::ising_generators::defects::{
    check_route, defect_observables, RouteError, DEFECTS, DEFECTS_SCHEMA, DEFECTS_V1,
};
use quip_solver_core::ising_generators::descriptor::Params;
use quip_solver_core::ising_generators::dipolar::PhysicalStatus;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::zephyr_lattice;
use quip_solver_core::ising_generators::instance::{Instance, InstanceLimits, InstanceParts};
use quip_solver_core::ising_generators::square::{Boundary, SquareLattice};
use quip_solver_cuda::families::dipolar::{
    DipolarError, Evidence, MoveFault, Problem, Request, Rng, State, MAX_TESTED_SIDE,
};

/// The parameters of a generated case.
#[derive(Clone, Copy, Debug)]
struct Case {
    m: u16,
    side: u16,
    pairs: u16,
    obstacles: u16,
    pins: u16,
    field_x: i32,
    field_y: i32,
}

impl Case {
    /// Canonical parameter bytes on `z(m, 4)`.
    fn bytes(self) -> Vec<u8> {
        let mut bytes = vec![0x01];
        bytes.extend_from_slice(&[1, 0, 2, 0, 0, 0]);
        bytes.extend_from_slice(&self.m.to_le_bytes());
        bytes.extend_from_slice(&[2, 0, 1, 0, 0, 0, 4]);
        for (tag, value) in [
            (3u8, self.side),
            (4, self.pairs),
            (5, self.obstacles),
            (6, self.pins),
        ] {
            bytes.extend_from_slice(&[tag, 0, 2, 0, 0, 0]);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for (tag, value) in [(7u8, self.field_x), (8, self.field_y)] {
            bytes.extend_from_slice(&[tag, 0, 4, 0, 0, 0]);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn instance(self, nonce: u8) -> Instance {
        let (nodes, edges) = zephyr_lattice(self.m, 4).expect("a lattice");
        let topology = BaseTopology::new(nodes, &edges).expect("a topology");
        export_instances(&DEFECTS_V1, &self.bytes(), &topology, &[[nonce; 32]])
            .expect("the case admits")
            .remove(0)
            .expect("the salt gives an instance")
    }

    fn problem(self, nonce: u8) -> (Instance, Problem) {
        let instance = self.instance(nonce);
        let problem =
            Problem::for_instance(&self.bytes(), &instance, Evidence::Analytic).expect("a problem");
        (instance, problem)
    }
}

/// The L = 2 cases: no constraint, two pins with mixed fields, every
/// boundary link pinned with the centre as an obstacle, and five pins.
const L2_CASES: [Case; 4] = [
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 0,
        pins: 0,
        field_x: 0,
        field_y: 0,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 0,
        pins: 2,
        field_x: 300,
        field_y: -200,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 1,
        pins: 8,
        field_x: 1000,
        field_y: -1000,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 1,
        pins: 5,
        field_x: -700,
        field_y: 450,
    },
];

/// L = 3 with one pair, one obstacle, and two pins.
const L3: Case = Case {
    m: 3,
    side: 3,
    pairs: 1,
    obstacles: 1,
    pins: 2,
    field_x: 1000,
    field_y: -1000,
};

fn state(bits: u32, n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| if bits >> i & 1 == 1 { 1 } else { -1 })
        .collect()
}

/// The instance energy with the offset, from the instance coefficients.
fn instance_energy(instance: &Instance, spins: &[i8]) -> i64 {
    let mut total = instance.offset_milli();
    for (&h, &s) in instance.h().iter().zip(spins) {
        total += i64::from(h) * i64::from(s);
    }
    for (&(a, b), &j) in instance.edges().iter().zip(instance.j()) {
        total += i64::from(j) * i64::from(spins[a as usize]) * i64::from(spins[b as usize]);
    }
    total
}

fn flip(problem: &Problem, spins: &[i8], links: &[u32]) -> Vec<i8> {
    let mut after = spins.to_vec();
    for &e in links {
        let p = problem.request().positions[e as usize] as usize;
        after[p] = -after[p];
    }
    after
}

/// Every one of the 4096 states of each L = 2 case agrees with the family:
/// the violations with the observables and the feasibility hook, the energy
/// with the instance, and the physical energy with the observables. Every
/// proposal from every state passes the direct check, keeps the violations,
/// and changes the score by the full recomputation.
#[test]
fn all_4096_l2_states_match_the_oracle() {
    let mut proposals = 0u64;
    for (k, case) in L2_CASES.iter().enumerate() {
        let (instance, problem) = case.problem(u8::try_from(k).unwrap());
        let bytes = case.bytes();
        let params = Params::parse(DEFECTS_SCHEMA, &bytes).unwrap();
        let hook = DEFECTS.feasible.expect("a hard family");
        assert_eq!(problem.nodes(), 12);
        assert_eq!(
            problem.physical_status(),
            PhysicalStatus::InsufficientPhysicalEvidence
        );
        let mut rng = Rng::new(31 + k as u64);
        let (mut max_feasible, mut min_infeasible) = (i64::MIN, i64::MAX);
        for bits in 0..4096u32 {
            let spins = state(bits, 12);
            let violations = problem.violations(&spins).unwrap();
            let seen = defect_observables(&params, &instance, &spins).expect("observables");
            assert_eq!(violations.labels, seen.violated_labels, "{case:?} {bits}");
            assert_eq!(violations.pins, seen.violated_pins, "{case:?} {bits}");
            assert_eq!(violations.feasible(), hook(&params, &instance, &spins));
            let energy = problem.energy(&spins).unwrap();
            assert_eq!(
                energy + problem.offset(),
                instance_energy(&instance, &spins)
            );
            assert_eq!(
                problem.physical_energy(&spins).unwrap(),
                seen.physical_energy
            );
            if violations.feasible() {
                assert_eq!(energy + problem.offset(), seen.physical_energy);
                max_feasible = max_feasible.max(energy);
            } else {
                min_infeasible = min_infeasible.min(energy);
            }
            let current = State::new(&problem, &spins).unwrap();
            assert_eq!(current.score(), energy);
            for _ in 0..3 {
                let Some(links) = problem.propose(&current, &mut rng) else {
                    continue;
                };
                proposals += 1;
                assert!(problem.check_move(&spins, &links).is_ok(), "{links:?}");
                let after = flip(&problem, &spins, &links);
                assert_eq!(problem.violations(&after).unwrap(), violations);
                let mut next = current.clone();
                let delta = next.apply(&problem, &links).unwrap();
                let full = problem.energy(&after).unwrap();
                assert_eq!(delta, full - energy, "{case:?} {bits} {links:?}");
                assert_eq!(next.score(), full);
                assert_eq!(next.spins(), after.as_slice());
            }
        }
        assert!(max_feasible < min_infeasible, "{case:?}: the penalty bound");
    }
    assert!(proposals > 10_000, "{proposals} proposals checked");
}

/// A boundary link joins two border vertices.
fn boundary_links(lattice: &SquareLattice) -> Vec<u32> {
    let side = u32::from(lattice.side());
    let border = |v: u32| {
        let (x, y) = (v % (side + 1), v / (side + 1));
        x == 0 || y == 0 || x == side || y == side
    };
    (0..lattice.link_count())
        .filter(|&e| {
            let link = lattice.link(e).unwrap();
            border(link.tail) && border(link.head)
        })
        .collect()
}

/// An analytic L = 2 fixture with the centre labeled `label` and two pins.
fn analytic(label: i8) -> Request {
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let mut labels = vec![None; 9];
    labels[4] = Some(label);
    let pins = boundary_links(&lattice);
    Request {
        side: 2,
        labels,
        positions: (0..12).rev().collect(),
        obstacles: Vec::new(),
        pins: vec![pins[1], pins[6]],
        field_x: 250,
        field_y: 400,
    }
}

#[test]
fn invalid_requests_fail_admission() {
    let refused = |request: Request| {
        matches!(
            Problem::new(request, Evidence::Analytic),
            Err(DipolarError::Request(_))
        )
    };
    let base = analytic(2);
    assert!(Problem::new(base.clone(), Evidence::Analytic).is_ok());

    let mut r = base.clone();
    r.labels[0] = Some(0);
    assert!(refused(r), "a boundary label");
    let mut r = base.clone();
    r.labels[4] = Some(1);
    assert!(refused(r), "an odd label");
    let mut r = base.clone();
    r.labels[4] = Some(4);
    assert!(refused(r), "a label outside -2, 0, +2");
    let mut r = base.clone();
    r.labels.pop();
    assert!(refused(r), "too few labels");

    let mut r = base.clone();
    r.obstacles = vec![4];
    assert!(refused(r), "an obstacle with a nonzero label");
    let mut r = analytic(0);
    r.obstacles = vec![0];
    assert!(refused(r), "an obstacle on the boundary");
    let mut r = analytic(0);
    r.obstacles = vec![4, 4];
    assert!(refused(r), "a repeated obstacle");
    let mut r = analytic(0);
    r.obstacles = vec![99];
    assert!(refused(r), "an obstacle outside the lattice");

    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let interior = (0..12)
        .find(|e| !boundary_links(&lattice).contains(e))
        .unwrap();
    let mut r = base.clone();
    r.pins = vec![interior];
    assert!(refused(r), "a pin that is not a boundary link");
    let mut r = base.clone();
    r.pins = vec![base.pins[1], base.pins[0]];
    assert!(refused(r), "pins out of order");
    let mut r = base.clone();
    r.pins = vec![40];
    assert!(refused(r), "a pin outside the lattice");

    let mut r = base.clone();
    r.positions[0] = r.positions[1];
    assert!(refused(r), "positions that repeat a node");

    for side in [1, MAX_TESTED_SIDE + 1] {
        let mut r = base.clone();
        r.side = side;
        assert_eq!(
            Problem::new(r, Evidence::Analytic).err(),
            Some(DipolarError::UntestedSide(side))
        );
    }
}

/// A copy of `instance` with new metadata or fields. Admission rebuilds the
/// instance, so a tampered copy fails the contract check.
fn tampered(instance: &Instance, metadata: Vec<u8>, h: Vec<i32>) -> Instance {
    let parts = InstanceParts {
        descriptor_digest: *instance.descriptor_digest(),
        topology_order_digest: *instance.topology_order_digest(),
        nonce: instance.nonce().copied(),
        node_ids: instance.node_ids().to_vec(),
        edges: instance.edges().to_vec(),
        h,
        j: instance.j().to_vec(),
        offset_milli: instance.offset_milli(),
        metadata,
    };
    Instance::new(
        parts,
        &InstanceLimits::FAM_37,
        DEFECTS_V1.metadata_kinds,
        None,
    )
    .expect("a well-formed instance")
}

#[test]
fn tampered_instances_fail_admission() {
    let instance = L3.instance(3);
    let bytes = L3.bytes();
    assert!(Problem::for_instance(&bytes, &instance, Evidence::Analytic).is_ok());
    let region = instance.metadata().to_vec();
    // The last four bytes are the last pin id, and the four before are the
    // first pin id: swap them so the pins are out of order.
    let mut swapped = region.clone();
    let end = swapped.len();
    let (first, last) = (
        swapped[end - 8..end - 4].to_vec(),
        swapped[end - 4..].to_vec(),
    );
    swapped[end - 8..end - 4].copy_from_slice(&last);
    swapped[end - 4..].copy_from_slice(&first);
    let mut obstacle = region.clone();
    // The obstacle id follows the 2-byte obstacle count after the commitment.
    let at = end - 4 * 2 - 2 - 4;
    obstacle[at] ^= 1;
    let mut h = instance.h().to_vec();
    h[0] += 1;
    for (what, bad) in [
        (
            "pins out of order",
            tampered(&instance, swapped, instance.h().to_vec()),
        ),
        (
            "another obstacle",
            tampered(&instance, obstacle, instance.h().to_vec()),
        ),
        ("a changed field", tampered(&instance, region, h)),
    ] {
        assert!(
            matches!(
                Problem::for_instance(&bytes, &bad, Evidence::Analytic),
                Err(DipolarError::Metadata(_))
            ),
            "{what}"
        );
    }
    assert_eq!(
        Problem::for_instance(&L2_CASES[1].bytes(), &instance, Evidence::Analytic).err(),
        Some(DipolarError::Metadata("the contract metadata check fails")),
        "the params of another side"
    );
    assert_eq!(
        Problem::for_instance(&[1, 2, 3], &instance, Evidence::Analytic).err(),
        Some(DipolarError::Params)
    );
}

/// The vertex that `e` points out of and into under the node state `spins`.
fn ends(problem: &Problem, spins: &[i8], e: u32) -> (u32, u32) {
    let link = problem.lattice().link(e).unwrap();
    let s = spins[problem.request().positions[e as usize] as usize];
    if s == 1 {
        (link.tail, link.head)
    } else {
        (link.head, link.tail)
    }
}

/// The route of a state whose flipped links are vertex-disjoint paths from
/// each `+2` vertex: follow the flipped links out of it.
fn route_of(problem: &Problem, spins: &[i8]) -> Vec<Vec<u32>> {
    let request = problem.request();
    let links = problem.link_spins(spins).unwrap();
    let mut route = Vec::new();
    for (v, label) in (0u32..).zip(&request.labels) {
        if *label != Some(2) {
            continue;
        }
        let mut path = vec![v];
        let mut at = v;
        while request.labels[at as usize] != Some(-2) {
            let next = problem
                .lattice()
                .incident(at)
                .iter()
                .filter(|&&(e, _)| links[e as usize] == -1)
                .map(|&(e, _)| {
                    let link = problem.lattice().link(e).unwrap();
                    if link.tail == at {
                        link.head
                    } else {
                        link.tail
                    }
                })
                .find(|w| !path.contains(w))
                .expect("the path continues");
            path.push(next);
            at = next;
        }
        route.push(path);
    }
    route
}

#[test]
fn invalid_moves_and_routes_fail_direct_checking() {
    let (instance, problem) = L3.problem(5);
    let request = problem.request().clone();
    let spins = problem.start_state().unwrap();
    let fault = |links: &[u32]| match problem.check_move(&spins, links) {
        Err(DipolarError::Move(fault)) => Some(fault),
        _ => None,
    };
    assert_eq!(fault(&[]), Some(MoveFault::Empty));
    assert_eq!(fault(&[999]), Some(MoveFault::UnknownLink(999)));
    assert_eq!(
        fault(&[request.pins[0]]),
        Some(MoveFault::Pinned(request.pins[0]))
    );
    let free_link = (0..problem.lattice().link_count())
        .find(|e| !request.pins.contains(e))
        .unwrap();
    assert_eq!(
        fault(&[free_link, free_link]),
        Some(MoveFault::RepeatedLink(free_link))
    );
    let too_long: Vec<u32> = (0..=problem.lattice().link_count()).collect();
    assert_eq!(fault(&too_long), Some(MoveFault::TooLong));

    // A step along a link into the head of the first link, and a step along
    // a link that does not touch that head.
    let movable: Vec<u32> = (0..problem.lattice().link_count())
        .filter(|e| !request.pins.contains(e))
        .collect();
    let (first, against) = movable
        .iter()
        .find_map(|&a| {
            let (_, w) = ends(&problem, &spins, a);
            movable
                .iter()
                .find(|&&b| b != a && ends(&problem, &spins, b).1 == w)
                .map(|&b| (a, b))
        })
        .expect("two links into one vertex");
    assert_eq!(
        fault(&[first, against]),
        Some(MoveFault::NotDirected(against))
    );
    let (_, w) = ends(&problem, &spins, first);
    let away = movable
        .iter()
        .find(|&&e| {
            let (a, b) = ends(&problem, &spins, e);
            a != w && b != w
        })
        .copied()
        .unwrap();
    assert_eq!(fault(&[first, away]), Some(MoveFault::NotAdjacent(away)));

    // A link from a labeled vertex to the border. The obstacle has the label
    // 0, so a link out of it changes its charge too.
    let obstacle = request.obstacles[0];
    let out_of_obstacle = problem
        .lattice()
        .incident(obstacle)
        .iter()
        .map(|&(e, _)| e)
        .find(|&e| ends(&problem, &spins, e).0 == obstacle)
        .unwrap();
    assert_eq!(
        fault(&[out_of_obstacle]),
        Some(MoveFault::ChargeChanged(obstacle))
    );
    let labeled_to_border = (0..problem.lattice().link_count()).find_map(|e| {
        let (a, b) = ends(&problem, &spins, e);
        let labeled = request.labels[a as usize].is_some() && a != obstacle;
        (labeled && request.labels[b as usize].is_none() && !request.pins.contains(&e))
            .then_some((e, a))
    });
    let (e, a) = labeled_to_border.expect("an interior link to the border");
    assert_eq!(fault(&[e]), Some(MoveFault::ChargeChanged(a)));

    // The start state of a pair is one path between the opposite labels that
    // avoids the obstacle, so the family accepts its route.
    let bytes = L3.bytes();
    let params = Params::parse(DEFECTS_SCHEMA, &bytes).unwrap();
    let route = route_of(&problem, &spins);
    assert_eq!(check_route(&params, &instance, &spins, &route), Ok(()));
    let mut reversed = route.clone();
    reversed[0].reverse();
    assert_eq!(check_route(&params, &instance, &spins, &reversed), Ok(()));
    let short = vec![vec![route[0][0]]];
    assert_eq!(
        check_route(&params, &instance, &spins, &short),
        Err(RouteError::Shape)
    );
    let mut wrong_end = route.clone();
    wrong_end[0].pop();
    let check = check_route(&params, &instance, &spins, &wrong_end);
    assert!(
        matches!(check, Err(RouteError::Endpoints | RouteError::Shape)),
        "{check:?}"
    );
    assert_eq!(
        check_route(&params, &instance, &vec![1; spins.len()], &route),
        Err(RouteError::Difference)
    );
}

#[test]
fn analytic_charges_and_calibration() {
    for label in [-2, 0, 2] {
        let problem = Problem::new(analytic(label), Evidence::Analytic).unwrap();
        for bits in 0..4096 {
            let spins = state(bits, 12);
            let links = problem.link_spins(&spins).unwrap();
            let charge: i32 = problem
                .lattice()
                .incident(4)
                .iter()
                .map(|&(e, sign)| i32::from(sign) * i32::from(links[e as usize]))
                .sum();
            assert_eq!(
                problem.violations(&spins).unwrap().labels == 0,
                charge == i32::from(label)
            );
            if problem.violations(&spins).unwrap().feasible() {
                assert_eq!(
                    problem.energy(&spins).unwrap() + problem.offset(),
                    problem.physical_energy(&spins).unwrap()
                );
            }
        }
        assert_eq!(
            Problem::new(analytic(label), Evidence::Physical).err(),
            Some(DipolarError::Uncalibrated)
        );
    }
}

#[test]
fn overflow_precision_and_memory_fail_before_cuda() {
    use quip_solver_cuda::families::dipolar::{check_integer_range, Schedule};
    assert!(check_integer_range(&[i64::MAX], &[], 0).is_err());
    assert!(check_integer_range(&[i64::MIN], &[], 0).is_err());
    assert!(check_integer_range(&[0], &[i64::MAX / 4], 0).is_err());
    assert!(check_integer_range(&[0], &[], i64::MIN).is_err());
    assert!(check_integer_range(&[i64::MAX / 16], &[], 0).is_ok());
    let problem = Problem::new(analytic(0), Evidence::Analytic).unwrap();
    let schedule = Schedule::default();
    assert!(problem.admit(&schedule, 4, usize::MAX).is_ok());
    assert!(problem.admit(&schedule, 4, 1).is_err());
    assert!(problem.admit(&schedule, usize::MAX, usize::MAX).is_err());
    assert!(problem
        .admit(
            &Schedule {
                max_move: 25,
                ..schedule
            },
            1,
            usize::MAX
        )
        .is_err());
}

#[test]
fn capabilities_and_plain_decline_need_no_cuda() {
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};
    use quip_solver_cuda::families::dipolar::DipolarSampler;
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-dipolar"))
        .arg("--capabilities")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cuda");
    assert_eq!(caps["algorithm"], "dipolar");
    assert_eq!(caps["solverId"], "quip-cuda-dipolar/1");
    for list in ["generatedFamilies", "acceptedFamilies"] {
        assert_eq!(caps[list].as_array().unwrap().len(), 1);
        assert_eq!(caps[list][0]["familyId"], "quip/defects-static");
        assert_eq!(caps[list][0]["revisions"], serde_json::json!([1]));
    }
    assert!(matches!(
        DipolarSampler::new(0).sample(
            &IsingGraph::new(vec![0.0], vec![], vec![]),
            &SampleParams::default()
        ),
        Err(SampleError::Declined(_))
    ));
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial]
fn gpu_every_proposal_matches_direct_checks_and_exhaustive_optimum() {
    use quip_solver_cuda::families::dipolar::{DipolarDevice, Schedule};
    for (k, case) in L2_CASES.iter().enumerate() {
        let (_, problem) = case.problem(u8::try_from(k).unwrap());
        let optimum = (0..4096)
            .map(|b| state(b, 12))
            .filter(|s| problem.violations(s).unwrap().feasible())
            .map(|s| problem.energy(&s).unwrap())
            .min()
            .unwrap();
        let starts = vec![problem.start_state().unwrap(); 4];
        let mut device = DipolarDevice::open(0, problem).unwrap();
        let schedule = Schedule {
            proposals: 512,
            ..Schedule::default()
        };
        let run = device.run(&starts, &schedule, 2032).unwrap();
        assert_eq!(run.reads.len(), 4);
        assert_eq!(run.seed, 2032);
        for (r, read) in run.reads.iter().enumerate() {
            assert_eq!(read.replica, r);
            assert!(read.error.is_none(), "{:?}", read.error);
            let start_energy = device.problem().energy(&read.start).unwrap();
            assert!(read.energy.as_ref().unwrap() <= &start_energy);
            assert_eq!(
                *read.energy.as_ref().unwrap(),
                device.problem().energy(&read.spins).unwrap()
            );
            assert_eq!(read.device_energy, *read.energy.as_ref().unwrap());
            assert!(device.problem().violations(&read.spins).unwrap().feasible());
            assert_eq!(read.trace.len(), schedule.proposals);
            for proposal in &read.trace {
                assert_eq!(
                    proposal.device_energy,
                    device.problem().energy(&proposal.spins).unwrap()
                );
                assert!(device
                    .problem()
                    .violations(&proposal.spins)
                    .unwrap()
                    .feasible());
            }
            assert_eq!(
                read.resolved,
                (start_energy - read.energy.as_ref().unwrap()).unsigned_abs()
                    > device.problem().rounding_bound_milli()
            );
        }
        assert!(
            run.reads
                .iter()
                .map(|r| *r.energy.as_ref().unwrap())
                .min()
                .unwrap()
                <= optimum + i64::try_from(device.problem().rounding_bound_milli()).unwrap()
        );
        let repeated = device.run(&starts, &schedule, 2032).unwrap();
        assert_eq!(run.reads, repeated.reads);
        let changed = device.run(&starts, &schedule, 2033).unwrap();
        assert_ne!(run.reads, changed.reads);
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial]
fn gpu_charges_obstacles_permutations_and_every_start_are_retained() {
    use quip_solver_cuda::families::dipolar::{DipolarDevice, Schedule};
    let (instance, problem) = L3.problem(1);
    let bytes = L3.bytes();
    let params = Params::parse(DEFECTS_SCHEMA, &bytes).unwrap();
    let starts = vec![problem.start_state().unwrap(), vec![-1; 24]];
    let mut device = DipolarDevice::open(0, problem).unwrap();
    let run = device
        .run(
            &starts,
            &Schedule {
                proposals: 256,
                ..Schedule::default()
            },
            7,
        )
        .unwrap();
    assert_eq!(run.reads.len(), starts.len());
    for read in &run.reads {
        assert!(read.error.is_none(), "{:?}", read.error);
        assert_eq!(read.supplied, starts[read.replica]);
        assert_eq!(
            read.start,
            device.problem().repair(&starts[read.replica]).unwrap()
        );
        assert_eq!(
            read.device_energy,
            device.problem().energy(&read.spins).unwrap()
        );
        assert!(read.device_energy <= device.problem().energy(&read.start).unwrap());
        for spins in read
            .trace
            .iter()
            .map(|p| &p.spins)
            .chain([&read.spins, &read.terminal])
        {
            let seen = defect_observables(&params, &instance, spins).unwrap();
            assert_eq!((seen.violated_labels, seen.violated_pins), (0, 0));
            assert_eq!(
                seen.physical_energy,
                device.problem().energy(spins).unwrap() + device.problem().offset()
            );
        }
        assert_eq!(
            *read.terminal_energy.as_ref().unwrap(),
            read.terminal_device_energy
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial]
fn gpu_analytic_charges_zero_work_and_bounded_walks() {
    use quip_solver_cuda::families::dipolar::{DipolarDevice, Schedule};
    for label in [-2, 0, 2] {
        let problem = Problem::new(analytic(label), Evidence::Analytic).unwrap();
        let starts = vec![problem.start_state().unwrap(); 2];
        let mut device = DipolarDevice::open(0, problem).unwrap();
        let still = device
            .run(
                &starts,
                &Schedule {
                    proposals: 0,
                    max_move: 1,
                },
                9,
            )
            .unwrap();
        for read in &still.reads {
            assert!(read.error.is_none());
            assert_eq!(read.spins, starts[read.replica]);
            assert_eq!(read.terminal, read.spins);
            assert!(!read.resolved);
            assert!(read.trace.is_empty());
        }
        let run = device
            .run(
                &starts,
                &Schedule {
                    proposals: 128,
                    max_move: 2,
                },
                9,
            )
            .unwrap();
        for read in &run.reads {
            assert!(read.error.is_none(), "{:?}", read.error);
            assert!(read
                .trace
                .iter()
                .all(|p| p.links.len() <= 2 && p.error.is_none()));
            assert_eq!(
                read.physical_energy.as_ref().unwrap(),
                &(read.device_energy + device.problem().offset())
            );
        }
    }
}
