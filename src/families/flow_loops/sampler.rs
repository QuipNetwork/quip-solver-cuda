//! Session adapter. CPU-12 flow_loops_sampler.rs:163-234 defines start order,
//! exact cold starts, and original-model result energies.

use super::{orient, FlowProblem, LoopError, Outcome};
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::flow::FLOW_V1;
use quip_solver_core::ising_generators::polariton::POLARITON_V1;
use quip_solver_core::{
    BackendIdentity, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, WarmStart,
};

/// Static identity. Capabilities require no CUDA context.
pub const CUDA_FLOW_LOOPS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cuda,
    algorithm: quip_proto::v1::Algorithm::FlowLoops,
    max_nodes: 8320,
    max_edges: 50000,
    features: &["streaming"],
    adapt: quip_solver_core::adapt::AdaptBounds {
        min_sweeps: 10,
        max_sweeps: 10,
        min_reads: 1,
        max_reads: 256,
        reads_solution_min_factor: 0,
        reads_solution_max_factor: 0,
        reads_solution_floor_factor: 0,
    },
    limits: quip_solver_core::ModelLimits {
        max_degree: Some(0),
        max_local_field_milli: None,
        exact_resolution_milli: Some(1),
    },
};

const FAMILIES: &[FamilyClaim] = &[
    FamilyClaim {
        revisions: &[FLOW_V1],
        only_params: &[],
    },
    FamilyClaim {
        revisions: &[POLARITON_V1],
        only_params: &[],
    },
];
const DECLINE: &str = "quip-cuda-flow-loops solves only validated quip/flow-static@1 and quip/polariton-static@1 jobs";

/// Serial family sampler. Device opening is deferred until an admitted job.
#[derive(Debug)]
pub struct FlowLoopsSampler {
    device: usize,
}

impl FlowLoopsSampler {
    /// Use the given visible CUDA device, with ten fixed passes per chain.
    #[must_use]
    pub const fn new(device: usize) -> Self {
        Self { device }
    }
}

fn failure(e: &LoopError) -> SampleError {
    match e {
        LoopError::Cuda(_) | LoopError::Driver(_) | LoopError::Rescore { .. } => {
            SampleError::DeviceFault(e.to_string())
        }
        _ => SampleError::Declined(e.to_string()),
    }
}

impl Sampler for FlowLoopsSampler {
    fn sample(
        &self,
        _graph: &IsingGraph,
        _params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        Err(SampleError::Declined(DECLINE.to_owned()))
    }
    fn accepts_warm_start() -> bool {
        true
    }
    fn accepts_family_metadata() -> bool {
        true
    }
    fn declaration() -> Declaration {
        Declaration {
            solver_id: "quip-cuda-flow-loops/1",
            generated: FAMILIES,
            accepted: FAMILIES,
        }
    }
    #[expect(
        clippy::float_cmp,
        reason = "require the exact coefficient conversion used by the session"
    )]
    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let problem =
            FlowProblem::from_instance(family.contract(), family.params(), family.instance())
                .map_err(|e| failure(&e))?;
        // Refuse mismatched direct API callers before opening a device.
        let i = family.instance();
        let fields_match = graph.h.len() == i.h().len()
            && graph
                .h
                .iter()
                .zip(i.h())
                .all(|(&a, &b)| a == f64::from(b) / 1000.0);
        let edges_match = graph.edges.len() == i.edges().len()
            && graph
                .edges
                .iter()
                .zip(i.edges())
                .all(|(&(a, b), &(u, v))| (a, b) == (u as usize, v as usize));
        let weights_match = graph.j.len() == i.j().len()
            && graph
                .j
                .iter()
                .zip(i.j())
                .all(|(&a, &b)| a == f64::from(b) / 1000.0);
        if !fields_match || !edges_match || !weights_match {
            return Err(SampleError::Declined(
                "family instance differs from job graph".into(),
            ));
        }
        let starts = if let Some(warm) = warm {
            warm.spins.clone()
        } else {
            match orient(problem.task(), &mut || false)
                .map_err(|e| SampleError::Declined(e.to_string()))?
            {
                Outcome::Optimal(o) => vec![
                    problem
                        .task()
                        .node_spins(&o.link_spins)
                        .expect("baseline permutation");
                    params.num_reads.max(1)
                ],
                Outcome::Infeasible(e) => {
                    return Err(SampleError::Declined(format!("infeasible labels: {e}")))
                }
            }
        };
        let run = problem
            .run(self.device, &starts, 10)
            .map_err(|e| failure(&e))?;
        tracing::info!(policy = "flow-loops-greedy-cuda/1", passes = 10, replicas = run.spins.len(), deterministic = true,
            launch_seconds = run.launch_time.as_secs_f64(), transfer_seconds = run.transfer_time.as_secs_f64(), validation_seconds = run.validation_time.as_secs_f64(),
            scores = ?run.scores, "flow-loop final states");
        Ok(run
            .spins
            .into_iter()
            .zip(run.energies)
            .map(|(spins, energy_milli)| SamplerResult {
                spins,
                energy_milli,
            })
            .collect())
    }
}
