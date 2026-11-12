//! Independent constrained CUDA chains for F-static and adapted P.
//!
//! CPU-12 ec84be8a, src/families/flow_loops.rs:1-55 defines the conserved
//! moves and score. This policy uses fixed work and strict improvements.
//! Every returned replica receives an independent direct check and exact
//! original-model rescore. Replica positions always retain start order.
//! The deterministic policy ignores the session seed and sweep count.
//! Cold reads are identical copies of the exact baseline orientation.

mod device;
mod sampler;
pub mod task;

pub use sampler::{FlowLoopsSampler, CUDA_FLOW_LOOPS_IDENTITY};
pub use task::{
    orient, parse_charges, parse_incidence, parse_link_costs, FlowError, LinkCosts, LinkTask, Mode,
    Outcome, TaskError,
};

use crate::capacity::{flow_loop_footprint, Footprint};
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::square::{Axis, Boundary};
use std::collections::BTreeMap;
use std::time::Duration;

/// Score ordered first by pin loss, then by mismatched link costs.
/// CPU-12 flow_loops.rs:94-136.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score {
    /// Sum of squared charge gaps at labeled vertices.
    pub gap_squares: u64,
    /// Cost of links whose spin differs from the requested direction.
    pub objective: u64,
}

/// Direct CPU-12 score. Invalid orientations return `None`.
#[must_use]
pub fn score(task: &LinkTask, spins: &[i8]) -> Option<Score> {
    let check = task.check(spins)?;
    let mut gap_squares = 0_u64;
    for (label, &charge) in task.labels().iter().zip(&check.charges) {
        if let Some(label) = label {
            let gap = i64::from(charge) - i64::from(*label);
            gap_squares = gap_squares.checked_add(gap.unsigned_abs().pow(2))?;
        }
    }
    Some(Score {
        gap_squares,
        objective: check.objective,
    })
}

/// Admission, execution, or validation failure. No invalid replica is returned.
#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    /// Canonical family decoding failed.
    #[error(transparent)]
    Task(#[from] TaskError),
    /// There are no starts.
    #[error("no input state")]
    Empty,
    /// A supplied start has the wrong length or an invalid spin.
    #[error("input state {index}: {why}")]
    Start {
        /// Position in start order.
        index: usize,
        /// Rejection detail.
        why: &'static str,
    },
    /// Invalid work count or integer range.
    #[error("flow-loop parameter: {0}")]
    Parameter(&'static str),
    /// The admitted shape exceeds usable free memory.
    #[error("flow-loop allocation needs {requested} bytes, budget {budget}")]
    Memory {
        /// Total estimated allocation including headroom.
        requested: usize,
        /// Usable memory budget.
        budget: usize,
    },
    /// A returned state disagrees with direct host checks.
    #[error("replica {replica} fails direct validation: {why}")]
    Rescore {
        /// Position in start order.
        replica: usize,
        /// Rejection detail.
        why: &'static str,
    },
    /// Compilation or module loading failed.
    #[error(transparent)]
    Cuda(#[from] crate::cuda_device::CudaError),
    /// A driver operation failed.
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}

/// Signed physical incidence and directed short cycles. Indices retain link order.
#[derive(Clone, Debug)]
pub struct LoopLayout {
    ends: Vec<i32>,
    incidence: Vec<i32>,
    labels: Vec<i32>,
    costs: Vec<i64>,
    targets: Vec<i8>,
    cycles: Vec<i32>,
}

impl LoopLayout {
    /// Pack the oracle lattice, including free vertices and parallel links.
    ///
    /// # Panics
    /// Only if `SquareLattice` breaks its bounded index contract.
    #[must_use]
    pub fn new(task: &LinkTask) -> Self {
        let lattice = task.lattice();
        let mut ends = Vec::new();
        for e in 0..lattice.link_count() {
            let link = lattice.link(e).expect("lattice link");
            ends.extend([int(link.tail), int(link.head)]);
        }
        let mut incidence = Vec::new();
        for v in 0..lattice.vertex_count() {
            let mut row = [0_i32; 4];
            for (slot, &(e, sign)) in row.iter_mut().zip(lattice.incident(v)) {
                *slot = int(e + 1) * i32::from(sign);
            }
            incidence.extend(row);
        }
        let links = task.positions().len();
        Self {
            ends,
            incidence,
            labels: task
                .labels()
                .iter()
                .map(|l| l.map_or(127, i32::from))
                .collect(),
            costs: task.costs().map_or_else(
                || vec![0; links],
                |c| c.costs.iter().map(|&v| i64::from(v)).collect(),
            ),
            targets: task
                .costs()
                .map_or_else(|| vec![1; links], |c| c.targets.clone()),
            cycles: short_cycles(task),
        }
    }

    /// Five integers per cycle: length, then signed one-based link indices.
    #[must_use]
    pub fn cycles(&self) -> &[i32] {
        &self.cycles
    }
}

fn int(v: u32) -> i32 {
    // SquareLattice bounds the side at 64: at most 8,320 links.
    i32::try_from(v).expect("square lattice index fits i32")
}

// CPU-12 flow_loops.rs:380-419, with signed one-based links in each row.
fn short_cycles(task: &LinkTask) -> Vec<i32> {
    let lattice = task.lattice();
    let side = u32::from(lattice.side());
    let width = match lattice.boundary() {
        Boundary::Open => side + 1,
        Boundary::Periodic => side,
    };
    let mut owned = vec![[0; 2]; lattice.vertex_count() as usize];
    let mut parallel = BTreeMap::<(u32, u32), Vec<(i32, i32)>>::new();
    for e in 0..lattice.link_count() {
        let l = lattice.link(e).expect("lattice link");
        owned[l.tail as usize][usize::from(l.axis == Axis::Y)] = int(e + 1);
        let key = (l.tail.min(l.head), l.tail.max(l.head));
        parallel
            .entry(key)
            .or_default()
            .push((int(e + 1), if l.tail < l.head { 1 } else { -1 }));
    }
    let at = |x: u32, y: u32| ((y % width) * width + x % width) as usize;
    let mut cycles = Vec::new();
    for y in 0..side {
        for x in 0..side {
            cycles.extend([
                4,
                owned[at(x, y)][0],
                owned[at(x + 1, y)][1],
                -owned[at(x, y + 1)][0],
                -owned[at(x, y)][1],
            ]);
        }
    }
    for links in parallel.values() {
        for (i, &(e, sense)) in links.iter().enumerate() {
            for &(f, other) in &links[i + 1..] {
                cycles.extend([2, e, -sense * other * f, 0, 0]);
            }
        }
    }
    cycles
}

/// Validated family task and the original integer model.
#[derive(Clone, Debug)]
pub struct FlowProblem {
    task: LinkTask,
    instance: Instance,
    layout: LoopLayout,
    weight: u64,
}

impl FlowProblem {
    /// Decode with CPU-12's canonical admission before packing device data.
    ///
    /// # Errors
    /// Returns typed metadata or accumulator errors, without opening CUDA.
    pub fn from_instance(
        contract: &Contract,
        params: &[u8],
        instance: &Instance,
    ) -> Result<Self, LoopError> {
        let (mode, task) = LinkTask::from_instance(contract, params, instance)?;
        let weight = match mode {
            Mode::FlowStatic => {
                let bytes =
                    section(instance.metadata(), task::KIND_FLOW).ok_or(TaskError::Metadata)?;
                u64::from_le_bytes(bytes[..8].try_into().map_err(|_| TaskError::Metadata)?)
            }
            Mode::AdaptedP => 250,
        };
        let mut gaps = 0_u64;
        for (v, label) in (0_u32..).zip(task.labels()) {
            if let Some(q) = label {
                let bound = u64::from(lattice_degree(&task, v)) + i64::from(*q).unsigned_abs();
                gaps = gaps
                    .checked_add(bound * bound)
                    .ok_or(LoopError::Parameter("gap range overflow"))?;
            }
        }
        let objective: u64 = task
            .costs()
            .map_or(0, |c| c.costs.iter().map(|&c| u64::from(c)).sum());
        let max_energy = i128::from(weight) * i128::from(gaps) / 2 + i128::from(objective);
        let model_bound: i128 = instance
            .h()
            .iter()
            .chain(instance.j())
            .map(|&c| i128::from(c).abs())
            .sum();
        if max_energy > i128::from(i64::MAX)
            || model_bound + i128::from(instance.offset_milli()).abs() > i128::from(i64::MAX)
        {
            return Err(LoopError::Parameter("exact milli accumulator exceeds i64"));
        }
        let layout = LoopLayout::new(&task);
        Ok(Self {
            task,
            instance: instance.clone(),
            layout,
            weight,
        })
    }

    /// The physical task in canonical link order.
    #[must_use]
    pub const fn task(&self) -> &LinkTask {
        &self.task
    }

    /// Check work counts, starts, integer dimensions, and memory before allocation.
    /// Starts are in original model node order.
    ///
    /// # Errors
    /// Returns a start, parameter, or memory error. This method makes no CUDA calls.
    pub fn admit(
        &self,
        starts: &[Vec<i8>],
        passes: u32,
        free_bytes: usize,
    ) -> Result<Footprint, LoopError> {
        if starts.is_empty() {
            return Err(LoopError::Empty);
        }
        if passes == 0 || passes > 10_000 || i32::try_from(starts.len()).is_err() {
            return Err(LoopError::Parameter(
                "passes must be in 1..=10000 and replicas must fit i32",
            ));
        }
        for (index, start) in starts.iter().enumerate() {
            if start.len() != self.task.positions().len()
                || start.iter().any(|&s| s != 1 && s != -1)
            {
                return Err(LoopError::Start {
                    index,
                    why: "expected one +1 or -1 per node",
                });
            }
        }
        let footprint = flow_loop_footprint(
            self.task.positions().len(),
            self.task.labels().len(),
            self.layout.cycles.len() / 5,
            starts.len(),
        );
        let budget = free_bytes / crate::capacity::MEMORY_HEADROOM_DEN
            * crate::capacity::MEMORY_HEADROOM_NUM;
        if footprint.total() > budget || footprint.total() > isize::MAX.unsigned_abs() {
            return Err(LoopError::Memory {
                requested: footprint.total(),
                budget,
            });
        }
        Ok(footprint)
    }

    /// Exact original-model energy, without the instance offset, in node order.
    ///
    /// # Errors
    /// Rejects invalid spin vectors or an overflowing result.
    pub fn energy(&self, spins: &[i8]) -> Result<i64, LoopError> {
        if spins.len() != self.instance.h().len() || spins.iter().any(|&s| s != 1 && s != -1) {
            return Err(LoopError::Parameter("invalid model orientation"));
        }
        let mut total = 0_i128;
        for (&s, &h) in spins.iter().zip(self.instance.h()) {
            total += i128::from(s) * i128::from(h);
        }
        for (&(u, v), &j) in self.instance.edges().iter().zip(self.instance.j()) {
            total += i128::from(j) * i128::from(spins[u as usize]) * i128::from(spins[v as usize]);
        }
        i64::try_from(total).map_err(|_| LoopError::Parameter("model energy exceeds i64"))
    }

    /// Physical loss including the instance offset.
    ///
    /// # Errors
    /// Rejects scores outside the exact integer range.
    pub fn score_energy(&self, score: Score) -> Result<i64, LoopError> {
        let loss = i128::from(score.objective)
            + i128::from(self.weight) * i128::from(score.gap_squares) / 2;
        i64::try_from(loss).map_err(|_| LoopError::Parameter("score energy exceeds i64"))
    }

    /// Run independent fixed-work chains. Every replica is returned in start order.
    ///
    /// # Errors
    /// Rejects admission, driver, compilation, or direct-check failures.
    pub fn run(
        &self,
        device: usize,
        starts: &[Vec<i8>],
        passes: u32,
    ) -> Result<FlowRun, LoopError> {
        self.admit(starts, passes, usize::MAX)?;
        device::run(self, device, starts, passes)
    }

    // CPU-12 flow_loops.rs:488-530, extended to compare each final replica
    // against its start and the device's running charges and exact score.
    fn validate(
        &self,
        replica: usize,
        start: &[i8],
        links: &[i8],
        charges: &[i32],
        reported: &[i64],
        status: i32,
    ) -> Result<(Vec<i8>, Score, i64), LoopError> {
        let bad = |why| LoopError::Rescore { replica, why };
        if status != 0 {
            return Err(bad("a device move failed its direct check"));
        }
        let check = self
            .task
            .check(links)
            .ok_or_else(|| bad("invalid returned orientation"))?;
        let direct = score(&self.task, links).ok_or_else(|| bad("direct score overflow"))?;
        if !check
            .charges
            .iter()
            .map(|&q| i32::from(q))
            .eq(charges.iter().copied())
        {
            return Err(bad("running charges differ"));
        }
        let values = [
            i64::try_from(direct.gap_squares).map_err(|_| bad("gap range"))?,
            i64::try_from(direct.objective).map_err(|_| bad("objective range"))?,
        ];
        if reported != values {
            return Err(bad("running score differs"));
        }
        let before = self.task.check(start).ok_or_else(|| bad("invalid start"))?;
        let mut flux = 0_i64;
        for ((label, &a), &b) in self
            .task
            .labels()
            .iter()
            .zip(&check.charges)
            .zip(&before.charges)
        {
            flux += i64::from(a) - i64::from(b);
            if let Some(label) = label {
                if (i64::from(a) - i64::from(*label)).abs()
                    > (i64::from(b) - i64::from(*label)).abs()
                {
                    return Err(bad("a pin moved further from its label"));
                }
            }
        }
        if flux != 0 || direct > score(&self.task, start).ok_or_else(|| bad("start score"))? {
            return Err(bad("flux or score worsened"));
        }
        let nodes = self
            .task
            .node_spins(links)
            .ok_or_else(|| bad("permutation failed"))?;
        let energy = self.energy(&nodes)?;
        if energy.checked_add(self.instance.offset_milli()) != Some(self.score_energy(direct)?) {
            return Err(bad("original model and physical score disagree"));
        }
        Ok((nodes, direct, energy))
    }
}

fn lattice_degree(task: &LinkTask, v: u32) -> u8 {
    task.lattice().degree(v).expect("lattice vertex")
}

/// Every replica final state with measured execution stages.
#[derive(Debug)]
pub struct FlowRun {
    /// Final states in original model node order.
    pub spins: Vec<Vec<i8>>,
    /// Direct host scores, in start order.
    pub scores: Vec<Score>,
    /// Exact original-model energies without the instance offset.
    pub energies: Vec<i64>,
    /// Synchronized kernel execution time, excluding compilation and transfers.
    pub launch_time: Duration,
    /// Synchronized upload and download time.
    pub transfer_time: Duration,
    /// Final direct host validation time.
    pub validation_time: Duration,
}

// CPU-12 otsuka_sampler.rs:68-85: counted little-endian TLV sections.
pub(crate) fn section(metadata: &[u8], kind: u16) -> Option<&[u8]> {
    let (count, mut rest) = metadata.split_first_chunk::<2>()?;
    for _ in 0..u16::from_le_bytes(*count) {
        let (head, tail) = rest.split_first_chunk::<6>()?;
        let len = usize::try_from(u32::from_le_bytes([head[2], head[3], head[4], head[5]])).ok()?;
        let (payload, next) = tail.split_at_checked(len)?;
        if u16::from_le_bytes([head[0], head[1]]) == kind {
            return Some(payload);
        }
        rest = next;
    }
    None
}
