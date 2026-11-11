//! Batched CUDA string and loop search for `quip/defects-static` revision 1.
//!
//! The family puts one dipolar island on each link of an open square
//! lattice. A state is feasible when every labeled vertex has its requested
//! charge and every pinned link has spin `+1`. The instance energy of a
//! feasible state is its physical energy: the kept dipolar couplings and the
//! field. The family has no research solver, so this module is a new search.
//!
//! A move flips a set of links that keeps every requested constraint:
//!
//! - a **loop** is a directed cycle in the current orientation, so every
//!   vertex on it keeps its charge;
//! - a **string** is a directed path between two free boundary vertices, so
//!   only free vertices change charge.
//!
//! A move is a list of canonical link ids of the open lattice. No move flips a
//! pinned link. A move may pass through an obstacle: an obstacle is a vertex
//! with the label 0, and a loop or a string keeps the charge of every vertex it
//! passes through. Obstacles constrain only the route evidence of the family.
//! [`Problem::check_move`] checks every move directly before the search scores
//! it.
//!
//! Each replica keeps the local field of every node and updates the full score
//! of the instance with admitted integer ranges. At the end of each read it
//! recomputes the score from the coefficients and compares the two. Every
//! coefficient is an integer in milli, so acceptance compares exact integers.
//! The dipolar table rounds each kept coupling by at most half a milli, so two
//! states can trade places by at most one milli for each kept coupling. An
//! improvement within that bound is reported as not resolved.
//!
//! Host rules are ported from CPU-13 ec84be8a31f24c722c83e4701efb1bf877a5a2f7,
//! src/families/dipolar_search.rs:439-1030, 1135-1160, 1235-1470.
//! The table has the oracle calibration. It supports analytic fixtures and
//! code tests only. A request for a physical claim fails with
//! [`DipolarError::Uncalibrated`] until a calibrated record exists.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use quip_solver_core::ising_generators::defects::{DEFECTS, DEFECTS_SCHEMA};
use quip_solver_core::ising_generators::descriptor::{Params, Value};
use quip_solver_core::ising_generators::dipolar::{
    physical_status, quantize, Axis as DipolarAxis, Calibration, Entry, Geometry, Island,
    PhysicalStatus, Table,
};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::square::{
    check_request, Axis, Boundary, Request as SquareRequest, SquareLattice,
};

/// The search policy, versioned apart from the family revision.
pub const DIPOLAR_POLICY: &str = "cuda-dipolar-strings/1";

/// The largest lattice side that the tests cover. Revision 1 admits sides 2
/// and 3 only.
pub const MAX_TESTED_SIDE: u16 = 3;

/// The dipolar constant `D` of revision 1.
const DIPOLAR_D: i32 = 1;
/// The dipolar scale `S` of revision 1: a table entry is milli of `D`.
const DIPOLAR_SCALE: u32 = 1000;
/// The squared truncation radius `R2` of revision 1.
const DIPOLAR_R2: u32 = 4;
/// The metadata value of a free vertex.
const FREE_LABEL: i8 = 127;
/// Metadata kind of the charge labels.
const KIND_CHARGES: u16 = 4;
/// Metadata kind of the window and the link positions.
const KIND_INCIDENCE: u16 = 5;
/// Metadata kind of the weight, the table summary, the obstacles, and the pins.
const KIND_DEFECTS: u16 = 8;
/// Parameter tags of revision 1.
const TAG_SIDE: u16 = 3;
const TAG_FIELD_X: u16 = 7;
const TAG_FIELD_Y: u16 = 8;

/// What the result may claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Evidence {
    /// Code tests and analytic fixtures. The table may be an oracle.
    #[default]
    Analytic,
    /// A measured physical claim. It needs a calibrated table.
    Physical,
}

/// Why a request, a state, or a run is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DipolarError {
    /// The parameter bytes do not parse or are out of bounds.
    Params,
    /// The instance metadata is not valid.
    Metadata(&'static str),
    /// The side is outside the tested sides.
    UntestedSide(u16),
    /// The request is malformed or has no feasible state.
    Request(String),
    /// A physical claim was requested, and the table has no calibration.
    Uncalibrated,
    /// The table commitment of the instance is not the rebuilt oracle table.
    CalibrationMismatch,
    /// The instance coefficients differ from the coefficients of the request.
    Coefficients,
    /// A state is not one `+1` or `-1` spin per node.
    Spins,
    /// A checked operation overflowed.
    Overflow,
    /// A move fails the direct check.
    Move(MoveFault),
    /// A state violates a requested constraint.
    Infeasible(Violations),
    /// The incremental score differs from the full recomputation.
    ScoreMismatch {
        /// The incremental score.
        incremental: i64,
        /// The recomputed score.
        full: i64,
    },
    /// A declared buffer budget is too small.
    Capacity {
        /// Memory domain that failed admission.
        domain: &'static str,
        /// Required bytes.
        needed: usize,
        /// Available bytes.
        available: usize,
    },
    /// A CUDA context, compiler, driver operation, or output check failed.
    Device(String),
}

impl fmt::Display for DipolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Params => f.write_str("the defects-static parameters do not parse"),
            Self::Metadata(why) => write!(f, "the defects-static metadata is invalid: {why}"),
            Self::UntestedSide(side) => write!(
                f,
                "side {side} is outside the tested sides 2 to {MAX_TESTED_SIDE}"
            ),
            Self::Request(why) => write!(f, "the defects-static request is refused: {why}"),
            Self::Uncalibrated => f.write_str(
                "insufficient physical evidence: the dipolar table has the oracle calibration, \
                 and a physical job needs a calibrated record",
            ),
            Self::CalibrationMismatch => {
                f.write_str("the table commitment is not the oracle table of the geometry")
            }
            Self::Coefficients => {
                f.write_str("the instance coefficients differ from the coefficients of the request")
            }
            Self::Spins => f.write_str("a state is not one +1 or -1 spin per node"),
            Self::Overflow => f.write_str("an energy overflows"),
            Self::Move(fault) => write!(f, "a move fails the direct check: {fault:?}"),
            Self::Infeasible(v) => write!(
                f,
                "a state violates {} labels and {} pins",
                v.labels, v.pins
            ),
            Self::ScoreMismatch { incremental, full } => write!(
                f,
                "the incremental score {incremental} differs from the full score {full}"
            ),
            Self::Capacity {
                domain,
                needed,
                available,
            } => write!(f, "{domain} capacity needs {needed} bytes, has {available}"),
            Self::Device(why) => write!(f, "CUDA dipolar failure: {why}"),
        }
    }
}

impl std::error::Error for DipolarError {}

/// Why a move fails the direct check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveFault {
    /// The move has no link.
    Empty,
    /// The move has more links than the lattice.
    TooLong,
    /// A link id is not a link of the lattice.
    UnknownLink(u32),
    /// A link appears twice.
    RepeatedLink(u32),
    /// A link is pinned.
    Pinned(u32),
    /// Two consecutive links share no vertex.
    NotAdjacent(u32),
    /// A link points against the walk in the current orientation.
    NotDirected(u32),
    /// The walk visits a vertex twice before it closes.
    RepeatedVertex(u32),
    /// The walk ends on a labeled vertex, so it changes that charge.
    ChargeChanged(u32),
}

/// The kind of a valid move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    /// A directed cycle.
    Loop,
    /// A directed path between two free vertices.
    String,
}

/// The violated constraints of a state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Violations {
    /// Labeled vertices whose charge differs from the label.
    pub labels: u32,
    /// Pinned links whose spin is not `+1`.
    pub pins: u32,
}

impl Violations {
    /// Whether the state meets every constraint.
    #[must_use]
    pub const fn feasible(self) -> bool {
        self.labels == 0 && self.pins == 0
    }
}

/// A defects-static request on the open lattice of side `side`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The lattice side `L`.
    pub side: u16,
    /// One label per vertex in vertex order. `None` is a free vertex.
    pub labels: Vec<Option<i8>>,
    /// The dense node position of each link, in link order.
    pub positions: Vec<u32>,
    /// Obstacle vertices, strictly ascending.
    pub obstacles: Vec<u32>,
    /// Pinned links, strictly ascending.
    pub pins: Vec<u32>,
    /// The field along `+x` in milli.
    pub field_x: i32,
    /// The field along `+y` in milli.
    pub field_y: i32,
}

/// The values that kind 8 stores.
struct Stored {
    weight: u64,
    dropped: u64,
    commitment: [u8; 32],
}

/// A little-endian reader over one metadata payload.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn ids(&mut self) -> Option<Vec<u32>> {
        let count = self.u16()?;
        (0..count).map(|_| self.u32()).collect()
    }
}

/// A signed parameter value.
fn signed(params: &Params<'_>, tag: u16) -> Option<i32> {
    match params.get(tag) {
        Some(Value::Signed(v)) => i32::try_from(v).ok(),
        _ => None,
    }
}

/// An unsigned parameter value.
fn unsigned(params: &Params<'_>, tag: u16) -> Option<u64> {
    match params.get(tag) {
        Some(Value::Unsigned(v)) => Some(v),
        _ => None,
    }
}

impl Request {
    /// The request of a validated instance, and what kind 8 stores.
    fn from_instance(
        params: &Params<'_>,
        instance: &Instance,
    ) -> Result<(Self, Stored), DipolarError> {
        let side = unsigned(params, TAG_SIDE)
            .and_then(|s| u16::try_from(s).ok())
            .ok_or(DipolarError::Params)?;
        let field_x = signed(params, TAG_FIELD_X).ok_or(DipolarError::Params)?;
        let field_y = signed(params, TAG_FIELD_Y).ok_or(DipolarError::Params)?;
        let region = instance.metadata();
        let missing = DipolarError::Metadata("a section is missing or malformed");

        let mut charges = Reader(section(region, KIND_CHARGES).ok_or_else(|| missing.clone())?);
        if charges.u16() != Some(side) {
            return Err(DipolarError::Metadata("the charge side is not the side"));
        }
        let labels = charges
            .0
            .iter()
            .map(|&b| {
                let label = i8::from_le_bytes([b]);
                (label != FREE_LABEL).then_some(label)
            })
            .collect();

        let mut incidence = Reader(section(region, KIND_INCIDENCE).ok_or_else(|| missing.clone())?);
        incidence.take(4).ok_or_else(|| missing.clone())?;
        let (chunks, tail) = incidence.0.as_chunks::<4>();
        if !tail.is_empty() {
            return Err(missing);
        }
        let positions = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();

        let mut defects = Reader(section(region, KIND_DEFECTS).ok_or_else(|| missing.clone())?);
        let weight = defects.u64().ok_or_else(|| missing.clone())?;
        let dropped = defects.u64().ok_or_else(|| missing.clone())?;
        let commitment: [u8; 32] = defects
            .take(32)
            .and_then(|c| c.try_into().ok())
            .ok_or_else(|| missing.clone())?;
        let obstacles = defects.ids().ok_or_else(|| missing.clone())?;
        let pins = defects.ids().ok_or_else(|| missing.clone())?;
        if !defects.0.is_empty() {
            return Err(missing);
        }
        Ok((
            Self {
                side,
                labels,
                positions,
                obstacles,
                pins,
                field_x,
                field_y,
            },
            Stored {
                weight,
                dropped,
                commitment,
            },
        ))
    }
}

/// Whether vertex `v` of the open lattice of side `side` is on the border.
fn on_border(side: u16, v: u32) -> bool {
    let width = u32::from(side) + 1;
    let (x, y) = (v % width, v / width);
    x == 0 || y == 0 || x == u32::from(side) || y == u32::from(side)
}

/// Whether `list` is strictly ascending.
fn ascending(list: &[u32]) -> bool {
    list.windows(2).all(|w| matches!(w, [a, b] if a < b))
}

/// The oracle table over the link midpoints of `lattice`.
fn oracle_table(lattice: &SquareLattice) -> Option<Table> {
    let width = u32::from(lattice.side()) + 1;
    let mut islands = Vec::new();
    for e in 0..lattice.link_count() {
        let link = lattice.link(e)?;
        let x = u16::try_from(link.tail % width).ok()?.checked_mul(2)?;
        let y = u16::try_from(link.tail / width).ok()?.checked_mul(2)?;
        islands.push(match link.axis {
            Axis::X => Island {
                x: x.checked_add(1)?,
                y,
                axis: DipolarAxis::PlusX,
                moment: 1,
            },
            Axis::Y => Island {
                x,
                y: y.checked_add(1)?,
                axis: DipolarAxis::PlusY,
                moment: 1,
            },
        });
    }
    let geometry = Geometry::new(islands, DIPOLAR_D, DIPOLAR_SCALE, DIPOLAR_R2).ok()?;
    let entries = quantize(&geometry).ok()?;
    Table::new(geometry, entries, Calibration::Oracle).ok()
}

/// Whether links `i` and `j` share a vertex.
fn share_vertex(lattice: &SquareLattice, i: u32, j: u32) -> Option<bool> {
    let (a, b) = (lattice.link(i)?, lattice.link(j)?);
    Some(
        [a.tail, a.head]
            .iter()
            .any(|v| *v == b.tail || *v == b.head),
    )
}

/// A validated request with its coefficients, ready to search.
#[derive(Debug)]
pub struct Problem {
    lattice: SquareLattice,
    request: Request,
    /// Per link.
    pinned: Vec<bool>,
    /// The vertices a walk may start from: every vertex.
    starts: Vec<u32>,
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    offset: i64,
    /// `(neighbour, J)` of each node.
    neighbours: Vec<Vec<(usize, i64)>>,
    kept: Vec<Entry>,
    weight: u64,
    dropped: u64,
    commitment: [u8; 32],
    truncation_bound: u128,
    status: PhysicalStatus,
    /// The inverse temperatures of the schedule, in 1 / milli.
    beta: (f64, f64),
}

impl Problem {
    // CPU-13:1192-1202 keeps an unrepairable supplied state unchanged.
    fn prepare_starts(&self, starts: &[Vec<i8>]) -> Result<Vec<Vec<i8>>, DipolarError> {
        starts
            .iter()
            .map(|start| match self.repair(start) {
                Ok(fixed) => Ok(fixed),
                Err(DipolarError::Infeasible(_)) => Ok(start.clone()),
                Err(other) => Err(other),
            })
            .collect()
    }

    /// Validate `request` and build its coefficients from the oracle table,
    /// as step 6 of the contract does.
    ///
    /// # Errors
    ///
    /// [`DipolarError::UntestedSide`] for a side above [`MAX_TESTED_SIDE`],
    /// [`DipolarError::Request`] for a malformed or infeasible request, and
    /// [`DipolarError::Uncalibrated`] when `evidence` asks for a physical
    /// claim, because the table has the oracle calibration.
    // Preserve the CPU-13 admission order for source comparison.
    #[allow(clippy::too_many_lines)]
    pub fn new(request: Request, evidence: Evidence) -> Result<Self, DipolarError> {
        if !(2..=MAX_TESTED_SIDE).contains(&request.side) {
            return Err(DipolarError::UntestedSide(request.side));
        }
        let refuse = |why: &str| Err(DipolarError::Request(why.to_owned()));
        let lattice = SquareLattice::new(request.side, Boundary::Open)
            .map_err(|e| DipolarError::Request(e.to_string()))?;
        let (vertices, links) = (lattice.vertex_count(), lattice.link_count());
        let n = links as usize;
        if request.labels.len() != vertices as usize {
            return refuse("the labels are not one per vertex");
        }
        for (v, label) in (0u32..).zip(&request.labels) {
            match (on_border(request.side, v), label) {
                (true, None) | (false, Some(-2 | 0 | 2)) => {}
                (true, Some(_)) => return refuse("a boundary vertex has a label"),
                (false, _) => return refuse("an interior label is not -2, 0, or +2"),
            }
        }
        let mut link_of = vec![None; n];
        if request.positions.len() != n {
            return refuse("the positions are not one per link");
        }
        for (e, &p) in (0u32..).zip(&request.positions) {
            match link_of.get_mut(p as usize) {
                Some(slot @ None) => *slot = Some(e),
                _ => return refuse("the positions are not a permutation of the nodes"),
            }
        }
        if !ascending(&request.obstacles) {
            return refuse("the obstacles are not strictly ascending");
        }
        for &v in &request.obstacles {
            let zero = request.labels.get(v as usize) == Some(&Some(0));
            if v >= vertices || on_border(request.side, v) || !zero {
                return refuse("an obstacle is not an interior vertex with the label 0");
            }
        }
        if !ascending(&request.pins) {
            return refuse("the pins are not strictly ascending");
        }
        let mut pinned = vec![false; n];
        for &e in &request.pins {
            let boundary = lattice.link(e).is_some_and(|link| {
                on_border(request.side, link.tail) && on_border(request.side, link.head)
            });
            if !boundary {
                return refuse("a pin is not a boundary link");
            }
            pinned[e as usize] = true;
        }
        let square = SquareRequest {
            charges: request.labels.clone(),
            fixed: request.pins.iter().map(|&e| (e, 1)).collect(),
            fields: Vec::new(),
        };
        check_request(&lattice, &square).map_err(|e| DipolarError::Request(e.to_string()))?;

        let table = oracle_table(&lattice).ok_or(DipolarError::Overflow)?;
        if evidence == Evidence::Physical && matches!(table.calibration(), Calibration::Oracle) {
            return Err(DipolarError::Uncalibrated);
        }
        let mut kept = Vec::new();
        let (mut dropped, mut kept_magnitude) = (0u64, 0u64);
        for entry in table.entries() {
            let magnitude = u64::from(entry.q.unsigned_abs());
            if share_vertex(&lattice, entry.i, entry.j).ok_or(DipolarError::Overflow)? {
                kept.push(*entry);
                kept_magnitude = kept_magnitude
                    .checked_add(magnitude)
                    .ok_or(DipolarError::Overflow)?;
            } else {
                dropped = dropped
                    .checked_add(magnitude)
                    .ok_or(DipolarError::Overflow)?;
            }
        }
        let weight =
            penalty_weight(&request, links, kept_magnitude).ok_or(DipolarError::Overflow)?;
        let built =
            Coefficients::build(&lattice, &request, &kept, weight).ok_or(DipolarError::Overflow)?;
        let mut neighbours = vec![Vec::new(); n];
        for (&(a, b), &j) in built.edges.iter().zip(&built.j) {
            neighbours[a as usize].push((b as usize, j));
            neighbours[b as usize].push((a as usize, j));
        }
        check_integer_range(&built.h, &built.j, built.offset)?;
        let beta = schedule(&kept, &request);
        let starts = (0..vertices).collect();
        Ok(Self {
            starts,
            status: physical_status(&table),
            truncation_bound: table.truncation_bound(),
            commitment: table.commitment(),
            lattice,
            request,
            pinned,
            h: built.h,
            edges: built.edges,
            j: built.j,
            offset: built.offset,
            neighbours,
            kept,
            weight,
            dropped,
            beta,
        })
    }

    /// The problem of a `quip/defects-static` revision 1 instance.
    ///
    /// The contract metadata check runs first. Then the request is read from
    /// kinds 4, 5, and 8, and validated by [`Problem::new`]. The stored
    /// weight, dropped magnitude, and table commitment must equal the rebuilt
    /// values, and the instance coefficients must equal the coefficients of
    /// the request.
    ///
    /// # Errors
    ///
    /// The first failed check.
    pub fn for_instance(
        params: &[u8],
        instance: &Instance,
        evidence: Evidence,
    ) -> Result<Self, DipolarError> {
        let parsed = Params::parse(DEFECTS_SCHEMA, params).map_err(|_| DipolarError::Params)?;
        parsed.check_bounds().map_err(|_| DipolarError::Params)?;
        if !(DEFECTS.check_metadata)(&parsed, instance) {
            return Err(DipolarError::Metadata("the contract metadata check fails"));
        }
        let (request, stored) = Request::from_instance(&parsed, instance)?;
        let problem = Self::new(request, evidence)?;
        if stored.commitment != problem.commitment {
            return Err(DipolarError::CalibrationMismatch);
        }
        if stored.weight != problem.weight || stored.dropped != problem.dropped {
            return Err(DipolarError::Metadata(
                "the weight or the dropped magnitude differs",
            ));
        }
        let same = instance
            .h()
            .iter()
            .map(|&v| i64::from(v))
            .eq(problem.h.iter().copied())
            && instance.edges() == problem.edges.as_slice()
            && instance
                .j()
                .iter()
                .map(|&v| i64::from(v))
                .eq(problem.j.iter().copied())
            && instance.offset_milli() == problem.offset;
        if !same {
            return Err(DipolarError::Coefficients);
        }
        Ok(problem)
    }

    /// The validated request.
    #[must_use]
    pub const fn request(&self) -> &Request {
        &self.request
    }

    /// The open lattice.
    #[must_use]
    pub const fn lattice(&self) -> &SquareLattice {
        &self.lattice
    }

    /// The number of nodes, one per link.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.h.len()
    }

    /// The fields in milli, in node order.
    #[must_use]
    pub fn h(&self) -> &[i64] {
        &self.h
    }

    /// The edges as dense node positions, ascending.
    #[must_use]
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// The couplings in milli, one per edge.
    #[must_use]
    pub fn j(&self) -> &[i64] {
        &self.j
    }

    /// The offset in milli.
    #[must_use]
    pub const fn offset(&self) -> i64 {
        self.offset
    }

    /// The penalty weight `U` in milli.
    #[must_use]
    pub const fn weight(&self) -> u64 {
        self.weight
    }

    /// The sum of `|q|` over the table entries that the layout drops.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The table commitment, which names the calibration.
    #[must_use]
    pub const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// The table bound on the pairs beyond `R2`, in units of `q`.
    #[must_use]
    pub const fn truncation_bound(&self) -> u128 {
        self.truncation_bound
    }

    /// What the table may claim about physics.
    #[must_use]
    pub const fn physical_status(&self) -> PhysicalStatus {
        self.status
    }

    /// The bound in milli on the rounding error of an energy difference
    /// between two states: each kept coupling is rounded by at most half a
    /// milli and its term changes by at most twice its value.
    #[must_use]
    pub fn rounding_bound_milli(&self) -> u64 {
        self.kept.len() as u64
    }

    /// The link spins of a node state.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a state of the wrong length or a value
    /// that is not `+1` or `-1`.
    pub fn link_spins(&self, spins: &[i8]) -> Result<Vec<i8>, DipolarError> {
        if spins.len() != self.nodes() || spins.iter().any(|&s| s != 1 && s != -1) {
            return Err(DipolarError::Spins);
        }
        Ok(self
            .request
            .positions
            .iter()
            .map(|&p| spins[p as usize])
            .collect())
    }

    /// The charge of every vertex for link spins `links`.
    fn charges(&self, links: &[i8]) -> Vec<i32> {
        (0..self.lattice.vertex_count())
            .map(|v| {
                self.lattice
                    .incident(v)
                    .iter()
                    .map(|&(e, sign)| i32::from(sign) * i32::from(links[e as usize]))
                    .sum()
            })
            .collect()
    }

    /// The violated labels and pins of a node state.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state.
    pub fn violations(&self, spins: &[i8]) -> Result<Violations, DipolarError> {
        let links = self.link_spins(spins)?;
        Ok(self.link_violations(&links))
    }

    fn link_violations(&self, links: &[i8]) -> Violations {
        let charges = self.charges(links);
        let labels = charges
            .iter()
            .zip(&self.request.labels)
            .filter(|(&q, label)| label.is_some_and(|want| i32::from(want) != q))
            .count();
        let pins = self
            .request
            .pins
            .iter()
            .filter(|&&e| links[e as usize] != 1)
            .count();
        Violations {
            labels: u32::try_from(labels).unwrap_or(u32::MAX),
            pins: u32::try_from(pins).unwrap_or(u32::MAX),
        }
    }

    /// The instance energy of a node state, without the offset, recomputed
    /// from every coefficient with checked arithmetic.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn energy(&self, spins: &[i8]) -> Result<i64, DipolarError> {
        self.link_spins(spins)?;
        let mut total = 0i64;
        for (&h, &s) in self.h.iter().zip(spins) {
            total = total
                .checked_add(h * i64::from(s))
                .ok_or(DipolarError::Overflow)?;
        }
        for (&(a, b), &j) in self.edges.iter().zip(&self.j) {
            let pair = i64::from(spins[a as usize]) * i64::from(spins[b as usize]);
            total = total
                .checked_add(j.checked_mul(pair).ok_or(DipolarError::Overflow)?)
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(total)
    }

    /// The physical energy of a node state in milli: the kept couplings and
    /// the field, without any penalty, from the table directly.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn physical_energy(&self, spins: &[i8]) -> Result<i64, DipolarError> {
        let links = self.link_spins(spins)?;
        let mut total = 0i64;
        for entry in &self.kept {
            let pair = i64::from(links[entry.i as usize]) * i64::from(links[entry.j as usize]);
            total = total
                .checked_add(i64::from(entry.q) * pair)
                .ok_or(DipolarError::Overflow)?;
        }
        for (e, &s) in (0u32..).zip(&links) {
            let field = self.field_of(e);
            total = total
                .checked_sub(i64::from(field) * i64::from(s))
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(total)
    }

    fn field_of(&self, e: u32) -> i32 {
        match self.lattice.link(e).map(|link| link.axis) {
            Some(Axis::X) => self.request.field_x,
            _ => self.request.field_y,
        }
    }

    /// The tail and head of link `e` in the orientation of spin `s`.
    fn oriented(&self, e: u32, s: i8) -> Option<(u32, u32)> {
        let link = self.lattice.link(e)?;
        Some(if s == 1 {
            (link.tail, link.head)
        } else {
            (link.head, link.tail)
        })
    }

    fn free(&self, v: u32) -> bool {
        self.request.labels.get(v as usize) == Some(&None)
    }

    /// Check a move against a node state. A move is a list of link ids that
    /// forms a directed walk in the current orientation. It must be a simple
    /// directed cycle, or a simple directed path between two free vertices.
    /// It may not flip a pin.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state, and
    /// [`DipolarError::Move`] with the first fault of the move.
    pub fn check_move(&self, spins: &[i8], links: &[u32]) -> Result<MoveKind, DipolarError> {
        let state = self.link_spins(spins)?;
        self.check_links(&state, links).map_err(DipolarError::Move)
    }

    fn check_links(&self, state: &[i8], links: &[u32]) -> Result<MoveKind, MoveFault> {
        let count = self.lattice.link_count();
        let first = *links.first().ok_or(MoveFault::Empty)?;
        if links.len() > count as usize {
            return Err(MoveFault::TooLong);
        }
        let mut used = vec![false; count as usize];
        for &e in links {
            let slot = used.get_mut(e as usize).ok_or(MoveFault::UnknownLink(e))?;
            if *slot {
                return Err(MoveFault::RepeatedLink(e));
            }
            *slot = true;
            if self.pinned[e as usize] {
                return Err(MoveFault::Pinned(e));
            }
        }
        let orient = |e: u32| {
            self.oriented(e, state[e as usize])
                .ok_or(MoveFault::UnknownLink(e))
        };
        let (start, _) = orient(first)?;
        let mut visited = vec![false; self.lattice.vertex_count() as usize];
        let mut at = start;
        for &e in links {
            let (from, to) = orient(e)?;
            if from != at {
                return Err(if to == at {
                    MoveFault::NotDirected(e)
                } else {
                    MoveFault::NotAdjacent(e)
                });
            }
            if visited[at as usize] {
                return Err(MoveFault::RepeatedVertex(at));
            }
            visited[at as usize] = true;
            at = to;
        }
        if at == start {
            return Ok(MoveKind::Loop);
        }
        if visited[at as usize] {
            return Err(MoveFault::RepeatedVertex(at));
        }
        for v in [start, at] {
            if !self.free(v) {
                return Err(MoveFault::ChargeChanged(v));
            }
        }
        Ok(MoveKind::String)
    }

    /// The node of link `e`.
    fn node(&self, e: u32) -> usize {
        self.request.positions[e as usize] as usize
    }

    /// Propose a bounded move: a random directed walk from a random vertex.
    /// The walk never takes a pinned link. It stops at the first closed cycle,
    /// or at the first
    /// directed path between two free vertices, and it gives up after as many
    /// steps as the lattice has links or at a vertex with no way out.
    pub fn propose(&self, state: &State, rng: &mut Rng) -> Option<Vec<u32>> {
        let vertices = self.lattice.vertex_count() as usize;
        let v0 = *self.starts.get(rng.index(self.starts.len().max(1)))?;
        let mut index: Vec<Option<usize>> = vec![None; vertices];
        let mut path = vec![v0];
        let mut links: Vec<u32> = Vec::new();
        index[v0 as usize] = Some(0);
        let mut first_free = self.free(v0).then_some(0usize);
        let mut out = Vec::with_capacity(4);
        while links.len() < self.lattice.link_count() as usize {
            let u = *path.last()?;
            out.clear();
            for &(e, _) in self.lattice.incident(u) {
                let s = state.spins[self.node(e)];
                let (from, to) = self.oriented(e, s)?;
                if from == u && !self.pinned[e as usize] {
                    out.push((e, to));
                }
            }
            if out.is_empty() {
                return None;
            }
            let (e, w) = out[rng.index(out.len())];
            links.push(e);
            if let Some(k) = index[w as usize] {
                return Some(links.split_off(k));
            }
            index[w as usize] = Some(path.len());
            path.push(w);
            if self.free(w) {
                match first_free {
                    Some(k) => return Some(links.split_off(k)),
                    None => first_free = Some(path.len() - 1),
                }
            }
        }
        None
    }

    /// The shortest directed path of non-pinned links from `v`, forward
    /// along out-links or backward along in-links, to a vertex that `target`
    /// accepts.
    fn bfs(
        &self,
        links: &[i8],
        v: u32,
        forward: bool,
        target: &dyn Fn(u32) -> bool,
    ) -> Option<Vec<u32>> {
        let vertices = self.lattice.vertex_count() as usize;
        let mut via: Vec<Option<(u32, u32)>> = vec![None; vertices];
        let mut seen = vec![false; vertices];
        seen[v as usize] = true;
        let mut queue = VecDeque::from([v]);
        while let Some(u) = queue.pop_front() {
            for &(e, _) in self.lattice.incident(u) {
                if self.pinned[e as usize] {
                    continue;
                }
                let (from, to) = self.oriented(e, links[e as usize])?;
                let next = match (forward, from == u) {
                    (true, true) => to,
                    (false, false) if to == u => from,
                    _ => continue,
                };
                if seen[next as usize] {
                    continue;
                }
                seen[next as usize] = true;
                via[next as usize] = Some((u, e));
                if target(next) {
                    let mut path = Vec::new();
                    let mut at = next;
                    while let Some((prev, link)) = via[at as usize] {
                        path.push(link);
                        at = prev;
                    }
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
        None
    }

    /// A feasible state near `spins`: every pin is set to `+1`, then each
    /// wrong label is fixed by flipping one shortest directed path to a
    /// vertex with the opposite error, or else to a free vertex. The result
    /// is checked directly.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state, and
    /// [`DipolarError::Infeasible`] when no path fixes a label.
    pub fn repair(&self, spins: &[i8]) -> Result<Vec<i8>, DipolarError> {
        let mut links = self.link_spins(spins)?;
        for &e in &self.request.pins {
            links[e as usize] = 1;
        }
        let rounds = links.len() * 4 + 4;
        for _ in 0..rounds {
            let charges = self.charges(&links);
            let error = |v: u32| -> i32 {
                self.request.labels[v as usize]
                    .map_or(0, |want| charges[v as usize] - i32::from(want))
            };
            let Some(v) = (0..self.lattice.vertex_count()).find(|&v| error(v) != 0) else {
                return self.finish_repair(&links);
            };
            // A deficit gains in-links along a path out of `v`, a surplus
            // loses them along a path into `v`.
            let forward = error(v) < 0;
            let opposite = |w: u32| if forward { error(w) > 0 } else { error(w) < 0 };
            let either = |w: u32| opposite(w) || self.free(w);
            let path = self
                .bfs(&links, v, forward, &opposite)
                .or_else(|| self.bfs(&links, v, forward, &either));
            let Some(path) = path else {
                return Err(DipolarError::Infeasible(self.link_violations(&links)));
            };
            for e in path {
                links[e as usize] = -links[e as usize];
            }
        }
        Err(DipolarError::Infeasible(self.link_violations(&links)))
    }

    fn finish_repair(&self, links: &[i8]) -> Result<Vec<i8>, DipolarError> {
        let violations = self.link_violations(links);
        if !violations.feasible() {
            return Err(DipolarError::Infeasible(violations));
        }
        let mut spins = vec![0i8; self.nodes()];
        for (e, &s) in links.iter().enumerate() {
            spins[self.node(u32::try_from(e).map_err(|_| DipolarError::Overflow)?)] = s;
        }
        Ok(spins)
    }

    /// The feasible start of a search read: the reference state, with every
    /// link spin `+1`, repaired by paths between opposite labels.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Infeasible`] when the repair fails.
    pub fn start_state(&self) -> Result<Vec<i8>, DipolarError> {
        self.repair(&vec![1; self.nodes()])
    }

    /// The checks of a returned state: the full score, the direct
    /// constraints, and the physical energy from the table.
    fn checked_final(&self, spins: &[i8], energy: i64) -> Result<(), DipolarError> {
        let full = self.energy(spins)?;
        if full != energy {
            return Err(DipolarError::ScoreMismatch {
                incremental: energy,
                full,
            });
        }
        let violations = self.violations(spins)?;
        if !violations.feasible() {
            return Err(DipolarError::Infeasible(violations));
        }
        let physical = self.physical_energy(spins)?;
        let shifted = energy
            .checked_add(self.offset)
            .ok_or(DipolarError::Overflow)?;
        if physical != shifted {
            return Err(DipolarError::ScoreMismatch {
                incremental: shifted,
                full: physical,
            });
        }
        Ok(())
    }
}

/// The generator seed of read `r`: `SplitMix64` of the job seed and the read
/// index. It never reads the instance nonce.
#[must_use]
pub const fn read_seed(seed: u64, r: u64) -> u64 {
    let mut z = seed.wrapping_add(r.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The weight `U` of the contract penalty bound.
fn penalty_weight(request: &Request, links: u32, kept_magnitude: u64) -> Option<u64> {
    let per_axis = u64::from(links / 2);
    let field = u64::from(request.field_x.unsigned_abs())
        .checked_add(u64::from(request.field_y.unsigned_abs()))?;
    let magnitude = kept_magnitude.checked_add(per_axis.checked_mul(field)?)?;
    let mut weight = magnitude.checked_mul(2)? / 2 + 1;
    if weight % 2 == 1 {
        weight = weight.checked_add(1)?;
    }
    Some(weight)
}

/// The inverse temperatures in 1 / milli: hot accepts an uphill move of the
/// largest coefficient with probability one half, cold accepts an uphill
/// move of the smallest nonzero coefficient with probability 1 / 1000.
fn schedule(kept: &[Entry], request: &Request) -> (f64, f64) {
    let magnitudes = kept
        .iter()
        .map(|e| e.q.unsigned_abs())
        .chain([
            request.field_x.unsigned_abs(),
            request.field_y.unsigned_abs(),
        ])
        .filter(|&m| m > 0);
    let (low, high) = magnitudes.fold((u32::MAX, 1), |(lo, hi), m| (lo.min(m), hi.max(m)));
    let low = if low == u32::MAX { 1 } else { low };
    (
        std::f64::consts::LN_2 / f64::from(high),
        1000f64.ln() / f64::from(low),
    )
}

/// The coefficients of step 6 of the contract.
struct Coefficients {
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    offset: i64,
}

impl Coefficients {
    fn build(
        lattice: &SquareLattice,
        request: &Request,
        kept: &[Entry],
        weight: u64,
    ) -> Option<Self> {
        let u = i64::try_from(weight).ok()?;
        let pos = |e: u32| request.positions.get(e as usize).map(|&p| p as usize);
        let mut h = vec![0i64; request.positions.len()];
        let mut pairs: BTreeMap<(u32, u32), i64> = BTreeMap::new();
        let mut offset = 0i64;
        let mut add_pair = |a: usize, b: usize, value: i64| -> Option<()> {
            let (a, b) = (u32::try_from(a).ok()?, u32::try_from(b).ok()?);
            let slot = pairs.entry((a.min(b), a.max(b))).or_default();
            *slot = slot.checked_add(value)?;
            Some(())
        };
        for e in 0..lattice.link_count() {
            let field = match lattice.link(e)?.axis {
                Axis::X => request.field_x,
                Axis::Y => request.field_y,
            };
            let slot = h.get_mut(pos(e)?)?;
            *slot = slot.checked_sub(i64::from(field))?;
        }
        for entry in kept {
            add_pair(pos(entry.i)?, pos(entry.j)?, i64::from(entry.q))?;
        }
        for (v, label) in (0u32..).zip(&request.labels) {
            let Some(q) = label.map(i64::from) else {
                continue;
            };
            let at = lattice.incident(v);
            let degree = i64::try_from(at.len()).ok()?;
            offset = offset.checked_add((u / 2).checked_mul(degree.checked_add(q * q)?)?)?;
            for (k, &(e, ie)) in at.iter().enumerate() {
                let slot = h.get_mut(pos(e)?)?;
                *slot = slot.checked_sub(u.checked_mul(q * i64::from(ie))?)?;
                for &(f, jf) in at.get(k + 1..)? {
                    add_pair(
                        pos(e)?,
                        pos(f)?,
                        u.checked_mul(i64::from(ie) * i64::from(jf))?,
                    )?;
                }
            }
        }
        for &e in &request.pins {
            let slot = h.get_mut(pos(e)?)?;
            *slot = slot.checked_sub(u)?;
            offset = offset.checked_add(u)?;
        }
        let (edges, j) = pairs.into_iter().filter(|&(_, v)| v != 0).unzip();
        Some(Self {
            h,
            edges,
            j,
            offset,
        })
    }
}

/// A search state: the node spins, the local field of each node, and the
/// instance energy without the offset.
#[derive(Clone, Debug)]
pub struct State {
    spins: Vec<i8>,
    field: Vec<i64>,
    score: i64,
}

impl State {
    /// The state of `spins` on `problem`, with its fields and score.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn new(problem: &Problem, spins: &[i8]) -> Result<Self, DipolarError> {
        let score = problem.energy(spins)?;
        let mut field = problem.h.clone();
        for (&(a, b), &j) in problem.edges.iter().zip(&problem.j) {
            let (a, b) = (a as usize, b as usize);
            field[a] = field[a]
                .checked_add(j * i64::from(spins[b]))
                .ok_or(DipolarError::Overflow)?;
            field[b] = field[b]
                .checked_add(j * i64::from(spins[a]))
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(Self {
            spins: spins.to_vec(),
            field,
            score,
        })
    }

    /// The node spins.
    #[must_use]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// The incremental score: the instance energy without the offset.
    #[must_use]
    pub const fn score(&self) -> i64 {
        self.score
    }

    /// The exact change of the score when the links of `links` flip.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Overflow`], or [`DipolarError::Move`] for an unknown
    /// link.
    pub fn delta(&self, problem: &Problem, links: &[u32]) -> Result<i64, DipolarError> {
        let overflow = DipolarError::Overflow;
        let mut flipped = vec![false; self.spins.len()];
        for &e in links {
            let node = *problem
                .request
                .positions
                .get(e as usize)
                .ok_or(DipolarError::Move(MoveFault::UnknownLink(e)))?;
            flipped[node as usize] = true;
        }
        let mut delta = 0i64;
        for (i, _) in flipped.iter().enumerate().filter(|(_, &f)| f) {
            let s = i64::from(self.spins[i]);
            delta = delta
                .checked_sub(
                    self.field[i]
                        .checked_mul(2 * s)
                        .ok_or_else(|| overflow.clone())?,
                )
                .ok_or_else(|| overflow.clone())?;
            // A coupling with both ends flipped does not change.
            for &(k, j) in &problem.neighbours[i] {
                if k > i && flipped[k] {
                    let pair = j * s * i64::from(self.spins[k]);
                    delta = delta
                        .checked_add(pair.checked_mul(4).ok_or_else(|| overflow.clone())?)
                        .ok_or_else(|| overflow.clone())?;
                }
            }
        }
        Ok(delta)
    }

    /// Flip the links of `links` and update the fields and the score.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Overflow`], or [`DipolarError::Move`] for an unknown
    /// link.
    pub fn apply(&mut self, problem: &Problem, links: &[u32]) -> Result<i64, DipolarError> {
        let delta = self.delta(problem, links)?;
        for &e in links {
            let i = problem.node(e);
            self.spins[i] = -self.spins[i];
            let now = i64::from(self.spins[i]);
            for &(k, j) in &problem.neighbours[i] {
                self.field[k] = self.field[k]
                    .checked_add(2 * j * now)
                    .ok_or(DipolarError::Overflow)?;
            }
        }
        self.score = self
            .score
            .checked_add(delta)
            .ok_or(DipolarError::Overflow)?;
        Ok(delta)
    }
}

// CPU-13 otsuka_sampler.rs:71-83, section framing shared by family metadata.
fn section(metadata: &[u8], kind: u16) -> Option<&[u8]> {
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

/// `SplitMix64` proposal randomness, independent of instance nonces.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Start a reproducible stream.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, count: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(count).expect("bounded lattice"))
            .expect("bounded index")
    }
}

/// Maximum admitted nodes and move length, matching CPU-13 sides 2 and 3.
pub const MAX_LINKS: usize = 24;
const MAX_VERTICES: usize = 16;
const KERNEL_THREADS: u32 = 32;
const MAX_INCIDENT_LINKS: usize = 4;
/// Maximum retained proposals per replica.
pub const MAX_PROPOSALS: usize = 8192;
/// Maximum independent replicas per batch.
pub const MAX_REPLICAS: usize = 4096;

/// Fixed proposal work. The physical coefficient range determines the betas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Proposals per independent replica, including walks without a move.
    pub proposals: usize,
    /// Maximum number of links in a walk, in 1 through 24.
    pub max_move: usize,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            proposals: 256,
            max_move: MAX_LINKS,
        }
    }
}

/// Prove that scores, local fields, delta reductions, and offsets fit i64.
/// The factor eight covers the CPU-13 delta formula, lines 1410-1440.
///
/// # Errors
/// Returns overflow if any absolute value or the conservative bound overflows.
pub fn check_integer_range(h: &[i64], j: &[i64], offset: i64) -> Result<(), DipolarError> {
    let mut bound = offset.checked_abs().ok_or(DipolarError::Overflow)?;
    for &v in h.iter().chain(j) {
        bound = bound
            .checked_add(v.checked_abs().ok_or(DipolarError::Overflow)?)
            .ok_or(DipolarError::Overflow)?;
    }
    bound.checked_mul(8).ok_or(DipolarError::Overflow)?;
    Ok(())
}

impl Problem {
    /// Validate all batch widths and memory before constructing device buffers.
    ///
    /// # Errors
    /// Returns a typed request, overflow, or capacity error.
    pub fn admit(
        &self,
        schedule: &Schedule,
        replicas: usize,
        free_bytes: usize,
    ) -> Result<crate::capacity::Footprint, DipolarError> {
        if self.nodes() > MAX_LINKS
            || self.nodes() > KERNEL_THREADS as usize
            || self.lattice.vertex_count() as usize > MAX_VERTICES
            || self.request.labels.len() > MAX_VERTICES
            || (0..self.lattice.vertex_count())
                .any(|v| self.lattice.incident(v).len() > MAX_INCIDENT_LINKS)
        {
            return Err(DipolarError::Request(
                "nodes, vertices, or incident links exceed the fixed kernel arrays".into(),
            ));
        }
        if replicas == 0
            || replicas > MAX_REPLICAS
            || schedule.proposals > MAX_PROPOSALS
            || schedule.max_move == 0
            || schedule.max_move > MAX_LINKS
        {
            return Err(DipolarError::Request(
                "replicas, proposals, or move length exceed the advertised limits".into(),
            ));
        }
        check_integer_range(&self.h, &self.j, self.offset)?;
        let footprint =
            crate::capacity::dipolar_footprint(self.nodes(), replicas, schedule.proposals);
        if footprint.total() > free_bytes {
            return Err(DipolarError::Capacity {
                domain: "device",
                needed: footprint.total(),
                available: free_bytes,
            });
        }
        let host_bytes = capture_bytes(self.nodes(), replicas, schedule.proposals);
        if host_bytes > MAX_CAPTURE_BYTES {
            return Err(DipolarError::Capacity {
                domain: "host capture",
                needed: host_bytes,
                available: MAX_CAPTURE_BYTES,
            });
        }
        Ok(footprint)
    }

    /// Dense symmetric coefficients in original node order, with zero diagonal.
    #[must_use]
    pub fn dense_matrix(&self) -> Vec<i64> {
        let n = self.nodes();
        let mut dense = vec![0; n * n];
        for (&(a, b), &j) in self.edges.iter().zip(&self.j) {
            let (a, b) = (a as usize, b as usize);
            dense[a * n + b] = j;
            dense[b * n + a] = j;
        }
        dense
    }
}

use crate::cuda_device::{compile_for_arch, device_arch, driver_version};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::defects::DEFECTS_V1;
use quip_solver_core::{
    BackendIdentity, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, WarmStart,
};
use std::sync::{Arc, RwLock};
use std::time::Instant;

const SOURCE: &str = include_str!("../../kernels/dipolar.cu");

/// A failed host check, identifying the replica and optional proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    /// Zero-based independent replica.
    pub replica: usize,
    /// Zero-based proposal, or the final or terminal state when absent.
    pub proposal: Option<usize>,
    /// Direct check or exact score failure.
    pub error: DipolarError,
}

/// A raw proposed state, retained whether accepted or rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    /// Canonical link ids, in directed walk order. Empty means no move.
    pub links: Vec<u32>,
    /// Proposed spins, in original node order.
    pub spins: Vec<i8>,
    /// Device score without the offset, in milli.
    pub device_energy: i64,
    /// Host rescore without the offset, in milli.
    pub host_energy: Result<i64, DipolarError>,
    /// Whether this proposal became the current state.
    pub accepted: bool,
    /// The direct move or score failure, retained for every proposal.
    pub error: Option<DipolarError>,
}

/// All retained data for one independent replica. An error prevents sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replica {
    /// Zero-based replica index.
    pub replica: usize,
    /// Derived seed, independent of the instance nonce.
    pub seed: u64,
    /// Supplied start before repair.
    pub supplied: Vec<i8>,
    /// Start after repair, or the unchanged state when repair fails.
    pub start: Vec<i8>,
    /// Best state visited, in original node order.
    pub spins: Vec<i8>,
    /// Host score of the best state, in milli without the offset.
    pub energy: Result<i64, DipolarError>,
    /// Device score of the best state, or the host score for an unsearched read.
    pub device_energy: i64,
    /// Current state after the last proposal.
    pub terminal: Vec<i8>,
    /// Host score of the terminal state.
    pub terminal_energy: Result<i64, DipolarError>,
    /// Device terminal score, or the host score for an unsearched read.
    pub terminal_device_energy: i64,
    /// Physical energy from the oracle table.
    pub physical_energy: Result<i64, DipolarError>,
    /// Whether the returned state satisfies the constraints.
    pub feasible: bool,
    /// Whether the improvement exceeds the rounding bound of CPU-13:668-674.
    pub resolved: bool,
    /// Every raw proposal, in proposal order.
    pub trace: Vec<Proposal>,
    /// A typed validation failure. Such a replica is never a valid sample.
    pub error: Option<ValidationError>,
}

/// Capture of every replica, including failed validations and host costs.
#[derive(Debug)]
pub struct Run {
    /// Job seed used to derive the independent replica seeds.
    pub seed: u64,
    /// Every replica in start order.
    pub reads: Vec<Replica>,
    /// Host setup time, including repair, compilation, and transfers.
    pub setup_seconds: f64,
    /// Device execution and download time, including synchronization.
    pub device_seconds: f64,
    /// Time spent replaying moves and rescoring all returned states.
    pub validation_seconds: f64,
}

/// Prepared oracle and on-demand kernel. No model buffers exist before admission.
pub struct DipolarDevice {
    problem: Problem,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    setup_seconds: f64,
}

impl DipolarDevice {
    /// Open only the selected CUDA device and compile through the JIT cache.
    ///
    /// # Errors
    /// Returns a typed device error when context creation or compilation fails.
    pub fn open(device: usize, problem: Problem) -> Result<Self, DipolarError> {
        let started = Instant::now();
        let ctx = CudaContext::new(device).map_err(device_error)?;
        let arch = device_arch(&ctx);
        let module = crate::jit_cache::load_or_compile(
            &ctx,
            "dipolar",
            SOURCE,
            &[],
            &format!("sm_{arch}"),
            driver_version().map_err(device_error)?,
            0,
            || compile_for_arch(SOURCE, 0, arch, &[]),
        )
        .map_err(device_error)?;
        let function = module.load_function("quip_dipolar").map_err(device_error)?;
        Ok(Self {
            problem,
            stream: ctx.default_stream(),
            function,
            setup_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// The admitted host oracle.
    #[must_use]
    pub const fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Search one independent replica per start and retain the entire batch.
    /// CPU-13 repair runs before upload. Unrepairable starts stay unchanged as
    /// infeasible reads. Every raw proposal is replayed through
    /// its direct move check, and every best and terminal state is rescored.
    ///
    /// # Errors
    /// Returns admission or device failures. Replica failures stay in the capture.
    pub fn run(
        &mut self,
        starts: &[Vec<i8>],
        schedule: &Schedule,
        seed: u64,
    ) -> Result<Run, DipolarError> {
        self.problem.admit(schedule, starts.len(), usize::MAX)?;
        let setup = Instant::now();
        let begin = self.problem.prepare_starts(starts)?;
        self.setup_seconds += setup.elapsed().as_secs_f64();
        self.run_prepared(starts, &begin, schedule, seed)
    }

    // Keep buffer allocation, the flat CUDA ABI, and downloads together.
    #[allow(clippy::too_many_lines)]
    fn run_prepared(
        &mut self,
        starts: &[Vec<i8>],
        begin: &[Vec<i8>],
        schedule: &Schedule,
        seed: u64,
    ) -> Result<Run, DipolarError> {
        let setup = Instant::now();
        let p = &self.problem;
        let n = p.nodes();
        let replicas = starts.len();
        let steps = schedule.proposals;
        p.admit(
            schedule,
            replicas,
            self.stream
                .context()
                .mem_get_info()
                .map_err(device_error)?
                .0,
        )?;
        let slots = replicas.checked_mul(steps).ok_or(DipolarError::Overflow)?;
        let enabled: Vec<i32> = begin
            .iter()
            .map(|start| p.violations(start).map(|v| i32::from(v.feasible())))
            .collect::<Result<_, _>>()?;
        let flat: Vec<i8> = begin.iter().flatten().copied().collect();
        let mut topology = Vec::with_capacity(n * 4 + MAX_VERTICES);
        for head in [false, true] {
            for e in 0..p.lattice.link_count() {
                let link = p.lattice.link(e).ok_or(DipolarError::Overflow)?;
                topology.push(
                    i32::try_from(if head { link.head } else { link.tail })
                        .map_err(|_| DipolarError::Overflow)?,
                );
            }
        }
        for &v in &p.request.positions {
            topology.push(i32::try_from(v).map_err(|_| DipolarError::Overflow)?);
        }
        topology.extend(p.pinned.iter().map(|&b| i32::from(b)));
        topology.extend(p.request.labels.iter().map(|l| i32::from(l.is_none())));
        topology.resize(n * 4 + MAX_VERTICES, 0);
        let h = self.stream.clone_htod(&p.h).map_err(device_error)?;
        let matrix = self
            .stream
            .clone_htod(&p.dense_matrix())
            .map_err(device_error)?;
        let topology = self.stream.clone_htod(&topology).map_err(device_error)?;
        let inputs = self.stream.clone_htod(&flat).map_err(device_error)?;
        let enabled = self.stream.clone_htod(&enabled).map_err(device_error)?;
        let mut finals = self
            .stream
            .alloc_zeros::<i8>(replicas * n)
            .map_err(device_error)?;
        let mut final_scores = self
            .stream
            .alloc_zeros::<i64>(replicas)
            .map_err(device_error)?;
        let mut terminal = self
            .stream
            .alloc_zeros::<i8>(replicas * n)
            .map_err(device_error)?;
        let mut terminal_scores = self
            .stream
            .alloc_zeros::<i64>(replicas)
            .map_err(device_error)?;
        let mut raw = self
            .stream
            .alloc_zeros::<i8>((slots * n).max(1))
            .map_err(device_error)?;
        let mut raw_scores = self
            .stream
            .alloc_zeros::<i64>(slots.max(1))
            .map_err(device_error)?;
        let mut moves = self
            .stream
            .alloc_zeros::<i32>((slots * MAX_LINKS).max(1))
            .map_err(device_error)?;
        let mut lengths = self
            .stream
            .alloc_zeros::<i32>(slots.max(1))
            .map_err(device_error)?;
        let mut accepted = self
            .stream
            .alloc_zeros::<i32>(slots.max(1))
            .map_err(device_error)?;
        let int = |v| i32::try_from(v).map_err(|_| DipolarError::Overflow);
        let (kn, kv, kp, km) = (
            int(n)?,
            int(p.request.labels.len())?,
            int(steps)?,
            int(schedule.max_move)?,
        );
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(&kn)
            .arg(&kv)
            .arg(&kp)
            .arg(&km)
            .arg(&h)
            .arg(&matrix)
            .arg(&topology)
            .arg(&inputs)
            .arg(&enabled)
            .arg(&seed)
            .arg(&p.beta.0)
            .arg(&p.beta.1)
            .arg(&mut finals)
            .arg(&mut final_scores)
            .arg(&mut terminal)
            .arg(&mut terminal_scores)
            .arg(&mut raw)
            .arg(&mut raw_scores)
            .arg(&mut moves)
            .arg(&mut lengths)
            .arg(&mut accepted);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(replicas).map_err(|_| DipolarError::Overflow)?,
                1,
                1,
            ),
            block_dim: (KERNEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let setup_seconds = setup.elapsed().as_secs_f64() + self.setup_seconds;
        self.setup_seconds = 0.0;
        let execution = Instant::now();
        // SAFETY: admission bounds replicas, nodes, vertices, and trace products.
        // Topology indices are validated by Problem::new. The ABI matches dipolar.cu.
        unsafe { launch.launch(config) }.map_err(device_error)?;
        let finals = self.stream.clone_dtoh(&finals).map_err(device_error)?;
        let scores = self
            .stream
            .clone_dtoh(&final_scores)
            .map_err(device_error)?;
        let terminal = self.stream.clone_dtoh(&terminal).map_err(device_error)?;
        let terminal_scores = self
            .stream
            .clone_dtoh(&terminal_scores)
            .map_err(device_error)?;
        let raw = self.stream.clone_dtoh(&raw).map_err(device_error)?;
        let raw_scores = self.stream.clone_dtoh(&raw_scores).map_err(device_error)?;
        let moves = self.stream.clone_dtoh(&moves).map_err(device_error)?;
        let lengths = self.stream.clone_dtoh(&lengths).map_err(device_error)?;
        let accepted = self.stream.clone_dtoh(&accepted).map_err(device_error)?;
        self.stream.synchronize().map_err(device_error)?;
        let device_seconds = execution.elapsed().as_secs_f64();
        let capture = DeviceCapture {
            finals,
            scores,
            terminal,
            terminal_scores,
            raw,
            raw_scores,
            moves,
            lengths,
            accepted,
        };
        let (reads, validation_seconds) =
            validate_capture(p, starts, begin, schedule, seed, &capture)?;
        Ok(Run {
            seed,
            reads,
            setup_seconds,
            device_seconds,
            validation_seconds,
        })
    }
}

// CPU-13:1220-1231 returns no moves and zero improvement.
fn unrepaired(
    p: &Problem,
    spins: &[i8],
    replica: usize,
    seed: u64,
) -> Result<Replica, DipolarError> {
    let energy = p.energy(spins)?;
    Ok(Replica {
        replica,
        seed: read_seed(seed, replica as u64),
        supplied: spins.to_vec(),
        start: spins.to_vec(),
        spins: spins.to_vec(),
        energy: Ok(energy),
        device_energy: energy,
        terminal: spins.to_vec(),
        terminal_energy: Ok(energy),
        terminal_device_energy: energy,
        physical_energy: p.physical_energy(spins),
        feasible: false,
        resolved: false,
        trace: Vec::new(),
        error: None,
    })
}

fn validate_capture(
    p: &Problem,
    starts: &[Vec<i8>],
    begin: &[Vec<i8>],
    schedule: &Schedule,
    seed: u64,
    capture: &DeviceCapture,
) -> Result<(Vec<Replica>, f64), DipolarError> {
    let validation = Instant::now();
    let n = p.nodes();
    let replicas = starts.len();
    let steps = schedule.proposals;
    let mut reads = Vec::with_capacity(replicas);
    for r in 0..replicas {
        if !p.violations(&begin[r])?.feasible() {
            reads.push(unrepaired(p, &starts[r], r, seed)?);
            continue;
        }
        let mut current = begin[r].clone();
        let mut best = (p.energy(&current)?, current.clone());
        let mut trace = Vec::with_capacity(steps);
        let mut error = None;
        for t in 0..steps {
            let slot = r * steps + t;
            let len = usize::try_from(capture.lengths[slot]).unwrap_or(usize::MAX);
            let links: Vec<u32> = capture.moves
                [slot * MAX_LINKS..slot * MAX_LINKS + len.min(MAX_LINKS)]
                .iter()
                .map(|&e| u32::try_from(e).unwrap_or(u32::MAX))
                .collect();
            let spins = capture.raw[slot * n..(slot + 1) * n].to_vec();
            let host_energy = p.energy(&spins);
            let check = check_proposal(
                p,
                &current,
                &links,
                &spins,
                len,
                schedule.max_move,
                capture.accepted[slot],
                capture.raw_scores[slot],
            );
            let proposal_error = check.as_ref().err().cloned();
            if let Err(fault) = check {
                error.get_or_insert(ValidationError {
                    replica: r,
                    proposal: Some(t),
                    error: fault,
                });
            }
            if capture.accepted[slot] != 0 {
                current.clone_from(&spins);
                if let Ok(energy) = host_energy {
                    if energy < best.0 {
                        best = (energy, spins.clone());
                    }
                }
            }
            trace.push(Proposal {
                links,
                spins,
                device_energy: capture.raw_scores[slot],
                host_energy,
                accepted: capture.accepted[slot] != 0,
                error: proposal_error,
            });
        }
        let spins = capture.finals[r * n..(r + 1) * n].to_vec();
        let terminal = capture.terminal[r * n..(r + 1) * n].to_vec();
        let energy = p.energy(&spins);
        let terminal_energy = p.energy(&terminal);
        let check = p
            .checked_final(&spins, capture.scores[r])
            .and_then(|()| p.checked_final(&terminal, capture.terminal_scores[r]))
            .and_then(|()| check_replay(&terminal, &current, &spins, &best));
        if let Err(fault) = check {
            error.get_or_insert(ValidationError {
                replica: r,
                proposal: None,
                error: fault,
            });
        }
        reads.push(Replica {
            replica: r,
            seed: read_seed(seed, r as u64),
            supplied: starts[r].clone(),
            start: begin[r].clone(),
            physical_energy: p.physical_energy(&spins),
            feasible: p.violations(&spins).is_ok_and(Violations::feasible),
            resolved: energy.as_ref().is_ok_and(|&e| {
                (p.energy(&begin[r]).expect("admitted start") - e).unsigned_abs()
                    > p.rounding_bound_milli()
            }),
            spins,
            energy,
            device_energy: capture.scores[r],
            terminal,
            terminal_energy,
            terminal_device_energy: capture.terminal_scores[r],
            trace,
            error,
        });
    }
    Ok((reads, validation.elapsed().as_secs_f64()))
}
fn check_replay(
    terminal: &[i8],
    current: &[i8],
    spins: &[i8],
    best: &(i64, Vec<i8>),
) -> Result<(), DipolarError> {
    if terminal == current && spins == best.1 {
        return Ok(());
    }
    Err(DipolarError::Device(
        "device final states differ from replay".into(),
    ))
}

// Keep malformed output in the capture and prevent unchecked link indexing.
#[allow(clippy::too_many_arguments)] // Flat proposal evidence is checked together.
fn check_proposal(
    p: &Problem,
    current: &[i8],
    links: &[u32],
    spins: &[i8],
    len: usize,
    max_move: usize,
    accepted: i32,
    score: i64,
) -> Result<(), DipolarError> {
    if len > max_move || len > p.nodes() {
        return Err(DipolarError::Move(MoveFault::TooLong));
    }
    if !(0..=1).contains(&accepted) {
        return Err(DipolarError::Device("invalid device decision".into()));
    }
    if !links.is_empty() {
        p.check_move(current, links)?;
    }
    let mut expected = current.to_vec();
    for &e in links {
        expected[p.node(e)] = -expected[p.node(e)];
    }
    if expected != spins || (links.is_empty() && accepted != 0) {
        return Err(DipolarError::Device(
            "device proposal differs from its move".into(),
        ));
    }
    p.checked_final(spins, score)
}

#[derive(Default)]
struct DeviceCapture {
    finals: Vec<i8>,
    scores: Vec<i64>,
    terminal: Vec<i8>,
    terminal_scores: Vec<i64>,
    raw: Vec<i8>,
    raw_scores: Vec<i64>,
    moves: Vec<i32>,
    lengths: Vec<i32>,
    accepted: Vec<i32>,
}

const MAX_CAPTURE_BYTES: usize = 256 * 1024 * 1024;

fn capture_bytes(n: usize, replicas: usize, proposals: usize) -> usize {
    let per_slot = n
        .saturating_mul(2)
        .saturating_add(208)
        .saturating_add(std::mem::size_of::<Proposal>());
    replicas.saturating_mul(
        proposals
            .max(1)
            .saturating_mul(per_slot)
            .saturating_add(std::mem::size_of::<Replica>())
            .saturating_add(n.saturating_mul(6))
            .saturating_add(32),
    )
}

fn device_error(error: impl fmt::Display) -> DipolarError {
    DipolarError::Device(error.to_string())
}

/// Capability identity, available without opening CUDA.
pub const CUDA_DIPOLAR_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cuda,
    algorithm: quip_proto::v1::Algorithm::Dipolar,
    max_nodes: 24,
    max_edges: 276,
    features: &["streaming"],
    adapt: quip_solver_core::adapt::AdaptBounds {
        min_sweeps: 16,
        max_sweeps: 256,
        min_reads: 4,
        max_reads: 16,
        reads_solution_min_factor: 0,
        reads_solution_max_factor: 0,
        reads_solution_floor_factor: 0,
    },
    limits: quip_solver_core::ModelLimits {
        max_degree: Some(23),
        max_local_field_milli: None,
        exact_resolution_milli: Some(1),
    },
};

const CLAIMS: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[DEFECTS_V1],
    only_params: &[],
}];

fn run_batch(
    problem: Problem,
    device: usize,
    starts: &[Vec<i8>],
    schedule: &Schedule,
    seed: u64,
) -> Result<Run, DipolarError> {
    let setup = Instant::now();
    problem.admit(schedule, starts.len(), usize::MAX)?;
    let begin = problem.prepare_starts(starts)?;
    if begin
        .iter()
        .all(|s| problem.violations(s).is_ok_and(|v| !v.feasible()))
    {
        let setup_seconds = setup.elapsed().as_secs_f64();
        let (reads, validation_seconds) = validate_capture(
            &problem,
            starts,
            &begin,
            schedule,
            seed,
            &DeviceCapture::default(),
        )?;
        return Ok(Run {
            seed,
            reads,
            setup_seconds,
            device_seconds: 0.0,
            validation_seconds,
        });
    }
    let repair_seconds = setup.elapsed().as_secs_f64();
    let mut device = DipolarDevice::open(device, problem)?;
    device.setup_seconds += repair_seconds;
    device.run_prepared(starts, &begin, schedule, seed)
}

/// Session sampler. Construction and capability discovery do not open a device.
/// Only validated defects-static revision 1 jobs can reach CUDA.
pub struct DipolarSampler {
    device: usize,
    evidence: RwLock<Result<Evidence, String>>,
}

impl DipolarSampler {
    /// Bind future family jobs to a visible CUDA device index.
    #[must_use]
    pub fn new(device: usize) -> Self {
        Self {
            device,
            evidence: RwLock::new(Ok(Evidence::Analytic)),
        }
    }
}

impl Sampler for DipolarSampler {
    fn sample(
        &self,
        _graph: &IsingGraph,
        _params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        Err(SampleError::Declined(
            "quip-cuda-dipolar accepts only validated quip/defects-static@1 jobs".into(),
        ))
    }

    fn accepts_warm_start() -> bool {
        true
    }
    fn accepts_family_metadata() -> bool {
        true
    }

    // An exact coefficient comparison is the admission contract.
    #[allow(clippy::float_cmp)]
    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let decline = |e: DipolarError| SampleError::Declined(e.to_string());
        if family.family_id() != DEFECTS_V1.family_id || family.revision() != DEFECTS_V1.revision {
            return self.sample(graph, params);
        }
        let evidence = self
            .evidence
            .read()
            .map_err(|_| SampleError::Declined("settings lock poisoned".into()))?
            .clone()
            .map_err(SampleError::Declined)?;
        let setup = Instant::now();
        let problem =
            Problem::for_instance(family.params(), family.instance(), evidence).map_err(decline)?;
        // Compare every coefficient and edge, not only the dimensions.
        if graph.h.len() != problem.nodes()
            || graph.edges.len() != problem.edges.len()
            || !graph
                .h
                .iter()
                .zip(&problem.h)
                .all(|(&a, &b)| a == milli_units(b))
            || !graph
                .j
                .iter()
                .zip(&problem.j)
                .all(|(&a, &b)| a == milli_units(b))
            || graph.j.len() != problem.j.len()
            || !graph
                .edges
                .iter()
                .zip(&problem.edges)
                .all(|(&(a, b), &(c, d))| a == c as usize && b == d as usize)
        {
            return Err(SampleError::Declined(
                "the family instance differs from the job graph or precision".into(),
            ));
        }
        let schedule = Schedule {
            proposals: params.num_sweeps.max(1).saturating_mul(problem.nodes()),
            ..Schedule::default()
        };
        let count = warm.map_or(params.num_reads, |w| w.spins.len());
        problem
            .admit(&schedule, count, usize::MAX)
            .map_err(decline)?;
        let starts = match warm {
            Some(w) => w.spins.clone(),
            None => vec![problem.start_state().map_err(decline)?; count],
        };
        let mut capture =
            run_batch(problem, self.device, &starts, &schedule, params.seed).map_err(decline)?;
        capture.setup_seconds =
            setup.elapsed().as_secs_f64() - capture.device_seconds - capture.validation_seconds;
        // Keep raw states, seeds, errors, and host costs in the session log.
        tracing::info!(policy=DIPOLAR_POLICY, seed=capture.seed, setup_seconds=capture.setup_seconds,
            device_seconds=capture.device_seconds, validation_seconds=capture.validation_seconds,
            reads=?capture.reads, "dipolar batch capture");
        let mut results = Vec::with_capacity(capture.reads.len());
        for read in capture.reads {
            if let Some(error) = read.error {
                return Err(SampleError::Declined(format!(
                    "replica {} proposal {:?}: {}",
                    error.replica, error.proposal, error.error
                )));
            }
            let canonical =
                quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges);
            if Ok(canonical) != read.energy {
                return Err(SampleError::Declined(format!(
                    "replica {} canonical score differs",
                    read.replica
                )));
            }
            results.push(SamplerResult {
                spins: read.spins,
                energy_milli: read.energy.map_err(decline)?,
            });
        }
        Ok(results)
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: "quip-cuda-dipolar/1",
            generated: CLAIMS,
            accepted: CLAIMS,
        }
    }

    fn apply_config(&self, backend_toml: &str) {
        #[derive(serde::Deserialize, Default)]
        struct Config {
            evidence: Option<String>,
            #[serde(flatten)]
            common: BTreeMap<String, toml::Value>,
        }
        let parsed = toml::from_str::<Config>(backend_toml)
            .map_err(|e| e.to_string())
            .and_then(|c| {
                quip_solver_core::config::warn_unknown_fields(
                    "cuda-dipolar",
                    c.common.keys().filter(|k| k.as_str() != "num_sweeps"),
                );
                match c.evidence.as_deref() {
                    None | Some("analytic") => Ok(Evidence::Analytic),
                    Some("physical") => Ok(Evidence::Physical),
                    Some(_) => Err("evidence must be analytic or physical".into()),
                }
            });
        if let Ok(mut current) = self.evidence.write() {
            *current = parsed;
        }
    }
}

// All admitted values are below 2^53 by the coefficient admission bound.
#[allow(clippy::cast_precision_loss)]
fn milli_units(value: i64) -> f64 {
    value as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> Problem {
        let mut labels = vec![None; 9];
        labels[4] = Some(0);
        Problem::new(
            Request {
                side: 2,
                labels,
                positions: (0..12).collect(),
                obstacles: vec![4],
                pins: vec![0],
                field_x: 0,
                field_y: 0,
            },
            Evidence::Analytic,
        )
        .unwrap()
    }

    #[test]
    fn admission_checks_kernel_arrays_independently_of_side() {
        let schedule = Schedule::default();
        let mut p = problem();
        assert!(p.request.side <= MAX_TESTED_SIDE);
        p.h.resize(25, 0);
        assert!(matches!(
            p.admit(&schedule, 1, usize::MAX),
            Err(DipolarError::Request(_))
        ));

        let mut p = problem();
        p.lattice = SquareLattice::new(4, Boundary::Open).unwrap();
        assert!(p.request.side <= MAX_TESTED_SIDE);
        assert_eq!(p.nodes(), 12);
        assert!(matches!(
            p.admit(&schedule, 1, usize::MAX),
            Err(DipolarError::Request(_))
        ));
    }

    #[test]
    fn replay_and_proposal_mismatches_are_device_errors() {
        let p = problem();
        let start = p.start_state().unwrap();
        let mut other = start.clone();
        other[0] = -other[0];
        let best = (p.energy(&start).unwrap(), start.clone());
        for (terminal, spins) in [(&other, &start), (&start, &other)] {
            assert!(matches!(
                check_replay(terminal, &start, spins, &best),
                Err(DipolarError::Device(_))
            ));
        }
        for (spins, accepted) in [(&start, 2), (&other, 0), (&start, 1)] {
            assert!(matches!(
                check_proposal(&p, &start, &[], spins, 0, 24, accepted, best.0),
                Err(DipolarError::Device(_))
            ));
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn an_unrepairable_start_keeps_the_other_gpu_replicas() {
        let mut p = problem();
        let good = p.start_state().unwrap();
        // Block every repair path at the centre to exercise CPU-13:1197-1200.
        // Revision-1 generation only pins boundary links, so use a host fixture.
        for &(e, _) in p.lattice.incident(4) {
            p.pinned[e as usize] = true;
        }
        let mut bad = good.clone();
        let e = p.lattice.incident(4)[0].0;
        bad[p.node(e)] *= -1;
        assert!(matches!(p.repair(&bad), Err(DipolarError::Infeasible(_))));
        let starts = vec![good.clone(), bad.clone(), good];
        let mut device = DipolarDevice::open(0, p).unwrap();
        let run = device.run(&starts, &Schedule::default(), 7).unwrap();
        assert_eq!(run.reads.len(), 3);
        for r in [0, 2] {
            assert!(run.reads[r].error.is_none());
            assert_eq!(run.reads[r].seed, read_seed(7, r as u64));
            assert_eq!(run.reads[r].trace.len(), 256);
        }
        let read = &run.reads[1];
        assert_eq!(read.spins, bad);
        assert_eq!(read.terminal, bad);
        assert_eq!(read.start, bad);
        assert_eq!(read.supplied, bad);
        assert!(read.trace.is_empty());
        assert!(!read.resolved);
        assert!(!read.feasible);
        assert!(read.error.is_none());
        assert_eq!(read.energy, device.problem().energy(&bad));
        assert_eq!(read.physical_energy, device.problem().physical_energy(&bad));
    }

    #[test]
    fn warm_start_preparation_finishes_before_opening_device() {
        let mut p = problem();
        let mut bad = p.start_state().unwrap();
        for &(e, _) in p.lattice.incident(4) {
            p.pinned[e as usize] = true;
        }
        let e = p.lattice.incident(4)[0].0;
        bad[p.node(e)] *= -1;
        let run = run_batch(p, usize::MAX, &[bad.clone()], &Schedule::default(), 7).unwrap();
        assert_eq!(run.reads.len(), 1);
        assert_eq!(run.reads[0].spins, bad);
        assert!(!run.reads[0].feasible);
        assert!(run.reads[0].trace.is_empty());
        assert!(run.reads[0].error.is_none());
        assert_eq!(run.device_seconds.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            run_batch(
                problem(),
                usize::MAX,
                &[vec![0; 12]],
                &Schedule::default(),
                7
            )
            .unwrap_err(),
            DipolarError::Spins
        );
    }

    #[test]
    fn shared_configuration_keeps_evidence_and_physical_admission() {
        let sampler = DipolarSampler::new(0);
        sampler.apply_config("num_sweeps = 64\nevidence = 'physical'");
        assert_eq!(*sampler.evidence.read().unwrap(), Ok(Evidence::Physical));
        assert_eq!(
            Problem::new(problem().request, Evidence::Physical).err(),
            Some(DipolarError::Uncalibrated)
        );
        sampler.apply_config("num_sweeps = 64");
        assert_eq!(*sampler.evidence.read().unwrap(), Ok(Evidence::Analytic));
        sampler.apply_config("evidence = 'unknown'");
        assert!(sampler.evidence.read().unwrap().is_err());
    }

    #[test]
    fn corrupt_device_output_keeps_every_state_and_replica_failure() {
        let p = problem();
        let start = p.start_state().unwrap();
        let score = p.energy(&start).unwrap();
        // Valid first replica, malformed second replica. Both must survive capture.
        let mut states = [start.clone(), start.clone()].concat();
        states[12] = 0;
        let capture = DeviceCapture {
            finals: states.clone(),
            scores: vec![score, score],
            terminal: states.clone(),
            terminal_scores: vec![score, score],
            raw: states.clone(),
            raw_scores: vec![score, score],
            moves: vec![-1; 2 * MAX_LINKS],
            lengths: vec![0, 0],
            accepted: vec![0, 0],
        };
        let starts = vec![start; 2];
        let (reads, _) = validate_capture(
            &p,
            &starts,
            &starts,
            &Schedule {
                proposals: 1,
                ..Schedule::default()
            },
            2,
            &capture,
        )
        .unwrap();
        assert_eq!(reads.len(), 2);
        assert!(reads[0].error.is_none());
        let error = reads[1].error.as_ref().unwrap();
        assert_eq!(error.replica, 1);
        assert_eq!(error.proposal, Some(0));
        assert_eq!(reads[1].trace[0].spins, states[12..]);
        assert_eq!(reads[1].spins, states[12..]);
        assert_eq!(reads[1].terminal, states[12..]);
        assert_eq!(reads[1].energy, Err(DipolarError::Spins));
        assert_eq!(reads[1].trace[0].host_energy, Err(DipolarError::Spins));
        assert!(reads[1].trace[0].error.is_some());
    }
}
