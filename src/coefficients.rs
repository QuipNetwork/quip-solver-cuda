//! Which kernel variant solves a job with its exact coefficients, and at what
//! integer scale, or why none does.
//!
//! The kernels hold integers. A job is exact on a path when every `h` and `J`,
//! in milli, is a multiple of one scale and the scaled values fit the path's
//! types. The scale is the GCD of the nonzero values, so a ±1 mining job is
//! scale 1000 and uploads the same bytes as before this module existed. The
//! beta schedule is multiplied by `scale_milli / 1000`, which keeps
//! `beta * delta` in the units the schedule was built for.
//!
//! A job no path holds is refused, never quantized: the caller turns the
//! `Err` into a Capacity reject, so the coordinator can route it to a CPU
//! solver.

use crate::KernelKind;
use quip_solver_core::IsingGraph;

/// Largest |delta| the int8 SA workspace holds: `2 * field <= 127`.
pub const INT8_MAX_DELTA: i64 = 127;

/// Largest |coefficient| an int8 slot holds.
const INT8_MAX_COEF: i64 = 127;

/// Largest local field, in scale units, the wide SA path holds. The neighbour
/// update in `kernels/sa.cu` adds `4 * J`, and `|J| <= field`, so
/// `4 * field <= i32::MAX` keeps every intermediate in `int`.
pub const WIDE_MAX_FIELD: i64 = 536_870_911;

/// Largest local field, in scale units, the weighted MSA lanes hold exactly.
pub const MSA_WEIGHTED_MAX_FIELD: i64 = (1 << 24) - 1;

/// Largest local field, in scale units, the int8 Gibbs variants sum exactly.
/// They add coefficients into a `float` field, and every partial sum stays a
/// whole number below 2^24, where f32 still holds each integer.
pub const GIBBS_FLOAT_EXACT_FIELD: i64 = (1 << 24) - 1;

/// Relative bound on the rounding error of `milli / 1000 * 1000`, scaled by
/// the rounded milli magnitude with a floor of one.
const MILLI_TOLERANCE: f64 = 4.0 * f64::EPSILON;

/// Which compiled kernel runs the job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoefficientPath {
    /// Today's kernels: `signed char` coefficients (and an int8 SA delta).
    Int8,
    /// i32 coefficients: wide SA delta, Gibbs field, or weighted MSA arithmetic.
    Wide,
    /// Scaled i8 coefficients with wide SA deltas or a weighted MSA lane.
    WeightedInt8,
    /// Real f32 coefficients, never selected for mining.
    Float32,
}

/// The path and integer scale one job runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoefficientPlan {
    /// The kernel variant.
    pub path: CoefficientPath,
    /// Milli per kernel integer unit. Every coefficient is a multiple of it.
    pub scale_milli: u64,
}

impl CoefficientPlan {
    /// Factor applied to every beta, so the kernel's integer delta (in units
    /// of `scale_milli` milli) sees the Boltzmann weights of the model in
    /// units. Exactly `1.0` at scale 1000.
    #[must_use]
    pub fn beta_factor(self) -> f64 {
        #[allow(clippy::cast_precision_loss)] // scale_milli <= i32::MAX, exact in f64
        let scale = self.scale_milli as f64;
        scale / 1000.0
    }
}

/// A coefficient in units as integer milli, or `None` when it is non-finite,
/// off the milli grid, or outside the `i32` wire range.
#[must_use]
pub fn to_milli(v: f64) -> Option<i64> {
    let x = v * 1000.0;
    let m = x.round();
    if !m.is_finite()
        || (v != 0.0 && m == 0.0)
        || (x - m).abs() > MILLI_TOLERANCE * m.abs().max(1.0)
        || m.abs() > f64::from(i32::MAX)
    {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)] // checked against i32::MAX above
    Some(m as i64)
}

fn milli_all(name: &str, values: &[f64]) -> Result<Vec<i64>, String> {
    values
        .iter()
        .map(|&v| {
            to_milli(v).ok_or_else(|| {
                format!("{name} coefficient {v} is not a whole milli value in the i32 wire range")
            })
        })
        .collect()
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Literal mining coefficients, whose integer scale is already known.
pub(crate) fn is_unit_mining(graph: &IsingGraph) -> bool {
    graph.j.len() == graph.edges.len()
        && graph.h.iter().all(|&v| {
            let magnitude = v.abs().to_bits();
            magnitude == 0 || magnitude == 1.0_f64.to_bits()
        })
        && graph
            .j
            .iter()
            .all(|&v| v.abs().to_bits() == 1.0_f64.to_bits())
}

/// `max_i(|h_i| + Σ_j |J_ij|)` in milli, each endpoint of every edge counted,
/// as `quip_solver_core::ModelLimits::check` counts it.
fn max_local_field(h: &[i64], j: &[i64], edges: &[(usize, usize)]) -> i64 {
    let mut field: Vec<i64> = h.iter().map(|v| v.abs()).collect();
    for (&(u, v), &jv) in edges.iter().zip(j) {
        for node in [u, v] {
            if let Some(f) = field.get_mut(node) {
                *f += jv.abs();
            }
        }
    }
    field.into_iter().max().unwrap_or(0)
}

/// Classify one job for `kind`.
///
/// # Errors
///
/// A message naming why no path holds the job exactly. Every refusal is
/// permanent for this job on this binary.
pub fn plan(kind: KernelKind, graph: &IsingGraph) -> Result<CoefficientPlan, String> {
    plan_with_unit(kind, graph).map(|(plan, _)| plan)
}

/// Preserve the literal-unit classification for the same job's beta resolution.
pub(crate) fn plan_with_unit(
    kind: KernelKind,
    graph: &IsingGraph,
) -> Result<(CoefficientPlan, bool), String> {
    let unit_mining = is_unit_mining(graph);
    // Unit Gibbs coefficients fit int8, and unit MSA uses its original kernel.
    // SA still needs its local-field bound, even with unit coefficients, and
    // so does Gibbs once a unit field, at most 1 + edges, can leave float range.
    let unit_field_exact =
        i64::try_from(graph.edges.len()).is_ok_and(|edges| edges < GIBBS_FLOAT_EXACT_FIELD);
    if unit_mining && (kind == KernelKind::Msa || (kind == KernelKind::Gibbs && unit_field_exact)) {
        return Ok((
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1000,
            },
            true,
        ));
    }
    let h = milli_all("h", &graph.h)?;
    let j = milli_all("J", &graph.j)?;

    let g = h
        .iter()
        .chain(&j)
        .fold(0, |acc, v| gcd(acc, v.unsigned_abs()));
    let scale_milli = if g == 0 { 1000 } else { g };
    let s = i64::try_from(scale_milli).unwrap_or(i64::MAX);
    let max_coef = h.iter().chain(&j).map(|v| v.abs() / s).max().unwrap_or(0);
    let field = max_local_field(&h, &j, &graph.edges) / s;
    let int8 = CoefficientPlan {
        path: CoefficientPath::Int8,
        scale_milli,
    };

    let plan = match kind {
        KernelKind::Msa => {
            let unit_h = h.iter().all(|v| matches!(v, -1000 | 0 | 1000));
            let unit_j = j.iter().all(|v| v.abs() == 1000);
            if unit_h && unit_j {
                Ok(CoefficientPlan {
                    path: CoefficientPath::Int8,
                    scale_milli: 1000,
                })
            } else if field > MSA_WEIGHTED_MAX_FIELD {
                Err(format!(
                    "msa: local field {field} times the model's {scale_milli} milli step exceeds the weighted path's limit of {MSA_WEIGHTED_MAX_FIELD}"
                ))
            } else {
                Ok(CoefficientPlan {
                    path: if max_coef <= INT8_MAX_COEF { CoefficientPath::WeightedInt8 } else { CoefficientPath::Wide },
                    scale_milli,
                })
            }
        }
        // Int8 Gibbs sums its field in float; past 2^24 it takes the integer wide path.
        KernelKind::Gibbs if max_coef <= INT8_MAX_COEF && field <= GIBBS_FLOAT_EXACT_FIELD => {
            Ok(int8)
        }
        KernelKind::Gibbs if field > WIDE_MAX_FIELD => Err(format!(
            "gibbs: local field {field} times the model's {scale_milli} milli step exceeds the wide path's limit of {WIDE_MAX_FIELD}"
        )),
        KernelKind::Gibbs => Ok(CoefficientPlan { path: CoefficientPath::Wide, scale_milli }),
        KernelKind::Sa if 2 * field <= INT8_MAX_DELTA => Ok(int8),
        KernelKind::Sa if field > WIDE_MAX_FIELD => Err(format!(
            "sa: local field {field} times the model's {scale_milli} milli step exceeds the wide path's limit of {WIDE_MAX_FIELD}"
        )),
        KernelKind::Sa => Ok(CoefficientPlan { path: if max_coef <= INT8_MAX_COEF { CoefficientPath::WeightedInt8 } else { CoefficientPath::Wide }, scale_milli }),
    };
    plan.map(|plan| (plan, unit_mining))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelKind;
    use quip_solver_core::IsingGraph;

    #[test]
    fn gibbs_plans_wide_for_wide_coefficients() {
        let g = graph(vec![0.0; 3], vec![1.0, 0.001], vec![(0, 1), (1, 2)]);
        assert_eq!(
            plan(KernelKind::Gibbs, &g).unwrap().path,
            CoefficientPath::Wide
        );
    }

    #[test]
    fn gibbs_mining_refuses_off_grid() {
        let g = graph(vec![0.0001, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(plan(KernelKind::Gibbs, &g).is_err());
    }

    #[test]
    fn gibbs_wide_checks_local_field_bound() {
        let fits = f64::from(i32::try_from(WIDE_MAX_FIELD).unwrap()) / 1000.0 - 0.001;
        let g = graph(vec![fits, 0.0], vec![0.001], vec![(0, 1)]);
        assert_eq!(
            plan(KernelKind::Gibbs, &g).unwrap().path,
            CoefficientPath::Wide
        );
        let g = graph(vec![fits + 0.001, 0.0], vec![0.001], vec![(0, 1)]);
        assert!(plan(KernelKind::Gibbs, &g).is_err());
    }

    /// A star whose hub sums one +1 and `leaves - 1` alternating ±126 couplings
    /// (milli step 1, every coefficient within i8) to leaves with h = -127.
    fn star(leaves: usize, j_milli: impl Fn(usize) -> i32) -> IsingGraph {
        let edges = (1..=leaves).map(|leaf| (0, leaf)).collect();
        let j = (0..leaves)
            .map(|i| f64::from(j_milli(i)) / 1000.0)
            .collect();
        let mut h = vec![-0.127; leaves + 1];
        h[0] = 0.0;
        graph(h, j, edges)
    }

    fn alternating(i: usize) -> i32 {
        match i {
            0 => 1,
            i if i % 2 == 1 => 126,
            _ => -126,
        }
    }

    #[test]
    fn gibbs_routes_int8_rows_past_float_exactness_to_wide() {
        // 126 * 133,152 + 1 = 16,777,153 < 2^24: float sums stay exact.
        let below = plan(KernelKind::Gibbs, &star(133_153, alternating)).unwrap();
        assert_eq!(below.path, CoefficientPath::Int8);
        // 126 * 133,153 + 1 = 16,777,279 >= 2^24: the hub needs integer sums.
        let above = plan(KernelKind::Gibbs, &star(133_154, alternating)).unwrap();
        assert_eq!(above.path, CoefficientPath::Wide);
        assert_eq!(above.scale_milli, 1);
        // The review's case: 280,001 leaves, a row sum of 35,280,001.
        let review = plan(KernelKind::Gibbs, &star(280_001, alternating)).unwrap();
        assert_eq!(review.path, CoefficientPath::Wide);
        // Admission sizes the rerouted model with the Int32 footprint.
        assert_eq!(
            crate::capacity::Precision::from_plan(review, false),
            crate::capacity::Precision::Int32
        );
    }

    #[test]
    fn gibbs_routes_unit_rows_past_float_exactness_to_wide() {
        let unit = |leaves| {
            let edges = (1..=leaves).map(|leaf| (0, leaf)).collect();
            graph(vec![0.0; leaves + 1], vec![1.0; leaves], edges)
        };
        // Hub field 2^24 - 1 stays on the mining int8 path; 2^24 does not.
        let below = plan(KernelKind::Gibbs, &unit((1 << 24) - 1)).unwrap();
        assert_eq!(below.path, CoefficientPath::Int8);
        let above = plan(KernelKind::Gibbs, &unit(1 << 24)).unwrap();
        assert_eq!(
            above,
            CoefficientPlan {
                path: CoefficientPath::Wide,
                scale_milli: 1000
            }
        );
    }

    #[test]
    fn sa_plans_weighted_int8_when_field_exceeds_i8_delta() {
        // Unit J on K_70: every coefficient fits i8, and 2 * 69 > INT8_MAX_DELTA.
        let g = complete(70, 0.0);
        assert_eq!(
            plan(KernelKind::Sa, &g).unwrap().path,
            CoefficientPath::WeightedInt8
        );
    }

    #[test]
    fn sa_plans_wide_only_for_wide_coefficients() {
        // gcd 1 milli, so J = 1.0 is 1,000 steps: above the i8 limit.
        let g = graph(vec![0.0; 3], vec![1.0, 0.001], vec![(0, 1), (1, 2)]);
        assert_eq!(
            plan(KernelKind::Sa, &g).unwrap().path,
            CoefficientPath::Wide
        );
    }

    #[test]
    fn sa_mining_refuses_off_grid() {
        let g = graph(vec![0.0001, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(plan(KernelKind::Sa, &g).is_err());
    }

    fn graph(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> IsingGraph {
        IsingGraph::new(h, j, edges)
    }

    /// `K_n` with all J = +1 and every h = `h`.
    fn complete(n: usize, h: f64) -> IsingGraph {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        let j = vec![1.0; edges.len()];
        graph(vec![h; n], j, edges)
    }

    #[test]
    fn int8_range_model_plans_weighted_int8() {
        let g = graph(
            vec![0.016, -0.992, 0.0],
            vec![0.032, -0.976],
            vec![(0, 1), (1, 2)],
        );
        let p = plan(KernelKind::Msa, &g).unwrap();
        assert_eq!(p.path, CoefficientPath::WeightedInt8);
        assert_eq!(p.scale_milli, 16);
    }

    #[test]
    fn int8_overflow_plans_wide() {
        let g = graph(vec![0.001, 0.0], vec![1.0], vec![(0, 1)]);
        let p = plan(KernelKind::Msa, &g).unwrap();
        assert_eq!(p.path, CoefficientPath::Wide);
        assert_eq!(p.scale_milli, 1);
    }

    #[test]
    fn binary_model_plan_unchanged() {
        let g = graph(vec![-1.0, 0.0, 1.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        assert_eq!(
            plan(KernelKind::Msa, &g).unwrap(),
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1000,
            }
        );
    }

    #[test]
    fn mining_model_is_int8_at_unit_scale() {
        let g = graph(vec![1.0, -1.0, 0.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
            let p = plan(kind, &g).expect("mining model plans");
            assert_eq!(
                p,
                CoefficientPlan {
                    path: CoefficientPath::Int8,
                    scale_milli: 1000
                }
            );
            assert_eq!(p.beta_factor().to_bits(), 1.0_f64.to_bits(), "{kind:?}");
        }
    }

    #[test]
    fn all_zero_model_is_int8_at_unit_scale() {
        let g = graph(vec![0.0; 3], vec![0.0, 0.0], vec![(0, 1), (1, 2)]);
        let p = plan(KernelKind::Sa, &g).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1000
            }
        );
    }

    #[test]
    fn quarter_unit_model_scales_to_250_milli() {
        let g = graph(vec![0.25, -0.75], vec![0.5], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 250
            }
        );
        assert!((p.beta_factor() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn thousand_x_mining_model_scales_but_stays_int8() {
        let g = graph(vec![1000.0, -1000.0], vec![-1000.0], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1_000_000
            }
        );
    }

    #[test]
    fn sa_int8_holds_k64_without_field_and_k65_needs_wide_delta() {
        // K_64, h = 0: field 63, delta 126 <= 127.
        assert_eq!(
            plan(KernelKind::Sa, &complete(64, 0.0)).unwrap().path,
            CoefficientPath::Int8
        );
        // K_64, h = 1: field 64, delta 128 > 127.
        assert_eq!(
            plan(KernelKind::Sa, &complete(64, 1.0)).unwrap().path,
            CoefficientPath::WeightedInt8
        );
        // K_65, h = 0: field 64.
        assert_eq!(
            plan(KernelKind::Sa, &complete(65, 0.0)).unwrap().path,
            CoefficientPath::WeightedInt8
        );
        assert_eq!(
            plan(KernelKind::Sa, &complete(200, 0.0)).unwrap().path,
            CoefficientPath::WeightedInt8
        );
        assert_eq!(
            plan(KernelKind::Sa, &complete(16, 1.0)).unwrap().path,
            CoefficientPath::Int8
        );
    }

    #[test]
    fn sa_wide_planning_leaves_node_admission_to_the_session() {
        let g = graph(vec![1.001; 20_000], vec![1.0], vec![(0, 1)]);
        assert_eq!(
            plan(KernelKind::Sa, &g).unwrap().path,
            CoefficientPath::Wide
        );
    }

    #[test]
    fn sa_wide_bound_is_a_quarter_of_i32() {
        // One milli step (scale 1), field exactly WIDE_MAX_FIELD: accepted.
        #[allow(clippy::cast_precision_loss)] // WIDE_MAX_FIELD < i32::MAX, exact in f64
        let fits = WIDE_MAX_FIELD as f64 / 1000.0;
        let g = graph(vec![fits - 0.001, 0.0], vec![0.001], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Wide,
                scale_milli: 1
            }
        );
        // One milli more: refused.
        let g = graph(vec![fits, 0.0], vec![0.001], vec![(0, 1)]);
        assert!(plan(KernelKind::Sa, &g).is_err());
    }

    #[test]
    fn gibbs_uses_int8_at_127_scale_units_and_wide_at_128() {
        let ok = graph(vec![127.0, 0.0], vec![1.0], vec![(0, 1)]);
        assert_eq!(plan(KernelKind::Gibbs, &ok).unwrap().scale_milli, 1000);
        assert_eq!(
            plan(KernelKind::Gibbs, &ok).unwrap().path,
            CoefficientPath::Int8
        );
        let wide = graph(vec![128.0, 0.0], vec![1.0], vec![(0, 1)]);
        assert_eq!(
            plan(KernelKind::Gibbs, &wide).unwrap().path,
            CoefficientPath::Wide
        );
        // 128 and 2 share the factor 2: scale 2000, values 64 and 1.
        let scaled = graph(vec![128.0, 0.0], vec![2.0], vec![(0, 1)]);
        assert_eq!(plan(KernelKind::Gibbs, &scaled).unwrap().scale_milli, 2000);
    }

    #[test]
    fn msa_unit_models_stay_on_the_unit_kernel() {
        let g = IsingGraph::new(vec![1.0, 0.0, -1.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        let p = plan(KernelKind::Msa, &g).unwrap();
        assert_eq!((p.path, p.scale_milli), (CoefficientPath::Int8, 1000));
    }

    #[test]
    fn msa_weighted_models_take_the_weighted_path_at_their_scale() {
        // The conformance driver's session fixture.
        let g = IsingGraph::new(
            vec![1.0, -1.0, 0.25],
            vec![0.5, -0.75],
            vec![(0, 1), (1, 2)],
        );
        let p = plan(KernelKind::Msa, &g).unwrap();
        assert_eq!(
            (p.path, p.scale_milli),
            (CoefficientPath::WeightedInt8, 250)
        );
    }

    #[test]
    fn msa_weighted_field_bound_is_two_to_the_24_minus_one() {
        // The 0.001 coupling pins the GCD scale at 1 milli. Node 1 touches both
        // edges, so its field is the sum of the two magnitudes in milli.
        let edges = vec![(0, 1), (1, 2)];
        let fits = IsingGraph::new(vec![0.0; 3], vec![16_777.214, 0.001], edges.clone());
        assert_eq!(
            plan(KernelKind::Msa, &fits).unwrap().path,
            CoefficientPath::Wide
        );
        let over = IsingGraph::new(vec![0.0; 3], vec![16_777.215, 0.001], edges);
        assert!(plan(KernelKind::Msa, &over).is_err());
    }

    #[test]
    fn msa_weighted_planning_leaves_node_admission_to_the_session() {
        let g = IsingGraph::new(vec![0.5, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(plan(KernelKind::Msa, &g).is_ok());
    }

    #[test]
    fn non_finite_and_sub_milli_are_refused() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0005,
            1e-7,
            1e-10,
            1.000_000_000_5,
            -2_147_483.648,
        ] {
            let g = graph(vec![bad, 0.0], vec![1.0], vec![(0, 1)]);
            for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
                assert!(plan(kind, &g).is_err(), "{kind:?} accepted {bad}");
            }
        }
    }

    #[test]
    fn to_milli_refuses_tiny_off_grid_values() {
        assert_eq!(to_milli(1e-10), None);
    }

    #[test]
    fn nonzero_values_rounding_to_zero_are_refused_by_all_paths() {
        use crate::topology::{fill_coefficients, SelfFeedingTopology};

        for value in [1e-19, -1e-19, f64::MIN_POSITIVE, -f64::MIN_POSITIVE] {
            assert_eq!(to_milli(value), None, "accepted {value}");
            for g in [
                graph(vec![value, 0.0], vec![1.0], vec![(0, 1)]),
                graph(vec![0.0, 0.0], vec![value], vec![(0, 1)]),
            ] {
                for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
                    assert!(plan(kind, &g).is_err(), "{kind:?} accepted {value}");
                }
                let topology = SelfFeedingTopology::build(&g);
                assert!(fill_coefficients::<i8>(&topology, &g, 1000).is_err());
                assert!(fill_coefficients::<i32>(&topology, &g, 1000).is_err());
            }
        }
    }

    #[test]
    fn to_milli_accepts_exact_signed_zero() {
        for zero in [0.0, -0.0] {
            assert_eq!(to_milli(zero), Some(0));
            let g = graph(vec![zero, zero], vec![1.0], vec![(0, 1)]);
            for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
                assert!(plan(kind, &g).is_ok());
            }
            let topology = crate::topology::SelfFeedingTopology::build(&g);
            assert_eq!(
                crate::topology::fill_coefficients::<i8>(&topology, &g, 1000)
                    .unwrap()
                    .1,
                [0, 0]
            );
            assert_eq!(
                crate::topology::fill_coefficients::<i32>(&topology, &g, 1000)
                    .unwrap()
                    .1,
                [0, 0]
            );
        }
    }

    #[test]
    fn to_milli_refuses_off_grid_values_near_one() {
        assert_eq!(to_milli(1.000_000_000_5), None);
    }

    #[test]
    fn to_milli_round_trips_the_wire_range() {
        for m in [
            i64::from(i32::MAX),
            -i64::from(i32::MAX),
            1,
            -1,
            0,
            999,
            123_456_789,
        ] {
            #[allow(clippy::cast_precision_loss)] // |m| <= i32::MAX is exact in f64
            let v = m as f64 / 1000.0;
            assert_eq!(to_milli(v), Some(m));
        }
    }

    #[test]
    fn field_counts_both_endpoints_like_the_sdk() {
        // A self-loop counts twice on its node, as ModelLimits::check does.
        // 32 self-loop terms of J=2 give field 128 at scale 2000 -> 64 units, delta 128 > 127.
        let g = graph(vec![0.0], vec![2.0; 32], vec![(0, 0); 32]);
        assert_eq!(
            plan(KernelKind::Sa, &g).unwrap().path,
            CoefficientPath::WeightedInt8
        );
        // A duplicate edge counts once per copy on each endpoint.
        let g = graph(vec![0.0, 0.0], vec![1.0; 64], vec![(0, 1); 64]);
        assert_eq!(
            plan(KernelKind::Sa, &g).unwrap().path,
            CoefficientPath::WeightedInt8
        );
    }
}
