//! RTX A4000 measured two-point slowdown per lane-update.
//! MSA: 128 reads, 48 nonces. SA: 256 reads, 48 nonces. Gibbs: 256 reads, 12 nonces.
//! Generated from bench-evidence/{msa,sa-gibbs}-vram-dense/summary.json.
//! SA and Gibbs entries are measured unless marked `projected`: a projected entry
//! sits at the probed limit, past the timed range (SA sparse 200,000 nodes, Gibbs
//! sparse 1,000,000, complete 4,000), and extends the last measured value of its series.

use crate::capacity::{Connectivity, Precision};
use crate::kernel::KernelKind;

#[allow(clippy::too_many_lines)] // measured series remain together in the generated table
fn samples(
    kernel: KernelKind,
    connectivity: Connectivity,
    precision: Precision,
) -> &'static [(usize, f64)] {
    match (kernel, connectivity, precision) {
        (KernelKind::Msa, Connectivity::Degree(20), Precision::Binary) => &[
            (5_791, 1.000_000_000_000),
            (16_000, 2.210_912_414_646),
            (64_000, 3.163_381_012_987),
            (200_000, 3.647_952_084_990),
            (2_145_598, 4.185_760_545_961),
        ],
        (KernelKind::Msa, Connectivity::Degree(20), Precision::Int8) => &[
            (5_791, 5.008_769_788_386),
            (16_000, 5.041_069_479_539),
            (64_000, 6.339_800_212_650),
            (200_000, 7.792_662_446_518),
            (2_145_512, 5.108_729_917_414),
        ],
        (KernelKind::Msa, Connectivity::Degree(20), Precision::Float32) => &[
            (5_791, 6.958_701_235_188),
            (16_000, 6.740_225_094_186),
            (64_000, 10.358_642_173_314),
            (200_000, 10.527_126_846_424),
            (869_715, 9.085_633_129_845),
        ],
        (KernelKind::Msa, Connectivity::Degree(60), Precision::Binary) => &[
            (5_791, 3.364_336_720_522),
            (16_000, 9.482_254_708_919),
            (64_000, 14.990_394_612_712),
            (1_096_198, 16.596_773_801_471),
        ],
        (KernelKind::Msa, Connectivity::Degree(60), Precision::Int8) => &[
            (5_791, 22.787_440_858_065),
            (16_000, 56.543_222_698_979),
            (64_000, 68.360_957_739_188),
            (1_096_153, 76.851_811_069_470),
        ],
        (KernelKind::Msa, Connectivity::Degree(60), Precision::Float32) => &[
            (5_791, 25.856_358_881_883),
            (16_000, 61.359_668_595_200),
            (64_000, 84.232_219_311_556),
            (345_027, 94.017_776_481_485),
        ],
        (KernelKind::Msa, Connectivity::Complete, Precision::Binary) => &[
            (64, 32.193_054_036_963),
            (256, 29.398_131_442_710),
            (1_000, 107.626_797_343_213),
            (2_000, 193.676_652_328_465),
            (4_000, 356.183_686_448_255),
            (8_000, 2_589.758_487_784_185),
            (9_595, 2_981.754_423_246_714),
        ],
        (KernelKind::Msa, Connectivity::Complete, Precision::Int8) => &[
            (64, 47.836_426_559_741),
            (256, 86.831_723_107_656),
            (1_000, 298.495_516_544_469),
            (2_000, 576.507_337_177_002),
            (4_000, 1_191.485_116_204_477),
            (8_000, 4_505.257_227_161_316),
            (9_595, 5_430.550_706_666_329),
        ],
        (KernelKind::Msa, Connectivity::Complete, Precision::Int32) => {
            &[(2_000, 675.571_402_904_511), (4_800, 1_887.653_415_218_227)]
        }
        (KernelKind::Msa, Connectivity::Complete, Precision::Float32) => &[
            (64, 36.887_692_984_998),
            (256, 67.242_381_497_459),
            (1_000, 232.183_869_321_150),
            (2_000, 484.137_525_389_070),
            (4_000, 950.505_250_894_777),
            (4_800, 1_382.427_432_912_693),
        ],
        (KernelKind::Sa, Connectivity::Degree(20), Precision::Binary) => &[
            (5_000, 1.0),
            (16_000, 0.386_471_224_144),
            (64_000, 0.395_014_021_601),
            (200_000, 0.446_364_527_795),
            (410_747, 0.446_364_527_795), // projected: flat from 200,000
        ],
        (KernelKind::Sa, Connectivity::Degree(20), Precision::Int8) => &[
            (5_000, 0.655_747_877_441),
            (16_000, 0.239_440_536_576),
            (64_000, 0.237_632_468_138),
            (191_788, 0.232_901_162_759),
        ],
        (KernelKind::Sa, Connectivity::Degree(20), Precision::Int32) => &[
            (5_000, 0.640_442_424_594),
            (16_000, 0.229_475_033_364),
            (64_000, 0.228_284_463_881),
            (169_546, 0.228_166_095_905),
        ],
        (KernelKind::Sa, Connectivity::Degree(20), Precision::Float32) => &[
            (5_000, 0.456_867_954_474),
            (16_000, 0.186_418_222_92),
            (64_000, 0.171_253_913_102),
            (169_553, 0.160_918_541_931),
        ],
        (KernelKind::Sa, Connectivity::Degree(60), Precision::Binary) => &[
            (5_000, 2.583_937_757_581),
            (16_000, 0.871_292_199_189),
            (64_000, 0.922_388_323_603),
            (200_000, 0.993_857_222_393),
            (347_112, 0.993_857_222_393), // projected: flat from 200,000
        ],
        (KernelKind::Sa, Connectivity::Degree(60), Precision::Int8) => &[
            (5_000, 1.947_447_130_951),
            (16_000, 1.113_933_161_427),
            (64_000, 1.298_547_995_335),
            (176_665, 1.305_577_407_343),
        ],
        (KernelKind::Sa, Connectivity::Degree(60), Precision::Int32) => &[
            (5_000, 1.702_310_267_555),
            (16_000, 0.975_694_214_217),
            (64_000, 1.022_429_145_506),
            (130_765, 1.283_859_264_138),
        ],
        (KernelKind::Sa, Connectivity::Degree(60), Precision::Float32) => &[
            (5_000, 1.418_507_224_89),
            (16_000, 0.491_784_891_73),
            (64_000, 0.461_196_502_909),
            (130_771, 0.459_118_722_674),
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Binary) => &[
            (64, 0.621_612_271_459),
            (256, 16.431_523_856_296),
            (1_000, 64.634_071_456_279),
            (2_000, 143.589_986_554_105),
            (4_000, 288.594_763_133_326),
            (9_370, 676.033_232_639_817), // projected: scaled by N from 4,000
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Int8) => &[
            (64, 0.636_079_148_866),
            (256, 8.683_983_315_114),
            (1_000, 34.354_114_513_211),
            (2_000, 75.795_565_972_916),
            (4_000, 152.954_016_466_178),
            (9_370, 358.294_783_572_021), // projected: scaled by N from 4,000
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Int32) => &[
            (64, 0.637_163_154_198),
            (256, 7.501_216_455_754),
            (1_000, 29.949_975_017_675),
            (2_000, 71.230_897_964_351),
            (4_000, 143.484_147_495_001),
            (4_741, 170.064_585_818_45), // projected: scaled by N from 4,000
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Float32) => &[
            (64, 0.432_402_085_884),
            (256, 6.080_538_018_996),
            (1_000, 20.247_950_405_666),
            (2_000, 50.591_996_048_58),
            (4_000, 104.324_087_736_175),
            (4_741, 123.650_124_989_301), // projected: scaled by N from 4,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Binary) => &[
            (4_800, 1.0),
            (16_000, 1.812_170_341_484),
            (64_000, 2.391_574_072_597),
            (200_000, 4.583_679_589_268),
            (400_000, 5.813_148_868_652),
            (1_000_000, 5.502_951_606_912),
            (6_491_600, 5.502_951_606_912), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Int8) => &[
            (4_800, 1.290_619_829_676),
            (16_000, 1.870_920_854_285),
            (64_000, 2.482_224_609_619),
            (200_000, 4.873_162_661_723),
            (400_000, 5.867_646_167_052),
            (1_000_000, 5.454_698_610_859),
            (6_491_600, 5.454_698_610_859), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Int32) => &[
            (4_800, 1.656_131_023_082),
            (16_000, 2.408_885_826_687),
            (64_000, 2.884_430_074_83),
            (200_000, 4.943_995_575_072),
            (400_000, 5.590_686_531_66),
            (1_000_000, 5.640_983_480_765),
            (3_077_188, 5.640_983_480_765), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Float32) => &[
            (4_800, 2.005_935_378_144),
            (16_000, 2.254_956_329_79),
            (64_000, 2.830_032_673_314),
            (200_000, 5.105_793_458_92),
            (400_000, 5.687_889_320_074),
            (1_000_000, 6.132_273_843_979),
            (3_077_188, 6.132_273_843_979), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Binary) => &[
            (4_800, 5.044_919_331_893),
            (16_000, 6.080_526_509_286),
            (64_000, 6.454_710_851_919),
            (200_000, 16.466_809_002_074),
            (400_000, 20.424_952_880_037),
            (1_000_000, 21.287_848_261_167),
            (3_641_281, 21.287_848_261_167), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Int8) => &[
            (4_800, 5.101_447_384_536),
            (16_000, 6.110_776_957_611),
            (64_000, 6.630_717_677_652),
            (200_000, 17.484_556_655_945),
            (400_000, 20.209_749_351_436),
            (1_000_000, 21.536_336_397_944),
            (3_641_281, 21.536_336_397_944), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Int32) => &[
            (4_800, 6.208_284_258_266),
            (16_000, 7.151_308_522_821),
            (64_000, 7.189_822_344_275),
            (200_000, 21.164_844_963_06),
            (400_000, 19.966_866_842_229),
            (1_000_000, 21.038_055_080_306),
            (1_296_797, 21.038_055_080_306), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Float32) => &[
            (4_800, 6.263_929_255_815),
            (16_000, 7.210_601_207_881),
            (64_000, 8.300_895_702_063),
            (200_000, 18.631_826_082_348),
            (400_000, 21.147_776_828_537),
            (1_000_000, 21.296_386_532_686),
            (1_296_797, 21.296_386_532_686), // projected: flat from 1,000,000
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Binary) => &[
            (64, 47.215_440_928_084),
            (256, 51.026_875_115_917),
            (1_000, 154.793_925_060_305),
            (2_000, 254.341_990_532_74),
            (4_000, 445.778_716_167_581),
            (19_177, 2_137.174_609_986_424), // projected: scaled by N from 4,000
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Int8) => &[
            (64, 47.097_141_384_651),
            (256, 48.488_788_964_088),
            (1_000, 156.915_651_872_326),
            (2_000, 254.400_637_825_26),
            (4_000, 446.361_816_275_26),
            (19_177, 2_139.970_137_677_665), // projected: scaled by N from 4,000
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Int32) => &[
            (64, 46.962_589_739_085),
            (256, 55.069_058_767_78),
            (1_000, 161.866_628_827_568),
            (2_000, 263.383_349_689_56),
            (4_000, 457.889_802_230_23),
            (9_592, 1_098.019_745_748_091), // projected: scaled by N from 4,000
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Float32) => &[
            (64, 48.473_246_520_027),
            (256, 57.285_004_927_757),
            (1_000, 166.422_246_488_486),
            (2_000, 278.948_364_332_079),
            (4_000, 482.313_969_493_723),
            (9_592, 1_156.588_898_845_948), // projected: scaled by N from 4,000
        ],
        _ => &[],
    }
}

/// Interpolate measured slowdown only within this kernel's measured read count and node range.
pub(crate) fn lookup(
    kernel: KernelKind,
    connectivity: Connectivity,
    precision: Precision,
    nodes: usize,
    reads: usize,
) -> Option<f64> {
    let measured_reads = match kernel {
        KernelKind::Msa => 128,
        KernelKind::Sa | KernelKind::Gibbs => 256,
    };
    if reads != measured_reads {
        return None;
    }
    interpolate(samples(kernel, connectivity, precision), nodes)
}

/// A calibrated slowdown and whether any part of it comes from a projected entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Slowdown {
    /// Slowdown per lane-update against the kernel's baseline.
    pub value: f64,
    /// True when the value is a projected entry or interpolates toward one.
    pub projected: bool,
}

/// [`lookup`] plus its source: past the last timed node count of a series that
/// ends in a projected entry, the value is projected.
pub(crate) fn lookup_with_source(
    kernel: KernelKind,
    connectivity: Connectivity,
    precision: Precision,
    nodes: usize,
    reads: usize,
) -> Option<Slowdown> {
    let value = lookup(kernel, connectivity, precision, nodes, reads)?;
    let projected = last_timed_before_projection(kernel, connectivity, precision)
        .is_some_and(|last_timed| nodes > last_timed);
    Some(Slowdown { value, projected })
}

/// The last timed node count of each series whose final entry is projected.
const LAST_TIMED_BEFORE_PROJECTION: &[(KernelKind, Connectivity, Precision, usize)] = &[
    (
        KernelKind::Sa,
        Connectivity::Degree(20),
        Precision::Binary,
        200_000,
    ),
    (
        KernelKind::Sa,
        Connectivity::Degree(60),
        Precision::Binary,
        200_000,
    ),
    (
        KernelKind::Sa,
        Connectivity::Complete,
        Precision::Binary,
        4_000,
    ),
    (
        KernelKind::Sa,
        Connectivity::Complete,
        Precision::Int8,
        4_000,
    ),
    (
        KernelKind::Sa,
        Connectivity::Complete,
        Precision::Int32,
        4_000,
    ),
    (
        KernelKind::Sa,
        Connectivity::Complete,
        Precision::Float32,
        4_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(20),
        Precision::Binary,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(20),
        Precision::Int8,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(20),
        Precision::Int32,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(20),
        Precision::Float32,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(60),
        Precision::Binary,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(60),
        Precision::Int8,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(60),
        Precision::Int32,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Degree(60),
        Precision::Float32,
        1_000_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Complete,
        Precision::Binary,
        4_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Complete,
        Precision::Int8,
        4_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Complete,
        Precision::Int32,
        4_000,
    ),
    (
        KernelKind::Gibbs,
        Connectivity::Complete,
        Precision::Float32,
        4_000,
    ),
];

fn last_timed_before_projection(
    kernel: KernelKind,
    connectivity: Connectivity,
    precision: Precision,
) -> Option<usize> {
    LAST_TIMED_BEFORE_PROJECTION
        .iter()
        .find(|&&(k, c, p, _)| (k, c, p) == (kernel, connectivity, precision))
        .map(|&(.., last_timed)| last_timed)
}

#[allow(clippy::cast_precision_loss)] // supported node counts fit exactly in f64
fn interpolate(samples: &[(usize, f64)], nodes: usize) -> Option<f64> {
    let &(first, _) = samples.first()?;
    let &(last, _) = samples.last()?;
    if nodes < first || nodes > last {
        return None;
    }
    for &(n, slowdown) in samples {
        if nodes == n {
            return Some(slowdown);
        }
    }
    for pair in samples.windows(2) {
        let (lo, slow_lo) = pair[0];
        let (hi, slow_hi) = pair[1];
        if nodes > lo && nodes < hi {
            let fraction = (nodes as f64 / lo as f64).ln() / (hi as f64 / lo as f64).ln();
            return Some(slow_lo + fraction * (slow_hi - slow_lo));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelKind;

    #[test]
    fn slowdown_baseline_is_one_per_kernel() {
        for (kernel, nodes, reads) in [
            (KernelKind::Sa, 5_000, 256),
            (KernelKind::Gibbs, 4_800, 256),
            (KernelKind::Msa, 5_791, 128),
        ] {
            assert_eq!(
                lookup(
                    kernel,
                    Connectivity::Degree(20),
                    Precision::Binary,
                    nodes,
                    reads
                ),
                Some(1.0)
            );
        }
    }

    #[test]
    fn slowdown_marks_values_past_the_timed_range_projected() {
        let at = |kernel, nodes, reads| {
            lookup_with_source(
                kernel,
                Connectivity::Degree(20),
                Precision::Binary,
                nodes,
                reads,
            )
            .map(|slowdown| slowdown.projected)
        };
        // 200,000 is the last timed SA cell; 410,747 is the projected probe limit.
        assert_eq!(at(KernelKind::Sa, 64_000, 256), Some(false));
        assert_eq!(at(KernelKind::Sa, 200_000, 256), Some(false));
        assert_eq!(at(KernelKind::Sa, 300_000, 256), Some(true));
        assert_eq!(at(KernelKind::Sa, 410_747, 256), Some(true));
        assert_eq!(at(KernelKind::Sa, 410_748, 256), None);
        assert_eq!(at(KernelKind::Msa, 2_145_598, 128), Some(false));
        assert_eq!(
            lookup_with_source(
                KernelKind::Sa,
                Connectivity::Degree(20),
                Precision::Binary,
                300_000,
                256
            )
            .map(|slowdown| slowdown.value),
            lookup(
                KernelKind::Sa,
                Connectivity::Degree(20),
                Precision::Binary,
                300_000,
                256
            )
        );
    }

    #[test]
    fn slowdown_outside_range_is_none() {
        for (kernel, first, reads) in [
            (KernelKind::Sa, 5_000, 256),
            (KernelKind::Gibbs, 4_800, 256),
            (KernelKind::Msa, 5_791, 128),
        ] {
            for nodes in [0, first - 1, usize::MAX] {
                assert_eq!(
                    lookup(
                        kernel,
                        Connectivity::Degree(20),
                        Precision::Binary,
                        nodes,
                        reads
                    ),
                    None
                );
            }
            assert_eq!(
                lookup(
                    kernel,
                    Connectivity::Degree(21),
                    Precision::Binary,
                    first,
                    reads
                ),
                None
            );
            for other_reads in [0, 64, reads / 2, reads * 2] {
                assert_eq!(
                    lookup(
                        kernel,
                        Connectivity::Degree(20),
                        Precision::Binary,
                        first,
                        other_reads
                    ),
                    None
                );
            }
        }
    }

    #[test]
    fn slowdown_interpolates_in_log_nodes_and_keeps_endpoints() {
        let points = [(100, 2.0), (10_000, 6.0)];
        assert_eq!(interpolate(&points, 100), Some(2.0));
        assert_eq!(interpolate(&points, 10_000), Some(6.0));
        assert!((interpolate(&points, 1000).unwrap() - 4.0).abs() < 1e-12);
        assert_eq!(interpolate(&points, 99), None);
        assert_eq!(interpolate(&points, 10_001), None);
        assert_eq!(interpolate(&[], 100), None);
        // 32,000 is the geometric midpoint of the measured 16,000 and 64,000 cells.
        for kernel in [KernelKind::Sa, KernelKind::Gibbs] {
            let at = |nodes| {
                lookup(
                    kernel,
                    Connectivity::Degree(20),
                    Precision::Binary,
                    nodes,
                    256,
                )
                .unwrap()
            };
            assert!((at(32_000) - f64::midpoint(at(16_000), at(64_000))).abs() < 1e-10);
        }
        for kernel in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
            let reads = if kernel == KernelKind::Msa { 128 } else { 256 };
            for connectivity in [
                Connectivity::Degree(20),
                Connectivity::Degree(60),
                Connectivity::Complete,
            ] {
                for precision in [
                    Precision::Binary,
                    Precision::Int8,
                    Precision::Int32,
                    Precision::Float32,
                ] {
                    let points = samples(kernel, connectivity, precision);
                    for &(nodes, value) in points {
                        assert_eq!(
                            lookup(kernel, connectivity, precision, nodes, reads),
                            Some(value)
                        );
                    }
                    if let (Some(&(first, _)), Some(&(last, _))) = (points.first(), points.last()) {
                        assert_eq!(
                            lookup(kernel, connectivity, precision, first - 1, reads),
                            None
                        );
                        assert_eq!(
                            lookup(kernel, connectivity, precision, last + 1, reads),
                            None
                        );
                    }
                }
            }
        }
    }
}
