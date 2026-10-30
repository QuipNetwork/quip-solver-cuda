//! RTX A4000 measured two-point slowdown per lane-update.
//! MSA: 128 reads, 48 nonces. SA: 256 reads, 48 nonces. Gibbs: 256 reads, 12 nonces.
//! Generated from bench-evidence/{msa,sa-gibbs}-vram-dense/summary.json.

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
            (410_747, 0.454_799_829_683),
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
            (347_112, 1.002_313_748_451),
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
            (9_370, 677.626_925_312_445),
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Int8) => &[
            (64, 0.636_079_148_866),
            (256, 8.683_983_315_114),
            (1_000, 34.354_114_513_211),
            (2_000, 75.795_565_972_916),
            (4_000, 152.954_016_466_178),
            (9_370, 568.570_481_100_219),
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Int32) => &[
            (64, 0.637_163_154_198),
            (256, 7.501_216_455_754),
            (1_000, 29.949_975_017_675),
            (2_000, 71.230_897_964_351),
            (4_000, 143.484_147_495_001),
            (4_741, 172.152_866_431_543),
        ],
        (KernelKind::Sa, Connectivity::Complete, Precision::Float32) => &[
            (64, 0.432_402_085_884),
            (256, 6.080_538_018_996),
            (1_000, 20.247_950_405_666),
            (2_000, 50.591_996_048_58),
            (4_000, 104.324_087_736_175),
            (4_741, 124.532_756_104_039),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Binary) => &[
            (4_800, 1.0),
            (16_000, 3.225_358_771_603),
            (64_000, 1.650_675_235_691),
            (200_000, 3.427_842_498_717),
            (6_491_600, 4.157_630_252_957),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Int8) => &[
            (4_800, 0.887_037_693_98),
            (16_000, 3.518_101_595_55),
            (64_000, 1.801_736_721_472),
            (200_000, 3.644_328_481_015),
            (6_491_600, 4.956_143_955_201),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Int32) => &[
            (4_800, 1.198_967_360_632),
            (16_000, 3.513_516_398_518),
            (64_000, 1.956_100_371_234),
            (200_000, 3.697_299_912_8),
            (3_077_188, 4.842_408_072_646),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(20), Precision::Float32) => &[
            (4_800, 1.259_109_380_92),
            (16_000, 3.593_751_985_627),
            (64_000, 2.030_729_608_495),
            (200_000, 3.818_298_261_758),
            (3_077_188, 4.882_893_382_911),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Binary) => &[
            (4_800, 3.851_358_817_246),
            (16_000, 10.505_325_106_023),
            (64_000, 4.877_235_789_449),
            (200_000, 12.314_479_364_511),
            (3_641_281, 17.296_040_993_616),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Int8) => &[
            (4_800, 3.730_876_129_322),
            (16_000, 10.369_919_952_757),
            (64_000, 4.905_400_864_473),
            (200_000, 13.075_588_118_507),
            (3_641_281, 17.954_876_116_877),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Int32) => &[
            (4_800, 4.361_785_740_343),
            (16_000, 10.896_258_033_447),
            (64_000, 5.443_793_852_032),
            (200_000, 15.827_841_721_966),
            (1_296_797, 18.186_608_793_556),
        ],
        (KernelKind::Gibbs, Connectivity::Degree(60), Precision::Float32) => &[
            (4_800, 4.780_424_708_051),
            (16_000, 11.004_751_847_271),
            (64_000, 5.672_291_635_346),
            (200_000, 13.933_557_970_177),
            (1_296_797, 16.118_025_333_368),
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Binary) => &[
            (64, 35.309_425_944_148),
            (256, 38.159_755_212_517),
            (1_000, 115.760_533_547_614),
            (2_000, 190.206_201_672_073),
            (4_000, 333.369_555_734_345),
            (19_177, 1_398.002_713_904_67),
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Int8) => &[
            (64, 35.220_957_238_022),
            (256, 36.261_681_970_877),
            (1_000, 117.347_238_114_386),
            (2_000, 190.250_060_252_896),
            (4_000, 333.805_619_226_826),
            (19_177, 1_398.212_566_373_457),
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Int32) => &[
            (64, 35.120_334_618_146),
            (256, 41.182_647_332_184),
            (1_000, 121.049_758_957_484),
            (2_000, 196.967_659_265_33),
            (4_000, 342.426_666_883_294),
            (9_592, 764.455_723_246_57),
        ],
        (KernelKind::Gibbs, Connectivity::Complete, Precision::Float32) => &[
            (64, 36.250_058_765_273),
            (256, 42.839_812_558_092),
            (1_000, 124.456_615_724_384),
            (2_000, 208.607_743_971_448),
            (4_000, 360.691_948_500_623),
            (9_592, 798.468_113_241_961),
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
