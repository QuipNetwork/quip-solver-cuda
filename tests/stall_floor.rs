//! CPU-only coverage of the streaming stall allowance.

use quip_solver_cuda::streaming::{stall_cutoff, StallConfig};
use std::time::Duration;

#[test]
fn small_sweep_counts_keep_ten_minute_floor() {
    for sweeps in [0, 1, 1024, 1_500_000] {
        assert_eq!(
            StallConfig::default().floor(sweeps),
            Duration::from_mins(10)
        );
    }
}

#[test]
fn deep_sweeps_scale_the_floor() {
    assert_eq!(
        StallConfig::default().floor(8_388_608),
        Duration::from_secs_f64(3355.4432)
    );
}

#[test]
fn custom_seconds_per_sweep_changes_the_floor() {
    let config = StallConfig {
        stall_secs_per_sweep: 0.001,
        ..StallConfig::default()
    };
    assert_eq!(config.floor(2_000_000), Duration::from_secs(2000));
}

#[test]
fn explicit_floor_replaces_the_scaled_floor_and_minimum() {
    let config = StallConfig {
        stall_secs_per_sweep: 0.001,
        stall_floor_secs: Some(1),
    };
    assert_eq!(config.floor(8_388_608), Duration::from_secs(1));
}

#[test]
fn cutoff_uses_the_larger_of_floor_and_four_times_longest_gap() {
    let floor = StallConfig::default().floor(8_388_608);
    assert_eq!(stall_cutoff(floor, Duration::ZERO), floor);
    assert_eq!(stall_cutoff(floor, Duration::from_secs(30)), floor);
    assert_eq!(
        stall_cutoff(floor, Duration::from_secs(1000)),
        Duration::from_secs(4000)
    );
    assert_eq!(stall_cutoff(floor, Duration::MAX), Duration::MAX);
}

#[test]
fn oversized_finite_allowance_saturates_instead_of_panicking() {
    let config = StallConfig {
        stall_secs_per_sweep: f64::MAX,
        ..StallConfig::default()
    };
    assert_eq!(config.floor(2), Duration::MAX);
}
