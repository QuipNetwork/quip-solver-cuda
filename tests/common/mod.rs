//! Shared helpers for quip-solver-cuda integration tests.

/// Path to a binary that Cargo builds for integration tests.
pub fn profile_bin(name: &str) -> String {
    match name {
        "quip-cuda-sa" => env!("CARGO_BIN_EXE_quip-cuda-sa"),
        "quip-cuda-msa" => env!("CARGO_BIN_EXE_quip-cuda-msa"),
        "quip-cuda-gibbs" => env!("CARGO_BIN_EXE_quip-cuda-gibbs"),
        _ => panic!("unknown test binary {name}"),
    }
    .to_owned()
}

pub fn ensure_built(package_bins: &[&str]) {
    for name in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(name)).exists(),
            "missing binary {name} at {}",
            profile_bin(name)
        );
    }
}
