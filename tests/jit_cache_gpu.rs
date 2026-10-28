//! GPU-gated round-trip for the NVRTC PTX cache (`src/jit_cache.rs`).
//!
//! Ignored by default: it needs a real CUDA device. Run with
//! `cargo test --release --test jit_cache_gpu -- --ignored`.

use quip_solver_cuda::cuda_device::CudaDevice;
use std::time::Instant;

// Both tests select a process-wide cache directory.
static CACHE_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "requires a CUDA GPU"]
fn second_open_uses_cached_ptx() {
    let _cache_env = CACHE_ENV.lock().unwrap();
    // Isolate the cache in a per-process temp dir so the run is hermetic.
    let dir = std::env::temp_dir().join(format!("quip-jitcache-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::env::set_var("QUIP_CUDA_CACHE", &dir);
    std::env::remove_var("QUIP_CUDA_CACHE_DISABLE");

    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    // Cold open: compiles every kernel via NVRTC and writes the cache.
    let t0 = Instant::now();
    let _cold_dev = CudaDevice::open(0).expect("first open");
    let cold = t0.elapsed();

    // Four base modules plus ten SA variants after the first open.
    let ptx: Vec<_> = std::fs::read_dir(&dir)
        .expect("cache dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "ptx"))
        .collect();
    assert_eq!(
        ptx.len(),
        14,
        "expected four base modules and ten SA variants, got {ptx:?}"
    );

    // Warm open: same process, same dir — loads the cached PTX, no NVRTC.
    let t1 = Instant::now();
    let _warm_dev = CudaDevice::open(0).expect("second open");
    let warm = t1.elapsed();

    eprintln!("cold open {cold:?}, warm open {warm:?}");
    assert!(
        warm < cold,
        "warm open ({warm:?}) should beat cold NVRTC compile ({cold:?})"
    );

    drop(std::fs::remove_dir_all(&dir));
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn msa_open_compiles_no_new_variants() {
    use quip_solver_cuda::KernelKind;
    use std::collections::BTreeSet;

    let _cache_env = CACHE_ENV.lock().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let previous_cache = std::env::var_os("QUIP_CUDA_CACHE");
    let previous_disable = std::env::var_os("QUIP_CUDA_CACHE_DISABLE");
    std::env::set_var("QUIP_CUDA_CACHE", dir.path());
    std::env::remove_var("QUIP_CUDA_CACHE_DISABLE");
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 5000).unwrap();
    let names: BTreeSet<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            assert_eq!(path.extension().unwrap(), "ptx");
            path.file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .split("-sm_")
                .next()
                .unwrap()
                .to_owned()
        })
        .collect();
    // Exact compile_all_kernels set at 0b12e8f, including weighted MSA modules.
    let expected = [
        "sa",
        "sa-wide",
        "msa",
        "gibbs",
        "msa-weighted-i32",
        "msa-weighted-f32",
        "msa-weighted-int8",
        "msa-real-f32",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    drop(device);
    for (key, previous) in [
        ("QUIP_CUDA_CACHE", previous_cache),
        ("QUIP_CUDA_CACHE_DISABLE", previous_disable),
    ] {
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }
    assert_eq!(names, expected);
}
