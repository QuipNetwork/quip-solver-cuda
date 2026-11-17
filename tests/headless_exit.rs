//! SPEC section 2 on a host with no visible device.
//!
//! `--capabilities` answers without opening the device and exits 0.
//! `--check` and `--solve` exit 69 (`EnvIncompatible`) because the device
//! does not open. `CUDA_VISIBLE_DEVICES=""` hides every GPU, so these run the
//! same on a GPU runner and on a host with no driver.

use std::process::{Command, Output, Stdio};

const BINARIES: [&str; 7] = [
    env!("CARGO_BIN_EXE_quip-cuda-sa"),
    env!("CARGO_BIN_EXE_quip-cuda-gibbs"),
    env!("CARGO_BIN_EXE_quip-cuda-msa"),
    env!("CARGO_BIN_EXE_quip-cuda-msa-descent"),
    env!("CARGO_BIN_EXE_quip-cuda-otsuka"),
    env!("CARGO_BIN_EXE_quip-cuda-dipolar"),
    env!("CARGO_BIN_EXE_quip-cuda-flow-loops"),
];

fn headless(binary: &str, flag: &str) -> Output {
    Command::new(binary)
        .arg(flag)
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn capabilities_answer_without_a_device() {
    for binary in BINARIES {
        let out = headless(binary, "--capabilities");
        assert_eq!(
            out.status.code(),
            Some(0),
            "{binary}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(caps["backend"], "cuda", "{binary}");
        assert!(caps["maxNodes"].as_u64().unwrap() > 0, "{binary}: {caps}");
        assert!(caps["maxEdges"].as_u64().unwrap() > 0, "{binary}: {caps}");
    }
}

#[test]
fn check_and_solve_exit_69_without_a_device() {
    for binary in BINARIES {
        for flag in ["--check", "--solve"] {
            let out = headless(binary, flag);
            assert_eq!(
                out.status.code(),
                Some(69),
                "{binary} {flag}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

#[test]
fn msa_descent_lists_only_sample_jobs() {
    let out = headless(
        env!("CARGO_BIN_EXE_quip-cuda-msa-descent"),
        "--capabilities",
    );
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["supportedKinds"], serde_json::json!(["ISING_SAMPLE"]));
    assert_eq!(caps["generators"], serde_json::json!([]));
    // An empty list is omitted from the JSON.
    assert!(caps.get("generatedFamilies").is_none(), "{caps}");
}
