//! CUDA Otsuka miner with validated family routing and CUDA SA fallback.

use clap::Parser;
use quip_solver_core::{run, CommonArgs, OpenError};
use quip_solver_cuda::cluster::sampler::{cuda_otsuka_identity, OtsukaSampler, HEADLESS_LIMITS};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::streaming::StallConfig;
use quip_solver_cuda::KernelKind;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 2"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    stall: StallConfig,
    /// CUDA ordinal. One process owns one device.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Maximum foreign GPU utilization when yielding.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Yield to other GPU processes.
    #[arg(long)]
    yielding: bool,
    /// Optional lower node ceiling.
    #[arg(long)]
    max_nodes: Option<usize>,
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("cuda-{}", cli.device));
    }
    let cap = cli.max_nodes.unwrap_or(usize::MAX);
    if cli.common.capabilities {
        return run(
            cuda_otsuka_identity(&HEADLESS_LIMITS, cap),
            &cli.common,
            || Err::<OtsukaSampler, _>(OpenError("capabilities must remain headless".into())),
        );
    }
    let mut device = match CudaDevice::open_with_nodes(cli.device, KernelKind::Sa, cap) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("device {}: {error}", cli.device);
            return ExitCode::FAILURE;
        }
    };
    let mut identity = cuda_otsuka_identity(&device.limits, cap.min(device.max_nodes));
    local_budget(&mut identity.adapt);
    if identity.max_nodes == 0 {
        eprintln!("device has no capacity for an Otsuka session");
        return ExitCode::FAILURE;
    }
    device.max_nodes = identity.max_nodes as usize;
    run(identity, &cli.common, || {
        let gov = UtilGovernor::start(&device.pci_bus_id, cli.utilization, cli.yielding);
        OtsukaSampler::new(device, gov, cli.stall).map_err(|e| OpenError(e.to_string()))
    })
}

/// Local build (`BudFox`, 2026-11-13): `QUIP_OTSUKA_SWEEP_MULT` scales the
/// sweep envelope the session derives per lease (stock 256..2048, and a
/// dimer lease sits at difficulty 0, so it runs the minimum). `QUIP_OTSUKA_READS`
/// pins reads per instance (stock 64 at difficulty 0, at most 256): one CUDA
/// block per read, so reads also set how many SMs one instance occupies.
/// Unset = upstream.
fn local_budget(adapt: &mut quip_solver_core::adapt::AdaptBounds) {
    if let Some(m) = std::env::var("QUIP_OTSUKA_SWEEP_MULT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        if m.is_finite() && m > 0.0 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let scale = |x: u32| ((f64::from(x) * m).round() as u32).max(1);
            adapt.min_sweeps = scale(adapt.min_sweeps);
            adapt.max_sweeps = scale(adapt.max_sweeps);
            eprintln!(
                "quip-otsuka: QUIP_OTSUKA_SWEEP_MULT={m} -> min {} max {}",
                adapt.min_sweeps, adapt.max_sweeps
            );
        }
    }
    if let Ok(v) = std::env::var("QUIP_OTSUKA_READS") {
        match v.trim().parse::<u32>() {
            Ok(r) if (1..=256).contains(&r) => {
                adapt.min_reads = r;
                adapt.max_reads = r;
                eprintln!("quip-otsuka: QUIP_OTSUKA_READS={r} applied");
            }
            _ => eprintln!("quip-otsuka: QUIP_OTSUKA_READS={v:?} ignored (1..=256)"),
        }
    }
}
