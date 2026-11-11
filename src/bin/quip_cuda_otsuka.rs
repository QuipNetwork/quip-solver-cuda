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
    let identity = cuda_otsuka_identity(&device.limits, cap.min(device.max_nodes));
    if identity.max_nodes == 0 {
        eprintln!("device has no capacity for an Otsuka session");
        return ExitCode::FAILURE;
    }
    device.max_nodes = identity.max_nodes as usize;
    run(identity, &cli.common, || {
        let gov = UtilGovernor::start(&device.pci_bus_id, cli.utilization, cli.yielding);
        Ok(OtsukaSampler::new(device, gov, cli.stall))
    })
}
