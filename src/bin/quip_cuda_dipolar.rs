//! D-static CUDA string and loop competitor for explicit family instances.
use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cuda::families::dipolar::{DipolarSampler, CUDA_DIPOLAR_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 2"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Visible CUDA device index. Device creation waits for an admitted job.
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CUDA_DIPOLAR_IDENTITY, &cli.common, || {
        Ok(DipolarSampler::new(cli.device))
    })
}
