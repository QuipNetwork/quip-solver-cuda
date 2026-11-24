//! Constrained CUDA loop search for adapted P and F-static revision 1.
//!
//! The deterministic policy uses up to ten passes and ignores the session seed
//! and sweep count. Cold reads are identical copies of the exact baseline.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cuda::families::flow_loops::{FlowLoopsSampler, CUDA_FLOW_LOOPS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Visible CUDA device index. Use `CUDA_VISIBLE_DEVICES` to select a physical GPU.
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CUDA_FLOW_LOOPS_IDENTITY, &cli.common, || {
        Ok(FlowLoopsSampler::new(cli.device))
    })
}
