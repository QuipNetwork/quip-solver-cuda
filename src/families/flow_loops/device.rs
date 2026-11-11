//! On-demand compilation and checked launch of the constrained chain kernel.

use super::{score, FlowProblem, FlowRun, LoopError};
use crate::cuda_device::{compile_for_arch, device_arch, driver_version};
use crate::jit_cache;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("../../../kernels/flow_loops.cu");

#[allow(clippy::too_many_lines)] // The flat launch sequence mirrors the CUDA ABI.
pub(super) fn run(
    problem: &FlowProblem,
    device: usize,
    starts: &[Vec<i8>],
    passes: u32,
) -> Result<FlowRun, LoopError> {
    let ctx = CudaContext::new(device)?;
    problem.admit(starts, passes, ctx.mem_get_info()?.0)?;
    let arch = device_arch(&ctx);
    let module = jit_cache::load_or_compile(
        &ctx,
        "flow_loops",
        SOURCE,
        &[],
        &format!("sm_{arch}"),
        driver_version()?,
        0,
        || compile_for_arch(SOURCE, 0, arch, &[]),
    )?;
    let function = module.load_function("quip_flow_loops")?;
    let stream = ctx.default_stream();
    let n = problem.task.positions().len();
    let v = problem.task.labels().len();
    let links: Vec<Vec<i8>> = starts
        .iter()
        .map(|s| problem.task.link_spins(s).expect("admitted start"))
        .collect();
    let flat: Vec<i8> = links.iter().flatten().copied().collect();
    let charges: Vec<i32> = links
        .iter()
        .flat_map(|s| {
            problem
                .task
                .check(s)
                .expect("admitted start")
                .charges
                .into_iter()
                .map(i32::from)
        })
        .collect();
    let scores: Vec<i64> = links
        .iter()
        .flat_map(|s| {
            let s = score(&problem.task, s).expect("admitted score");
            [
                i64::try_from(s.gap_squares).expect("admitted gap"),
                i64::try_from(s.objective).expect("admitted objective"),
            ]
        })
        .collect();
    let t = Instant::now();
    let d = &problem.layout;
    let ends = stream.clone_htod(&d.ends)?;
    let incidence = stream.clone_htod(&d.incidence)?;
    let labels = stream.clone_htod(&d.labels)?;
    let costs = stream.clone_htod(&d.costs)?;
    let targets = stream.clone_htod(&d.targets)?;
    let cycle_data: &[i32] = if d.cycles.is_empty() { &[0] } else { &d.cycles };
    let cycles = stream.clone_htod(cycle_data)?;
    let mut spins = stream.clone_htod(&flat)?;
    let mut charges = stream.clone_htod(&charges)?;
    let mut scores = stream.clone_htod(&scores)?;
    let mut scratch = stream.alloc_zeros::<i32>(v * 3 * starts.len())?;
    let mut status = stream.alloc_zeros::<i32>(starts.len())?;
    stream.synchronize()?;
    let mut transfer_time = t.elapsed();
    let n_i = i32::try_from(n).map_err(|_| LoopError::Parameter("link count exceeds i32"))?;
    let v_i = i32::try_from(v).map_err(|_| LoopError::Parameter("vertex count exceeds i32"))?;
    let c_i = i32::try_from(d.cycles.len() / 5)
        .map_err(|_| LoopError::Parameter("cycle count exceeds i32"))?;
    let passes = i32::try_from(passes).map_err(|_| LoopError::Parameter("passes exceed i32"))?;
    let blocks = u32::try_from(starts.len())
        .map_err(|_| LoopError::Parameter("replica count exceeds u32"))?;
    let t = Instant::now();
    let mut launch = stream.launch_builder(&function);
    launch
        .arg(&n_i)
        .arg(&v_i)
        .arg(&c_i)
        .arg(&passes)
        .arg(&ends)
        .arg(&incidence)
        .arg(&labels)
        .arg(&costs)
        .arg(&targets)
        .arg(&cycles)
        .arg(&mut spins)
        .arg(&mut charges)
        .arg(&mut scores)
        .arg(&mut scratch)
        .arg(&mut status);
    // SAFETY: canonical lattice bounds every endpoint and signed incidence.
    // Admission checks all products and memory before allocation. Each block
    // owns one replica's slices. Argument order matches quip_flow_loops.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    stream.synchronize()?;
    let launch_time = t.elapsed();
    let t = Instant::now();
    let spins = stream.clone_dtoh(&spins)?;
    let charges = stream.clone_dtoh(&charges)?;
    let scores = stream.clone_dtoh(&scores)?;
    let status = stream.clone_dtoh(&status)?;
    stream.synchronize()?;
    transfer_time += t.elapsed();
    let t = Instant::now();
    let mut result = FlowRun {
        spins: Vec::with_capacity(starts.len()),
        scores: Vec::with_capacity(starts.len()),
        energies: Vec::with_capacity(starts.len()),
        launch_time,
        transfer_time,
        validation_time: Duration::ZERO,
    };
    for (k, start) in links.iter().enumerate() {
        let (state, score, energy) = problem.validate(
            k,
            start,
            &spins[k * n..(k + 1) * n],
            &charges[k * v..(k + 1) * v],
            &scores[k * 2..k * 2 + 2],
            status[k],
        )?;
        result.spins.push(state);
        result.scores.push(score);
        result.energies.push(energy);
    }
    result.validation_time = t.elapsed();
    Ok(result)
}
