//! GPU generator and structural scorer of `quip/twisted-ice-dimer@1` salts (`kernels/gen_s0.cu`).
//!
//! Given the ChaCha stream keys of a batch of nonces it returns the S0 structural score of each, the same score
//! `structural_score` gives for the canonical instance. It is a screen only: a nonce that passes is rebuilt by the canonical
//! CPU generator before anything is solved or submitted.

use super::lean_gen::{GpuTables, Triangles};
use crate::cuda_device::{compile_for_arch, device_arch, driver_version};
use crate::jit_cache;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const SOURCE: &str = include_str!("../../kernels/gen_s0.cu");
const THREADS: u32 = 128;

pub(crate) struct GenGpu {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    ints: [i32; 10],
    bufs: Tables,
    scratch: CudaSlice<u8>,
    prof: CudaSlice<u64>,
    blocks: u32,
    shared_bytes: u32,
}

struct Tables {
    cand_edges: CudaSlice<u32>,
    adj_off: CudaSlice<i32>,
    adj_r: CudaSlice<u16>,
    adj_c: CudaSlice<u32>,
    eu: CudaSlice<u16>,
    ev: CudaSlice<u16>,
    tri: CudaSlice<u16>,
    fw: CudaSlice<f64>,
    fmu: CudaSlice<f64>,
    fsd: CudaSlice<f64>,
}

fn s<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
}

impl GenGpu {
    pub(crate) fn new(device: usize, gt: &GpuTables, tri: &Triangles) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(s)?;
        let arch = device_arch(&ctx);
        let module = jit_cache::load_or_compile(&ctx, "gen_s0", SOURCE, &[], &format!("sm_{arch}"), driver_version().map_err(s)?, 0, || {
            compile_for_arch(SOURCE, 0, arch, &[])
        })
        .map_err(s)?;
        let func = module.load_function("quip_gen_s0").map_err(s)?;
        let rup = |x: usize| (x + 1) & !1usize;
        let pair_cap = gt.edges - 6 * gt.sel_expected;
        // Same layout as the kernel: pair order | left order | rank | matched | stamp | selected bits | used bits | 10 counters.
        let mut bytes = 2 * rup(pair_cap) + 2 * rup(gt.left) + 2 * rup(gt.right) + 2 * rup(gt.right) + 2 * rup(gt.right);
        bytes += 4 * ((gt.cands + 31) / 32) + 4 * ((gt.nodes + 31) / 32) + 4 * 10;
        let shared_bytes = u32::try_from(bytes).map_err(s)?;
        let optin = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).map_err(s)?;
        if i64::from(shared_bytes) > i64::from(optin) {
            return Err(format!("generator needs {shared_bytes} B of shared memory, device allows {optin}"));
        }
        func.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, i32::try_from(shared_bytes).map_err(s)?).map_err(s)?;
        let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(s)?;
        let per_sm = (i64::from(ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR).map_err(s)?) / i64::from(shared_bytes + 1024)).clamp(1, 4);
        let blocks = u32::try_from(i64::from(sms) * per_sm).map_err(s)?;
        let stream = ctx.new_stream().map_err(s)?;
        let up = |v: &[u8]| -> Result<CudaSlice<u8>, String> { stream.clone_htod(v).map_err(s) };
        let _ = up;
        let bufs = Tables {
            cand_edges: stream.clone_htod(&gt.cand_edges).map_err(s)?,
            adj_off: stream.clone_htod(&gt.adj_off).map_err(s)?,
            adj_r: stream.clone_htod(&gt.adj_r).map_err(s)?,
            adj_c: stream.clone_htod(&gt.adj_c).map_err(s)?,
            eu: stream.clone_htod(&gt.eu).map_err(s)?,
            ev: stream.clone_htod(&gt.ev).map_err(s)?,
            tri: stream.clone_htod(&tri.flat_u16()).map_err(s)?,
            fw: stream.clone_htod(&crate::cluster::sampler::lease_run::W).map_err(s)?,
            fmu: stream.clone_htod(&crate::cluster::sampler::lease_run::MU).map_err(s)?,
            fsd: stream.clone_htod(&crate::cluster::sampler::lease_run::SD).map_err(s)?,
        };
        let scratch = stream.alloc_zeros::<u8>(blocks as usize * gt.edges).map_err(s)?;
        let prof = stream.alloc_zeros::<u64>(8).map_err(s)?;
        let ints = [
            gt.left as i32,
            gt.right as i32,
            gt.cands as i32,
            gt.edges as i32,
            gt.nodes as i32,
            tri.len() as i32,
            gt.weak,
            gt.pairs as i32,
            gt.sel_expected as i32,
            pair_cap as i32,
        ];
        Ok(Self { stream, func, ints, bufs, scratch, prof, blocks, shared_bytes })
    }

    /// Cycles spent in each phase of the kernel, summed over blocks (left+right shuffle, matching, gauges, weak signs, pairs, score).
    pub(crate) fn phase_cycles(&self) -> Vec<u64> {
        self.stream.clone_dtoh(&self.prof).unwrap_or_default()
    }

    /// Score a batch. `keys` holds 40 ChaCha key words per nonce. Returns `(score, status)` per nonce; status 0 is a valid instance.
    pub(crate) fn score(&mut self, keys: &[u32]) -> Result<Vec<(f64, i32)>, String> {
        let batch = keys.len() / 40;
        let d_keys = self.stream.clone_htod(keys).map_err(s)?;
        let mut d_score = self.stream.alloc_zeros::<f64>(batch).map_err(s)?;
        let mut d_status = self.stream.alloc_zeros::<i32>(batch).map_err(s)?;
        let batch_i = batch as i32;
        let cfg = LaunchConfig { grid_dim: (self.blocks.min(batch as u32).max(1), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: self.shared_bytes };
        let mut b = self.stream.launch_builder(&self.func);
        for v in &self.ints {
            b.arg(v);
        }
        b.arg(&self.bufs.cand_edges).arg(&self.bufs.adj_off).arg(&self.bufs.adj_r).arg(&self.bufs.adj_c).arg(&self.bufs.eu).arg(&self.bufs.ev).arg(&self.bufs.tri);
        b.arg(&self.bufs.fw).arg(&self.bufs.fmu).arg(&self.bufs.fsd);
        b.arg(&d_keys).arg(&batch_i).arg(&mut self.scratch).arg(&mut d_score).arg(&mut d_status).arg(&mut self.prof);
        // SAFETY: the arguments match `quip_gen_s0` in order and type; the buffers outlive the synchronized launch.
        unsafe { b.launch(cfg) }.map_err(s)?;
        self.stream.synchronize().map_err(s)?;
        let sc = self.stream.clone_dtoh(&d_score).map_err(s)?;
        let st = self.stream.clone_dtoh(&d_status).map_err(s)?;
        Ok(sc.into_iter().zip(st).collect())
    }
}
