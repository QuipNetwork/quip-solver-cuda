//! Answer quip-solver-core without an open device.
//!
//! SPEC section 2: `--capabilities` never opens the device, and a device that
//! does not open exits 69 (`EnvIncompatible`) in `--check`, `--solve` and
//! session mode. The binaries that advertise memory-derived limits open the
//! device before [`run`], because `run` takes the identity before it calls
//! `open`. When the device does not open, or `--capabilities` asks for the
//! static answer, they hand quip-solver-core an identity at
//! [`HEADLESS_LIMITS`] and an `open` that fails, so quip-solver-core maps
//! every mode.

use crate::capacity::DeviceLimits;
use crate::cuda_device::CudaDevice;
use crate::KernelKind;
use quip_solver_core::{run, BackendIdentity, CommonArgs, OpenError, Sampler};
use std::process::ExitCode;

/// Conservative device budgets for an identity built without a device.
///
/// `--capabilities` advertises limits at these budgets. A live session
/// advertises the limits of the device it opened.
pub const HEADLESS_LIMITS: DeviceLimits = DeviceLimits {
    shared_bytes_per_block: 48 << 10,
    shared_bytes_per_block_optin: 48 << 10,
    free_bytes: 64 << 20,
    sm_count: 1,
    threads_per_sm: 2048,
};

/// Open the device, except for `--capabilities`, which must not open it.
///
/// # Errors
///
/// The reason the device is not open, for [`run_unopened`].
pub fn open_unless_capabilities(
    common: &CommonArgs,
    device: usize,
    kernel: KernelKind,
    max_nodes: usize,
) -> Result<CudaDevice, String> {
    if common.capabilities {
        return Err("--capabilities answers without opening the device".into());
    }
    CudaDevice::open_with_nodes(device, kernel, max_nodes)
        .map_err(|e| format!("device {device}: {e}"))
}

/// Run quip-solver-core with a device that is not open.
///
/// `--capabilities` prints `id`. `--check`, `--solve` and session mode call
/// `open`, which fails with `reason`, so they exit 69.
#[must_use]
pub fn run_unopened<S: Sampler>(
    id: BackendIdentity,
    common: &CommonArgs,
    reason: String,
) -> ExitCode {
    run(id, common, || Err::<S, _>(OpenError(reason)))
}

/// Fail `open` unless `device` is a visible CUDA ordinal.
///
/// For binaries that create their device per job. Their `open` touches no
/// device, so without this check `--check` would pass on a host with no GPU.
///
/// # Errors
///
/// An [`OpenError`] when a CUDA library does not load, or when the driver
/// reports no devices or fewer than `device + 1`.
pub fn require_visible(device: usize) -> Result<(), OpenError> {
    crate::cuda_device::require_libraries()
        .map_err(|e| OpenError(format!("device {device}: {e}")))?;
    let count =
        CudaDevice::device_count().map_err(|e| OpenError(format!("device {device}: {e}")))?;
    if device < count {
        Ok(())
    } else {
        Err(OpenError(format!(
            "device {device}: {count} CUDA device(s) visible"
        )))
    }
}

/// Exit code 64 (`ConfigInvalid`), for a flag the device cannot honour.
#[must_use]
pub fn config_invalid() -> ExitCode {
    ExitCode::from(quip_solver_core::ExitCode::ConfigInvalid as u8)
}

/// Exit code 69 (`EnvIncompatible`), for a device that does not open.
#[must_use]
pub fn env_incompatible() -> ExitCode {
    ExitCode::from(quip_solver_core::ExitCode::EnvIncompatible as u8)
}
