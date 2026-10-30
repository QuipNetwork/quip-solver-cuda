//! Kernel capacity grid derived from the session allocation model.

use crate::capacity::{
    footprint, gibbs_shared_spins, instances, max_model, max_reads, msa_dynamic_shared_bytes,
    msa_spin_layout, route_storage, Connectivity, DeviceLimits, Precision, Storage,
    MSA_DEFAULT_BETAS, SA_DEFAULT_NODES,
};
use crate::cuda_device::CudaDevice;
use crate::kernel::KernelKind;

/// One capacity cell at fixed connectivity, precision, reads, and beta count.
#[derive(Debug, Clone)]
pub struct ProbeCell {
    /// Kernel whose capacity and storage tiers this cell describes.
    pub kernel: KernelKind,
    /// Topology family.
    pub connectivity: Connectivity,
    /// Coefficient storage and arithmetic tier.
    pub precision: Precision,
    /// Sparse CSR or dense matrix.
    pub storage: Storage,
    /// Whether sparse spins extend beyond local or shared memory.
    pub spill: bool,
    /// Largest supported variable count.
    pub nodes: usize,
    /// Couplers at that variable count.
    pub couplers: usize,
    /// Global bytes allocated per instance.
    pub per_instance_bytes: usize,
    /// RTX A4000 slowdown per lane-update relative to this kernel's degree-20 Binary baseline.
    /// Available only at the measured read count and within the measured node range.
    pub expected_slowdown: Option<f64>,
}

impl ProbeCell {
    /// Display tier for the model's spin and coefficient storage.
    #[must_use]
    pub fn tier(&self) -> &'static str {
        match (self.kernel, self.storage, self.spill) {
            (_, Storage::Dense, _) => "dense",
            (KernelKind::Sa, Storage::Sparse, true) => "global",
            (KernelKind::Sa, Storage::Sparse, false) => "local",
            (KernelKind::Gibbs | KernelKind::Msa, Storage::Sparse, true) => "spill",
            (KernelKind::Gibbs | KernelKind::Msa, Storage::Sparse, false) => "shared",
        }
    }
}

/// Degree 20, degree 60, and complete models by Binary, Int8, Int32, and Float32.
#[must_use]
pub fn probe(kernel: KernelKind, limits: &DeviceLimits, reads: usize) -> Vec<ProbeCell> {
    let mut cells = Vec::with_capacity(12);
    for connectivity in [
        Connectivity::Degree(20),
        Connectivity::Degree(60),
        Connectivity::Complete,
    ] {
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let shape = max_model(kernel, connectivity, precision, reads, limits);
            // MSA complete-model searches retain dense accounting even below degree 63.
            let routing_shape = if kernel == KernelKind::Msa {
                connectivity.shape(usize::MAX)
            } else {
                shape
            };
            let storage = route_storage(
                kernel,
                routing_shape,
                precision,
                reads,
                MSA_DEFAULT_BETAS,
                limits,
            );
            let footprint = footprint(
                kernel,
                shape,
                precision,
                storage,
                reads,
                MSA_DEFAULT_BETAS,
                limits,
            );
            let on_chip_spins = match kernel {
                KernelKind::Sa => SA_DEFAULT_NODES,
                KernelKind::Gibbs => {
                    gibbs_shared_spins(shape.nodes, limits.shared_bytes_per_block_optin, None)
                }
                KernelKind::Msa => {
                    msa_spin_layout(
                        shape.nodes,
                        reads,
                        msa_dynamic_shared_bytes(limits.shared_bytes_per_block_optin),
                        None,
                    )
                    .1
                }
            };
            cells.push(ProbeCell {
                kernel,
                connectivity,
                precision,
                storage,
                spill: storage == Storage::Sparse && on_chip_spins < shape.nodes,
                nodes: shape.nodes,
                couplers: shape.couplers,
                per_instance_bytes: footprint.per_instance_bytes,
                expected_slowdown: crate::slowdown_a4000::lookup(
                    kernel,
                    connectivity,
                    precision,
                    shape.nodes,
                    reads,
                ),
            });
        }
    }
    cells
}

/// Print the device capacity grid as a table or a single JSON object.
pub fn print_probe(device: &CudaDevice, json: bool) {
    let limits = &device.limits;
    let kernel = device.kernel;
    let reads = max_reads(kernel, limits);
    let instance_count = instances(kernel, limits.sm_count);
    let per_instance_budget = limits.usable_bytes() / instance_count;
    let slowdown_reference = match kernel {
        KernelKind::Msa => {
            "RTX A4000, 128 reads, 48 nonces, per lane-update against degree-20 Binary N=5791"
        }
        KernelKind::Sa => {
            "RTX A4000, 256 reads, 48 nonces, per lane-update against degree-20 Binary N=5000"
        }
        KernelKind::Gibbs => {
            "RTX A4000, 256 reads, 12 nonces, per lane-update against degree-20 Binary N=4800"
        }
    };
    let cells = probe(kernel, limits, reads);
    let name = device
        .name()
        .unwrap_or_else(|_| format!("cuda-{}", device.device_index));
    let connectivity = |conn| match conn {
        Connectivity::Degree(d) => format!("degree {d}"),
        Connectivity::Complete => "complete".to_owned(),
    };
    if json {
        let cells: Vec<_> = cells.iter().map(|cell| serde_json::json!({
            "connectivity": connectivity(cell.connectivity),
            "precision": format!("{:?}", cell.precision),
            "storage": match cell.storage { Storage::Sparse => "Sparse", Storage::Dense => "Dense" },
            "tier": cell.tier(), "spill": cell.spill,
            "nodes": cell.nodes, "couplers": cell.couplers,
            "per_instance_bytes": cell.per_instance_bytes,
            "expected_slowdown": cell.expected_slowdown,
        })).collect();
        println!(
            "{}",
            serde_json::json!({
                "kernel": kernel.name(), "device_name": name, "sm_count": limits.sm_count,
                "free_bytes": limits.free_bytes, "usable_bytes": limits.usable_bytes(),
                "instances": instance_count,
                "per_instance_budget": per_instance_budget,
                "reads": reads, "betas": MSA_DEFAULT_BETAS, "cells": cells,
                "slowdown_reference": slowdown_reference,
            })
        );
        return;
    }
    println!("{name}: SMs={} free_bytes={} usable_bytes={} instances={} per_instance_budget={} reads={reads} betas={MSA_DEFAULT_BETAS}",
        limits.sm_count, limits.free_bytes, limits.usable_bytes(),
        instance_count, per_instance_budget);
    println!(
        "{:<12} | {:<48} | {:<48} | {:<48} | {:<48}",
        "Connectivity", "Binary", "Int8", "Int32", "Float32"
    );
    for row in cells.chunks(4) {
        print!("{:<12}", connectivity(row[0].connectivity));
        for cell in row {
            print!(
                " | {:<48}",
                format!(
                    "{} vars, {} couplers, {}, slowdown {}",
                    cell.nodes,
                    cell.couplers,
                    cell.tier(),
                    cell.expected_slowdown
                        .map_or_else(|| "unknown".to_owned(), |value| format!("{value:.2}x"))
                )
            );
        }
        println!();
    }
    println!("Bytes per instance:");
    println!(
        "{:<12} | {:<16} | {:<16} | {:<16} | {:<16}",
        "Connectivity", "Binary", "Int8", "Int32", "Float32"
    );
    for row in cells.chunks(4) {
        println!(
            "{:<12} | {:<16} | {:<16} | {:<16} | {:<16}",
            connectivity(row[0].connectivity),
            row[0].per_instance_bytes,
            row[1].per_instance_bytes,
            row[2].per_instance_bytes,
            row[3].per_instance_bytes
        );
    }
    println!(
        "Slowdown: {slowdown_reference}. Unknown for other read counts or outside measured ranges."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::{msa_footprint, DeviceLimits, Precision};

    fn a4000_limits() -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49_152,
            shared_bytes_per_block_optin: 101_376,
            free_bytes: 15 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        }
    }

    #[test]
    fn probe_has_twelve_cells_for_every_kernel() {
        use crate::capacity::{footprint, gibbs_shared_spins, max_reads, SA_DEFAULT_NODES};
        let limits = a4000_limits();
        for kernel in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
            let reads = max_reads(kernel, &limits);
            let cells = probe(kernel, &limits, reads);
            assert_eq!(cells.len(), 12);
            let d20 = cells
                .iter()
                .find(|c| {
                    c.connectivity == Connectivity::Degree(20) && c.precision == Precision::Binary
                })
                .unwrap();
            let d60 = cells
                .iter()
                .find(|c| {
                    c.connectivity == Connectivity::Degree(60) && c.precision == Precision::Binary
                })
                .unwrap();
            assert!(d20.nodes > d60.nodes);
            for cell in &cells {
                let shape = max_model(kernel, cell.connectivity, cell.precision, reads, &limits);
                assert_eq!((cell.nodes, cell.couplers), (shape.nodes, shape.couplers));
                let storage = route_storage(
                    kernel,
                    shape,
                    cell.precision,
                    reads,
                    MSA_DEFAULT_BETAS,
                    &limits,
                );
                assert_eq!(cell.storage, storage);
                assert_eq!(
                    cell.per_instance_bytes,
                    footprint(
                        kernel,
                        shape,
                        cell.precision,
                        storage,
                        reads,
                        MSA_DEFAULT_BETAS,
                        &limits
                    )
                    .per_instance_bytes
                );
                let tier = match (kernel, storage) {
                    (_, Storage::Dense) => "dense",
                    (KernelKind::Sa, Storage::Sparse) if shape.nodes <= SA_DEFAULT_NODES => "local",
                    (KernelKind::Sa, Storage::Sparse) => "global",
                    (KernelKind::Gibbs, Storage::Sparse)
                        if gibbs_shared_spins(
                            shape.nodes,
                            limits.shared_bytes_per_block_optin,
                            None,
                        ) == shape.nodes =>
                    {
                        "shared"
                    }
                    (KernelKind::Gibbs | KernelKind::Msa, Storage::Sparse) => "spill",
                };
                assert_eq!(cell.tier(), tier);
            }
        }
    }

    #[test]
    fn probe_small_sparse_models_use_local_or_shared_tier() {
        let shape = Connectivity::Degree(20).shape(1000);
        for kernel in [KernelKind::Sa, KernelKind::Gibbs] {
            let mut limits = a4000_limits();
            // Keep SA's resident local frame below its first global model.
            limits.threads_per_sm = 256;
            let reads = max_reads(kernel, &limits);
            let storage = route_storage(kernel, shape, Precision::Binary, reads, 64, &limits);
            assert_eq!(storage, Storage::Sparse);
            let bytes = footprint(
                kernel,
                shape,
                Precision::Binary,
                storage,
                reads,
                64,
                &limits,
            )
            .total();
            limits.free_bytes = bytes.div_ceil(4) * 5;
            let cells = probe(kernel, &limits, reads);
            let cell = cells
                .iter()
                .find(|c| {
                    c.connectivity == Connectivity::Degree(20) && c.precision == Precision::Binary
                })
                .unwrap();
            assert_eq!(cell.nodes, shape.nodes);
            assert_eq!(
                cell.tier(),
                if kernel == KernelKind::Sa {
                    "local"
                } else {
                    "shared"
                }
            );
        }
    }

    #[test]
    fn probe_grid_has_twelve_cells_and_lower_degree_gives_more_nodes() {
        let limits = a4000_limits();
        let cells = probe(KernelKind::Msa, &limits, 128);
        assert_eq!(cells.len(), 12);
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let row: Vec<_> = cells.iter().filter(|c| c.precision == precision).collect();
            assert!(row[0].nodes > row[1].nodes && row[1].nodes > row[2].nodes);
            assert!(row[0].couplers < row[1].couplers);
        }
        for row in cells.chunks(4) {
            assert!(
                row[0].nodes >= row[1].nodes
                    && row[1].nodes > row[2].nodes
                    && row[2].nodes <= row[3].nodes
            );
        }
    }

    #[test]
    fn probe_int32_boundary_includes_integer_acceptance_tables() {
        let mut limits = a4000_limits();
        let conn = Connectivity::Degree(20);
        let shape = conn.shape(10_000);
        let storage =
            |precision| route_storage(KernelKind::Msa, shape, precision, 128, 64, &limits);
        let float = msa_footprint(
            shape,
            Precision::Float32,
            storage(Precision::Float32),
            128,
            64,
            &limits,
        );
        let integer = msa_footprint(
            shape,
            Precision::Int32,
            storage(Precision::Int32),
            128,
            64,
            &limits,
        );
        assert_eq!(integer.total() - float.total(), 525_056);
        limits.free_bytes = (float.total() + 100).div_ceil(4) * 5;
        assert!(float.total() <= limits.usable_bytes());
        assert!(integer.total() > limits.usable_bytes());
        let cells = probe(KernelKind::Msa, &limits, 128);
        let cell = |precision| {
            cells
                .iter()
                .find(|c| c.connectivity == conn && c.precision == precision)
                .unwrap()
        };
        assert_eq!(cell(Precision::Float32).nodes, shape.nodes);
        assert!(cell(Precision::Int32).nodes < shape.nodes);
        assert_eq!(
            cell(Precision::Int32).nodes,
            max_model(KernelKind::Msa, conn, Precision::Int32, 128, &limits).nodes
        );
    }

    #[test]
    fn probe_attaches_calibration_at_measured_reads() {
        for kernel in [KernelKind::Sa, KernelKind::Gibbs] {
            let mut limits = a4000_limits();
            limits.free_bytes = 512 << 20;
            let cells = probe(kernel, &limits, 256);
            let complete_binary = cells
                .iter()
                .find(|cell| {
                    cell.connectivity == Connectivity::Complete
                        && cell.precision == Precision::Binary
                })
                .unwrap();
            assert!(complete_binary.expected_slowdown.is_some(), "{kernel:?}");
            assert!(probe(kernel, &limits, 128)
                .iter()
                .all(|cell| cell.expected_slowdown.is_none()));
        }
        let mut limits = a4000_limits();
        limits.free_bytes = 1 << 30;
        let cells = probe(KernelKind::Msa, &limits, 128);
        assert_eq!(
            cells
                .iter()
                .filter(|cell| cell.expected_slowdown.is_some())
                .count(),
            9
        );
        assert!(probe(KernelKind::Msa, &limits, 64)
            .iter()
            .all(|cell| cell.expected_slowdown.is_none()));
    }
}
