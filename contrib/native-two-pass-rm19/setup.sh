#!/bin/sh
# One-time setup: the solver needs a small patch to the published `ising-generators` 0.0.15 crate (a flag that skips the generator's
# metadata self-check, cached per-topology tables, and gpugen.rs, the table export used by the GPU screening code). The output of
# the generator is bit-identical to the unpatched crate (checked on 20,000 salts).
set -eu
cd "$(dirname "$0")"
# 1. fetch the published crate with the patch section disabled
sed -i 's/^#\(\[patch.crates-io\]\)/\1/; s/^#\(ising-generators = { path = "vendor\/ising-generators" }\)/\1/' Cargo.toml
sed -i 's/^\[patch.crates-io\]/#[patch.crates-io]/; s/^ising-generators = { path = "vendor\/ising-generators" }/#ising-generators = { path = "vendor\/ising-generators" }/' Cargo.toml
cargo fetch
REG=$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/ising-generators-0.0.15 | head -1)
[ -d "$REG" ] || { echo "ising-generators 0.0.15 not found in the cargo registry"; exit 1; }
# 2. copy and patch it
rm -rf vendor/ising-generators; mkdir -p vendor; cp -r "$REG" vendor/ising-generators; chmod -R u+w vendor/ising-generators
( cd vendor/ising-generators && patch -p1 < ../../patches/ising-generators-0.0.15.patch )
# 3. enable the [patch] section
sed -i 's/^#\[patch.crates-io\]/[patch.crates-io]/; s/^#ising-generators = { path = "vendor\/ising-generators" }/ising-generators = { path = "vendor\/ising-generators" }/' Cargo.toml
echo "ready: cargo build --release"
