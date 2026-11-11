# Local + CI entry points. Everything runs inside the CUDA 12.9 CI image so a
# local run and a pipeline run exercise the same toolkit (see the spec:
# quip-solver docs/superpowers/specs/2026-08-14-cuda-version-target-design.md).
CI_IMAGE := quip-solver-cuda-ci
# `-e CI=1` marks the container as the pinned-toolkit environment. tests/arch_coverage.rs
# skips without it, because off CUDA 12.9 those tests report the toolkit, not the kernels.
# GitLab sets CI itself; docker run inherits nothing, so the flag is what keeps
# `make test-archs` a real run rather than a silent skip.
RUN_CI   := docker run --rm -e CI=1 -v $(PWD):/src -v quip-cuda-cargo:/opt/cargo/registry \
            -v quip-cuda-target:/src/target -w /src $(CI_IMAGE)
RUN_GPU  := docker run --rm --gpus all -e CI=1 -v $(PWD):/src -v quip-cuda-cargo:/opt/cargo/registry \
            -v quip-cuda-target:/src/target -w /src $(CI_IMAGE)

.PHONY: ci-image check test test-archs test-gpu test-long bench-arch

ci-image:
	docker build -t $(CI_IMAGE) - < .gitlab/ci-toolchain.Dockerfile

check: ci-image
	$(RUN_CI) sh -c 'cargo fmt --all --check && cargo clippy --release --all-targets --all-features -- -D warnings'

test: ci-image
	$(RUN_CI) cargo test --release

# Arch coverage needs NVRTC 12.9 but no GPU.
test-archs: ci-image
	$(RUN_CI) cargo test --release --test arch_coverage -- --include-ignored

# test-gpu runs the fast GPU suite with CI time limits; test-long adds manual long tests.
test-gpu: ci-image
	$(RUN_GPU) env RUSTC_BOOTSTRAP=1 RUST_TEST_TIME_UNIT=10000,30000 \
	    RUST_TEST_TIME_INTEGRATION=10000,30000 RUST_TEST_TIME_DOCTEST=10000,30000 \
	    cargo test --release -- --include-ignored --test-threads=1 -Z unstable-options --ensure-time

test-long: ci-image
	$(RUN_GPU) cargo test --release --features long-tests -- --include-ignored

# Benchmark for the spec's D4 gate. OUT differs per run: make bench-arch OUT=bench-out/baseline
OUT ?= bench-out/run
bench-arch: ci-image
	mkdir -p $(OUT)
	$(RUN_GPU) sh -c 'cargo run --release --bin quip-cuda-sa -- bench run \
	    --reads 8 --sweeps 1024 --sweeps-per-beta 4 --nodes 512 --repeats 5 --out /src/$(OUT)'
