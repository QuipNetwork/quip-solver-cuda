# CI toolchain image: the CUDA 12.9 toolkit that the crate's cuda-12090 cudarc
# pin targets, the build tools, and Rust 1.97.1 with clippy and rustfmt. A
# CUDA-less image cannot catch a pin and toolkit mismatch.
#
# The `build-toolchain-image` job in `.gitlab-ci.yml` builds and pushes this
# image. It runs automatically on a branch push that changes this file and is
# manual otherwise. The jobs pull the image by digest, so a rebuild takes
# effect only when a commit updates TOOLCHAIN_IMAGE. `make ci-image` builds the
# same file for local runs.

# Nothing watches these pins for updates. On a rebuild, check the nvidia/cuda
# tags (the toolkit moves only with the cudarc pin) and keep the Rust version
# equal to rust-toolchain.toml.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The toolchain lives outside CARGO_HOME, because the CI jobs point CARGO_HOME
# at the checkout for the registry cache. rustup finds the toolchain through
# RUSTUP_HOME and the proxies through PATH.
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
            --default-toolchain 1.97.1 --component clippy,rustfmt \
    && chmod -R a+rwX /opt/rustup /opt/cargo

WORKDIR /src

# Fail the image build here, not in a CI job.
RUN cargo --version && cargo clippy --version && cargo fmt --version \
    && nvcc --version | tail -1 \
    && ls /usr/local/cuda/lib64/libnvrtc.so.12
