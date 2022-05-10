# This Dockerfile demonstrates how to build the openvino bindings using an installation of OpenVINO.
# For instructions to install OpenVINO see the OpenVINO documentation, e.g.
# https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html.
ARG OS=ubuntu18
ARG VERSION=2020.4
FROM openvino/${OS}_runtime:${VERSION} AS builder

# OpenVINO's images use a default user, `openvino`, that disallows root access.
USER root

# Install Rust.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN rustup component add rustfmt

# Install build dependencies (for bindgen).
RUN apt update && apt install -y clang libclang-dev

# Copy in source code.
WORKDIR /usr/src/openvino-rs
COPY . .

# Build openvino libraries.
RUN OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo build -vv

# Test; note that we need to setup the library paths before using them since the
# OPENVINO_INSTALL_DIR can only affect the build library search path.
RUN ["/bin/bash", "-c", "source /opt/intel/openvino/setupvars.sh && OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo test -v"]
