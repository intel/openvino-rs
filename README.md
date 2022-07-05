# openvino-rs

[![Build Status](https://github.com/intel/openvino-rs/workflows/CI/badge.svg)][ci]
[![Documentation Status](https://docs.rs/openvino/badge.svg)][docs]

This repository contains the [openvino-sys] crate (low-level, unsafe bindings) and the [openvino]
crate (high-level, ergonomic bindings) for accessing OpenVINO™ functionality in Rust.

[openvino]: crates/openvino
[openvino-sys]: crates/openvino-sys
[openvino-finder]: crates/openvino-finder
[upstream]: crates/openvino-sys/upstream
[docs]: https://docs.rs/openvino
[ci]: https://github.com/abrown/openvino-rs/actions?query=workflow%3ACI



### Prerequisites

The [openvino-sys] crate creates bindings to the OpenVINO™ C API using `bindgen`; this requires a
local installation of `libclang`. Also, be sure to retrieve all Git submodules.

This repo currently uses [git-lfs](https://git-lfs.github.com/) for large file storage. If you
[install it](https://github.com/git-lfs/git-lfs/wiki/Installation) before cloning this repository,
it should have downloaded all large files. To check this, verify that `find crates/openvino -name
*.bin | xargs ls -lhr` returns `.bin` files of tens of megabytes. If not, download the large files
with:

```shell
git lfs fetch
git lfs checkout
```


### Build from an OpenVINO™ installation

```shell script
cargo build
source /opt/intel/openvino/setupvars.sh
cargo test
```

The quickest method to build the [openvino] and [openvino-sys] crates is with a local installation
of OpenVINO™ (see, e.g., [installing from an apt repository][install-apt]). The build script will
attempt to locate an existing installation (see [openvino-finder]) and link against its shared
libraries. Provide the `OPENVINO_INSTALL_DIR` environment variable to point at a specific
installation. Ensure that the correct libraries are available on the system's load path; OpenVINO™'s
`setupvars.sh` script will do this automatically.

[install-apt]: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html



### Build for runtime linking

```shell script
cargo build --features openvino-sys/runtime-linking
source /opt/intel/openvino/setupvars.sh
cargo test --features openvino-sys/runtime-linking
```

The `openvino-rs` crates also support linking from a shared library at runtime (i.e.
`dlopen`-style). This allow building the crates with no OpenVINO™ installation or source code
present and only later--at runtime--providing the OpenVINO™ shared libraries. All underlying system
calls are wrapped so that a call to `openvino_sys::library::load` will link them to their shared
library implementation (using the logic in [openvino-finder] to locate the shared libraries). For
high-level users, call `openvino::Core::new` first to automatically load and link the libraries.



### Build from OpenVINO™ sources

First, build OpenVINO by cloning the [openvino] repository and following the [OpenVINO™ build
documentation]. Then, using the top-level directory as `<openvino-repo>` (not the CMake build
directory), build this crate:

```shell script
OPENVINO_BUILD_DIR=<openvino-repo> cargo build
OPENVINO_BUILD_DIR=<openvino-repo> cargo test
```

Some important notes about the path passed in `OPENVINO_BUILD_DIR`:
- `<openvino-repo>` should be an absolute path (or at least a path relative to the
  `crates/openvino-sys` directory, which is the current directory when used at build time)
- `<openvino-repo>` should either be outside of this crate's tree or in the `target` directory (see
  the limitations on [`cargo:rustc-link-search`]).

The various OpenVINO libraries and dependencies are found using the [openvino-finder] crate. Turn on
logging to troubleshoot any issues finding the right libraries, e.g., `RUST_LOG=debug
OPENVINO_BUILD_DIR=... cargo build -vv`.

[openvino]: https://github.com/openvinotoolkit/openvino
[OpenVINO™ build documentation]: https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md
[`cargo:rustc-link-search`]: https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-search



### Build without linking to OpenVINO™

```shell script
OPENVINO_SKIP_LINKING=1 cargo build -vv
```

In some environments it may be necessary to build the [openvino-sys] crate without linking to the
OpenVINO libraries (e.g. for *docs.rs* builds). In these cases, use the `OPENVINO_SKIP_LINKING`
environment variable to skip linking entirely. The compiled crate will likely not work as expected
(e.g., for inference), but it should build.



### Use

After building:
  - peruse the documentation for the [openvino crate][docs]; this is the library you likely want to
    interact with from Rust.
  - follow along one with of the [classification examples](crates/openvino/tests); these examples
    classifies an image using pre-built models. The examples (and all tests) are runnable using
    `cargo test` (or `OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo test` when building from an
    installation).



### Development

Run `cargo xtask --help` to read up on the in-tree development tools.



### License

`openvino-rs` is released under the same license as OpenVINO™: the [Apache License Version
2.0][license]. By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

[license]: LICENSE
