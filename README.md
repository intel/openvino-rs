# openvino-rs

[![Build Status](https://github.com/intel/openvino-rs/workflows/CI/badge.svg)][ci]
[![Documentation Status](https://docs.rs/openvino/badge.svg)][docs]

This repository contains the [openvino-sys] crate (low-level, unsafe bindings) and the [openvino]
crate (high-level, ergonomic bindings) for accessing OpenVINO™ functionality in Rust.

[openvino-sys]: crates/openvino-sys
[openvino]: crates/openvino
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

The quickest method to build [openvino] and [openvino-sys] is with a local installation of OpenVINO™
(see, e.g., [installing from an apt repository][install-apt]). The build script will attempt to
locate an existing installation (see [openvino-finder]) and link against its shared libraries.
Provide the `OPENVINO_INSTALL_DIR` environment variable to point at a specific installation. Ensure
that the correct libraries are available on the system's load path; OpenVINO™'s `setupvars.sh`
script will do this automatically.

[install-apt]: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html
[openvino-finder]: crates/openvino-finder



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

```shell script
git submodule update --init --recursive
cargo build -vv --features openvino-sys/from-source
cargo test --features openvino-sys/from-source
```

[openvino] and [openvino-sys] can also be built directly from OpenVINO™'s source code using CMake.
This is not tested across all OS and OpenVINO™ versions--use at your own risk! Also, this build
process can be quite slow and there are quite a few dependencies. Some notes:
 - first, install the necessary packages to build OpenVINO™; steps are included in the [CI
   workflow](.github/workflows)
   but reference the [OpenVINO™ build documentation](https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md)
   for the full documentation
 - OpenVINO™ has a plugin system for device-specific libraries (e.g. GPU); building all of these
   libraries along with the core inference libraries can take >20 minutes. To avoid over-long build
   times, [openvino-sys] exposes several Cargo features. By default, [openvino-sys] will only build
   the CPU plugin; to build all plugins, use `--features all` (see
   [Cargo.toml](crates/openvino-sys/Cargo.toml)).
 - OpenVINO™ includes other libraries (e.g. ngraph, tbb); see the
   [build.rs](crates/openvino-sys/build.rs) file for how these are linked to these libraries.



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
