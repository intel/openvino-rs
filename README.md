# openvino-rs

[![Build Status](https://github.com/abrown/openvino-rs/workflows/Rust/badge.svg)][ci]
[![Documentation Status](https://docs.rs/openvino-rs/badge.svg)][docs]

This repository exposes both [low-level bindings][openvino-sys] and [high-level, ergonomic, Rust bindings][openvino] for 
OpenVINO.

[openvino-sys]: crates/openvino-sys
[openvino]: crates/openvino
[upstream]: crates/upstream
[docs]: https://docs.rs/openvino-rs
[ci]: https://github.com/abrown/openvino-rs/actions?query=workflow%3ARust



### Build

```shell script
git submodule update --init --recursive
cargo build -vv
```

[openvino-sys] will attempt to build OpenVINO from [source][upstream] which can take quite some time. The `-vv` on the 
`cargo build` will print the output CMake emits as it compiles OpenVINO.

If the build fails due to missing system dependencies, take a first look at this project's [CI workflow](.github/workflows) 
for a quick overview but refer to the [OpenVINO build documentation](https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md)
for the details.



### Use

After building:
  - peruse the documentation for the [openvino crate][docs]; this is the library you likely want to interact with from
  Rust.
  - follow along with the [classification example](crates/openvino/tests/classify.rs); this example classifies an image 
  using a [pre-built model](crates/openvino/tests/fixture). The examples (and all tests) are runnable using `cargo test`
  but please note that the dependencies required for running these tests (e.g. `opencv`) may not build easily on your 
  system.



### License

`openvino-rs` is released under the same license as OpenVINO: the [Apache License Version 2.0][license]. By 
contributing to the project, you agree to the license and copyright terms therein and release your contribution under
these terms.

[license]: LICENSE
