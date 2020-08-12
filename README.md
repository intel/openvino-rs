# openvino-rs

This repository exposes both [low-level bindings][openvino-sys] and [high-level, ergonomic, Rust bindings][openvino] for 
OpenVINO.

[openvino-sys]: crates/openvino-sys
[openvino]: crates/openvino
[upstream]: crates/upstream

### Build

```shell script
git submodule update --init --recursive
cargo build -vv
```

If OpenVINO's inference libraries are not present on the system (and they are likely not), then [openvino-sys] will
attempt to build OpenVINO from [source][upstream], which can take quite some time. The `-vv` on the cargo build will
print the output CMake emits as it compiles OpenVINO.
