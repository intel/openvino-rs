# openvino-sys

The [openvino-sys] crate provides low-level, unsafe, Rust bindings to OpenVINO™ using its [C API]. If you are looking to use OpenVINO™ from Rust, you likely should look at the ergonomic, safe bindings in [openvino], which depends on this crate. See the repository [README] for more information, including build instructions. 

> #### WARNING
> This crate is currently [under review]--its source location and API surface are subject to change based on that review.

[C API]: https://docs.openvinotoolkit.org/2020.1/ie_c_api/groups.html
[openvino-sys]: https://crates.io/crates/openvino-sys
[openvino]: https://crates.io/crates/openvino
[README]: https://github.com/abrown/openvino/blob/rust-bridge/inference-engine/ie_bridges/rust/README.md
[under review]: https://github.com/openvinotoolkit/openvino/pull/2342
[upstream]: upstream
