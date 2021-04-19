openvino-finder
===============

A utility for locating OpenVINO™ libraries on a host system.

> #### WARNING
> This crate is currently experimental--its API surface is subject to change.

### Use

```Rust
let path = openvino_finder::find("inference_engine_c_api").unwrap();
```
