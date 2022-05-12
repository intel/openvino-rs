openvino-finder
===============

A utility for locating OpenVINO™ libraries on a host system. It will attempt to find the OpenVINO
shared libraries in:
- the `OPENVINO_BUILD_DIR` environment variable (pointed at a locally-built OpenVINO repository)
- the `OPENVINO_INSTALL_DIR` environment variable (pointed at the top-level OpenVINO installation,
  e.g., `/opt/intel/openvino`)
- the `INTEL_OPENVINO_DIR` environment variable (same as above; this is set by OpenVINO setup
  scripts)
- the environment's library path (e.g., `LD_LIBRARY_PATH` in Linux; this is also set by the OpenVINO
  setup scripts)
- OpenVINO's default installation paths for the OS (a best effort attempt)

### Use

```Rust
let path = openvino_finder::find("inference_engine_c_api").unwrap();
```
