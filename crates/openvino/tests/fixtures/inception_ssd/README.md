# Inception SSD Test Fixture

In order to test the use of `openvino-rs` in the real world, here we include the necessary files for
performing single-shot detection (SSD) in [detect-inception.rs](../detect-inception.rs). The
artifacts are included in-tree and can be rebuilt using the [build.sh] script (with the right system
dependencies). The artifacts include:
 - the Inception SSD inference model, converted to OpenVINO IR format (`*.bin`, `*.mapping`, `*.xml`)
 - an image from the COCO dataset transformed into the correct tensor format (`tensor-*.bgr`)

The [mod.sh] `Fixture` provides the correct artifact paths in Rust.
