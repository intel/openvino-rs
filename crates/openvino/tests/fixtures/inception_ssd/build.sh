#!/bin/bash

# The following script rebuilds the test fixture committed in this directory. It relies on external tools that not all
# systems will have readily available and as such it should be used mainly as an example script. The artifacts created
# are checked in to save future users the setup time.
set -e
TMP_DIR=${1:-$(mktemp -d -t ci-XXXXXXXXXX)}
FIXTURE_DIR=$(dirname "$0" | xargs realpath)
OPENVINO_DIR=${OPENVINO_DIR:-$(realpath $FIXTURE_DIR/../../../../../../openvino)}
PYTHON=${PYTHON:-python3}
pushd $TMP_DIR

# Retrieve the Inception model from the TensorFlow 
wget --no-clobber http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar xzvf ssd_*.tar.gz
ln -sf ssd_inception_v2_coco_2018_01_28 model

# Convert the model to OpenVINO IR using the model-optimizer.
pip install --user -r $OPENVINO_DIR/model-optimizer/requirements_tf.txt 
$PYTHON $OPENVINO_DIR/model-optimizer/mo_tf.py \
  --input_model model/frozen_inference_graph.pb \
  --transformations_config $OPENVINO_DIR/model-optimizer/extensions/front/tf/ssd_v2_support.json \
  --tensorflow_object_detection_api_pipeline_config model/pipeline.config 
cp $TMP_DIR/frozen_inference_graph.bin $FIXTURE_DIR/inception-ssd.bin
cp $TMP_DIR/frozen_inference_graph.mapping $FIXTURE_DIR/inception-ssd.mapping
cp $TMP_DIR/frozen_inference_graph.xml $FIXTURE_DIR/inception-ssd.xml

# Retrieve the first 10 images of the COCO dataset.
wget --no-clobber http://images.cocodataset.org/zips/val2017.zip
rm -rf val2017
unzip -Z1 val2017.zip | head -n 10 | xargs unzip val2017.zip
popd

# Convert an image to raw tensor format
cargo run -p openvino-tensor-converter -- $TMP_DIR/val2017/000000062808.jpg $FIXTURE_DIR/tensor-1x3x640x481-u8.bgr 481x640x3xu8

# Clean up.
rm -rf $TMP_DIR
