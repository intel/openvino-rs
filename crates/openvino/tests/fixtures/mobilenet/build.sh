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

# Retrieve the MobileNet model, following
# https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v2-1.0-224/model.yml.
wget --no-clobber https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
tar zxvf mobilenet_v2_1.0_224.tgz

# Convert the model to OpenVINO IR using the model-optimizer.
pip install --user -r $OPENVINO_DIR/model-optimizer/requirements_tf.txt 
$PYTHON $OPENVINO_DIR/model-optimizer/mo_tf.py \
  --reverse_input_channels \
  --input_shape=[1,224,224,3] \
  --input=input \
  --mean_values=input[127.5,127.5,127.5] \
  --scale_values=input[127.5] \
  --output=MobilenetV2/Predictions/Reshape_1 \
  --input_model=$TMP_DIR/mobilenet_v2_1.0_224_frozen.pb

cp $TMP_DIR/mobilenet_v2_1.0_224_frozen.bin $FIXTURE_DIR/mobilenet.bin
cp $TMP_DIR/mobilenet_v2_1.0_224_frozen.mapping $FIXTURE_DIR/mobilenet.mapping
cp $TMP_DIR/mobilenet_v2_1.0_224_frozen.xml $FIXTURE_DIR/mobilenet.xml

# Retrieve the first 10 images of the COCO dataset.
wget --no-clobber http://images.cocodataset.org/zips/val2017.zip
rm -rf val2017
unzip -Z1 val2017.zip | head -n 10 | xargs unzip val2017.zip
popd

# Convert an image to raw tensor format. Weirdly, this actually produces the correct output tensor:
# I would have expected to have to transpose tensor dimensions for MobileNet (i.e. 224x224x3 to
# 3x224x224) but this works. Perhaps the model-optimizer sets things to an OpenVINO-expected shape.
cargo run -p openvino-tensor-converter -- $TMP_DIR/val2017/000000062808.jpg $FIXTURE_DIR/tensor-1x224x224x3-f32.bgr 224x224x3xfp32

# Clean up.
rm -rf $TMP_DIR
