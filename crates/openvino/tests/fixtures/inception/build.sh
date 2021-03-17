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
wget --no-clobber https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar xzvf inception_*.tar.gz

# Convert the model to OpenVINO IR using the model-optimizer; see
# https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/googlenet-v3/model.yml.
pip install --user -r $OPENVINO_DIR/model-optimizer/requirements_tf.txt 
$PYTHON $OPENVINO_DIR/model-optimizer/mo_tf.py \
  --reverse_input_channels \
  --input_shape=[1,299,299,3] \
  --input=input \
  --mean_values=input[127.5,127.5,127.5] \
  --scale_values=input[127.5] \
  --output=InceptionV3/Predictions/Softmax \
  --input_model=inception_v3_2016_08_28_frozen.pb
cp $TMP_DIR/inception_v3_2016_08_28_frozen.bin $FIXTURE_DIR/inception.bin
cp $TMP_DIR/inception_v3_2016_08_28_frozen.mapping $FIXTURE_DIR/inception.mapping
cp $TMP_DIR/inception_v3_2016_08_28_frozen.xml $FIXTURE_DIR/inception.xml

# Retrieve the first 10 images of the COCO dataset.
wget --no-clobber http://images.cocodataset.org/zips/val2017.zip
rm -rf val2017
unzip -Z1 val2017.zip | head -n 10 | xargs unzip val2017.zip
popd

# Convert an image to raw tensor format
cargo run -p openvino-tensor-converter -- $TMP_DIR/val2017/000000062808.jpg $FIXTURE_DIR/tensor-1x3x299x299-f32.bgr 299x299x3xfp32

# Clean up.
rm -rf $TMP_DIR
