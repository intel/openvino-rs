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

# Retrieve the AlexNet model, following
# https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/alexnet/model.yml.
wget --no-clobber https://raw.githubusercontent.com/BVLC/caffe/88c96189bcbf3853b93e2b65c7b5e4948f9d5f67/models/bvlc_alexnet/deploy.prototxt
wget --no-clobber http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
sed -i 's/dim: 10/dim: 1/' deploy.prototxt

# Convert the model to OpenVINO IR using the model-optimizer.
pip install --user -r $OPENVINO_DIR/model-optimizer/requirements_caffe.txt 
$PYTHON $OPENVINO_DIR/model-optimizer/mo_caffe.py \
  --input_shape=[1,3,227,227] \
  --input=data \
  --mean_values=data[104.0,117.0,123.0] \
  --output=prob \
  --input_model=$TMP_DIR/bvlc_alexnet.caffemodel \
  --input_proto=$TMP_DIR/deploy.prototxt
cp $TMP_DIR/bvlc_alexnet.bin $FIXTURE_DIR/alexnet.bin
cp $TMP_DIR/bvlc_alexnet.mapping $FIXTURE_DIR/alexnet.mapping
cp $TMP_DIR/bvlc_alexnet.xml $FIXTURE_DIR/alexnet.xml

# Retrieve the first 10 images of the COCO dataset.
wget --no-clobber http://images.cocodataset.org/zips/val2017.zip
rm -rf val2017
unzip -Z1 val2017.zip | head -n 10 | xargs unzip val2017.zip
popd

# Convert an image to raw tensor format
cargo run -p openvino-tensor-converter -- $TMP_DIR/val2017/000000062808.jpg $FIXTURE_DIR/tensor-1x3x227x227-f32.bgr 227x227x3xfp32

# Clean up.
rm -rf $TMP_DIR
