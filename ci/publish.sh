#!/bin/bash

# This script publishes the OpenVINO crates; it depends on having logged in to Cargo with `cargo
# login`.
set -e
PROJECT_DIR=$(dirname "$0" | xargs dirname)

function get_version {
  cargo pkgid $1 | cut -f2 -d'#'
}

function publish {
  pushd $PROJECT_DIR/crates/$1
  OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo publish
  popd
}

if [ $(get_version openvino) != $(get_version openvino-sys) ]; then
  echo "Package versions are not the same, aborting."
  echo " openvino = $(get_version openvino)"
  echo " openvino-sys = $(get_version openvino-sys)"
  exit 1
fi

publish openvino-sys
sleep 10
publish openvino