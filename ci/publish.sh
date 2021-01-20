#!/bin/bash

# This script publishes the OpenVINO crates; it depends on:
# - having logged in to Cargo with `cargo login`.
# - having `cargo workspaces` installed (e.g. `cargo install cargo-workspaces`)
#
# In order to publish the openvino-rs crates, we must:
#  - update the versions of both `openvino` and `openvino-sys`, which should match
#  - ensure `openvino` is pointing to the new version of `openvino-sys`
#  - commit the changes and tag the commit with the newly published version
#  - publish `openvino-sys` (ensuring that it will fit within 10MB!), then publish `openvino`
#  - push the commit and tag (this is left for the user)
#
# Also can be run in dry-run mode: `DRY_RUN=1 ci/publish.sh`.

set -e
PROJECT_DIR=$(dirname "$0" | xargs dirname)
DRY_RUN=${DRY_RUN:-0}
if [ $DRY_RUN ]; then
  CARGO_WORKSPACE_OPTIONS="--no-git-commit"
  CARGO_PUBLISH_OPTIONS="--dry-run --allow-dirty"
else
  CARGO_WORKSPACE_OPTIONS="--no-git-push"
  CARGO_PUBLISH_OPTIONS=""
fi

# Extract the version of a Cargo package.
function get_version {
  cargo pkgid $1 | cut -f2 -d'#'
}

# Extract the version of openvino-sys when used as a dependency by openvino.
function get_openvino_sys_dependency_version {
  cat crates/openvino/Cargo.toml | sed -n 's/openvino-sys .* version = "\([0-9.]*\)".*/\1/p'
}

# Publish a crate.
function publish {
  pushd $PROJECT_DIR/crates/$1
  OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo publish $CARGO_PUBLISH_OPTIONS
  popd
}

# Bump the versions of the Cargo.toml files.
cargo workspaces version patch --force openvino* $CARGO_WORKSPACE_OPTIONS

# Check that the versions of openvino and openvino-sys match.
if [ $(get_version openvino) != $(get_version openvino-sys) ]; then
  echo "Package versions are not the same, aborting."
  echo " openvino = $(get_version openvino)"
  echo " openvino-sys = $(get_version openvino-sys)"
  exit 1
fi

# Update the use of openvino-sys as a dependency to the latest version.
sed -i "s/openvino-sys .* version = \"\([0-9.]*\)\".*/openvino-sys = { path = \"..\/openvino-sys\", version = \"$(get_version openvino)\" }/g" crates/openvino/Cargo.toml
cargo fetch
if [[ "$DRY_RUN" == 0 ]]; then
  git commit --amend
fi

# Check that openvino-sys is used with the latest version.
if [ $(get_version openvino-sys) != $(get_openvino_sys_dependency_version) ]; then
  echo "Dependency version is not the same, aborting."
  echo " openvino-sys defined as = $(get_version openvino-sys)"
  echo " openvino-sys used as = $(get_openvino_sys_dependency_version)"
  exit 1
fi

# Publish the crates, waiting in between to allow time for the crate to become available on
# crates.io.
publish openvino-sys
sleep 10
publish openvino

echo ""
echo "Successfully published version $(get_version openvino)"
echo "Do not forget to check the 'git log' and pushing the version commit and tags with 'git push && git push --tags'"
