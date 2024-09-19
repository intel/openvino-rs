//! To realistically test `openvino-rs`, this module retrieves the necessary files for running
//! inference integration tests (e.g., `classify-*.rs`).
//!
//! The [`download`] function does all the work, relying on `curl` being installed to download the
//! files. Files are retrieved as they are used by `Fixture` structures inside each sub-module and
//! are saved in the same directory structure as on the remote server. This means each fixture
//! directory (e.g., the `alexnet` target directory) must be present in the Git tree to avoid
//! errors.
//!
//! The reason for this retrieval process is to avoid bandwidth costs: the files are large and the
//! cost to retrieve them on each test run can add up. Also, some of the files are too large for
//! GitHub's 100MB limit, requiring Git LFS support with its additional cost. This "retrieve on
//! demand" approach removes the Git LFS dependency and allows us to use GitHub action caching.

#![allow(dead_code)] // Rust finds it hard to see that the sub-module functions are used in tests.

use std::path::PathBuf;
use std::process::Command;

const BASE_FIXTURES_URL: &str = "https://download.01.org/openvinotoolkit/fixtures";

/// Download `from` a relative URL path `to` the filesystem using `curl`.
///
/// This will:
/// - skip the download if the file already exists
/// - append `to` to the `BASE_FIXTURES_URL` to create the URL
/// - download the file using `curl`
/// - store the file in the current directory.
///
/// This relies on the fixtures being stored remotely in the same directory structure as here.
pub fn download(from: &str) -> anyhow::Result<PathBuf> {
    let to = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(from);
    if to.exists() {
        println!("> skipping: {}", from);
        return Ok(to);
    }

    let url = format!("{BASE_FIXTURES_URL}/{from}");
    let mut curl = Command::new("curl");
    curl.arg("--location").arg(url).arg("--output").arg(&to);
    println!("> downloading: {:?}", &curl);
    let result = curl.output().unwrap();
    if !result.status.success() {
        panic!(
            "curl failed: {}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
    }

    Ok(to)
}

/// Retrieve the files necessary for running the `alexnet` classification example.
///
/// The artifacts, stored remotely, were built using the remote `build.sh` script (with the right
/// system dependencies). The artifacts include:
/// - the AlexNet inference model, converted to OpenVINO IR format (`*.bin`, `*.mapping`, `*.xml`)
/// - an image from the COCO dataset transformed into the correct tensor format (`tensor-*.bgr`)
pub mod alexnet {
    use super::download;
    use std::path::PathBuf;
    pub fn graph() -> PathBuf {
        download("alexnet/alexnet.xml").unwrap()
    }
    pub fn weights() -> PathBuf {
        download("alexnet/alexnet.bin").unwrap()
    }
    pub fn tensor() -> PathBuf {
        download("alexnet/tensor-1x3x227x227-f32.bgr").unwrap()
    }
}

/// Retrieve the files necessary for running the `inception` classification example.
///
/// The artifacts, stored remotely, were built using the remote `build.sh` script (with the right
/// system dependencies). The artifacts include:
/// - the Inception v3 inference model, converted to OpenVINO IR format (`*.bin`, `*.mapping`,
///   `*.xml`)
/// - an image from the COCO dataset transformed into the correct tensor format (`tensor-*.bgr`)
pub mod inception {
    use super::download;
    use std::path::PathBuf;
    pub fn graph() -> PathBuf {
        download("inception/inception.xml").unwrap()
    }
    pub fn weights() -> PathBuf {
        download("inception/inception.bin").unwrap()
    }
    pub fn tensor() -> PathBuf {
        download("inception/tensor-1x3x299x299-f32.bgr").unwrap()
    }
}

/// Retrieve the files necessary for running the `mobilenet` classification example.
///
/// The artifacts, stored remotely, were built using the remote `build.sh` script (with the right
/// system dependencies). The artifacts include:
/// - the MobileNet inference model, converted to OpenVINO IR format (`*.bin`, `*.mapping`, `*.xml`)
///   using [this guide]
/// - an image from the COCO dataset transformed into the correct tensor format (`tensor-*.bgr`)
///
/// [this guide]:
///     https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v2-1.0-224/model.yml
pub mod mobilenet {
    use super::download;
    use std::path::PathBuf;
    pub fn graph() -> PathBuf {
        download("mobilenet/mobilenet.xml").unwrap()
    }
    pub fn weights() -> PathBuf {
        download("mobilenet/mobilenet.bin").unwrap()
    }
    pub fn tensor() -> PathBuf {
        download("mobilenet/tensor-1x224x224x3-f32.bgr").unwrap()
    }
}
