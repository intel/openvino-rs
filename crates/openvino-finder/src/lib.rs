use std::env;
use std::path::PathBuf;

/// Find the path to an OpenVINO library. This will try:
/// - the `OPENVINO_INSTALL_DIR` environment variable with several subdirectories appended
/// - the `INTEL_OPENVINO_DIR` environment variable with several subdirectories appended
/// - the environment's library path (e.g. `LD_LIBRARY_PATH` in Linux)
/// - OpenVINO's default installation paths for the OS
pub fn find(library_name: &str) -> Option<PathBuf> {
    let file = format!(
        "{}{}{}",
        env::consts::DLL_PREFIX,
        library_name,
        env::consts::DLL_SUFFIX
    );

    // Search using the `OPENVINO_INSTALL_DIR` environment variable; this may be set by users of the
    // openvino-rs library.
    if let Some(install_dir) = env::var_os(ENV_OPENVINO_INSTALL_DIR) {
        let install_dir = PathBuf::from(install_dir);
        for lib_dir in KNOWN_INSTALLATION_SUBDIRECTORIES {
            let search_path = install_dir.join(lib_dir).join(&file);
            if search_path.is_file() {
                return Some(search_path);
            }
        }
    }

    // Search using the `INTEL_OPENVINO_DIR` environment variable; this is set up by an OpenVINO
    // installation (e.g. `source /opt/intel/openvino/bin/setupvars.sh`).
    if let Some(install_dir) = env::var_os(ENV_INTEL_OPENVINO_DIR) {
        let install_dir = PathBuf::from(install_dir);
        for lib_dir in KNOWN_INSTALLATION_SUBDIRECTORIES {
            let search_path = install_dir.join(lib_dir).join(&file);
            if search_path.is_file() {
                return Some(search_path);
            }
        }
    }

    // Search in the OS library path (i.e. `LD_LIBRARY_PATH` on Linux, `PATH` on Windows, and
    // `DYLD_LIBRARY_PATH` on MacOS). See
    // https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html
    if let Some(path) = env::var_os(ENV_LIBRARY_PATH) {
        for lib_dir in env::split_paths(&path) {
            let search_path = lib_dir.join(&file);
            if search_path.is_file() {
                return Some(search_path);
            }
        }
    }

    // Search in OpenVINO's default installation directories (if they exist).
    for default_dir in DEFAULT_INSTALLATION_DIRECTORIES
        .iter()
        .map(PathBuf::from)
        .filter(|d| d.is_dir())
    {
        for lib_dir in KNOWN_INSTALLATION_SUBDIRECTORIES {
            let search_path = default_dir.join(lib_dir).join(&file);
            if search_path.is_file() {
                return Some(search_path);
            }
        }
    }

    None
}

const ENV_OPENVINO_INSTALL_DIR: &'static str = "OPENVINO_INSTALL_DIR";

const ENV_INTEL_OPENVINO_DIR: &'static str = "INTEL_OPENVINO_DIR";

#[cfg(target_os = "linux")]
const ENV_LIBRARY_PATH: &'static str = "LD_LIBRARY_PATH";
#[cfg(target_os = "macos")]
const ENV_LIBRARY_PATH: &'static str = "DYLD_LIBRARY_PATH";
#[cfg(target_os = "windows")]
const ENV_LIBRARY_PATH: &'static str = "PATH";

#[cfg(any(target_os = "linux", target_os = "macos"))]
const DEFAULT_INSTALLATION_DIRECTORIES: &'static [&'static str] =
    &["/opt/intel/openvino", "/opt/intel/openvino_2021"];
#[cfg(target_os = "windows")]
const DEFAULT_INSTALLATION_DIRECTORIES: &'static [&'static str] = &[
    "C:\\Program Files (x86)\\Intel\\openvino",
    "C:\\Program Files (x86)\\Intel\\openvino_2021",
];

const KNOWN_INSTALLATION_SUBDIRECTORIES: &'static [&'static str] = &[
    "deployment_tools/ngraph/lib",
    "deployment_tools/inference_engine/lib/intel64",
    "deployment_tools/inference_engine/external/hddl/lib",
    "deployment_tools/inference_engine/external/gna/lib",
    "deployment_tools/inference_engine/external/tbb/lib",
];

#[cfg(test)]
mod test {
    use super::*;

    /// This test uses `find` to search for the `inference_engine_c_api` library on the local
    /// system.
    #[test]
    fn find_inference_engine_c_api_locally() {
        assert!(find("inference_engine_c_api").is_some());
    }
}
