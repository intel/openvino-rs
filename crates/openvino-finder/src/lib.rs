//! Provides a mechanism for locating the OpenVINO shared libraries installed on a system.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::must_use_candidate)]

use cfg_if::cfg_if;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// We search for the library in various different places and early-return if we find it.
macro_rules! check_and_return {
    ($path: expr) => {
        log::debug!("Searching in: {}", $path.display());
        if $path.is_file() {
            log::info!("Found library at path: {}", $path.display());
            return Some($path);
        }
    };
}

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
    log::info!("Attempting to find library: {}", file);

    // Search using the `OPENVINO_BUILD_DIR` environment variable; this may be set by users of the
    // `openvino-rs` library.
    if let Some(build_dir) = env::var_os(ENV_OPENVINO_BUILD_DIR) {
        let install_dir = PathBuf::from(build_dir);
        for lib_dir in KNOWN_BUILD_SUBDIRECTORIES {
            let search_path = install_dir.join(lib_dir).join(&file);
            check_and_return!(search_path);
        }
    }

    // Search using the `OPENVINO_INSTALL_DIR` environment variable; this may be set by users of the
    // `openvino-rs` library.
    if let Some(install_dir) = env::var_os(ENV_OPENVINO_INSTALL_DIR) {
        let install_dir = PathBuf::from(install_dir);
        for lib_dir in KNOWN_INSTALLATION_SUBDIRECTORIES {
            let search_path = install_dir.join(lib_dir).join(&file);
            check_and_return!(search_path);
        }
    }

    // Search using the `INTEL_OPENVINO_DIR` environment variable; this is set up by an OpenVINO
    // installation (e.g. `source /opt/intel/openvino/setupvars.sh`).
    if let Some(install_dir) = env::var_os(ENV_INTEL_OPENVINO_DIR) {
        let install_dir = PathBuf::from(install_dir);
        for lib_dir in KNOWN_INSTALLATION_SUBDIRECTORIES {
            let search_path = install_dir.join(lib_dir).join(&file);
            check_and_return!(search_path);
        }
    }

    // Search in the OS library path (i.e. `LD_LIBRARY_PATH` on Linux, `PATH` on Windows, and
    // `DYLD_LIBRARY_PATH` on MacOS).
    if let Some(path) = env::var_os(ENV_LIBRARY_PATH) {
        for lib_dir in env::split_paths(&path) {
            let search_path = lib_dir.join(&file);
            check_and_return!(search_path);
        }
    }

    // Search in OpenVINO's installation directories; after v2022.3, Linux packages will be
    // installed in the system's default library locations.
    for install_dir in SYSTEM_INSTALLATION_DIRECTORIES
        .iter()
        .map(PathBuf::from)
        .filter(|d| d.is_dir())
    {
        // Check if the file is located in the installation directory.
        let search_path = install_dir.join(&file);
        check_and_return!(search_path);

        // Otherwise, check for version terminators: e.g., `libfoo.so.3.1.2`.
        let filenames = list_directory(&install_dir).expect("cannot list installation directory");
        let versions = get_suffixes(filenames, &file);
        if let Some(path) = build_latest_version(&install_dir, &file, versions) {
            check_and_return!(path);
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
            check_and_return!(search_path);
        }
    }

    None
}

const ENV_OPENVINO_INSTALL_DIR: &str = "OPENVINO_INSTALL_DIR";
const ENV_OPENVINO_BUILD_DIR: &str = "OPENVINO_BUILD_DIR";
const ENV_INTEL_OPENVINO_DIR: &str = "INTEL_OPENVINO_DIR";
const ENV_OPENVINO_PLUGINS_XML: &str = "OPENVINO_PLUGINS_XML";

cfg_if! {
    if #[cfg(any(target_os = "linux"))] {
        const ENV_LIBRARY_PATH: &str = "LD_LIBRARY_PATH";
    } else if #[cfg(target_os = "macos")] {
        const ENV_LIBRARY_PATH: &str = "DYLD_LIBRARY_PATH";
    } else if #[cfg(target_os = "windows")] {
        const ENV_LIBRARY_PATH: &str = "PATH";
    } else {
        // This may not work but seems like a sane default for target OS' not listed above.
        const ENV_LIBRARY_PATH: &str = "LD_LIBRARY_PATH";
    }
}

cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        const DEFAULT_INSTALLATION_DIRECTORIES: &[&str] = &[
            "/opt/intel/openvino_2022",
            "/opt/intel/openvino",
        ];
    } else if #[cfg(target_os = "windows")] {
        const DEFAULT_INSTALLATION_DIRECTORIES: &[&str] = &[
            "C:\\Program Files (x86)\\Intel\\openvino_2022",
            "C:\\Program Files (x86)\\Intel\\openvino",
        ];
    } else {
        const DEFAULT_INSTALLATION_DIRECTORIES: &[&str] = &[];
    }
}

cfg_if! {
    if #[cfg(target_os = "linux")] {
        const SYSTEM_INSTALLATION_DIRECTORIES: &[&str] = &[
            "/usr/lib/x86_64-linux-gnu", // DEB-installed package (OpenVINO >= 2022.3)
            "/lib/x86_64-linux-gnu", // DEB-installed package (TBB)
            "/usr/lib64", // RPM-installed package >= 2022.3
        ];
    } else {
        const SYSTEM_INSTALLATION_DIRECTORIES: &[&str] = &[];
    }
}

const KNOWN_INSTALLATION_SUBDIRECTORIES: &[&str] =
    &["runtime/lib/intel64", "runtime/3rdparty/tbb/lib"];

const KNOWN_BUILD_SUBDIRECTORIES: &[&str] = &[
    "bin/intel64/Debug/lib",
    "bin/intel64/Release/lib",
    "temp/tbb/lib",
];

/// Find the path to the `plugins.xml` configuration file.
///
/// OpenVINO records the location of its plugin libraries in a `plugins.xml` file. This file is
/// examined by OpenVINO on initialization; not knowing the location to this file can lead to
/// inference errors later (e.g., `Inference(GeneralError)`). OpenVINO uses the `plugins.xml` file
/// to load target-specific libraries on demand for performing inference. The `plugins.xml` file
/// maps targets (e.g., CPU) to their target-specific implementation library.
///
/// This file can be found in multiple locations, depending on the installation mechanism. For TAR
/// installations, it is found in the same directory as the OpenVINO libraries themselves. For
/// DEB/RPM installations, it is found in a version-suffixed directory beside the OpenVINO libraries
/// (e.g., `openvino-2022.3.0/plugins.xml`).
///
/// This function will check:
/// - the `OPENVINO_PLUGINS_XML` environment variable--this is specific to this library
/// - the same directory as the `openvino_c` shared library, as discovered by [find]
/// - the latest version directory beside the `openvino_c` shared library (i.e.,
///   `openvino-<version>/`)
pub fn find_plugins_xml() -> Option<PathBuf> {
    const FILE_NAME: &str = "plugins.xml";

    // The `OPENVINO_PLUGINS_XML` should point directly to the file.
    if let Some(path) = env::var_os(ENV_OPENVINO_PLUGINS_XML) {
        return Some(PathBuf::from(path));
    }

    // Check in the same directory as the `openvino_c` library; e.g.,
    // `/opt/intel/openvino_.../runtime/lib/intel64/plugins.xml`.
    let library = find("openvino_c")?;
    let library_parent_dir = library.parent()?;
    check_and_return!(library_parent_dir.join(FILE_NAME));

    // Check in OpenVINO's special system installation directory; e.g.,
    // `/usr/lib/x86_64-linux-gnu/openvino-2022.3.0/plugins.xml`.
    let filenames = list_directory(library_parent_dir)?;
    let versions = get_suffixes(filenames, "openvino-");
    let path = build_latest_version(library_parent_dir, "openvino-", versions)?.join("plugins.xml");
    check_and_return!(path);

    None
}

#[inline]
fn list_directory(dir: &Path) -> Option<impl IntoIterator<Item = String>> {
    let traversal = fs::read_dir(dir).ok()?;
    Some(
        traversal
            .filter_map(Result::ok)
            .filter_map(|f| f.file_name().to_str().map(ToString::to_string)),
    )
}

#[inline]
fn get_suffixes(filenames: impl IntoIterator<Item = String>, prefix: &str) -> Vec<String> {
    filenames
        .into_iter()
        .filter_map(|f| f.strip_prefix(prefix).map(ToString::to_string))
        .collect()
}

#[inline]
fn build_latest_version(dir: &Path, prefix: &str, mut versions: Vec<String>) -> Option<PathBuf> {
    if versions.is_empty() {
        return None;
    }
    versions.sort();
    versions.reverse();
    let latest_version = versions
        .first()
        .expect("already checked that a version exists");
    let filename = format!("{}{}", prefix, latest_version);
    Some(dir.join(filename))
}

#[cfg(test)]
mod test {
    use super::*;

    /// This test uses `find` to search for the `openvino_c` library on the local
    /// system.
    #[test]
    fn find_openvino_c_locally() {
        pretty_env_logger::init();
        assert!(find("openvino_c").is_some());
    }

    /// This test shows how the finder would discover the latest shared library on an
    /// APT installation.
    #[test]
    fn find_latest_library() {
        let path = build_latest_version(
            &PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            "libopenvino.so.",
            vec!["2022.1.0".into(), "2022.3.0".into()],
        );
        assert_eq!(
            path,
            Some(PathBuf::from(
                "/usr/lib/x86_64-linux-gnu/libopenvino.so.2022.3.0"
            ))
        );
    }

    /// This test shows how the finder would discover the latest `plugins.xml` directory on an
    /// APT installation.
    #[test]
    fn find_latest_plugin_xml() {
        let path = build_latest_version(
            &PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            "openvino-",
            vec!["2022.3.0".into(), "2023.1.0".into(), "2022.1.0".into()],
        );
        assert_eq!(
            path,
            Some(PathBuf::from("/usr/lib/x86_64-linux-gnu/openvino-2023.1.0"))
        );
    }
}
