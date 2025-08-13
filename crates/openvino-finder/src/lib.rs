//! This crate provides a mechanism for locating the OpenVINO files installed on a system.
//!
//! OpenVINO can be installed several ways: [from an archive][install-archive], [from an APT
//! repository][install-apt], [via Python `pip`][install-pip]. The Rust bindings need to be able to:
//!  1. locate the shared libraries (e.g., `libopenvino_c.so` on Linux) &mdash; see [`find`]
//!  2. locate the plugin configuration file (i.e., `plugins.xml`) &mdash; see [`find_plugins_xml`].
//!
//! These files are located in different locations based on the installation method, so this crate
//! encodes "how to find" OpenVINO files. This crate's goal is to locate __only the latest version__
//! of OpenVINO; older versions may continue to be supported on a best-effort basis.
//!
//! [install-archive]: https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
//! [install-apt]: https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_apt.html
//! [install-pip]: https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_pip.html
//!
//! Problems with the OpenVINO bindings are most likely to be due to "finding" the right files. Both
//! [`find`] and [`find_plugins_xml`] provide various ways of configuring the search paths, first by
//! examining _special environment variables_ and then by looking in _known installation locations_.
//! When [installing from an archive][install-archive], OpenVINO provides a setup script (e.g.,
//! `source /opt/intel/openvino/setupvars.sh`) that sets these special environment variables. Note
//! that you may need to have the OpenVINO environment ready both when building (`cargo build`) and
//! running (e.g., `cargo test`) when the libraries are linked at compile-time (the default). By
//! using the `runtime-linking` feature, the libraries are only searched for at run-time.
//!
//! If you do run into problems, the following chart summarizes some of the known installation
//! locations of the OpenVINO files as of version `2022.3.0`:
//!
//! | Installation Method | Path                                               | Available on            | Notes                            |
//! | ------------------- | -------------------------------------------------- | ----------------------- | -------------------------------- |
//! | Archive (`.tar.gz`) | `<extracted folder>/runtime/lib/<arch>`            | Linux                   | `<arch>`: `intel64,armv7l,arm64` |
//! | Archive (`.tar.gz`) | `<extracted folder>/runtime/lib/<arch>/Release`    | `MacOS`                 | `<arch>`: `intel64,armv7l,arm64` |
//! | Archive (`.zip`)    | `<unzipped folder>/runtime/bin/<arch>/Release`     | Windows                 | `<arch>`: `intel64,armv7l,arm64` |
//! | `PyPI`              | `<pip install folder>/site-packages/openvino/libs` | Linux, `MacOS`, Windows | Find install folder with `pip show openvino` |
//! | DEB                 | `/usr/lib/x86_64-linux-gnu/openvino-<version>/`    | Linux (APT-based)       | This path is for plugins; the libraries are one directory above |
//! | RPM                 | `/usr/lib64/`                                      | Linux (YUM-based)       |                                  |

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

/// Distinguish which kind of library to link to.
///
/// The difference is important on Windows, e.g., which [requires] `*.lib` libraries when linking
/// dependent libraries.
///
/// [requires]:
///     https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Linking {
    /// Find _static_ libraries: OpenVINO comes with static libraries on some platforms (e.g.,
    /// Windows).
    Static,
    /// Find _dynamic_ libraries.
    Dynamic,
}

/// Find the path to an OpenVINO library.
///
/// Because OpenVINO can be installed in quite a few ways (see module documentation), this function
/// attempts the difficult and thankless task of locating the installation's shared libraries for
/// use in the Rust bindings (i.e., [openvino] and [openvino-sys]). It uses observations from
/// various OpenVINO releases across several operating systems and conversations with the OpenVINO
/// development team, but it may not perfectly locate the libraries in every environment &mdash;
/// hence the `Option<PathBuf>` return type.
///
/// [openvino]: https://docs.rs/openvino
/// [openvino-sys]: https://docs.rs/openvino-sys
///
/// This function will probe:
/// - the `OPENVINO_BUILD_DIR` environment variable with known build subdirectories appended &mdash;
///   this is useful for finding libraries built from source
/// - the `OPENVINO_INSTALL_DIR`, `INTEL_OPENVINO_DIR`, and `LD_LIBRARY_PATH` (or OS-equivalent)
///   environment variables with known install subdirectories appended &mdash; one of these is set
///   by a version of OpenVINO's environment script (e.g., `source
///   /opt/intel/openvino/setupvars.sh`)
/// - OpenVINO's package installation paths for the OS (e.g., `/usr/lib64`) &mdash; this is useful
///   for DEB or RPM installations
/// - OpenVINO's documented extract paths &mdash; this is useful for users who extract the TAR or
///   ZIP archive to the default locations or use the Docker images
///
/// The locations above may change over time. As OpenVINO has released new versions, the documented
/// locations of the shared libraries has changed. New versions of this function will reflect this,
/// removing older, unused locations over time.
///
/// # Panics
///
/// Panics if it cannot list the contents of a search directory.
pub fn find(library_name: &str, kind: Linking) -> Option<PathBuf> {
    let suffix = if kind == Linking::Static {
        // This is a bit rudimentary but works for the top three supported platforms: `linux`,
        // `macos`, and `windows`.
        if cfg!(target_os = "windows") {
            ".lib"
        } else {
            ".a"
        }
    } else {
        env::consts::DLL_SUFFIX
    };
    let file = format!("{}{}{}", env::consts::DLL_PREFIX, library_name, suffix);
    log::info!("Attempting to find library: {file}");

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
            "/lib", // DEB-installed package (OpenVINO >= 2023.2)
            "/usr/lib/x86_64-linux-gnu", // DEB-installed package (OpenVINO >= 2022.3)
            "/lib/x86_64-linux-gnu", // DEB-installed package (TBB)
            "/usr/lib64", // RPM-installed package >= 2022.3
        ];
    } else {
        const SYSTEM_INSTALLATION_DIRECTORIES: &[&str] = &[];
    }
}

const KNOWN_INSTALLATION_SUBDIRECTORIES: &[&str] = &[
    "runtime/lib/intel64/Release",
    "runtime/lib/intel64",
    "runtime/lib/arm64/Release",
    "runtime/lib/arm64",
    "runtime/lib/aarch64/Release",
    "runtime/lib/aarch64",
    "runtime/lib/armv7l",
    "runtime/lib/armv7l/Release",
    "runtime/bin/intel64/Release",
    "runtime/bin/intel64",
    "runtime/bin/arm64/Release",
    "runtime/bin/arm64",
    "runtime/3rdparty/tbb/bin",
    "runtime/3rdparty/tbb/lib",
];

const KNOWN_BUILD_SUBDIRECTORIES: &[&str] = &[
    "bin/intel64/Debug/lib",
    "bin/intel64/Debug",
    "bin/intel64/Release/lib",
    "bin/arm64/Debug/lib",
    "bin/arm64/Debug",
    "bin/arm64/Release/lib",
    "bin/aarch64/Debug/lib",
    "bin/aarch64/Debug",
    "bin/aarch64/Release/lib",
    "bin/armv7l/Debug/lib",
    "bin/armv7l/Debug",
    "bin/armv7l/Release/lib",
    "temp/tbb/lib",
    "temp/tbb/bin",
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
/// This function will probe:
/// - the `OPENVINO_PLUGINS_XML` environment variable &mdash; this is specific to this library
/// - the same directory as the `openvino_c` shared library, as discovered by [find]
/// - the latest version directory beside the `openvino_c` shared library (i.e.,
///   `openvino-<latest version>/`)
pub fn find_plugins_xml() -> Option<PathBuf> {
    const FILE_NAME: &str = "plugins.xml";

    // The `OPENVINO_PLUGINS_XML` should point directly to the file.
    if let Some(path) = env::var_os(ENV_OPENVINO_PLUGINS_XML) {
        return Some(PathBuf::from(path));
    }

    // Check in the same directory as the `openvino_c` library; e.g.,
    // `/opt/intel/openvino_.../runtime/lib/intel64/plugins.xml`.
    let library = find("openvino_c", Linking::Dynamic)?;
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
    let filename = format!("{prefix}{latest_version}");
    Some(dir.join(filename))
}

#[cfg(test)]
mod test {
    use super::*;

    /// This test uses `find` to search for the `openvino_c` library on the local
    /// system.
    #[test]
    fn find_openvino_c_locally() {
        env_logger::init();
        assert!(find("openvino_c", Linking::Dynamic).is_some());
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
