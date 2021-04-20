use cmake;
use std::env;
use std::path::{Path, PathBuf};

// These are the libraries we expect to be available to dynamically link to:
const LIBRARIES: &'static [&'static str] = &[
    "inference_engine",
    "inference_engine_legacy",
    "inference_engine_transformations",
    "inference_engine_c_api",
    "ngraph",
    "tbb",
];

// A user-specified environment variable indicating that `build.rs` should not attempt to link
// against any libraries (e.g. a doc build, user may link them later).
const ENV_OPENVINO_SKIP_LINKING: &'static str = "OPENVINO_SKIP_LINKING";

// A build.rs-specified environment variable that must be populated with the location of the
// inference engine library that OpenVINO is being linked to in this script.
const ENV_OPENVINO_LIB_PATH: &'static str = "OPENVINO_LIB_PATH";

fn main() {
    // Trigger rebuild on changes to build.rs and Cargo.toml and every source file.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let cb = |p: PathBuf| println!("cargo:rerun-if-changed={}", p.display());
    visit_dirs(Path::new("src"), &cb).expect("to visit source files");

    // Use the dynamic linking feature for conditional compilation if no linking method was
    // specified.
    if !cfg!(feature = "runtime-linking") && !cfg!(feature = "dynamic-linking") {
        println!("cargo:rustc-cfg=feature=\"dynamic-linking\"");
    }

    // Determine what linking method to use: avoid dynamic linking when we have either specified
    // runtime linking or no linking at all. It turns out we may not always want to link this crate
    // against its dynamic libraries (e.g. building documentation)--these environment variables
    // provide an escape hatch.
    let linking = if cfg!(feature = "runtime-linking")
        || env::var_os(ENV_OPENVINO_SKIP_LINKING).is_some()
    {
        assert!(!cfg!(feature = "from-source"), "When building from source, the build script must always try to dynamically link the built libraries.");
        Linking::None
    } else {
        Linking::Dynamic
    };

    // Find the OpenVINO libraries to link to, either from a pre-installed location or by building
    // from source.
    let (c_api_library_path, library_search_paths) = if cfg!(feature = "from-source") {
        build_from_source_using_cmake()
    } else if linking == Linking::None {
        (openvino_finder::find("inference_engine_c_api"), vec![])
    } else if let Some(path) = openvino_finder::find("inference_engine_c_api") {
        (Some(path), find_libraries_in_existing_installation())
    } else {
        panic!("Unable to find an OpenVINO installation on your system; build with runtime linking using `--features runtime-linking` or build from source with `--features from-source`.")
    };

    // Capture the path to the library we are using. The reason we do this is to provide a mechanism
    // for finding the `plugins.xml` file at runtime (usually it is found in the same directory as
    // the inference engine libraries).
    if let Some(path) = c_api_library_path {
        record_library_path(path);
    } else {
        println!("cargo:warning=openvino-sys cannot find the `inference_engine_c_api` library in any of the library search paths: {:?}", &library_search_paths);
        println!("cargo:warning=Proceeding with an empty value of {}; users must specify this location at runtime, e.g. `Core::new(Some(...))`.", ENV_OPENVINO_LIB_PATH);
        record_library_path(PathBuf::new());
    }

    // If necessary, dynamically link the necessary OpenVINO libraries.
    if linking == Linking::Dynamic {
        library_search_paths
            .iter()
            .for_each(add_library_search_path);
        LIBRARIES
            .iter()
            .cloned()
            .for_each(add_dynamically_linked_library)
    }
}

/// Enumerate the possible linking states for this build script:
/// - either we don't want to link to anything during compile time
/// - or we want to link to the OpenVINO libraries dynamically.
#[derive(Eq, PartialEq)]
enum Linking {
    None,
    Dynamic,
}

/// Canonicalize a path as well as verify that it exists.
fn dir<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if !path.exists() || !path.is_dir() {
        panic!("Unable to find directory: {}", path.display())
    }
    path.canonicalize()
        .expect("to be able to canonicalize the path")
}

/// Helper for recursively visiting the files in this directory; see https://doc.rust-lang.org/std/fs/fn.read_dir.html.
fn visit_dirs(dir: &Path, cb: &dyn Fn(PathBuf)) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(path);
            }
        }
    }
    Ok(())
}

/// Record the path to the shared library we link against in an environment variable.
fn record_library_path(library_path: PathBuf) {
    println!(
        "cargo:rustc-env={}={}",
        ENV_OPENVINO_LIB_PATH,
        library_path.display()
    );
}

/// Ensure a path is valid and add it to the build-time library search path.
fn add_library_search_path<P: AsRef<Path>>(path: P) {
    let path = path.as_ref();
    assert!(
        path.is_dir(),
        "Invalid library search path: {}",
        path.display()
    );
    println!("cargo:rustc-link-search=native={}", path.display());
}

/// Add a dynamically-linked library.
fn add_dynamically_linked_library(library: &str) {
    println!("cargo:rustc-link-lib=dylib={}", library);
}

/// Find all of the necessary libraries to link using the `openvino_finder`. This will return the
/// directories that should contain the necessary libraries to link to.
///
/// It would be preferable to use pkg-config here to retrieve the libraries when they are
/// installed system-wide but there are issues:
///  - OpenVINO does not install itself a system library, e.g., through ldconfig.
///  - OpenVINO relies on a `plugins.xml` file for finding target-specific libraries
///    and it is unclear how we would discover this in a system-install scenario.
fn find_libraries_in_existing_installation() -> Vec<PathBuf> {
    let mut dirs = vec![];
    for library in LIBRARIES {
        if let Some(path) = openvino_finder::find(library) {
            println!(
                "cargo:warning=Found library to link against: {}",
                path.display()
            );
            let dir = path.parent().unwrap().to_owned();
            if !dirs.iter().any(|d| d == &dir) {
                dirs.push(dir);
            }
        } else {
            panic!(
                "Unable to find an existing installation of library: {}",
                library
            );
        }
    }
    dirs
}

/// Build OpenVINO with CMake. TODO this currently will not work when the crate is published
/// because the `upstream` directory will not fit inside the 10MB crate limit. To solve this, we
/// could retrieve the sources (cringe), e.g., with `git2`.
fn build_from_source_using_cmake() -> (Option<PathBuf>, Vec<PathBuf>) {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    fn cmake(out_dir: &str) -> cmake::Config {
        let mut config = cmake::Config::new("upstream");
        config
            .very_verbose(true)
            .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
            .define("ENABLE_OPENCV", "OFF")
            .define("ENABLE_CPPLINT", "OFF")
            // As described in https://github.com/intel/openvino-rs/issues/8, the OpenVINO source
            // includes redundant moves. These were previously warnings but newer compilers treat
            // them as errors.
            .cxxflag("-Wno-error=redundant-move")
            // Because OpenVINO by default wants to build its binaries in its own tree, we must
            // specify that we actually want them in Cargo's output directory.
            .define("OUTPUT_ROOT", out_dir);
        config
    }

    // Specifying the build targets reduces the build time somewhat; this one will trigger
    // builds for other necessary shared libraries (e.g. inference_engine).
    let build_path = cmake(out.to_str().unwrap())
        .build_target("inference_engine_c_api")
        .build();

    // Unfortunately, `inference_engine_c_api` will not build the OpenVINO plugins used for
    // the actual computation. Here we re-run CMake for each plugin the user specifies using
    // Cargo features (see `Cargo.toml`).
    for plugin in get_plugin_target_from_features() {
        cmake(out.to_str().unwrap()).build_target(plugin).build();
    }

    // Collect the locations of the libraries. Note that ngraph should also be found with the
    // built OpenVINO libraries.
    let openvino_libraries =
        find_and_append_cmake_build_type(build_path.join("bin/intel64")).join("lib");

    // Copy the TBB libraries into the OpenVINO library directory. Since ngraph already exists
    // here and because the TBB directory is weirdly downloaded in-tree rather than under target
    // (meaning that the TBB path would be stripped from LD_LIBRARY_PATH, see
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#dynamic-library-paths),
    // copying the files over makes some sense. Also, I have noticed compatibility issues with
    // pre-installed libtbb (on some systems, the nodes_count symbol is not present in the
    // system-provided libtbb) so it may be important to include OpenVINO's version of libtbb
    // here.
    let tbb_libraries = dir("upstream/inference-engine/temp/tbb/lib");
    visit_dirs(&tbb_libraries, &|from: PathBuf| {
        let to = openvino_libraries.join(from.file_name().unwrap());
        println!("Copying {} to {}", from.display(), to.display());
        std::fs::copy(from, to).expect("failed copying TBB libraries");
    })
    .expect("failed visiting TBB directory");

    let c_api = format!(
        "{}inference_engine_c_api{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    );
    (
        Some(openvino_libraries.join(c_api)),
        vec![openvino_libraries],
    )
}

/// Determine CMake targets for the various OpenVINO plugins. The plugin mapping is available in
/// OpenVINO's `plugins.xml` file and, usign that, this function wires up the exposed Cargo
/// features of openvino-sys to the correct CMake target.
fn get_plugin_target_from_features() -> Vec<&'static str> {
    let mut plugins = vec![];
    if cfg!(feature = "all") {
        plugins.push("ie_plugins")
    } else {
        if cfg!(feature = "cpu") {
            plugins.push("MKLDNNPlugin")
        }
        if cfg!(feature = "gpu") {
            plugins.push("clDNNPlugin")
        }
        if cfg!(feature = "gna") {
            plugins.push("GNAPlugin")
        }
        if cfg!(feature = "hetero") {
            plugins.push("HeteroPlugin")
        }
        if cfg!(feature = "multi") {
            plugins.push("MultiDevicePlugin")
        }
        if cfg!(feature = "myriad") {
            plugins.push("myriadPlugin")
        }
    }
    assert!(!plugins.is_empty());
    plugins
}

/// According to https://docs.rs/cmake/0.1.44/cmake/struct.Config.html#method.profile, the cmake
/// crate will tries to infer the appropriate CMAKE_BUILD_TYPE from a combination of Rust opt-level
/// and debug. To avoid duplicating https://docs.rs/cmake/0.1.44/src/cmake/lib.rs.html#553-559, this
/// helper searches for build type directories and appends it to the path if a result is found; this
/// will panic otherwise.
fn find_and_append_cmake_build_type(build_path: PathBuf) -> PathBuf {
    let types = ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"];
    let found: Vec<_> = types
        .iter()
        .filter(|&&t| build_path.join(t).is_dir())
        .collect();
    match found.len() {
        0 => panic!(
            "No CMake build directory found in {}; expected one of {:?}",
            build_path.display(),
            types
        ),
        1 => build_path.join(found[0]),
        _ => panic!(
            "Too many CMake build directories found in {}",
            build_path.display()
        ),
    }
}
