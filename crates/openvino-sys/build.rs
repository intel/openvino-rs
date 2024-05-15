use std::env;
use std::path::{Path, PathBuf};

// These are the libraries we expect to be available to dynamically link to:
const LIBRARIES: &[&str] = &["openvino", "openvino_c"];

// A user-specified environment variable indicating that `build.rs` should not attempt to link
// against any libraries (e.g. a doc build, user may link them later).
const ENV_OPENVINO_SKIP_LINKING: &str = "OPENVINO_SKIP_LINKING";

// A build.rs-specified environment variable that must be populated with the location of the
// inference engine library that OpenVINO is being linked to in this script.
const ENV_OPENVINO_LIB_PATH: &str = "OPENVINO_LIB_PATH";

// An environment variable for building against a from-source build of OpenVINO. See
// `openvino-finder` for how this is used to find library paths.
const ENV_OPENVINO_BUILD_DIR: &str = "OPENVINO_BUILD_DIR";

fn main() {
    // This allows us to log the `openvino-finder` search paths, for troubleshooting.
    let _ = env_logger::try_init();

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
        assert!(env::var_os(ENV_OPENVINO_BUILD_DIR).is_none(), "When building from source, the build script must always try to dynamically link the built libraries.");
        Linking::None
    } else {
        Linking::Dynamic
    };

    // Find the OpenVINO libraries to link to, either from a pre-installed location or by building
    // from source. We always look for the dynamic libraries here.
    let link_kind = openvino_finder::Linking::Dynamic;
    let (c_api_library_path, library_search_paths) = if linking == Linking::None {
        // Why try to find the library if we're not going to link against it? Well, this is for the
        // helpful Cargo warnings that get printed below if we can't find the library on the system.
        (openvino_finder::find("openvino_c", link_kind), vec![])
    } else if let Some(path) = openvino_finder::find("openvino_c", link_kind) {
        (Some(path), find_libraries_in_existing_installation())
    } else {
        panic!("Unable to find an OpenVINO installation on your system; build with runtime linking using `--features runtime-linking` or build from source with `OPENVINO_BUILD_DIR`.")
    };

    // Capture the path to the library we are using. The reason we do this is to provide a mechanism
    // for finding the `plugins.xml` file at runtime (usually it is found in the same directory as
    // the inference engine libraries).
    if let Some(path) = c_api_library_path {
        record_library_path(path);
    } else {
        println!("cargo:warning=openvino-sys cannot find the `openvino_c` library in any of the library search paths: {:?}", &library_search_paths);
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
/// It would be preferable to use pkg-config here to retrieve the libraries when they are installed
/// system-wide but there are issues:
///  - OpenVINO does not install itself as a system library, e.g., through `ldconfig`;
///  - OpenVINO relies on a `plugins.xml` file for finding target-specific libraries and it is
///    unclear how we would discover this in a system-install scenario.
fn find_libraries_in_existing_installation() -> Vec<PathBuf> {
    let mut dirs = vec![];
    let link_kind = if cfg!(target_os = "windows") {
        // Retrieve `*.lib` files on Windows. This is important because, when linking, Windows
        // expects `*.lib` files. See
        // https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation#creating-an-import-library.
        openvino_finder::Linking::Static
    } else {
        openvino_finder::Linking::Dynamic
    };
    for library in LIBRARIES {
        if let Some(path) = openvino_finder::find(library, link_kind) {
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
