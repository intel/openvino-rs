use bindgen;
use cmake;
use std::path::{Path, PathBuf};

fn main() {
    // Trigger rebuild on changes to build.rs and Cargo.toml and every source file.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let cb = |p: PathBuf| println!("cargo:rerun-if-changed={}", p.display());
    visit_dirs(Path::new("src"), &cb).expect("to visit source files");

    // Generate bindings from C header.
    let openvino_c_api_header =
        file("../upstream/inference-engine/ie_bridges/c/include/c_api/ie_c_api.h");
    let bindings = bindgen::Builder::default()
        .header(openvino_c_api_header.to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("generate C API bindings");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // Build OpenVINO using CMake.
    let build_path = cmake::Config::new("../upstream")
        .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
        .define("ENABLE_OPENCV", "OFF")
        .cxxflag("-Wno-pessimizing-move")
        .cxxflag("-Wno-redundant-move")
        .very_verbose(true)
        .build();

    // Set up the link search path to the manually-built library. It would be preferable to use
    // pkg-config here to retrieve the libraries if they were installed system-wide (see initial
    // commit), but OpenVINO relies on a `plugin.xml` file for finding target-specific libraries
    // and it is unclear how pkg-config where this would be in the system-install scenario.
    let library_path = build_path.join("deployment_tools/inference_engine/lib/intel64");
    println!("cargo:rustc-link-search=native={}", library_path.display());
    let third_party_path = build_path.join("lib64");
    println!(
        "cargo:rustc-link-search=native={}",
        third_party_path.display()
    );

    // Dynamically link the OpenVINO libraries.
    let libraries = vec![
        "inference_engine",
        "inference_engine_legacy",
        "inference_engine_transformations",
        "ngraph",
        "inference_engine_c_api",
    ];
    for library in &libraries {
        println!("cargo:rustc-link-lib=dylib={}", library);
    }

    // Output the location of the libraries to the environment.
    println!("cargo:rustc-env=LIBRARY_PATH={}", library_path.display());
}

/// Canonicalize a path as well as verify that it exists.
fn file<P: AsRef<Path>>(path: P) -> PathBuf {
    let canonicalized = path
        .as_ref()
        .canonicalize()
        .expect("to be able to canonicalize the path");
    if !canonicalized.exists() || !canonicalized.is_file() {
        panic!("Unable to find file: {}", canonicalized.display())
    }
    canonicalized
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
                cb(entry.path());
            }
        }
    }
    Ok(())
}
