use bindgen;
use cmake;
use pkg_config;
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

    // Link to the system-installed library (using pkg-config) or build it manually (using cmake).
    let libraries = vec![
        "inference_engine",
        "inference_engine_legacy",
        "inference_engine_transformations",
        "ngraph",
        "inference_engine_c_api",
    ];
    if let Ok(lib) = pkg_config::probe_library(libraries[0]) {
        if let Some(path) = lib.link_paths.get(0) {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    } else {
        let path = cmake::Config::new("../upstream")
            .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
            .define("ENABLE_OPENCV", "OFF")
            .cxxflag("-Wno-pessimizing-move")
            .cxxflag("-Wno-redundant-move")
            .very_verbose(true)
            .build();

        println!(
            "cargo:rustc-link-search=native={}",
            path.join("deployment_tools/inference_engine/lib/intel64")
                .display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            path.join("lib64").display()
        );
    }
    for library in &libraries {
        println!("cargo:rustc-link-lib=dylib={}", library);
    }
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
