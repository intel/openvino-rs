use std::path::{Path, PathBuf};

use bindgen;
use cmake;

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
        // While understanding the warnings in https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum
        // that these enums could result in unspecified behavior if constructed from an invalid
        // value, the expectation here is that OpenVINO only returns valid layout and precision
        // values. This assumption is reasonable because otherwise OpenVINO itself would be broken.
        .rustified_enum("layout_e")
        .rustified_enum("precision_e")
        .rustified_enum("resize_alg_e")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("generate C API bindings");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // Build OpenVINO using CMake.
    let library_path = if let Some(path) = std::env::var_os("OPENVINO_LIB_DIR") {
        let library_path = dir(path);
        assert!(
            library_path.join("plugins.xml").is_file(),
            "The OPENVINO_LIB_DIR should contain a plugins.xml file"
        );
        library_path
    } else {
        fn cmake(out_dir: &str) -> cmake::Config {
            let mut config = cmake::Config::new("../upstream");
            config
                .very_verbose(true)
                .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
                .define("ENABLE_OPENCV", "OFF")
                .define("ENABLE_CPPLINT", "OFF")
                // Because OpenVINO by default wants to build its binaries in its own tree, we must specify
                // that we actually want them in Cargo's output directory.
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

        // Set up the link search path to the manually-built library. It would be preferable to use
        // pkg-config here to retrieve the libraries if they were installed system-wide (see initial
        // commit), but OpenVINO relies on a `plugin.xml` file for finding target-specific libraries
        // and it is unclear how pkg-config where this would be in the system-install scenario.
        build_path.join("bin/intel64/Debug/lib")
        // TODO we can document the assumption that OpenVINO will likely compile best on x86-64 but
        // we must figure out what path the binaries have been created in: Debug/Release/RelWithDebInfo/MinSizeRel.
        // The cmake crate is automatically inferring this: https://docs.rs/cmake/0.1.44/cmake/struct.Config.html#method.profile
    };
    println!("cargo:rustc-link-search=native={}", library_path.display());

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

/// Canonicalize a path as well as verify that it exists.
fn dir<P: AsRef<Path>>(path: P) -> PathBuf {
    let canonicalized = path
        .as_ref()
        .canonicalize()
        .expect("to be able to canonicalize the path");
    if !canonicalized.exists() || !canonicalized.is_dir() {
        panic!("Unable to find directory: {}", canonicalized.display())
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
