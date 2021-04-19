fn main() {
    println!("Hello, world!");
}

// use std::path::PathBuf;

// use bindgen;

// const OUT_DIR: &'static str = "src/generated";
// const TYPES_FILE: &'static str = "types.rs";
// const FUNCTIONS_FILE: &'static str = "functions.rs";

// fn main() {
//     let out_dir = PathBuf::from(OUT_DIR);
//     // Generate bindings from C header.
//     let openvino_c_api_header =
//         file("upstream/inference-engine/ie_bridges/c/include/c_api/ie_c_api.h");
//     let type_bindings = bindgen::Builder::default()
//         .header(openvino_c_api_header.to_string_lossy())
//         //.allowlist_function("ie_.*")
//         .allowlist_type("ie_.*")
//         // Enumerations.
//         .allowlist_type("precision_e")
//         .allowlist_type("layout_e")
//         .allowlist_type("resize_alg_e")
//         .allowlist_type("colorformat_e")
//         // Custom types.
//         .allowlist_type("dimensions_t")
//         .allowlist_type("input_shapes_t")
//         .allowlist_type("tensor_desc_t")
//         .allowlist_type("roi_t")
//         .allowlist_type("IEStatusCode")
//         //.blocklist_type("__uint8_t")
//         //.blocklist_type("__int64_t")
//         .size_t_is_usize(true)
//         // While understanding the warnings in https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum
//         // that these enums could result in unspecified behavior if constructed from an invalid
//         // value, the expectation here is that OpenVINO only returns valid layout and precision
//         // values. This assumption is reasonable because otherwise OpenVINO itself would be broken.
//         .rustified_enum("layout_e")
//         .rustified_enum("precision_e")
//         .rustified_enum("resize_alg_e")
//         .rustified_enum("colorformat_e")
//         .with_codegen_config(CodegenConfig::TYPES)
//         .parse_callbacks(Box::new(bindgen::CargoCallbacks))
//         .generate()
//         .expect("generate C API bindings");
//     type_bindings
//         .write_to_file(out_dir.join(TYPES_FILE))
//         .expect(concat!("failed to write ", TYPES_FILE));

//     let function_bindings = bindgen::Builder::default()
//         .header(openvino_c_api_header.to_string_lossy())
//         .allowlist_function("ie_.*")
//         .blocklist_type("__uint8_t")
//         .blocklist_type("__int64_t")
//         .size_t_is_usize(true)
//         //.raw_line("use crate::*;")
//         // While understanding the warnings in https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum
//         // that these enums could result in unspecified behavior if constructed from an invalid
//         // value, the expectation here is that OpenVINO only returns valid layout and precision
//         // values. This assumption is reasonable because otherwise OpenVINO itself would be broken.
//         .rustified_enum("layout_e")
//         .rustified_enum("precision_e")
//         .rustified_enum("resize_alg_e")
//         .rustified_enum("colorformat_e")
//         .with_codegen_config(CodegenConfig::FUNCTIONS)
//         .parse_callbacks(Box::new(bindgen::CargoCallbacks))
//         .generate()
//         .expect("generate C API bindings");
//     let out = PathBuf::from(env::var("src/generated").unwrap());
//     function_bindings
//         .write_to_file(out_dir.join(FUNCTIONS_FILE))
//         .expect(concat!("failed to write ", FUNCTIONS_FILE));
// }
