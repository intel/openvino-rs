use crate::util::path_to_crates;
use anyhow::{anyhow, ensure, Context, Result};
use clap::Args;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub struct CodegenCommand {
    /// The path to OpenVINO's Inference Engine C API header; by default,
    /// `.../openvino-sys/upstream/inference-engine/ie_bridges/c/include/c_api/ie_c_api.h`.
    #[arg(short = 'i', long = "input-header-file")]
    header_file: Option<PathBuf>,

    /// The path to the directory in which to output the generated files; by default,
    /// `.../openvino-sys/crates/src/generated`.
    #[arg(short = 'o', long = "output-directory")]
    output_directory: Option<PathBuf>,
}

impl CodegenCommand {
    /// Because of how the linking is implemented (i.e. a `link!` macro that must wrap around the
    /// foreign functions), we must split the bindgen generation of types and functions into
    /// separate files. This means that, at least for the function output, we also need a prefix
    /// (e.g. to add the `link!` macro) and suffix to make things compile.
    pub fn execute(&self) -> Result<()> {
        let header_file = self.path_to_header_file()?;
        let output_directory = self.path_to_output_directory()?;

        // Generate the type bindings into `.../types.rs`.
        let type_bindings = Self::generate_type_bindings(&header_file)?;
        let type_bindings_path = output_directory.join(TYPES_FILE);
        type_bindings
            .write_to_file(&type_bindings_path)
            .with_context(|| {
                format!("Failed to write types to: {}", type_bindings_path.display())
            })?;

        // Generate the function bindings into `.../functions.rs`, with a prefix and suffix.
        let function_bindings = Self::generate_function_bindings(&header_file)?;
        let function_bindings_path = output_directory.join(FUNCTIONS_FILE);
        {
            let mut function_bindings_file = Box::new(File::create(&function_bindings_path)?);
            function_bindings_file.write_all(b"use super::types::*;\n")?;
            function_bindings_file.write_all(b"use crate::link;\n")?;
            function_bindings_file.write_all(b"link! {\n")?;
            function_bindings_file.write_all(b"\n")?;
            function_bindings
                .write(function_bindings_file)
                .with_context(|| {
                    format!(
                        "Failed to write functions to: {}",
                        &function_bindings_path.display()
                    )
                })?;
        }

        let mut function_bindings_file = OpenOptions::new()
            .append(true)
            .open(&function_bindings_path)?;
        function_bindings_file.write_all(b"\n")?;
        function_bindings_file.write_all(b"}\n")?;
        Ok(())
    }

    fn path_to_header_file(&self) -> Result<PathBuf> {
        Ok(match self.header_file.clone() {
            Some(path) => {
                ensure!(
                    path.is_file(),
                    "The input header file must be an actual file."
                );
                path
            }
            None => {
                // Equivalent to:
                // crates/xtask/src/codegen.rs/../../../openvino-sys/upstream/inference-engine...
                path_to_crates()?.join(DEFAULT_HEADER_FILE)
            }
        })
    }

    fn path_to_output_directory(&self) -> Result<PathBuf> {
        Ok(match self.output_directory.clone() {
            Some(path) => {
                ensure!(
                    path.is_dir(),
                    "The output directory must be an actual directory."
                );
                path
            }
            None => {
                // Equivalent to: crates/xtask/src/codegen.rs/../../../openvino-sys/src/generated
                path_to_crates()?.join(DEFAULT_OUTPUT_DIRECTORY)
            }
        })
    }

    fn generate_type_bindings<P: AsRef<Path>>(header_file: P) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg("-I/home/rahul/repos/openvino-binding/openvino-rs/crates/openvino-sys/upstream/src/bindings/c/include")
            // .allowlist_type("ie_.*")
            .allowlist_type("ov_.*")
            // Enumerations.
            .allowlist_type("precision_e")
            .allowlist_type("layout_e")
            .allowlist_type("resize_alg_e")
            .allowlist_type("colorformat_e")
            // Custom types.
            .allowlist_type("dimensions_t")
            .allowlist_type("input_shapes_t")
            .allowlist_type("tensor_desc_t")
            .allowlist_type("roi_t")
            .allowlist_type("IEStatusCode")
            // Convert C's `size_t` to `usize`.
            .size_t_is_usize(true)
            // While understanding the warnings in
            // https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum that
            // these enums could result in unspecified behavior if constructed from an invalid
            // value, the expectation here is that OpenVINO only returns valid layout and precision
            // values. This assumption is reasonable because otherwise OpenVINO itself would be
            // broken.
            .rustified_enum("layout_e")
            .rustified_enum("precision_e")
            .rustified_enum("resize_alg_e")
            .rustified_enum("colorformat_e")
            // Generate only the types.
            .with_codegen_config(bindgen::CodegenConfig::TYPES)
            .generate()
            .map_err(|_| anyhow!("unable to generate type bindings"))
    }

    fn generate_function_bindings<P: AsRef<Path>>(header_file: P) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg("-I/home/rahul/repos/openvino-binding/openvino-rs/crates/openvino-sys/upstream/src/bindings/c/include")
            //.allowlist_function("ie_.*")
            .allowlist_function("ov_.*")
            .blocklist_type("__uint8_t")
            .blocklist_type("__int64_t")
            .size_t_is_usize(true)
            // While understanding the warnings in
            // https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum that
            // these enums could result in unspecified behavior if constructed from an invalid
            // value, the expectation here is that OpenVINO only returns valid layout and precision
            // values. This assumption is reasonable because otherwise OpenVINO itself would be
            // broken.
            .rustified_enum("layout_e")
            .rustified_enum("precision_e")
            .rustified_enum("resize_alg_e")
            .rustified_enum("colorformat_e")
            // Generate only functions.
            .with_codegen_config(bindgen::CodegenConfig::FUNCTIONS)
            .generate()
            .map_err(|_| anyhow!("unable to generate function bindings"))
    }
}

const TYPES_FILE: &str = "types.rs";
const FUNCTIONS_FILE: &str = "functions.rs";
const DEFAULT_OUTPUT_DIRECTORY: &str = "openvino-sys/src/generated";
//const DEFAULT_HEADER_FILE: &str = "openvino-sys/upstream/src/bindings/c/include/c_api/ie_c_api.h";
const DEFAULT_HEADER_FILE: &str =
    "openvino-sys/upstream/src/bindings/c/include/openvino/c/openvino.h";
