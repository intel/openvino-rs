use crate::util::path_to_crates;
use anyhow::{anyhow, ensure, Context, Result};
use clap::{Args, ValueEnum};
use openvino_finder::Linking;
use regex::Regex;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, ValueEnum)]
enum CrateTarget {
    /// Generate bindings for openvino-sys (default).
    Sys,
    /// Generate bindings for openvino-genai-sys.
    Genai,
}

#[derive(Debug, Args)]
pub struct CodegenCommand {
    /// Which crate to generate bindings for.
    #[arg(short = 'c', long = "crate", default_value = "sys")]
    target: CrateTarget,

    /// The path to the C API header; overrides the default for the selected crate.
    #[arg(short = 'i', long = "input-header-file")]
    header_file: Option<PathBuf>,

    /// The path to the directory in which to output the generated files; overrides the default for
    /// the selected crate.
    #[arg(short = 'o', long = "output-directory")]
    output_directory: Option<PathBuf>,
}

impl CodegenCommand {
    /// Because of how the linking is implemented (i.e. a `link!` macro that must wrap around the
    /// foreign functions), we must split the bindgen generation of types and functions into
    /// separate files. This means that, at least for the function output, we also need a prefix
    /// (e.g. to add the `link!` macro) and suffix to make things compile.
    pub fn execute(&self) -> Result<()> {
        match self.target {
            CrateTarget::Sys => self.execute_sys(),
            CrateTarget::Genai => self.execute_genai(),
        }
    }

    fn execute_sys(&self) -> Result<()> {
        let header_file = self.resolve_path(OV_SYS_HEADER)?;
        let output_directory = self.resolve_output(OV_SYS_OUTPUT)?;
        let include_directory =
            Self::resolve_include_dir(OV_SYS_INCLUDE, "openvino_c", "openvino/c/openvino.h")?;

        // Generate the type bindings into `.../types.rs`.
        let type_bindings = Self::generate_sys_type_bindings(&header_file, &include_directory)?;
        let type_bindings_path = output_directory.join(TYPES_FILE);
        type_bindings
            .write_to_file(&type_bindings_path)
            .with_context(|| {
                format!("Failed to write types to: {}", type_bindings_path.display())
            })?;

        // Generate the function bindings into `.../functions.rs`, with a prefix and suffix.
        let function_bindings =
            Self::generate_sys_function_bindings(&header_file, &include_directory)?;

        // Runtime linking doesn't work yet with variadic args (...), so we need to convert them
        // to a fixed pair of args (property_key, property_value) for a few select functions.
        // This is a workaround until the runtime linking is updated to support variadic args.
        let functions_to_modify = vec!["ov_core_set_property", "ov_compiled_model_set_property"];
        let mut function_bindings_string = function_bindings.to_string();
        for function in &functions_to_modify {
            let re = Regex::new(&format!(r"(?s){function}.*?\.\.\.")).unwrap();
            if re.is_match(&function_bindings_string) {
                function_bindings_string = re.replace(&function_bindings_string, |caps: &regex::Captures| {
                    caps[0].replace("...", "property_key: *const ::std::os::raw::c_char,\n        property_value: *const ::std::os::raw::c_char")
                }).to_string();
            }
        }

        Self::write_functions_file(
            &output_directory.join(FUNCTIONS_FILE),
            &function_bindings_string,
            b"use super::types::*;\nuse crate::link;\ntype wchar_t = ::std::os::raw::c_char;\n",
        )?;

        Ok(())
    }

    fn execute_genai(&self) -> Result<()> {
        let header_file = self.resolve_path(GENAI_HEADER)?;
        let output_directory = self.resolve_output(GENAI_OUTPUT)?;
        let genai_include_directory = Self::resolve_include_dir(
            GENAI_INCLUDE,
            "openvino_genai_c",
            "openvino/genai/c/llm_pipeline.h",
        )?;
        let openvino_include_directory =
            Self::resolve_include_dir(OV_SYS_INCLUDE, "openvino_c", "openvino/c/openvino.h")?;

        // Generate the type bindings — only GenAI-specific types, blocklisting core OV types.
        let type_bindings = Self::generate_genai_type_bindings(
            &header_file,
            &genai_include_directory,
            &openvino_include_directory,
        )?;
        let type_bindings_path = output_directory.join(TYPES_FILE);
        type_bindings
            .write_to_file(&type_bindings_path)
            .with_context(|| {
                format!("Failed to write types to: {}", type_bindings_path.display())
            })?;

        // Generate the function bindings.
        let function_bindings = Self::generate_genai_function_bindings(
            &header_file,
            &genai_include_directory,
            &openvino_include_directory,
        )?;
        let function_bindings_string = function_bindings.to_string();

        Self::write_functions_file(
            &output_directory.join(FUNCTIONS_FILE),
            &function_bindings_string,
            b"use super::types::*;\nuse crate::link;\nuse openvino_sys::{ov_status_e, ov_tensor_t};\n",
        )?;

        Ok(())
    }

    /// Write a functions.rs file wrapped in the `link! { }` macro.
    fn write_functions_file(path: &Path, functions: &str, prefix: &[u8]) -> Result<()> {
        {
            let mut f = Box::new(File::create(path)?);
            f.write_all(prefix)?;
            f.write_all(b"link! {\n\n")?;
            f.write_all(functions.as_bytes())
                .context(format!("Failed to write functions to: {}", path.display()))?;
        }
        let mut f = OpenOptions::new().append(true).open(path)?;
        f.write_all(b"\n}\n")?;
        Ok(())
    }

    fn resolve_path(&self, default: &str) -> Result<PathBuf> {
        Ok(match self.header_file.clone() {
            Some(path) => {
                ensure!(
                    path.is_file(),
                    "The input header file must be an actual file."
                );
                path
            }
            None => path_to_crates()?.join(default),
        })
    }

    fn resolve_output(&self, default: &str) -> Result<PathBuf> {
        Ok(match self.output_directory.clone() {
            Some(path) => {
                ensure!(
                    path.is_dir(),
                    "The output directory must be an actual directory."
                );
                path
            }
            None => path_to_crates()?.join(default),
        })
    }

    fn resolve_include_dir(default: &str, library_name: &str, header: &str) -> Result<PathBuf> {
        let include_dir = path_to_crates()?.join(default);
        if include_dir.join(header).is_file() {
            return Ok(include_dir);
        }

        let library_path = openvino_finder::find(library_name, Linking::Dynamic)
            .with_context(|| format!("Unable to find installed library for {library_name}"))?;
        Self::find_include_dir_from_library(&library_path, header).with_context(|| {
            format!(
                "Unable to derive include directory for {library_name} from {}",
                library_path.display()
            )
        })
    }

    fn find_include_dir_from_library(library_path: &Path, header: &str) -> Option<PathBuf> {
        let mut current = library_path.parent();
        while let Some(dir) = current {
            let include_dir = dir.join("include");
            if include_dir.join(header).is_file() {
                return Some(include_dir);
            }
            current = dir.parent();
        }
        None
    }

    // --- openvino-sys bindgen ---

    fn generate_sys_type_bindings<P: AsRef<Path>>(
        header_file: P,
        include_directory: &Path,
    ) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg(format!("-I{}", include_directory.display()))
            .allowlist_type("ov_.*")
            .size_t_is_usize(true)
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .with_codegen_config(bindgen::CodegenConfig::TYPES)
            .generate()
            .map_err(|_| anyhow!("unable to generate type bindings"))
    }

    fn generate_sys_function_bindings<P: AsRef<Path>>(
        header_file: P,
        include_directory: &Path,
    ) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg(format!("-I{}", include_directory.display()))
            .allowlist_function("ov_.*")
            .blocklist_type("__uint8_t")
            .blocklist_type("__int64_t")
            .size_t_is_usize(true)
            .with_codegen_config(bindgen::CodegenConfig::FUNCTIONS)
            .generate()
            .map_err(|_| anyhow!("unable to generate function bindings"))
    }

    // --- openvino-genai-sys bindgen ---

    fn generate_genai_type_bindings<P: AsRef<Path>>(
        header_file: P,
        genai_include_directory: &Path,
        openvino_include_directory: &Path,
    ) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg(format!("-I{}", genai_include_directory.display()))
            .clang_arg(format!("-I{}", openvino_include_directory.display()))
            .allowlist_type("ov_genai_.*|streamer_callback|StopCriteria")
            // Core OV types are provided by openvino-sys.
            .blocklist_type("ov_status_e")
            .blocklist_type("ov_tensor_t")
            .size_t_is_usize(true)
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .with_codegen_config(bindgen::CodegenConfig::TYPES)
            .generate()
            .map_err(|_| anyhow!("unable to generate genai type bindings"))
    }

    fn generate_genai_function_bindings<P: AsRef<Path>>(
        header_file: P,
        genai_include_directory: &Path,
        openvino_include_directory: &Path,
    ) -> Result<bindgen::Bindings> {
        bindgen::Builder::default()
            .header(header_file.as_ref().to_string_lossy())
            .clang_arg(format!("-I{}", genai_include_directory.display()))
            .clang_arg(format!("-I{}", openvino_include_directory.display()))
            .allowlist_function("ov_genai_.*")
            .blocklist_function("ov_genai_(llm|vlm|whisper)_pipeline_create")
            // Core OV types are provided by openvino-sys.
            .blocklist_type("ov_status_e")
            .blocklist_type("ov_tensor_t")
            .blocklist_type("__uint8_t")
            .blocklist_type("__int64_t")
            .size_t_is_usize(true)
            .with_codegen_config(bindgen::CodegenConfig::FUNCTIONS)
            .generate()
            .map_err(|_| anyhow!("unable to generate genai function bindings"))
    }
}

const TYPES_FILE: &str = "types.rs";
const FUNCTIONS_FILE: &str = "functions.rs";

const OV_SYS_OUTPUT: &str = "openvino-sys/src/generated";
const OV_SYS_INCLUDE: &str = "openvino-sys/upstream/src/bindings/c/include";
const OV_SYS_HEADER: &str = "openvino-sys/upstream/src/bindings/c/include/openvino/c/openvino.h";

const GENAI_OUTPUT: &str = "openvino-genai-sys/src/generated";
const GENAI_INCLUDE: &str = "openvino-genai-sys/upstream/src/c/include";
const GENAI_HEADER: &str = "openvino-genai-sys/genai_all.h";
