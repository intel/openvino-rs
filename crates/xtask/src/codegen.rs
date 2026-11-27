use crate::util::path_to_crates;
use anyhow::{anyhow, ensure, Context, Result};
use clap::{Args, ValueEnum};
use openvino_finder::Linking;
use std::fs::File;
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

        let function_bindings_string = function_bindings.to_string();

        Self::write_functions_file(
            &output_directory.join(FUNCTIONS_FILE),
            &function_bindings_string,
            "use super::types::*;\ntype wchar_t = ::std::os::raw::c_char;\n",
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
            "use super::types::*;\nuse openvino_sys::{ov_status_e, ov_tensor_t};\n",
        )?;

        Ok(())
    }

    /// Write a functions.rs file, splitting the bindings between the `link! { }` and
    /// `link_variadic! { }` macros.
    ///
    /// The two macros exist because a C-variadic function cannot be proxied through a generated
    /// Rust `fn`: Rust can *declare* a C-variadic function but cannot *define* one on stable,
    /// so there is no way to forward the varargs. `link!` generates such proxies and would silently
    /// drop the `...`, leaving the function declared with the wrong arity. This is an ABI mismatch that
    /// crashes at run time on targets where the variadic calling convention differs from the fixed
    /// one. That's the case for macOS aarch64, which passes varargs on the stack rather than in registers.
    /// `link_variadic!` binds those symbols to function pointers instead, preserving the `...`.
    ///
    /// `macro_rules!` cannot dispatch on the presence of `...` within a single macro (a `ty`
    /// fragment may not be followed by a `tt`), hence the split here rather than in the macro.
    fn write_functions_file(path: &Path, functions: &str, prefix: &str) -> Result<()> {
        let (fixed, variadic) = Self::partition_variadic_declarations(functions);

        let mut f = File::create(path)?;
        f.write_all(prefix.as_bytes())?;

        // The macros are invoked by path rather than imported, so that a crate whose C API has no
        // variadic functions (currently `openvino-genai-sys`, which does not define
        // `link_variadic!`) never names a macro it does not have.
        f.write_all(b"crate::link! {\n\n")?;
        f.write_all(fixed.as_bytes())
            .context(format!("Failed to write functions to: {}", path.display()))?;
        f.write_all(b"\n}\n")?;

        if !variadic.trim().is_empty() {
            f.write_all(b"\ncrate::link_variadic! {\n\n")?;
            f.write_all(variadic.as_bytes()).context(format!(
                "Failed to write variadic functions to: {}",
                path.display()
            ))?;
            f.write_all(b"\n}\n")?;
        }

        Ok(())
    }

    /// Split bindgen's output into the `extern "C" { … }` blocks that declare only fixed-arity
    /// functions and those that declare a C-variadic one (a `...` parameter).
    ///
    /// Blocks that contain no function declaration at all (bindgen's leading comment, for example)
    /// stay with the fixed-arity output so nothing is lost.
    fn partition_variadic_declarations(functions: &str) -> (String, String) {
        let mut fixed = String::with_capacity(functions.len());
        let mut variadic = String::new();

        let mut rest = functions;
        while let Some(start) = rest.find("unsafe extern \"C\" {") {
            // Everything before this block (comments, blank lines) belongs with the fixed output.
            fixed.push_str(&rest[..start]);
            rest = &rest[start..];

            // `extern` blocks are not nested, so the first closing brace at the start of a line
            // ends this one.
            let end = rest
                .find("\n}\n")
                .map_or(rest.len(), |offset| offset + "\n}\n".len());
            let (block, remainder) = rest.split_at(end);
            rest = remainder;

            if Self::declares_variadic_function(block) {
                variadic.push_str(block);
            } else {
                fixed.push_str(block);
            }
        }
        fixed.push_str(rest);

        (fixed, variadic)
    }

    /// Whether an `extern "C" { … }` block declares a C-variadic function.
    ///
    /// bindgen emits the variadic marker as a parameter on its own line, so match that line
    /// exactly: a `...` elsewhere in the block, such as in a doc comment ("@param ... property
    /// pairs"), must not move a fixed-arity function into `link_variadic!`.
    fn declares_variadic_function(block: &str) -> bool {
        block.lines().any(|line| line.trim() == "...")
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

#[cfg(test)]
mod tests {
    use super::CodegenCommand;
    use std::path::PathBuf;

    /// A fixed-arity declaration, formatted the way bindgen emits it.
    const FIXED: &str = "unsafe extern \"C\" {
    #[doc = \" @brief Frees a core.\"]
    pub fn ov_core_free(core: *mut ov_core_t);
}
";

    /// A C-variadic declaration, formatted the way bindgen emits it.
    const VARIADIC: &str = "unsafe extern \"C\" {
    #[doc = \" @brief Sets properties.\\n @param ... property pairs\"]
    pub fn ov_core_set_property(
        core: *const ov_core_t,
        device_name: *const ::std::os::raw::c_char,
        ...
    ) -> ov_status_e;
}
";

    /// A fixed-arity declaration whose documentation mentions `...`.
    const FIXED_WITH_ELLIPSIS_IN_DOC: &str = "unsafe extern \"C\" {
    #[doc = \" @brief Gets a value.\\n @param ... is not a parameter here\"]
    pub fn ov_get_value(
        value: *mut ::std::os::raw::c_char,
    ) -> ov_status_e;
}
";

    const BINDGEN_HEADER: &str = "\n/* automatically generated by rust-bindgen 0.72.0 */\n\n";

    #[test]
    fn partition_splits_fixed_and_variadic_blocks() {
        let input = format!("{FIXED}{VARIADIC}{FIXED}");
        let (fixed, variadic) = CodegenCommand::partition_variadic_declarations(&input);
        assert_eq!(fixed, format!("{FIXED}{FIXED}"));
        assert_eq!(variadic, VARIADIC);
    }

    #[test]
    fn partition_keeps_text_outside_blocks_with_fixed_output() {
        let input = format!("{BINDGEN_HEADER}{VARIADIC}{FIXED}\n");
        let (fixed, variadic) = CodegenCommand::partition_variadic_declarations(&input);
        assert_eq!(fixed, format!("{BINDGEN_HEADER}{FIXED}\n"));
        assert_eq!(variadic, VARIADIC);
    }

    #[test]
    fn partition_loses_nothing() {
        let input =
            format!("{BINDGEN_HEADER}{FIXED}{VARIADIC}{FIXED_WITH_ELLIPSIS_IN_DOC}{VARIADIC}");
        let (fixed, variadic) = CodegenCommand::partition_variadic_declarations(&input);
        assert_eq!(fixed.len() + variadic.len(), input.len());
        assert_eq!(variadic, format!("{VARIADIC}{VARIADIC}"));
    }

    #[test]
    fn partition_without_variadic_functions_is_identity() {
        let input = format!("{BINDGEN_HEADER}{FIXED}{FIXED}");
        let (fixed, variadic) = CodegenCommand::partition_variadic_declarations(&input);
        assert_eq!(fixed, input);
        assert_eq!(variadic, "");
    }

    #[test]
    fn partition_of_text_without_blocks_is_identity() {
        let (fixed, variadic) = CodegenCommand::partition_variadic_declarations(BINDGEN_HEADER);
        assert_eq!(fixed, BINDGEN_HEADER);
        assert_eq!(variadic, "");
    }

    /// A `...` in a doc comment is not a variadic parameter: moving such a function into
    /// `link_variadic!()` would fail to compile (that macro requires a `, ...` parameter).
    #[test]
    fn ellipsis_in_documentation_is_not_variadic() {
        let (fixed, variadic) =
            CodegenCommand::partition_variadic_declarations(FIXED_WITH_ELLIPSIS_IN_DOC);
        assert_eq!(fixed, FIXED_WITH_ELLIPSIS_IN_DOC);
        assert_eq!(variadic, "");
    }

    /// Writes `functions` through [`CodegenCommand::write_functions_file()`] and returns the result.
    fn write_and_read(name: &str, functions: &str) -> String {
        let path =
            std::env::temp_dir().join(format!("xtask-codegen-{}-{name}.rs", std::process::id()));
        CodegenCommand::write_functions_file(&path, functions, "use super::types::*;\n").unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        written
    }

    #[test]
    fn write_functions_file_emits_both_macros() {
        let written = write_and_read("both", &format!("{FIXED}{VARIADIC}"));
        assert_eq!(
            written,
            format!(
                "use super::types::*;\ncrate::link! {{\n\n{FIXED}\n}}\n\ncrate::link_variadic! {{\n\n{VARIADIC}\n}}\n"
            )
        );
    }

    /// `openvino-genai-sys` has no variadic functions and does not define `link_variadic!()`: the
    /// generated file must not name it, or that crate would fail to compile.
    #[test]
    fn write_functions_file_omits_link_variadic_when_unneeded() {
        let written = write_and_read("fixed-only", FIXED);
        assert_eq!(
            written,
            format!("use super::types::*;\ncrate::link! {{\n\n{FIXED}\n}}\n")
        );
        assert!(!written.contains("link_variadic"));
    }

    /// The macros are invoked by path so that the generated file does not depend on an import.
    #[test]
    fn write_functions_file_does_not_import_macros() {
        let written = write_and_read("no-import", &format!("{FIXED}{VARIADIC}"));
        assert!(!written.contains("use crate::link"));
    }

    fn generated_functions(crate_name: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(crate_name)
            .join("src/generated/functions.rs");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
    }

    /// Return the body of the `name! { … }` invocation in a generated file, if present.
    fn macro_body<'a>(file: &'a str, name: &str) -> Option<&'a str> {
        let open = format!("crate::{name}! {{\n");
        let start = file.find(&open)? + open.len();
        // `write_functions_file()` closes the macro on its own line after a blank one; a bare
        // `\n}\n` would instead match the end of the first `extern` block inside it.
        let len = file[start..]
            .find("\n\n}\n")
            .expect("unterminated macro invocation");
        Some(&file[start..start + len])
    }

    /// Guard the checked-in `openvino-sys` bindings: every variadic function must live in
    /// `link_variadic!()` and nothing else may.
    #[test]
    fn checked_in_sys_bindings_are_partitioned() {
        let file = generated_functions("openvino-sys");
        let fixed = macro_body(&file, "link").expect("missing `crate::link!`");
        let variadic = macro_body(&file, "link_variadic").expect("missing `crate::link_variadic!`");

        let (_, misplaced) = CodegenCommand::partition_variadic_declarations(fixed);
        assert!(
            misplaced.is_empty(),
            "variadic functions found in `link!`:\n{}",
            misplaced
        );
        let (stray, _) = CodegenCommand::partition_variadic_declarations(variadic);
        assert!(
            !stray.contains("pub fn"),
            "fixed-arity functions found in `link_variadic!`:\n{}",
            stray
        );
        assert!(variadic.contains("pub fn ov_core_set_property("));
    }

    #[test]
    fn checked_in_genai_bindings_have_no_variadic_functions() {
        let file = generated_functions("openvino-genai-sys");
        let fixed = macro_body(&file, "link").expect("missing `crate::link!`");
        let (_, variadic) = CodegenCommand::partition_variadic_declarations(fixed);
        assert!(
            variadic.is_empty(),
            "unexpected variadic functions:\n{}",
            variadic
        );
        assert!(!file.contains("link_variadic"));
        assert!(!file.contains("use crate::link"));
    }
}
