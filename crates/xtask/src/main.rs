mod codegen;

use anyhow::Result;
use codegen::CodegenCommand;
use structopt::{clap::AppSettings, StructOpt};

fn main() -> Result<()> {
    let command = XtaskCommand::from_args();
    command.execute()?;
    Ok(())
}

#[derive(StructOpt, Debug)]
#[structopt(
    version = env!("CARGO_PKG_VERSION"),
    global_settings = &[
        AppSettings::VersionlessSubcommands,
        AppSettings::ColoredHelp
    ],
)]
enum XtaskCommand {
    /// Generate the Rust bindings for OpenVINO to use in the openvino-sys crate.
    Codegen(CodegenCommand),
}

impl XtaskCommand {
    fn execute(&self) -> Result<()> {
        match self {
            Self::Codegen(codegen) => codegen.execute(),
        }
    }
}
