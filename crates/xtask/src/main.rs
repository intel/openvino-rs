mod bump;
mod codegen;
mod util;

use anyhow::Result;
use bump::BumpCommand;
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
    /// Increment the version of each of the publishable crates.
    Bump(BumpCommand),
}

impl XtaskCommand {
    fn execute(&self) -> Result<()> {
        match self {
            Self::Codegen(codegen) => codegen.execute(),
            Self::Bump(bump) => bump.execute(),
        }
    }
}
