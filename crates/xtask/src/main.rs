#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]

mod bump;
mod codegen;
mod publish;
mod util;

use anyhow::Result;
use bump::BumpCommand;
use clap::{Parser, Subcommand};
use codegen::CodegenCommand;
use publish::PublishCommand;

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.command.execute()?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: XtaskCommand,
}

// #[structopt(
//     version = env!("CARGO_PKG_VERSION"),
//     global_settings = &[
//         AppSettings::VersionlessSubcommands,
//         AppSettings::ColoredHelp
//     ],
// )]
#[derive(Debug, Subcommand)]
enum XtaskCommand {
    /// Generate the Rust bindings for OpenVINO to use in the openvino-sys crate.
    Codegen(CodegenCommand),
    /// Increment the version of each of the publishable crates.
    Bump(BumpCommand),
    /// Publish all public crates to crates.io and add a Git release tag.
    Publish(PublishCommand),
}

impl XtaskCommand {
    fn execute(&self) -> Result<()> {
        match self {
            Self::Codegen(codegen) => codegen.execute(),
            Self::Bump(bump) => bump.execute(),
            Self::Publish(publish) => publish.execute(),
        }
    }
}
