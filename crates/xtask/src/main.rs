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
use codegen::CodegenCommand;
use publish::PublishCommand;
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
