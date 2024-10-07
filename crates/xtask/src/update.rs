use crate::util::{exec, path_to_crates};
use anyhow::Result;
use clap::Args;
use std::process::Command;

#[derive(Debug, Args)]
pub struct UpdateCommand {
    /// The upstream tag to update to; see <https://github.com/openvinotoolkit/openvino/releases>.
    #[arg(name = "TAG")]
    tag: String,
}

impl UpdateCommand {
    /// Retrieve the requested tag and checkout the upstream submodule using `git`.
    pub fn execute(&self) -> Result<()> {
        let submodule = path_to_crates()?.join("openvino-sys/upstream");
        let submodule = submodule.to_string_lossy();
        exec(Command::new("git").args(["-C", &submodule, "fetch", "origin", "tag", &self.tag]))?;
        exec(Command::new("git").args(["-C", &submodule, "checkout", &self.tag]))?;
        println!("> to use the updated headers, run `cargo xtask codegen`");
        Ok(())
    }
}
