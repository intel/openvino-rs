use anyhow::{anyhow, Context, Result};
use semver::Version;
use std::path::PathBuf;
use std::{fs, process::Command};
use toml::Value;

/// Convenience wrapper for executing commands.
pub fn exec(command: &mut Command) -> Result<()> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("failed to execute: {:?}", &command))
    }
}

/// Determine the path to the `crates` directory.
pub fn path_to_crates() -> Result<PathBuf> {
    Ok(PathBuf::from(file!())
        .parent()
        .with_context(|| "Failed to get parent of path.".to_string())?
        .parent()
        .with_context(|| "Failed to get parent of path.".to_string())?
        .parent()
        .with_context(|| "Failed to get parent of path.".to_string())?
        .into())
}

/// Determine the path to the top-level `Cargo.toml` file.
pub fn get_top_level_cargo_toml() -> Result<PathBuf> {
    let crates_dir = path_to_crates()?;
    let top_level_dir = crates_dir
        .parent()
        .with_context(|| "Failed to get parent of path.".to_string())?;
    let cargo_toml = top_level_dir.join("Cargo.toml");
    assert!(cargo_toml.is_file());
    Ok(cargo_toml)
}

/// Parse the top-level `Cargo.toml` for the workspace version.
pub fn get_top_level_version() -> Result<Version> {
    let path = get_top_level_cargo_toml()?;
    let contents = fs::read_to_string(&path)?;
    let toml: Value = contents
        .parse()
        .with_context(|| format!("unable to parse TOML of {}", &path.display()))?;

    let version = toml["workspace"]["package"]["version"]
        .as_str()
        .with_context(|| "No top-level package version in workspace Cargo.toml")?
        .to_owned();
    Ok(Version::parse(&version)?)
}

/// Retrieve information about all of the crates found in the `crates` directory.
pub fn get_crates() -> Result<Vec<Crate>> {
    let crates_dir = path_to_crates()?;
    assert!(crates_dir.is_dir());

    let mut crates = Vec::new();
    for entry in fs::read_dir(crates_dir)? {
        let path = entry?.path().join("Cargo.toml");
        let contents = fs::read_to_string(&path)?;
        let toml: Value = contents
            .parse()
            .with_context(|| format!("unable to parse TOML of {}", &path.display()))?;
        let name = toml["package"]["name"]
            .as_str()
            .with_context(|| "Every Cargo.toml should have a package name")?
            .to_owned();

        let publish = toml["package"]
            .get("publish")
            .unwrap_or(&Value::Boolean(true))
            .as_bool()
            .unwrap();

        crates.push(Crate {
            name,
            path,
            publish,
        });
    }

    Ok(crates)
}

/// A wrapper for describing found crates.
#[derive(Debug)]
pub struct Crate {
    pub name: String,
    pub path: PathBuf,
    pub publish: bool,
}
