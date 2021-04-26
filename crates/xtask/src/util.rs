use anyhow::{Context, Result};
use semver::Version;
use std::fs;
use std::path::PathBuf;
use toml::Value;

/// Determine the path to the `crates` directory`.
pub fn path_to_crates() -> Result<PathBuf> {
    Ok(PathBuf::from(file!())
        .parent()
        .with_context(|| format!("Failed to get parent of path."))?
        .parent()
        .with_context(|| format!("Failed to get parent of path."))?
        .parent()
        .with_context(|| format!("Failed to get parent of path."))?
        .into())
}

/// Retrieve information about all of the crates found in the `crates` directory.
pub fn get_crates() -> Result<Vec<Crate>> {
    let crates_dir = path_to_crates()?;
    assert!(crates_dir.is_dir());

    let mut crates = Vec::new();
    for entry in fs::read_dir(crates_dir)? {
        let path = entry?.path().join("Cargo.toml");
        let contents: Vec<u8> = fs::read(&path)?;
        let toml: Value = toml::from_slice(&contents)
            .with_context(|| format!("unable to parse TOML of {}", &path.display()))?;

        let name = toml["package"]["name"]
            .as_str()
            .with_context(|| "Every Cargo.toml should have a package name")?
            .to_owned();

        let version = toml["package"]["version"]
            .as_str()
            .with_context(|| "Every Cargo.toml should have a package name")?
            .to_owned();
        Version::parse(&version)?;

        let publish = toml["package"]
            .get("publish")
            .unwrap_or(&Value::Boolean(true))
            .as_bool()
            .unwrap();

        crates.push(Crate {
            name,
            version,
            publish,
            path,
        })
    }

    Ok(crates)
}

/// A wrapper for describing found crates.
#[derive(Debug)]
pub struct Crate {
    pub name: String,
    pub path: PathBuf,
    pub version: String,
    pub publish: bool,
}
