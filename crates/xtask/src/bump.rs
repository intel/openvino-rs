use crate::util::path_to_crates;
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bump")]
pub struct BumpCommand {
    /// Do not modify the Cargo.toml files; instead, simply print the actions that would have been
    /// taken.
    #[structopt(long = "dry-run")]
    dry_run: bool,
    /// What part of the semver version to change: major | minor | patch | <version string>
    #[structopt(name = "BUMP")]
    bump: Bump,
}

impl BumpCommand {
    /// Because of how the linking is implemented (i.e. a `link!` macro that must wrap around the
    /// foreign functions), we must split the bindgen generation of types and functions into
    /// separate files. This means that, at least for the function output, we also need a prefix
    /// (e.g. to add the `link!` macro) and suffix to make things compile.
    pub fn execute(&self) -> Result<()> {
        // Find the publishable crates.
        let publishable_crates: Vec<Crate> =
            get_crates()?.into_iter().filter(|c| c.publish).collect();

        // Check that all of the versions are the same.
        if !publishable_crates
            .windows(2)
            .all(|w| w[0].version == w[1].version)
        {
            anyhow!(
                "Not all crate versions are the same: {:?}",
                publishable_crates
            );
        }

        // Change the version.
        let mut next_version = semver::Version::parse(&publishable_crates[0].version)?;
        match &self.bump {
            Bump::Major => next_version.increment_major(),
            Bump::Minor => next_version.increment_minor(),
            Bump::Patch => next_version.increment_patch(),
            Bump::Custom(v) => next_version = semver::Version::parse(v)?,
        }

        // Update the Cargo.toml files.
        let next_version_str = &next_version.to_string();
        for c in publishable_crates.iter() {
            update_version(c, &publishable_crates, next_version_str, self.dry_run)?;
        }

        // Update the Cargo.lock file.
        if !self.dry_run {
            assert!(Command::new("cargo").arg("fetch").status()?.success());
        }

        Ok(())
    }
}

/// Enumerate the ways a version can change.
#[derive(Debug)]
pub enum Bump {
    Major,
    Minor,
    Patch,
    Custom(String),
}

impl std::str::FromStr for Bump {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "major" => Self::Major,
            "minor" => Self::Minor,
            "patch" => Self::Patch,
            _ => Self::Custom(s.into()),
        })
    }
}

/// Retrieve information about all of the crates found in the `crates` directory.
fn get_crates() -> Result<Vec<Crate>> {
    let crates_dir = path_to_crates()?;
    assert!(crates_dir.is_dir());

    let mut crates = Vec::new();
    for entry in fs::read_dir(crates_dir)? {
        let path = entry?.path().join("Cargo.toml");
        let contents = fs::read(&path)?;
        let toml: toml::Value = toml::from_slice(&contents)
            .with_context(|| format!("unable to parse TOML of {}", &path.display()))?;

        let name = toml["package"]["name"]
            .as_str()
            .with_context(|| "Every Cargo.toml should have a package name")?
            .to_owned();

        let version = toml["package"]["version"]
            .as_str()
            .with_context(|| "Every Cargo.toml should have a package name")?
            .to_owned();
        semver::Version::parse(&version)?;

        let publish = toml["package"]
            .get("publish")
            .unwrap_or(&toml::Value::Boolean(true))
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

#[derive(Debug)]
struct Crate {
    name: String,
    path: PathBuf,
    version: String,
    publish: bool,
}

/// Update the version of `krate` and any dependencies in `crates` to match the version passed in
/// `next_version`. Adapted from
/// https://github.com/bytecodealliance/wasmtime/blob/main/scripts/publish.rs
fn update_version(
    krate: &Crate,
    crates: &[Crate],
    next_version: &str,
    dry_run: bool,
) -> Result<()> {
    let contents = fs::read_to_string(&krate.path)?;
    let mut new_contents = String::new();
    let mut reading_dependencies = false;
    for line in contents.lines() {
        let mut rewritten = false;

        // Update top-level `version = "..."` line.
        if !reading_dependencies && line.starts_with("version =") {
            println!(
                "bump `{}` {} => {}",
                krate.name, krate.version, next_version
            );
            new_contents.push_str(&line.replace(&krate.version.to_string(), next_version));
            rewritten = true;
        }

        // Check whether we have reached the `[dependencies]` section.
        reading_dependencies = if line.starts_with("[") {
            line.contains("dependencies")
        } else {
            reading_dependencies
        };

        // Find dependencies and update them as well.
        for other in crates {
            if !reading_dependencies || !line.starts_with(&format!("{} ", other.name)) {
                continue;
            }
            if !line.contains(&other.version.to_string()) {
                if !line.contains("version =") {
                    continue;
                }
                panic!(
                    "{:?} has a dependency on {} but doesn't list version {}",
                    krate.path, other.name, other.version
                );
            }
            println!(
                "  bump dependency `{}` {} => {}",
                other.name, other.version, next_version
            );
            rewritten = true;
            new_contents.push_str(&line.replace(&other.version, next_version));
            break;
        }

        // All other lines are printed as-is.
        if !rewritten {
            new_contents.push_str(line);
        }

        new_contents.push_str("\n");
    }

    if !dry_run {
        fs::write(&krate.path, new_contents)?;
    }

    Ok(())
}
