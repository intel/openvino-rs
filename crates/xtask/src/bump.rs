use crate::util::{get_crates, Crate};
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::process::Command;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bump")]
pub struct BumpCommand {
    /// Do not modify the Cargo.toml files; instead, simply print the actions that would have been
    /// taken.
    #[structopt(long = "dry-run")]
    dry_run: bool,
    /// Add a conventional Git commit message for the bump changes; equivalent to `git commit -a -m
    /// 'Release v[bumped version]'`.
    #[structopt(long)]
    git: bool,
    /// What part of the semver version to change: major | minor | patch | <version string>
    #[structopt(name = "KIND")]
    bump: Bump,
}

impl BumpCommand {
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

        // Add a Git commit.
        let commit_message = format!("Release v{}", next_version_str);
        if self.git {
            println!("> add Git commit: {}", &commit_message);
            if !self.dry_run && self.git {
                assert!(Command::new("git")
                    .arg("commit")
                    .arg("-a")
                    .arg("-m")
                    .arg(&commit_message)
                    .status()
                    .with_context(|| format!("failed to run `git commit` command"))?
                    .success());
            }
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
                "> bump `{}` {} => {}",
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
                ">   bump dependency `{}` {} => {}",
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
