use crate::util::{get_crates, get_top_level_cargo_toml, get_top_level_version, Crate};
use anyhow::{Context, Result};
use clap::Args;
use semver::{BuildMetadata, Prerelease};
use std::fs;
use std::process::Command;

#[derive(Debug, Args)]
pub struct BumpCommand {
    /// Do not modify the Cargo.toml files; instead, simply print the actions that would have been
    /// taken.
    #[arg(long = "dry-run")]
    dry_run: bool,
    /// Add a conventional Git commit message for the bump changes; equivalent to `git commit -a -m
    /// 'Release v[bumped version]'`.
    #[arg(long)]
    git: bool,
    /// What part of the semver version to change: major | minor | patch | [version string]
    #[arg(name = "KIND")]
    bump: Bump,
}

impl BumpCommand {
    pub fn execute(&self) -> Result<()> {
        // Find the publishable crates.
        let publishable_crates: Vec<Crate> =
            get_crates()?.into_iter().filter(|c| c.publish).collect();

        // Change the version. Unless specified with a custom version, the `pre` and `build`
        // metadata are cleared.
        let current_version = get_top_level_version()?;
        let mut next_version = current_version.clone();
        next_version.pre = Prerelease::EMPTY;
        next_version.build = BuildMetadata::EMPTY;
        match &self.bump {
            Bump::Major => {
                next_version.major += 1;
                next_version.minor = 0;
                next_version.patch = 0;
            }
            Bump::Minor => {
                next_version.minor += 1;
                next_version.patch = 0;
            }
            Bump::Patch => {
                next_version.patch += 1;
            }
            Bump::Custom(v) => next_version = semver::Version::parse(v)?,
        }

        // Update the top-level Cargo.toml version. We expect all the crates use the top-level
        // workspace version.
        assert!(publishable_crates.iter().all(uses_workspace_version));
        let current_version_str = current_version.to_string();
        let next_version_str = next_version.to_string();
        update_version(
            &publishable_crates,
            &current_version_str,
            &next_version_str,
            self.dry_run,
        )?;

        // Update the Cargo.lock file.
        if !self.dry_run {
            assert!(Command::new("cargo").arg("fetch").status()?.success());
        }

        // Add a Git commit.
        let commit_message = format!("Release v{next_version_str}");
        if self.git {
            println!("> add Git commit: {}", &commit_message);
            if !self.dry_run && self.git {
                assert!(Command::new("git")
                    .arg("commit")
                    .arg("-a")
                    .arg("-m")
                    .arg(&commit_message)
                    .status()
                    .with_context(|| "failed to run `git commit` command".to_string())?
                    .success());
            }
        }

        Ok(())
    }
}

/// Enumerate the ways a version can change.
#[derive(Clone, Debug)]
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

/// Check that a publishable crate pulls its version from the workspace version.
fn uses_workspace_version(krate: &Crate) -> bool {
    let contents = fs::read_to_string(&krate.path).unwrap();
    let toml: toml::Value = contents.parse().unwrap();
    let version_workspace = &toml["package"]["version"]["workspace"];
    *version_workspace == toml::Value::Boolean(true)
}

/// Update the version in the top-level Cargo.toml and any dependencies in its
/// `[workspace.dependencies]` to match the version passed in `next_version`. Adapted from
/// Wasmtime's [publish.rs] script.
///
/// [publish.rs]: https://github.com/bytecodealliance/wasmtime/blob/main/scripts/publish.rs
fn update_version(
    crates: &[Crate],
    current_version: &str,
    next_version: &str,
    dry_run: bool,
) -> Result<()> {
    let top_level_cargo_toml_path = get_top_level_cargo_toml()?;
    let contents = fs::read_to_string(&top_level_cargo_toml_path)?;
    let mut new_contents = String::new();
    let mut reading_dependencies = false;
    for line in contents.lines() {
        let mut rewritten = false;

        // Update top-level `version = "..."` line.
        if !reading_dependencies && line.starts_with("version =") {
            let modified_line = line.replace(current_version, next_version);
            println!(
                "> bump: {} => {}",
                line.trim_start_matches("version =").trim(),
                modified_line.trim_start_matches("version =").trim()
            );
            new_contents.push_str(&modified_line);
            rewritten = true;
        }

        // Check whether we have reached the `[dependencies]` section.
        reading_dependencies = if line.starts_with('[') {
            line.contains("dependencies")
        } else {
            reading_dependencies
        };

        // Find dependencies and update them as well.
        for other in crates {
            if !reading_dependencies || !line.starts_with(&format!("{} ", other.name)) {
                continue;
            }
            if !line.contains(current_version) {
                if !line.contains("version =") {
                    continue;
                }
                panic!(
                    "workspace dependency {} doesn't list version {}",
                    other.name, current_version
                );
            }
            println!(
                ">   bump dependency `{}` {} => {}",
                other.name, current_version, next_version
            );
            rewritten = true;
            new_contents.push_str(&line.replace(current_version, next_version));
            break;
        }

        // All other lines are printed as-is.
        if !rewritten {
            new_contents.push_str(line);
        }

        new_contents.push('\n');
    }

    if !dry_run {
        fs::write(top_level_cargo_toml_path, new_contents)?;
    }

    Ok(())
}
