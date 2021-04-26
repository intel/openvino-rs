use crate::util::{get_crates, path_to_crates, Crate};
use anyhow::{anyhow, Context, Result};
use std::{process::Command, thread::sleep, time::Duration};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bump")]
pub struct PublishCommand {
    /// Tag the current commit and push the tags to the default upstream; equivalent to `git tag
    /// v[version]` and `git push v[version]`.
    #[structopt(long)]
    git: bool,
    /// Do not publish any crates; instead, simply print the actions that would have been taken.
    #[structopt(long = "dry-run")]
    dry_run: bool,
}

impl PublishCommand {
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

        // Check that all of the publishable crates are in `PUBLICATION_ORDER`.
        assert_eq!(publishable_crates.len(), PUBLICATION_ORDER.len());
        assert!(publishable_crates
            .iter()
            .all(|c| PUBLICATION_ORDER.iter().any(|d| d == &c.name)));

        // Publish each crate.
        let crates_dir = path_to_crates()?;
        for krate in PUBLICATION_ORDER {
            println!("> publish {}", krate);
            if !self.dry_run {
                assert!(Command::new("cargo")
                    .arg("publish")
                    .current_dir(crates_dir.clone().join(krate))
                    .arg("--no-verify")
                    .status()
                    .with_context(|| format!("failed to run cargo publish on '{}' crate", krate))?
                    .success());

                // Hopefully this gives crates.io enough time for subsequent publications to work.
                sleep(Duration::from_secs(20));
            }
        }

        // Tag the repository.
        let tag = format!("v{}", publishable_crates[0].version);
        if self.git {
            println!("> push Git tag: {}", tag);
            if !self.dry_run {
                assert!(Command::new("git")
                    .arg("tag")
                    .arg(&tag)
                    .status()
                    .with_context(|| format!("failed to run `git tag {}` command", &tag))?
                    .success());
                assert!(Command::new("git")
                    .arg("push")
                    .arg(&tag)
                    .status()
                    .with_context(|| format!("failed to run `git push {}` command", &tag))?
                    .success());
            }
        }

        Ok(())
    }
}

const PUBLICATION_ORDER: &[&str] = &["openvino-finder", "openvino-sys", "openvino"];
