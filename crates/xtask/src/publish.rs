use crate::util::{exec, get_crates, path_to_crates, Crate};
use anyhow::{anyhow, Result};
use std::{process::Command, thread::sleep, time::Duration};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bump")]
pub struct PublishCommand {
    /// Tag the current commit and push the tags to the default upstream; equivalent to `git tag
    /// v[version] && git push origin v[version]`.
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
            let krate_dir = crates_dir.clone().join(krate);
            let mut command = Command::new("cargo");
            command.current_dir(&krate_dir).arg("publish");
            if self.dry_run {
                command.arg("--dry-run");
            } else {
                command.arg("--no-verify");
            }

            let exec_result = exec(&mut command);

            // We want to continue even if a crate does not publish: this allows us to re-run the
            // `publish` command if uploading one or more crates fails. In `--dry-run` mode,
            // however, we do want to fail the process immediately to identify any issues.
            if let Err(e) = exec_result {
                if self.dry_run {
                    panic!("Failed to publish crate {}:\n  {}", krate, e);
                } else {
                    println!("Failed to publish crate {}, continuing:\n  {}", krate, e);
                }
            }

            // Hopefully this gives crates.io enough time for subsequent publications to work.
            if !self.dry_run {
                sleep(Duration::from_secs(20));
            }
        }

        // Tag the repository.
        let tag = format!("v{}", publishable_crates[0].version);
        if self.git {
            println!("> push Git tag: {}", tag);
            if !self.dry_run {
                exec(Command::new("git").arg("tag").arg(&tag))?;
                exec(Command::new("git").arg("push").arg("origin").arg(&tag))?;
            }
        }

        Ok(())
    }
}

const PUBLICATION_ORDER: &[&str] = &["openvino-finder", "openvino-sys", "openvino"];
