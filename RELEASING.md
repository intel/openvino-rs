# Release Process

To publish a new version of these crates:

1. Bump the crate versions using semantic versioning. As described in Cargo's [semver compatibility]
   documentation and because this crate is not yet released as "1.0", we typically release major
   breaking changes as a _minor_ semver bump and anything less than that as a _patch_ semver bump.
   After bumping, open a [PR]:

   ```shell script
   git checkout -b bump-version
   cargo xtask bump patch --git
   git push -u <FORK>
   # open PR
   ```

[semver compatibility]: https://doc.rust-lang.org/cargo/reference/semver.html#change-categories
[PR]: https://github.com/intel/openvino-rs/pulls

2. Once the bump PR is merged, publish the crates:

   ```shell script
   git checkout main
   git pull
   cargo xtask publish --git
   ```

   You may want to check that everything looks right on [crates.io] after this step. Note that the
   `--git` flag is equivalent to `git tag v<VERSION>; git push origin v<VERSION>`. If your remotes
   won't work with this default `origin` for any reason, those commands can be run manually.

[crates.io]: https://crates.io/crates/openvino

3. Once published, [draft a new release] on GitHub. Use the newly-published tag and use "Generate
   release notes" for a sane changelog description.

[draft a new release]: https://github.com/intel/openvino-rs/releases/new

4. Eventually, check that [docs.rs] was able to build the crate documentation correctly.

[docs.rs]: https://docs.rs/openvino
