use anyhow::{Context, Result};
use std::path::PathBuf;

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
