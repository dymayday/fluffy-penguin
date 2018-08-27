//! Utility repository.

use std::io::Error;
use std::{fs, path::Path};

/// Create the all the missing directory tree of file path if it does not exist.
/// Recursively create all of the parent components of a file if they are missing.
pub fn create_parent_directory(file_path: &str) -> Result<(), Error> {
    let basename: &str = Path::new(file_path).parent().unwrap().to_str().unwrap();
    if !Path::new(basename).exists() {
        fs::create_dir_all(basename)?;
    }
    Ok(())
}
