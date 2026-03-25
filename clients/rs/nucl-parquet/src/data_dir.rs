use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Version tag for the data archive (matches crate version).
const DATA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// GitHub release URL pattern.
#[cfg(feature = "fetch")]
const RELEASE_URL: &str = "https://github.com/exoma-ch/nucl-parquet/releases/download";

/// Resolved data directory with auto-download support.
///
/// Locates the nucl-parquet data files on disk, optionally downloading them
/// from a GitHub Release when the `fetch` feature is enabled.
///
/// # Resolution order
///
/// 1. `$NUCL_PARQUET_DATA` environment variable (if set and valid)
/// 2. `~/.nucl-parquet/v{VERSION}/` cache directory
#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
}

impl DataDir {
    /// Resolve data directory without downloading.
    ///
    /// Returns an error if no data is found. Use [`ensure()`](Self::ensure)
    /// (requires the `fetch` feature) to auto-download.
    pub fn resolve() -> Result<Self> {
        if let Ok(env) = std::env::var("NUCL_PARQUET_DATA") {
            let p = PathBuf::from(env);
            if p.is_dir() {
                return Ok(Self { root: p });
            }
        }
        let cache = Self::cache_dir();
        if cache.join("meta").is_dir() {
            return Ok(Self { root: cache });
        }
        Err(Error::DataNotFound)
    }

    /// Ensure data is available, downloading from GitHub Releases if needed.
    ///
    /// Tries [`resolve()`](Self::resolve) first; downloads only when no local
    /// data is found.
    #[cfg(feature = "fetch")]
    pub fn ensure() -> Result<Self> {
        if let Ok(d) = Self::resolve() {
            return Ok(d);
        }
        Self::download()?;
        Self::resolve()
    }

    /// Path to the data root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the `meta/` subdirectory (EPDL97, EADL, EEDL, XCOM, etc.).
    pub fn meta(&self) -> PathBuf {
        self.root.join("meta")
    }

    /// Path to the `stopping/` subdirectory.
    pub fn stopping(&self) -> PathBuf {
        self.root.join("stopping")
    }

    // -- convenience openers ------------------------------------------------

    /// Open the photon cross-section database (EPDL97).
    pub fn photon_db(&self) -> Result<crate::PhotonDb> {
        crate::PhotonDb::open(self.meta())
    }

    /// Open the atomic relaxation database (EADL).
    pub fn relaxation_db(&self) -> Result<crate::RelaxationDb> {
        crate::RelaxationDb::open(self.meta())
    }

    /// Open the subshell photoelectric database.
    pub fn subshell_pe_db(&self) -> Result<crate::SubshellPeDb> {
        crate::SubshellPeDb::open(self.meta())
    }

    /// Open the XCOM total attenuation database.
    pub fn xcom_db(&self) -> Result<crate::XcomDb> {
        crate::XcomDb::open(self.meta())
    }

    /// Open the electron cross-section database (EEDL).
    pub fn electron_db(&self) -> Result<crate::ElectronDb> {
        crate::ElectronDb::open(self.meta())
    }

    /// Open the stopping power database (PSTAR/ASTAR/ESTAR/CatIMA).
    pub fn stopping_db(&self) -> Result<crate::StoppingDb> {
        crate::StoppingDb::open(self.stopping())
    }

    /// Open the isotopic abundances database.
    pub fn abundances_db(&self) -> Result<crate::AbundancesDb> {
        crate::AbundancesDb::open(self.meta())
    }

    /// Open the radioactive decay database.
    pub fn decay_db(&self) -> Result<crate::DecayDb> {
        crate::DecayDb::open(self.meta())
    }

    /// Open the dose rate constants database.
    pub fn dose_db(&self) -> Result<crate::DoseDb> {
        crate::DoseDb::open(self.meta())
    }

    // -- internals ----------------------------------------------------------

    /// Cache directory: `~/.nucl-parquet/v{VERSION}/`
    fn cache_dir() -> PathBuf {
        home_dir()
            .join(".nucl-parquet")
            .join(format!("v{DATA_VERSION}"))
    }

    #[cfg(feature = "fetch")]
    fn download() -> Result<()> {
        let url = format!(
            "{RELEASE_URL}/v{DATA_VERSION}/nucl-parquet-data-v{DATA_VERSION}.tar.zst"
        );
        let cache = Self::cache_dir();
        std::fs::create_dir_all(&cache)?;

        eprintln!("Downloading nucl-parquet data from {url} ...");

        let resp =
            reqwest::blocking::get(&url).map_err(|e| Error::Download(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(Error::Download(format!("HTTP {}", resp.status())));
        }

        // Stream directly into zstd decoder to avoid buffering the full
        // compressed archive in memory.
        let decoder = zstd::stream::Decoder::new(resp)
            .map_err(|e| Error::Download(format!("zstd: {e}")))?;

        let mut archive = tar::Archive::new(decoder);
        archive
            .unpack(&cache)
            .map_err(|e| Error::Download(format!("tar: {e}")))?;

        eprintln!("Data extracted to {}", cache.display());
        Ok(())
    }
}

/// Best-effort home directory lookup.
fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_contains_version() {
        let cd = DataDir::cache_dir();
        let version = env!("CARGO_PKG_VERSION");
        assert!(cd.ends_with(format!("v{version}")));
    }

    #[test]
    fn meta_and_stopping_paths() {
        let dd = DataDir {
            root: PathBuf::from("/tmp/fake"),
        };
        assert_eq!(dd.meta(), PathBuf::from("/tmp/fake/meta"));
        assert_eq!(dd.stopping(), PathBuf::from("/tmp/fake/stopping"));
    }
}
