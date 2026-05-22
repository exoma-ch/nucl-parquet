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

    /// Ensure data is available, downloading the full tarball if needed.
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

    /// Ensure data is available in lazy mode — fetch only `catalog.json`.
    ///
    /// Individual Parquet files are downloaded on first access via
    /// [`fetch_file()`](Self::fetch_file). This is ideal for consumers that
    /// need <10% of the full dataset per session.
    #[cfg(feature = "fetch")]
    pub fn ensure_lazy() -> Result<Self> {
        if let Ok(d) = Self::resolve() {
            return Ok(d);
        }
        let cache = Self::cache_dir();
        std::fs::create_dir_all(&cache)?;

        // Fetch catalog.json from main to discover data_version + base_url.
        // The catalog's base_url template resolves to a versioned tag, so
        // actual data files are fetched from the pinned data release, not main.
        let catalog_url = format!(
            "https://raw.githubusercontent.com/exoma-ch/nucl-parquet/main/data/catalog.json"
        );
        let catalog_path = cache.join("catalog.json");
        if !catalog_path.exists() {
            eprintln!("Fetching catalog from {catalog_url} ...");
            Self::fetch_url(&catalog_url, &catalog_path)?;
        }

        // Parse catalog to build versioned base_url
        let catalog_text =
            std::fs::read_to_string(&catalog_path).map_err(|e| Error::Download(e.to_string()))?;
        let catalog: serde_json::Value =
            serde_json::from_str(&catalog_text).map_err(|e| Error::Download(e.to_string()))?;

        let data_version = catalog["data_version"].as_str().unwrap_or("latest");
        let base_template = catalog["base_url"].as_str().unwrap_or(
            "https://raw.githubusercontent.com/exoma-ch/nucl-parquet/data-{version}/data",
        );
        let base_url = base_template.replace("{version}", data_version);

        // Write lazy marker for fetch_file()
        let marker = cache.join(".lazy_base_url");
        std::fs::write(&marker, &base_url)?;

        // Create meta/ directory so resolve() finds it
        std::fs::create_dir_all(cache.join("meta"))?;

        eprintln!(
            "Lazy mode: catalog at {}, files on demand from {base_url}",
            cache.display()
        );
        Ok(Self { root: cache })
    }

    /// Fetch a single file from the lazy HTTP base if not already cached.
    ///
    /// Returns the local path to the file. No-op if the file exists on disk.
    #[cfg(feature = "fetch")]
    pub fn fetch_file(&self, rel_path: &str) -> Result<std::path::PathBuf> {
        let dest = self.root.join(rel_path);
        if dest.exists() {
            return Ok(dest);
        }

        let marker = self.root.join(".lazy_base_url");
        if !marker.exists() {
            return Err(Error::DataDirNotFound(dest));
        }

        let base_url =
            std::fs::read_to_string(&marker).map_err(|e| Error::Download(e.to_string()))?;
        let url = format!("{}/{}", base_url.trim(), rel_path);
        eprintln!("  Fetching {rel_path} ...");
        Self::fetch_url(&url, &dest)?;
        Ok(dest)
    }

    #[cfg(feature = "fetch")]
    fn fetch_url(url: &str, dest: &Path) -> Result<()> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let resp = reqwest::blocking::get(url).map_err(|e| Error::Download(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Download(format!("HTTP {} for {url}", resp.status())));
        }
        let bytes = resp.bytes().map_err(|e| Error::Download(e.to_string()))?;
        // Atomic write: tmp file + rename to avoid partial files on failure
        let tmp = dest.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, dest)?;
        Ok(())
    }

    /// Create a DataDir from an existing root path (no resolution or download).
    pub fn from_root(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
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

    /// Open the stopping power database (NIST PSTAR/ASTAR/ESTAR + dSTAR/tSTAR + CatIMA).
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
        let url =
            format!("{RELEASE_URL}/v{DATA_VERSION}/nucl-parquet-data-v{DATA_VERSION}.tar.zst");
        let cache = Self::cache_dir();
        std::fs::create_dir_all(&cache)?;

        eprintln!("Downloading nucl-parquet data from {url} ...");

        let resp = reqwest::blocking::get(&url).map_err(|e| Error::Download(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(Error::Download(format!("HTTP {}", resp.status())));
        }

        // Stream directly into zstd decoder to avoid buffering the full
        // compressed archive in memory.
        let decoder =
            zstd::stream::Decoder::new(resp).map_err(|e| Error::Download(format!("zstd: {e}")))?;

        let mut archive = tar::Archive::new(decoder);
        // Filter out macOS resource fork files (._*) that may be in the archive
        for entry in archive
            .entries()
            .map_err(|e| Error::Download(format!("tar: {e}")))?
        {
            let mut entry = entry.map_err(|e| Error::Download(format!("tar: {e}")))?;
            let path = entry
                .path()
                .map_err(|e| Error::Download(format!("tar: {e}")))?;
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("._") {
                continue;
            }
            entry
                .unpack_in(&cache)
                .map_err(|e| Error::Download(format!("tar: {e}")))?;
        }

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
