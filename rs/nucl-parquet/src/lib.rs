//! # nucl-parquet
//!
//! Fast, thread-safe access to nuclear interaction data stored as Parquet files.
//!
//! Designed as the physics data backbone for Monte Carlo particle transport codes.
//! All data structures are `Send + Sync` — load once, share across threads via `Arc`.
//!
//! ## Data sources
//!
//! - **EPDL97**: Photon cross-sections (photoelectric, Compton, Rayleigh, pair production)
//! - **EADL**: Atomic relaxation (fluorescence X-ray and Auger transition probabilities)
//! - **EEDL**: Electron cross-sections (elastic, bremsstrahlung, ionization)
//! - **XCOM**: Total mass attenuation coefficients (µ/ρ, µ_en/ρ)
//!
//! ## Usage
//!
//! ```no_run
//! use nucl_parquet::PhotonDb;
//! use std::sync::Arc;
//!
//! # fn main() -> nucl_parquet::Result<()> {
//! let db = Arc::new(PhotonDb::open("path/to/nucl-parquet/meta")?);
//!
//! // Thread-safe lookups (no locks, data is immutable)
//! let xs = db.cross_section(29, 511.0, nucl_parquet::Process::Photoelectric);
//! let ff = db.form_factor(29, 0.5);  // Rayleigh form factor at q=0.5
//! # Ok(())
//! # }
//! ```

mod error;
mod interp;
mod photon;
mod relaxation;

pub use error::Error;
pub use photon::{PhotonDb, Process};
pub use relaxation::{RelaxationDb, Transition, TransitionType};

/// Result type for nucl-parquet operations.
pub type Result<T> = std::result::Result<T, Error>;
