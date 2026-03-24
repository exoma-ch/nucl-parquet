use std::path::PathBuf;

/// Errors from loading or querying nuclear data.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("data directory not found: {0}")]
    DataDirNotFound(PathBuf),

    #[error("no data files found in {0}")]
    NoDataFiles(PathBuf),

    #[error("parquet read error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("missing column '{column}' in {file}")]
    MissingColumn { file: PathBuf, column: String },

    #[error("element Z={0} not loaded")]
    ElementNotLoaded(u8),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
