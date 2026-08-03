# nucl-parquet

## What this project is

A **data-representation** project. The job is to carry nuclear data faithfully in
Parquet, replacing ENDF-6, EXFOR/X4 and other fixed-width or txt legacy formats.

It is **not** a transport, shielding, or activation code. Consumer use cases
(hyrr, shielding calculations, activation analysis) are *evidence about schema
adequacy* — never the design target. A use case a schema cannot serve is a symptom
of a representation defect: use it as a probe, then fix the representation.

So the question for any data-layer change is "is this the right Parquet shape for
this data?", not "does this serve my simulation?".

## Representation principles

The legacy formats encode **identity in position** (ENDF's fixed-column MF/MT
records, X4's BIB/COMMON/DATA blocks) and **metadata in prose**. Moving to a
columnar format is only half the win; the other half is making every row
self-describing.

1. **One schema per concept.** There should be exactly one spelling of
   "cross section vs energy". Adding a variant is a defect, not a feature.
2. **Keep the canonical identity.** ENDF's MT number *is* the channel identity.
   MT → residual is derivable (`scripts/fetch_endf_libs.py::mt_to_residual`);
   residual → MT is not. Never store the derivable half and drop the primitive.
3. **Nulls, not sentinels.** `residual_Z = residual_A = 0` for "this channel names
   no product" is a magic value that collides with real Z=0 and makes (n,tot),
   (n,el) and (n,f) indistinguishable. Parquet has nulls — use them, and let
   `WHERE product_Z IS NULL` be the query.
4. **Long, never wide.** A new channel must be new rows, not a new column.
   `xs_total_mb` / `xs_elastic_mb` is how a table stops being extensible.
5. **Rows are self-describing.** Library and projectile must not live only in the
   directory and filename — a glob read should never require regexing a path to
   attribute a row. Prefer Hive partitioning (`library=…/projectile=…/`), which
   Arrow, DuckDB and Polars all surface as columns for free, and keep the columns
   in the payload too.
6. **Units in column names** (`energy_MeV`, `xs_mb`). Unambiguous, and something
   ENDF never gives you.
7. **Uncertainty is first-class and nullable.** Evaluated rows having no
   `xs_err_mb` is a *data* gap, not a reason for a separate schema.
8. **Matrices get their own table.** Covariance is not a scalar column; key it by
   the same identity tuple.

## Working notes

- The devShell is load-bearing on NixOS: `nix develop -c .venv/bin/python …`.
  Bare `.venv/bin/python` fails to import numpy/h5py/duckdb (missing libstdc++).
- `data/catalog.json` is the single source of truth. Bumping `data_version` and
  merging to main auto-tags and publishes a release, so treat that edit as the
  release trigger it is.
- After changing anything under `data/`, recompute `catalog.json::data_sha256`
  (`nucl_parquet.download.compute_data_sha256`) and regenerate the README
  (`scripts/build_readme.py --write`) or the suite fails.
- Pre-existing test failures unrelated to your change: `test_fetch_strata.py`
  and the `test_g4_gammas.py` errors (missing local data files).
