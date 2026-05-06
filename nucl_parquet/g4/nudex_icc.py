"""NUDEX per-shell internal-conversion factors (issue #123).

Imports the G4NUDEXLIB1.0 ``ICC_factors.dat`` table — the per-shell,
per-multipolarity internal-conversion lookup that BrIcc uses to compute
``α(Z, E_γ, shell, multipolarity)``. Where v0.11's ``radiation`` view
ships a single per-gamma ``icc_total`` (sum over shells, single-multipolarity
choice baked in upstream), this view exposes the underlying tables for
consumers that need partial ICCs or want to override the multipolarity
assignment.

Source: ``nuclear/nudex_icc_factors.parquet`` (strata, gate-class 1 raw-file
copy of G4NUDEXLIB1.0/ICC_factors.dat).

Schema (long format, one row per (Z, shell, E_γ, multipolarity)):

- ``Z``                 Int32   atomic number (ICC factors are an *atomic*
                                property, common across isotopes; the
                                upstream-listed ``A`` is a representative
                                isotope label and is preserved as ``A_ref``)
- ``A_ref``             Int32   representative mass number from upstream header
- ``symbol``            String  element symbol (Ne, Pb, ...)
- ``shell``             String  shell label — one of K, L1-3, M1-5, N1-7,
                                O1-8, P1-6, Q1-3, R1-2 (35 distinct labels)
- ``binding_energy_eV`` Float64 shell binding energy (eV)
- ``gamma_energy_keV``  Float64 gamma energy on the lookup grid (keV)
- ``multipolarity``     String  E1, E2, E3, E4, E5, M1, M2, M3, M4, M5
- ``alpha``             Float64 BrIcc factor (interpolation input — see notes)

**Note on ``alpha`` units**: G4NUDEXLIB ships these as the BrIcc ``ICC
factors`` lookup, *not* directly-usable α values. They are inputs to BrIcc's
log-log interpolation routine. For a directly-usable per-gamma ICC, use
``radiation.icc_total`` (single value baked in upstream) or wait for the
NUDEX known-levels-gammas import (#122) which surfaces the per-transition
α from the upstream evaluation. Cross-validation of the α reconstruction
against ``radiation.icc_total`` is left as a follow-up once #122 lands.

**Upstream contamination filter**: strata's parser of the raw ``ICC_factors.dat``
mis-tags the per-isotope ``Total`` block as belonging to the last-seen
``<shell>`` (the raw file uses ``96 TOT`` in the last column for totals
rows, but strata's column-tagging falls back to the previous shell-block
header). This produces 6404 duplicate ``(Z, shell, E_γ)`` rows in the
upstream parquet — same energy, same shell label, but values from the
totals block instead of the per-shell block. We deduplicate by keeping
the first occurrence of each ``(Z, shell, E_γ)`` (raw-file order: actual
per-shell block comes before the totals block). Filed upstream as a
strata bug (strata#610); once fixed and the catalog SHA bumps, the dedup
becomes a no-op and can be removed (tracked as nucl-parquet follow-up).

Use cases:
- Statistical-model decay calculations needing per-shell branching.
- Sensitivity studies overriding the upstream multipolarity assignment.
- Reproducing BrIcc lookups in user-side code (combine with shell-specific
  binding energies, which are surfaced as ``binding_energy_eV``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# The 10 multipolarity columns in the upstream wide table.
_MULTIPOLARITIES = ("E1", "E2", "E3", "E4", "E5", "M1", "M2", "M3", "M4", "M5")

# Upstream-contamination invariant — see module docstring. If strata fixes
# the totals-vs-shell tagging in a future release, this count will drop and
# the assertion in :func:`build` will surface that, prompting removal of
# the workaround.
_EXPECTED_CONTAMINATION_TUPLES = 3202


def build(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Read NUDEX wide-format ICC factors and write deduped long-format parquet."""
    raw = pl.read_parquet(strata_path)

    # Lock down the upstream contamination count — surfaces both regressions
    # (strata fixed it but we still filter) and worse regressions (more
    # contamination than expected).
    n_dup_tuples = raw.group_by(["z", "shell", "e_gamma_mev"]).len().filter(pl.col("len") > 1).height
    if n_dup_tuples != _EXPECTED_CONTAMINATION_TUPLES:
        logger.warning(
            "upstream contamination count drift: expected %d duplicate (Z, shell, E_γ) "
            "tuples, got %d — see module docstring; check whether strata fixed the "
            "totals-vs-shell tagging or introduced new contamination.",
            _EXPECTED_CONTAMINATION_TUPLES,
            n_dup_tuples,
        )

    # Keep the first occurrence per (Z, shell, e_gamma_mev) — raw file order
    # places the legitimate per-shell row before the contaminated totals row.
    # Sort by row index first to guarantee read order is preserved across
    # any future Polars internal reordering.
    deduped = (
        raw.with_row_index("_row_idx")
        .sort("_row_idx")
        .unique(subset=["z", "shell", "e_gamma_mev"], keep="first")
        .drop("_row_idx")
    )

    long = deduped.unpivot(
        index=["z", "a", "symbol", "shell", "binding_energy_ev", "e_gamma_mev"],
        on=list(_MULTIPOLARITIES),
        variable_name="multipolarity",
        value_name="alpha",
    )

    df = long.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A_ref"),
        pl.col("symbol"),
        pl.col("shell"),
        pl.col("binding_energy_ev").alias("binding_energy_eV"),
        (pl.col("e_gamma_mev") * 1000.0).alias("gamma_energy_keV"),
        pl.col("multipolarity"),
        pl.col("alpha"),
    ).sort("Z", "shell", "gamma_energy_keV", "multipolarity")

    if not df["alpha"].is_finite().all():
        raise ValueError("non-finite (NaN/Inf) ICC factor in NUDEX upstream data")
    if (df["alpha"] < 0).any():
        raise ValueError("negative ICC factor in NUDEX upstream data")
    if not df["binding_energy_eV"].is_finite().all() or (df["binding_energy_eV"] <= 0).any():
        raise ValueError("non-positive or non-finite shell binding energy in NUDEX upstream data")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "wrote %d ICC factor rows (%d Z × %d shells × %d multipolarities) to %s",
        df.height,
        df["Z"].n_unique(),
        df["shell"].n_unique(),
        df["multipolarity"].n_unique(),
        out_path,
    )
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-nuclear-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/meta"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(
        args.strata_nuclear_dir / "nudex_icc_factors.parquet",
        args.out_dir / "icc_factors.parquet",
    )
