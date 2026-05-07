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
                                O1-8, P1-6, Q1-3, R1-2 (35 per-shell labels),
                                plus ``TOT`` for upstream-pre-summed totals
- ``binding_energy_eV`` Float64 shell binding energy (eV; 0 for TOT rows)
- ``gamma_energy_keV``  Float64 gamma energy on the lookup grid (keV)
- ``multipolarity``     String  E1, E2, E3, E4, E5, M1, M2, M3, M4, M5
- ``alpha``             Float64 BrIcc factor (interpolation input — see notes)

**Note on ``alpha`` units**: G4NUDEXLIB ships these as the BrIcc ``ICC
factors`` lookup, *not* directly-usable α values. They are inputs to BrIcc's
log-log interpolation routine. For a directly-usable per-gamma ICC, use
``radiation.icc_total`` (single value baked in upstream) or the per-transition
α from ``nudex_level_gammas`` (#122).

**TOT rows**: strata distinguishes ``shell='TOT'`` (per-isotope sum across
shells, ~3,181 rows) from per-shell rows. Filter ``WHERE shell != 'TOT'``
for per-shell queries; ``WHERE shell = 'TOT'`` for total-XS queries.

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


def build(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Read NUDEX wide-format ICC factors and write long-format parquet."""
    raw = pl.read_parquet(strata_path)

    # Sanity: strata#610 was fixed — (z, shell, e_gamma_mev) must be unique
    # in the upstream now (TOT rows have shell='TOT', distinct from per-shell).
    n_dup = raw.group_by(["z", "shell", "e_gamma_mev"]).len().filter(pl.col("len") > 1).height
    if n_dup:
        raise ValueError(f"upstream contains {n_dup} duplicate (Z, shell, E_γ) tuples — strata#610 regression?")

    long = raw.unpivot(
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
    # Per-shell rows must have positive binding energy; TOT rows carry 0.
    per_shell = df.filter(pl.col("shell") != "TOT")
    if per_shell.height and (per_shell["binding_energy_eV"] <= 0).any():
        raise ValueError("non-positive shell binding energy in per-shell rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "wrote %d ICC factor rows (%d Z × %d shell labels × %d multipolarities) to %s",
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
