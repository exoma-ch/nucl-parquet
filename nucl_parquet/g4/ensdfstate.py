"""Build ``meta/ensdf/nuclides.parquet`` and ``meta/ensdf/ground_states.parquet``
from strata's G4ENSDFSTATE3.0 mirror, joined with AME2020 (mass excess) and
IUPAC (abundances) auxiliaries.

Per ADR-0002 the v0.10.x output schema is preserved:

- ``level_keV``      <- ``excitation_kev`` (rename only)
- ``half_life_s``    <- ``convert_mean_life(mean_life_ns)`` (see below)
- ``jp``             <- ``encode_jp(spin_x2)`` (J-only string; G4ENSDFSTATE
  does not ship parity, only ``floating_level_flag``)
- ``state``          <- ``''`` for ground; ``'m'``, ``'m2'`` ... for ascending
  long-lived isomers per (Z, A) using the canonical >1 ms threshold
- Dual-carry: ``spin_x2`` (Int16) + ``floating_level_flag`` (str) shipped beside
  ``jp`` for the v1.0 cleanup path.
- Bonus: ``magnetic_moment_jt`` (Float64).

``ground_states.parquet`` is a strict ground-state filter LEFT-JOINed onto AME2020
and IUPAC compositions on (Z, A); rows with no aux match keep NULL.

Half-life conversion (binding spec from ADR-0002):

==================================  =============================
G4 ``mean_life_ns`` value           Output ``half_life_s``
==================================  =============================
``> 0``                             ``mean_life_ns * ln(2) * 1e-9``
``-1`` (G4 stable sentinel)         ``+inf`` (IEEE-754 infinity)
``0`` (prompt sentinel)             ``0.0``
NULL / missing                      NULL
==================================  =============================

Crucial: the ``-1`` sentinel MUST be intercepted before multiplication. A naive
``-1 * ln(2) * 1e-9`` yields a *negative* half-life that silently corrupts
downstream queries — the boundary is unit-tested explicitly.

Usage
-----
::

    uv run python -m nucl_parquet.g4.ensdfstate
    # or programmatically:
    from nucl_parquet.g4.ensdfstate import build
    build()
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from nucl_parquet.download import data_dir as _resolve_data_dir

LN2 = math.log(2.0)

# ENSDF convention: "long-lived" isomer threshold = T_1/2 > 1 ms. Converted to
# mean-life ns: 1 ms / ln(2) = 1.4427e-3 s = 1.4427e6 ns.
LONG_LIVED_MEAN_LIFE_NS_THRESHOLD = 1.4427e6

# G4 ENSDFSTATE sentinels (per strata's gen_ensdfstate_parquet.py).
G4_STABLE_MEAN_LIFE_NS = -1.0
G4_UNKNOWN_SPIN_X2 = -2  # parser uses -2 when token cannot be parsed.

# Default relative input/output paths under the data root.
_STRATA_ENSDFSTATE = Path("g4_raw/strata-nuclear/ensdfstate.parquet")
_AME2020 = Path("auxiliary/ame2020.parquet")
_IUPAC = Path("auxiliary/iupac_compositions.parquet")
_ELEMENTS = Path("meta/elements.parquet")
_OUT_NUCLIDES = Path("meta/ensdf/nuclides.parquet")
_OUT_GROUND_STATES = Path("meta/ensdf/ground_states.parquet")


# ---------------------------------------------------------------------------
# Pure transform primitives (unit-tested in tests/test_g4_ensdfstate.py).
# ---------------------------------------------------------------------------


def _half_life_expr(col: str = "mean_life_ns") -> pl.Expr:
    """Polars expression mirror of ``convert_mean_life`` for batch transforms.

    Single source of truth — both the Python primitive (used in unit tests)
    and the production batch path (``_build_states_frame``) consume the
    same logic; a future optimization on either branch is forced to update
    the other or break the round-trip test in ``tests/test_g4_ensdfstate``.
    """
    return (
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(pl.col(col) == G4_STABLE_MEAN_LIFE_NS)
        .then(pl.lit(math.inf))
        .when(pl.col(col) == 0.0)
        .then(pl.lit(0.0))
        .otherwise(pl.col(col) * LN2 * 1e-9)
    )


def convert_mean_life(mean_life_ns: float | None) -> float | None:
    """G4 ``mean_life_ns`` -> ``half_life_s`` per ADR-0002.

    The ``-1`` G4 stable-sentinel MUST be intercepted before multiplication;
    otherwise a naive product yields a negative half-life.
    """
    if mean_life_ns is None:
        return None
    if mean_life_ns == G4_STABLE_MEAN_LIFE_NS:
        return math.inf
    if mean_life_ns == 0.0:
        return 0.0
    return mean_life_ns * LN2 * 1e-9


def encode_jp(spin_x2: int | None, floating_level_flag: str | None = None) -> str | None:
    """Encode ``spin_x2`` to a v0.10.x-style ``jp`` string.

    G4ENSDFSTATE does not carry parity; only spin (as 2J) and a
    ``floating_level_flag`` token are available. Therefore ``jp`` is J-only.

    - ``spin_x2 = -2`` (G4 parser unknown sentinel) -> ``None``
    - even ``spin_x2`` -> integer J (e.g. 12 -> "6")
    - odd ``spin_x2``  -> half-integer J (e.g. 3 -> "3/2")
    - ``spin_x2 = 0``  -> "0"
    - any non-trivial ``floating_level_flag`` (anything other than '-') is
      preserved verbatim as a suffix in parens, e.g. spin_x2=3 + flag '+X'
      -> "3/2(+X)". Strata's '-' indicates "not floating" and is dropped.
    """
    if spin_x2 is None or spin_x2 == G4_UNKNOWN_SPIN_X2:
        return None
    if spin_x2 < 0:
        return None
    if spin_x2 % 2 == 0:
        j = str(spin_x2 // 2)
    else:
        j = f"{spin_x2}/2"
    if floating_level_flag and floating_level_flag != "-":
        return f"{j}({floating_level_flag})"
    return j


def assign_state_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Label rows in ``df`` with ``state`` ('' / 'm' / 'm2' / ...).

    ``df`` must contain ``Z``, ``A``, ``level_keV``, ``mean_life_ns``. The
    ground state (lowest ``level_keV`` per (Z,A), expected to be 0) gets ``''``;
    long-lived isomers (``mean_life_ns > 1.44e6`` ns ≈ T½ > 1 ms) above ground
    are labeled ``m``, ``m2``, ... in ascending ``level_keV`` order.

    Excited levels that do not meet the long-lived threshold are NOT included
    in the returned frame; this filter mirrors the v0.10.x ``nuclides.parquet``
    convention.
    """
    # Ground state per (Z,A): the row with the lowest level_keV. We assume
    # exactly one ground per (Z,A) — strata's input guarantees this.
    is_ground = pl.col("level_keV") == pl.col("level_keV").min().over(["Z", "A"])

    long_lived = (pl.col("mean_life_ns").is_not_null()) & (
        (pl.col("mean_life_ns") > LONG_LIVED_MEAN_LIFE_NS_THRESHOLD)
        | (pl.col("mean_life_ns") == G4_STABLE_MEAN_LIFE_NS)
    )

    keep = is_ground | long_lived
    filtered = df.filter(keep).sort(["Z", "A", "level_keV"])

    # Within each (Z,A): first row is ground (state=''), subsequent rows are
    # m, m2, m3, ... in ascending level_keV order.
    state_idx = pl.int_range(0, pl.len()).over(["Z", "A"])
    state_label = (
        pl.when(state_idx == 0)
        .then(pl.lit(""))
        .when(state_idx == 1)
        .then(pl.lit("m"))
        .otherwise(pl.lit("m") + state_idx.cast(pl.String))
    )
    return filtered.with_columns(state_label.alias("state"))


# ---------------------------------------------------------------------------
# Build entry-point.
# ---------------------------------------------------------------------------


def _load_inputs(
    data_dir: Path,
    *,
    strata_path: Path | None = None,
    ame_path: Path | None = None,
    iupac_path: Path | None = None,
    elements_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    paths = {
        "strata": strata_path or (data_dir / _STRATA_ENSDFSTATE),
        "ame": ame_path or (data_dir / _AME2020),
        "iupac": iupac_path or (data_dir / _IUPAC),
        "elements": elements_path or (data_dir / _ELEMENTS),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ensdfstate converter requires the following inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun `uv run python scripts/fetch_strata_nuclear.py` first."
        )
    return (
        pl.read_parquet(paths["strata"]),
        pl.read_parquet(paths["ame"]),
        pl.read_parquet(paths["iupac"]),
        pl.read_parquet(paths["elements"]),
    )


def _build_states_frame(strata: pl.DataFrame, elements: pl.DataFrame) -> pl.DataFrame:
    """Apply ADR-0002 transforms to the raw strata frame (without filtering)."""
    converted = strata.rename({"z": "Z", "a": "A", "excitation_kev": "level_keV"}).with_columns(
        Z=pl.col("Z").cast(pl.Int32),
        A=pl.col("A").cast(pl.Int32),
        # Half-life conversion via the shared expression — same logic the
        # convert_mean_life Python primitive (unit-tested) implements.
        half_life_s=_half_life_expr("mean_life_ns"),
        # jp encoding via map_elements — the per-row branching is awkward in
        # pure-Polars and the row count (28 k) is small enough.
        jp=pl.struct(["spin_x2", "floating_level_flag"]).map_elements(
            lambda s: encode_jp(s["spin_x2"], s["floating_level_flag"]),
            return_dtype=pl.String,
        ),
        # Dual-carry parity column per ADR-0002. G4ENSDFSTATE doesn't expose
        # parity (column 4 is floating_level_flag, not parity), so we ship
        # all-NULL here. PhotonEvaporation (#70) populates parity from jpi's
        # sign for level rows. Keeping the column ensures schema stability:
        # v1.0 cleanup is a column drop, not a column re-introduction.
        parity=pl.lit(None, dtype=pl.Int8),
    )
    return converted.join(elements.select(["Z", "symbol"]), on="Z", how="left")


def build_nuclides(strata: pl.DataFrame, elements: pl.DataFrame) -> pl.DataFrame:
    """Construct the v0.10.x-compatible ``nuclides.parquet`` frame."""
    states = _build_states_frame(strata, elements)
    labelled = assign_state_labels(states)

    # decay_1 / decay_1_pct / decay_2 / decay_2_pct are populated by the
    # downstream ``radioactive_decay`` converter (#71). At this layer we leave
    # them NULL so the schema stays identical to v0.10.x.
    return labelled.select(
        pl.col("Z").cast(pl.Int32),
        pl.col("A").cast(pl.Int32),
        pl.col("state").cast(pl.String),
        pl.col("symbol").cast(pl.String),
        pl.col("jp").cast(pl.String),
        pl.col("half_life_s").cast(pl.Float64),
        pl.col("level_keV").cast(pl.Float64),
        pl.lit(None, dtype=pl.String).alias("decay_1"),
        pl.lit(None, dtype=pl.Float64).alias("decay_1_pct"),
        pl.lit(None, dtype=pl.String).alias("decay_2"),
        pl.lit(None, dtype=pl.Float64).alias("decay_2_pct"),
        # Dual-carry per ADR-0002. parity is all-NULL on this branch
        # (G4ENSDFSTATE doesn't expose parity; PhotonEvaporation #70 will
        # populate it from jpi-sign for level rows).
        pl.col("spin_x2").cast(pl.Int16),
        pl.col("parity").cast(pl.Int8),
        pl.col("floating_level_flag").cast(pl.String),
        # Bonus.
        pl.col("magnetic_moment_jt").cast(pl.Float64),
    ).sort(["Z", "A", "level_keV"])


def build_ground_states(
    strata: pl.DataFrame,
    ame: pl.DataFrame,
    iupac: pl.DataFrame,
    elements: pl.DataFrame,
) -> pl.DataFrame:
    """Construct the v0.10.x-compatible ``ground_states.parquet`` frame."""
    states = _build_states_frame(strata, elements)
    # Ground state := lowest level_keV per (Z, A). G4ENSDFSTATE always ships
    # excitation_kev=0 for ground; we use min() to be defensive.
    ground_only = states.sort(["Z", "A", "level_keV"]).group_by(["Z", "A"], maintain_order=True).first()

    ame_proj = ame.select(
        pl.col("Z").cast(pl.Int32),
        pl.col("A").cast(pl.Int32),
        pl.col("N").cast(pl.Int64),
        "mass_excess_keV",
        "mass_excess_unc_keV",
        "binding_energy_per_A_keV",
        "binding_energy_per_A_unc_keV",
        "beta_decay_energy_keV",
        "beta_decay_energy_unc_keV",
        "atomic_mass_microu",
        "atomic_mass_unc_microu",
        pl.col("is_estimated").alias("ame_is_estimated"),
    )
    iupac_proj = iupac.select(
        pl.col("Z").cast(pl.Int32),
        pl.col("A").cast(pl.Int32),
        pl.col("isotopic_composition").alias("abundance"),
        pl.col("isotopic_composition_unc_last_digit").alias("unc_abundance"),
        pl.col("relative_atomic_mass_u").alias("atomic_mass_u"),
        pl.col("standard_atomic_weight"),
    )

    joined = ground_only.join(ame_proj, on=["Z", "A"], how="left").join(iupac_proj, on=["Z", "A"], how="left")

    return joined.select(
        pl.col("Z").cast(pl.Int32),
        # Compute N from A - Z when AME doesn't supply it.
        pl.coalesce([pl.col("N"), (pl.col("A") - pl.col("Z")).cast(pl.Int64)]).alias("N"),
        pl.col("A").cast(pl.Int32),
        pl.col("symbol").cast(pl.String),
        pl.col("jp").cast(pl.String),
        pl.col("half_life_s").cast(pl.Float64),
        pl.col("level_keV").cast(pl.Float64),
        pl.col("abundance").cast(pl.Float64),
        pl.col("unc_abundance").cast(pl.Float64),
        pl.col("standard_atomic_weight").cast(pl.Float64),
        pl.col("atomic_mass_u").cast(pl.Float64),
        pl.col("mass_excess_keV").cast(pl.Float64),
        pl.col("mass_excess_unc_keV").cast(pl.Float64),
        pl.col("binding_energy_per_A_keV").cast(pl.Float64),
        pl.col("binding_energy_per_A_unc_keV").cast(pl.Float64),
        pl.col("beta_decay_energy_keV").cast(pl.Float64),
        pl.col("beta_decay_energy_unc_keV").cast(pl.Float64),
        pl.col("atomic_mass_microu").cast(pl.Float64),
        pl.col("atomic_mass_unc_microu").cast(pl.Float64),
        pl.col("ame_is_estimated").cast(pl.Boolean),
        # Dual-carry per ADR-0002. parity is all-NULL on this branch
        # (G4ENSDFSTATE doesn't expose parity; PhotonEvaporation #70 will
        # populate it from jpi-sign for level rows).
        pl.col("spin_x2").cast(pl.Int16),
        pl.col("parity").cast(pl.Int8),
        pl.col("floating_level_flag").cast(pl.String),
        # Bonus.
        pl.col("magnetic_moment_jt").cast(pl.Float64),
    ).sort(["Z", "A"])


def build(
    data_dir: Path | None = None,
    *,
    strata_path: Path | None = None,
    ame_path: Path | None = None,
    iupac_path: Path | None = None,
    elements_path: Path | None = None,
    nuclides_out: Path | None = None,
    ground_states_out: Path | None = None,
) -> tuple[Path, Path]:
    """Build both parquet files.

    Returns the (nuclides_path, ground_states_path) actually written.
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    strata, ame, iupac, elements = _load_inputs(
        data_dir,
        strata_path=strata_path,
        ame_path=ame_path,
        iupac_path=iupac_path,
        elements_path=elements_path,
    )

    nuclides = build_nuclides(strata, elements)
    ground_states = build_ground_states(strata, ame, iupac, elements)

    nuc_out = nuclides_out or (data_dir / _OUT_NUCLIDES)
    gs_out = ground_states_out or (data_dir / _OUT_GROUND_STATES)
    nuc_out.parent.mkdir(parents=True, exist_ok=True)
    gs_out.parent.mkdir(parents=True, exist_ok=True)

    nuclides.write_parquet(nuc_out, compression="zstd")
    ground_states.write_parquet(gs_out, compression="zstd")
    return nuc_out, gs_out


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    nuc, gs = build(data_dir=args.data_dir)
    print(f"wrote {nuc}")
    print(f"wrote {gs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
