"""Convert strata's photon_evap_levels.parquet → meta/ensdf/levels/{Symbol}.parquet.

Per epic #66 / issue #70 / ADR-0002. This is a pure schema/encoding transform on
strata's already-parsed Geant4 PhotonEvaporation6.1.2 level data; we do not
reach into G4 ASCII files ourselves.

Input schema (strata):
    z UInt8, a UInt16, level_idx UInt16, floating_flag String,
    excitation_kev Float64, half_life_s Float64, jpi Float32, n_gammas UInt16

Output schema (per element ``meta/ensdf/levels/{Symbol}.parquet``):
    Z Int32, A Int32, level_idx Int32, energy_keV Float64,
    half_life_s Float64, jp Utf8 (nullable), n_gammas Int32,
    floating_flag Utf8 (nullable), jpi Float32  (raw, dual-carry per ADR)

Transform rules (ADR-0002 §"Transform spec"):
    - excitation_kev → energy_keV (rename only).
    - half_life_s passes through, EXCEPT G4 stable sentinel -1 → +inf
      (matches ADR-0002 stable-vs-unknown distinction; PhotonEvaporation
      stores half-life directly in seconds, no ns→s conversion).
    - jpi (Float32, J × sign(parity); 99 = unknown) → jp string:
        * |J| half-integer (0.5, 1.5, ...) → "1/2±", "3/2±", ...
        * |J| integer                       → "0±", "1±", "2±", ...
        * jpi == 99                          → NULL
        * sign of jpi (incl. signed zero) gives parity.
    - jpi raw value preserved as a Float32 column for v1.0 dual-carry.
    - Partition by element via Z→symbol lookup; write one parquet per element.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Z → IUPAC symbol, 1..118. Capitalized first letter only; matches the
# convention used by data/meta/ensdf/{coincidences,levels}/*.parquet today.
Z_TO_SYMBOL: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}

# G4 sentinel values from PhotonEvaporation README-LevelGammaData.
JPI_UNKNOWN: float = 99.0
HALF_LIFE_STABLE_SENTINEL: float = -1.0


def encode_jp(jpi: float) -> str | None:
    """Encode G4 PhotonEvaporation jpi (J × sign(parity)) into ENSDF-style string.

    Examples:
        >>> encode_jp(4.5)
        '9/2+'
        >>> encode_jp(-0.5)
        '1/2-'
        >>> encode_jp(-8.0)
        '8-'
        >>> encode_jp(0.0)
        '0+'
        >>> encode_jp(99.0) is None
        True

    The sign bit of ``jpi`` (including IEEE-754 signed zero) carries parity:
    positive → ``+``, negative → ``-``. Magnitudes 0.5, 1.5, 2.5, … encode as
    fractions ``1/2``, ``3/2``, ``5/2``, …; integer magnitudes encode as bare
    integers. The sentinel ``99`` (G4 "unknown") returns ``None``.

    NaN inputs map to ``None`` defensively even though the strata input
    schema declares ``jpi`` non-nullable today.
    """
    if jpi is None or math.isnan(jpi):
        return None
    if jpi == JPI_UNKNOWN:
        return None
    # math.copysign distinguishes +0.0 from -0.0; bool(jpi >= 0) does not.
    parity = "+" if math.copysign(1.0, jpi) > 0 else "-"
    magnitude = abs(jpi)
    # Half-integer spins live at .5 offsets (within Float32 precision).
    # All G4 jpi values are exact multiples of 0.5, so the integer test is robust.
    doubled = magnitude * 2.0
    rounded = round(doubled)
    if rounded % 2 == 0:
        # Integer J.
        return f"{rounded // 2}{parity}"
    # Half-integer J.
    return f"{rounded}/2{parity}"


def convert_half_life(half_life_s: float) -> float:
    """Map G4 PhotonEvaporation half_life_s to nucl-parquet's half_life_s.

    PhotonEvaporation stores half-life directly in seconds (per its README),
    so we only translate the stable-sentinel ``-1`` to ``+inf`` per ADR-0002
    (stable-by-observation, distinct from "unknown"). Negative values other
    than the sentinel and NaN are passed through unchanged for inspection.
    """
    if half_life_s == HALF_LIFE_STABLE_SENTINEL:
        return math.inf
    return half_life_s


def transform(src: Path) -> pl.DataFrame:
    """Load + transform strata's photon_evap_levels.parquet into one DataFrame.

    The transform itself is pure: column rename + cast + sentinel rewrite +
    per-row jp encoding (Python list comp; ~190k rows runs in <1 s).
    """
    raw = pl.read_parquet(src)
    jp_encoded = [encode_jp(float(v)) for v in raw.get_column("jpi").to_list()]

    return raw.with_columns(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("level_idx").cast(pl.Int32).alias("level_idx"),
        pl.col("excitation_kev").alias("energy_keV"),
        pl.when(pl.col("half_life_s") == HALF_LIFE_STABLE_SENTINEL)
        .then(pl.lit(math.inf))
        .otherwise(pl.col("half_life_s"))
        .alias("half_life_s"),
        pl.col("n_gammas").cast(pl.Int32).alias("n_gammas"),
        pl.col("floating_flag").alias("floating_flag"),
        pl.col("jpi").cast(pl.Float32).alias("jpi"),
        pl.Series("jp", jp_encoded, dtype=pl.Utf8),
    ).select(
        "Z",
        "A",
        "level_idx",
        "energy_keV",
        "half_life_s",
        "jp",
        "n_gammas",
        "floating_flag",
        "jpi",
    )


def write_per_element(df: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Partition by Z → write meta/ensdf/levels/{Symbol}.parquet. Returns paths written.

    On case-insensitive filesystems (macOS APFS default) ``open`` of an
    existing file preserves the original on-disk casing, so a stray
    lowercase ``n.parquet`` from an older build would silently capture
    fresh writes for nitrogen. We unlink first to force the canonical
    capitalized symbol to land on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for z, group in sorted(df.group_by("Z"), key=lambda kg: kg[0][0]):
        z_int = int(z[0])
        symbol = Z_TO_SYMBOL.get(z_int)
        if symbol is None:
            logger.warning("Skipping Z=%d: no symbol mapping (extend Z_TO_SYMBOL?)", z_int)
            continue
        path = out_dir / f"{symbol}.parquet"
        path.unlink(missing_ok=True)
        # Sort within each element by (A, level_idx) for stable, browseable output.
        group.sort(["A", "level_idx"]).write_parquet(path)
        written.append(path)
    return written


def build(src: Path, out_dir: Path) -> list[Path]:
    """End-to-end: read strata input, transform, write per-element parquet files."""
    if not src.exists():
        raise FileNotFoundError(f"strata input not found at {src}; run scripts/fetch_strata_nuclear.py first")
    df = transform(src)
    paths = write_per_element(df, out_dir)
    logger.info("Wrote %d per-element level files to %s", len(paths), out_dir)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--src",
        type=Path,
        default=repo_root / "data" / "g4_raw" / "strata-nuclear" / "photon_evap_levels.parquet",
        help="Path to strata's photon_evap_levels.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "meta" / "ensdf" / "levels",
        help="Directory to write per-element parquet files into",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    build(args.src, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
