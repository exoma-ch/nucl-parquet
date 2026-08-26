"""The one place that knows how to spell a canonical cross-section row (#359).

`CANONICAL_XS_SCHEMA` lives in `nucl_parquet/_schemas.py`, but *producing* a
frame in that shape needs more than the column list: the projectile code has to
resolve to (Z, A), the file stem has to resolve to a target Z, legacy column
names have to be renamed rather than silently dropped, and 0/0 residual
sentinels have to become nulls.

That logic existed once, inside `migrate_xs_schema.py::migrate_file`, reachable
only by rewriting a parquet already on disk. So the builders had two options:
write the legacy 6-column form and hope someone remembers to run the migration
afterwards, or reimplement the transform. `fetch_endf_libs.py` did the first —
which meant a plain re-ingest silently reverted a library to the legacy shape and
dropped twelve of eighteen columns, with the run exiting 0 (#359).

Same reasoning as `_paths.py` in #341: one place to be right, imported rather
than re-derived.

    sys.path.insert(0, str(Path(__file__).parent))
    from _canonical import LIGHT_ION, canonical_frame, parse_stem
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nucl_parquet._schemas import CANONICAL_XS_SCHEMA  # noqa: E402

#: Light-ion projectile code -> (Z, A). Photons carry ZA = 0, per ENDF.
LIGHT_ION: dict[str, tuple[int, int]] = {
    "n": (0, 1),
    "p": (1, 1),
    "d": (1, 2),
    "t": (1, 3),
    "h": (2, 3),
    "a": (2, 4),
    "g": (0, 0),
}

ELEMENTS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()
SYMBOL_TO_Z: dict[str, int] = {s.lower(): i for i, s in enumerate(ELEMENTS)}

#: Z -> element symbol. `Z<number>` is the fallback the file stems use for
#: elements with no symbol here, so the two directions stay consistent.
Z_TO_SYMBOL: dict[int, str] = {i: s for i, s in enumerate(ELEMENTS) if i}

#: Legacy -> canonical column names. `exfor_entry` is source-specific naming for
#: what is really "which record did this datum come from"; the canonical schema
#: generalises it so measurements from any source share one provenance column.
RENAMES: dict[str, str] = {"exfor_entry": "source_entry"}

# Heavy-ion projectile stem, e.g. 'ar40' -> ('Ar', 40).
_HEAVY_ION = re.compile(r"^([a-z]{1,2})(\d{1,3})$")
# File stem: <projectile>_<Element>. The target is either an element symbol or,
# for elements the builders have no symbol for (transuranics, Tc, Pm), the
# explicit 'Z<number>' form — e.g. 'p_Z61', 'd_Z105'.
_STEM = re.compile(r"^([a-z]{1,2}\d{0,3})_(Z\d{1,3}|[A-Za-z]{1,2})$")


def element_stem(target_z: int) -> str:
    """Element token used in a file stem: 'Fe' where we know the symbol, else 'Z61'."""
    return Z_TO_SYMBOL.get(target_z, f"Z{target_z}")


def parse_stem(stem: str) -> tuple[str, int, int, int] | None:
    """'p_Cu' -> ('p', 1, 1, 29);  'ar40_Ac' -> ('ar40', 18, 40, 89);
    'p_Z61' -> ('p', 1, 1, 61)."""
    m = _STEM.match(stem)
    if not m:
        return None
    proj, elem = m.group(1), m.group(2)
    if elem.startswith("Z") and elem[1:].isdigit():
        target_z = int(elem[1:])
    else:
        target_z = SYMBOL_TO_Z.get(elem.lower())
    if target_z is None:
        return None
    if proj in LIGHT_ION:
        pz, pa = LIGHT_ION[proj]
    else:
        hm = _HEAVY_ION.match(proj)
        if hm is None:
            return None
        pz = SYMBOL_TO_Z.get(hm.group(1).lower())
        if pz is None:
            return None
        pa = int(hm.group(2))
    return proj, pz, pa, target_z


def canonical_frame(
    df,  # noqa: ANN001 — polars.DataFrame, imported lazily by callers
    *,
    library: str,
    kind: str,
    projectile: str,
    proj_z: int,
    proj_a: int,
    target_z: int,
):
    """Return `df` in exactly `CANONICAL_XS_SCHEMA` — columns, order and dtypes.

    Fills identity columns the caller supplies, renames legacy spellings, adds
    typed nulls for anything absent, and converts the legacy 0/0 residual
    sentinel to nulls. Columns already present in `df` win over the arguments,
    so a builder that knows its own per-row `target_Z` (heavy ions, natural
    targets) is not overwritten by a stem-derived guess.
    """
    import polars as pl

    renames = {old: new for old, new in RENAMES.items() if old in df.columns and new not in df.columns}
    if renames:
        df = df.rename(renames)

    have = set(df.columns)
    literals = {
        "library": pl.lit(library, dtype=pl.Utf8),
        "kind": pl.lit(kind, dtype=pl.Utf8),
        "projectile": pl.lit(projectile, dtype=pl.Utf8),
        "proj_Z": pl.lit(proj_z, dtype=pl.Int32),
        "proj_A": pl.lit(proj_a, dtype=pl.Int32),
        "target_Z": pl.lit(target_z, dtype=pl.Int32),
    }
    df = df.with_columns(
        *[
            pl.col(col).cast(getattr(pl, CANONICAL_XS_SCHEMA[col])) if col in have else expr.alias(col)
            for col, expr in literals.items()
        ]
    )

    for col, dtype in CANONICAL_XS_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=getattr(pl, dtype)).alias(col))

    # A 0/0 residual is the legacy sentinel for "this channel names none".
    # Nulls say that truthfully and do not collide with a real Z=0 product.
    if {"residual_Z", "residual_A"} <= set(df.columns):
        no_residual = (pl.col("residual_Z") == 0) & (pl.col("residual_A") == 0)
        df = df.with_columns(
            pl.when(no_residual).then(None).otherwise(pl.col("residual_Z")).cast(pl.Int32).alias("residual_Z"),
            pl.when(no_residual).then(None).otherwise(pl.col("residual_A")).cast(pl.Int32).alias("residual_A"),
        )

    return df.select([pl.col(c).cast(getattr(pl, t)) for c, t in CANONICAL_XS_SCHEMA.items()])
