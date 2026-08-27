"""Regenerate cross-language golden fixtures from the Python loader.

The Python loader is the reference implementation per #176. Run this script
when a schema-additive change lands; commit the resulting JSON.

Usage:
    uv run python tests/golden/generate.py

Outputs `tests/golden/fixtures/*.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from normalize import normalize_coincidences, normalize_emissions, normalize_gamma_candidates

import nucl_parquet

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "golden" / "fixtures"


def _write(name: str, data: object, *, allow_empty: bool = False) -> Path:
    """Write one fixture, refusing to write an unexpectedly empty one.

    Every generator here filters on `state`, and when the 2026.8.5 release moved
    `meta/ensdf/radiation` from `''` to `'g'` two of them silently matched
    nothing. Re-running this script would then have *blessed* the emptiness:
    zero-row fixtures, golden tests green, and the cross-language check for
    `identify_gamma` and `emissions` reduced to comparing [] against []. The
    instruction in the failure message is "re-run generate.py", so that is
    exactly the hole a stale filter falls into.

    `allow_empty=True` is for the fixtures whose *point* is that they match
    nothing — `sr90_y90_negative` asserts Sr-90 has no mixed β/γ pairs. "Empty
    on purpose" and "empty by accident" have to be told apart at the point of
    writing, because afterwards they are the same two bytes on disk.
    """
    rows = len(data) if isinstance(data, list) else 1
    if rows == 0 and not allow_empty:
        raise SystemExit(
            f"refusing to write an empty fixture {name!r}. The query matched nothing — "
            "check its `state` filter against nucl_parquet/state_vocabulary.py before "
            "re-running, rather than committing a fixture that asserts nothing."
        )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)} ({rows} rows)")
    return path


def gen_co60_beta_gamma(db) -> object:
    """Canonical Co-60 → Ni-60 β⁻ ⊗ γ pairs. Acceptance: pair_intensity ≈ 0.9986
    for the 317 keV β endpoint × 1173 keV γ row.
    """
    rel = nucl_parquet.coincidences(
        db,
        z=28,
        a=60,
        parent_decay_mode="beta-",
        emission1_rad_type="beta",
        emission2_rad_type="gamma",
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_coincidences(rows)


def gen_y86_kshell_xray_gamma(db) -> object:
    """Y-86 KshellEC X-ray ⊗ γ pairs (prompt-γ PET workflow)."""
    rel = nucl_parquet.coincidences(
        db,
        z=38,
        a=86,
        parent_decay_mode="KshellEC",
        emission1_rad_type="xray",
        emission2_rad_type="gamma",
        min_intensity=1e-5,
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_coincidences(rows)


def gen_co60_gamma_gamma(db) -> object:
    """v0.11 Co-60 1173/1333 γ-γ cascade regression."""
    rel = nucl_parquet.coincidences(
        db,
        z=28,
        a=60,
        parent_decay_mode="beta-",
        emission1_rad_type="gamma",
        emission2_rad_type="gamma",
        min_intensity=0.5,
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_coincidences(rows)


def gen_sr90_y90_negative(db) -> object:
    """Sr-90 → Y-90 pure β⁻: zero mixed pairs."""
    # All rows where emission1_rad_type != 'gamma'. Should be empty.
    rel = db.sql(
        """
        SELECT Z, A, parent_state, parent_decay_mode, daughter_ex_keV,
               emission1_rad_type, emission1_energy_keV, emission1_intensity, emission1_shell,
               emission2_rad_type, emission2_energy_keV, emission2_intensity, emission2_shell,
               pair_intensity
          FROM coincidences
         WHERE Z=39 AND A=90 AND parent_decay_mode='beta-'
           AND emission1_rad_type != 'gamma'
         ORDER BY pair_intensity DESC NULLS LAST
        """
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_coincidences(rows)


def gen_identify_gamma_1173(db) -> object:
    """`identify_gamma(1173.2)` — must include Ni-60 (Co-60 daughter).

    Lean shape (matches Rust `GammaCandidate`): (Z, A, state, energy_keV,
    intensity_pct, delta_keV) — signed `delta_keV` for consistency with the
    typed client surfaces. The richer Python `IDENTIFY_GAMMA_SQL` join to
    `nuclides` (for half-life + symbol) lives in the loader for ad-hoc use
    but isn't part of the cross-language contract.
    """
    rel = db.sql(
        """
        SELECT r.Z, r.A, r.state, r.energy_keV, MAX(r.intensity_pct) AS intensity_pct,
               r.energy_keV - 1173.2 AS delta_keV
          FROM radiation r
         WHERE r.rad_type='gamma' AND r.state='g'
           AND r.energy_keV BETWEEN 1172.2 AND 1174.2
           AND r.intensity_pct > 0.1
        GROUP BY r.Z, r.A, r.state, r.energy_keV
        -- Deterministic final tiebreak so the golden fixture is stable
        -- across DuckDB sort implementations.
        ORDER BY ABS(r.energy_keV - 1173.2) ASC, intensity_pct DESC, r.Z, r.A
        LIMIT 20
        """
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_gamma_candidates(rows)


def gen_ni60_emissions(db) -> object:
    """All Ni-60 emissions (radiation rows) — for the cross-language `emissions`
    accessor parity. Includes the canonical 1173 + 1333 keV γ from Co-60 β⁻
    daughter cascade (daughter-keyed convention).
    """
    rel = db.sql(
        """
        SELECT Z, A, state, rad_type, energy_keV, intensity_pct,
               decay_mode, rad_subtype, icc_total, vacancy_shell
          FROM radiation
         WHERE Z=28 AND A=60 AND state='g'
           AND intensity_pct >= 5.0
         ORDER BY rad_type, energy_keV
        """
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_emissions(rows)


def gen_catima_stopping(db) -> object:
    """CatIMA per-isotope mass stopping power — guards the Rust `catima_dedx`
    (proj_Z, proj_A, target_Z) lookup against the isotope-merge regression (#246).

    Reads the catima master directly and log-log-interpolates with the loader's
    method. Deliberately includes isotope pairs of the same Z (He-3/He-4,
    C-12/C-13) at low energy, where reduced-mass-dependent nuclear stopping makes
    them differ by several percent — collapsing them onto (Z, target_Z) corrupts
    the result.
    """
    import numpy as np

    from nucl_parquet.loader import _interp_loglog

    triples = [
        (2, 3),
        (2, 4),  # He-3 vs He-4
        (6, 12),
        (6, 13),  # C-12 vs C-13
        (82, 208),  # heavy reference
    ]
    targets = [13, 26, 79]
    energies = [0.001, 0.005, 0.05, 0.5, 5.0, 50.0]  # MeV/u, low→high

    rows: list[dict[str, Any]] = []
    for proj_z, proj_a in triples:
        for tz in targets:
            # Read the federated per-isotope shards via the catima_stopping view
            # (post-#252; the single monolith was removed).
            rel = db.execute(
                "SELECT energy_MeV_u, dedx FROM catima_stopping "
                "WHERE proj_Z=? AND proj_A=? AND target_Z=? ORDER BY energy_MeV_u",
                [proj_z, proj_a, tz],
            ).fetchall()
            if not rel:
                print(f"  skip catima ({proj_z},{proj_a},{tz}) — not in inventory")
                continue
            e = np.array([r[0] for r in rel], dtype=float)
            s = np.array([r[1] for r in rel], dtype=float)
            log_e, log_s = np.log(e), np.log(s)
            for eu in energies:
                if eu < e[0] or eu > e[-1]:
                    continue  # avoid edge-clamp ambiguity; interior points only
                dedx = float(_interp_loglog(log_e, log_s, np.array([eu]))[0])
                rows.append(
                    {
                        "proj_Z": proj_z,
                        "proj_A": proj_a,
                        "target_Z": tz,
                        "energy_MeV_u": eu,
                        "dedx": round(dedx, 6),
                    }
                )
    return rows


def main() -> int:
    db = nucl_parquet.connect()
    _write("co60_beta_gamma", gen_co60_beta_gamma(db))
    _write("y86_kshell_xray_gamma", gen_y86_kshell_xray_gamma(db))
    _write("co60_gamma_gamma", gen_co60_gamma_gamma(db))
    # Empty by design: the assertion is that there are no mixed pairs.
    _write("sr90_y90_negative", gen_sr90_y90_negative(db), allow_empty=True)
    _write("identify_gamma_1173keV", gen_identify_gamma_1173(db))
    _write("ni60_emissions", gen_ni60_emissions(db))
    _write("catima_stopping", gen_catima_stopping(db))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
