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

from normalize import normalize_coincidences, normalize_emissions, normalize_gamma_candidates

import nucl_parquet

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "golden" / "fixtures"


def _write(name: str, data: object) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(data) if isinstance(data, list) else 1} rows)")
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
         WHERE r.rad_type='gamma' AND r.state=''
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
         WHERE Z=28 AND A=60 AND state=''
           AND intensity_pct >= 5.0
         ORDER BY rad_type, energy_keV
        """
    )
    rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
    return normalize_emissions(rows)


def main() -> int:
    db = nucl_parquet.connect()
    _write("co60_beta_gamma", gen_co60_beta_gamma(db))
    _write("y86_kshell_xray_gamma", gen_y86_kshell_xray_gamma(db))
    _write("co60_gamma_gamma", gen_co60_gamma_gamma(db))
    _write("sr90_y90_negative", gen_sr90_y90_negative(db))
    _write("identify_gamma_1173keV", gen_identify_gamma_1173(db))
    _write("ni60_emissions", gen_ni60_emissions(db))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
