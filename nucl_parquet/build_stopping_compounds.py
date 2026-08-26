"""Fetch NIST PSTAR/ASTAR *compound* stopping-power tables.

NIST publishes proton and α-particle stopping-power tables for ~50 standard
ICRU-37 compounds (water, A-150 plastic, polyethylene, tissue substitutes,
air, …) via the same `ap_table.pl` CGI as the elemental tables, but with
matno values in the 99–276 range instead of 1–98.

This script probes every matno in that range. Compound matnos are sparse —
only ~50 of the 180 candidate matnos return real data; the rest return
a short "no data" page that we skip. The full mapping isn't published in a
machine-readable form so we discover it by trial.

Output (per program):
- `data/stopping/compounds/PSTAR_compounds.parquet`
- `data/stopping/compounds/ASTAR_compounds.parquet`

These live in a `compounds/` subdir so they don't collide with the elemental
`stopping` view glob (`_register_glob` is non-recursive).

Schema: `source | matno | compound | energy_MeV | dedx`
  - source: "PSTAR_compound" / "ASTAR_compound"
  - matno: NIST's compound identifier (kept as-is for audit + future re-fetch)
  - compound: NIST's verbatim material name (uppercase, e.g. "WATER_LIQUID")
  - energy_MeV / dedx: as in the elemental tables

Reference densities are listed by NIST but not parsed here; consumers needing
density should look up by matno in the NIST PML methods page.

Issue #160; design discussed in https://github.com/exoma-ch/nucl-parquet/issues/160

Usage:
    uv run python -m nucl_parquet.build_stopping_compounds
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .build_stopping import _APSTAR_CGI, _fetch_apstar, _parse_apstar_html
from .download import writable_data_dir as _resolve_data_dir

# Probe range. NIST's compound matnos start at 99 and most known entries fall
# in 99-276; we go to 280 to be safe. ~180 probes per program at 0.3s = ~55s.
_MATNO_MIN = 99
_MATNO_MAX = 280
_THROTTLE_S = 0.3
# Retry transient HTTP failures so a single 5xx doesn't silently drop a matno.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 1.0


def _extract_name(html: str) -> str | None:
    """Return the compound name from a NIST CGI response, or None if matno is empty.

    NIST puts the material name in the second <h*> tag of a successful response;
    failed lookups return a short "no data" page with only the program-title <h*>.
    The `len < 5000` check is a fast-path; the `len(hs) < 2` guard below is the
    load-bearing one and would catch a future NIST layout change correctly.
    """
    if len(html) < 5000:
        return None
    hs = re.findall(r"<h[1-3][^>]*>([^<]+)</h[1-3]>", html, re.I)
    if len(hs) < 2:
        return None
    return hs[1].strip()


def _fetch_with_retry(prog: str, matno: int) -> str:
    """Fetch one matno, retrying transient HTTP failures with backoff.

    Real NIST CGI flakes occasionally (5xx mid-run, connection reset). Retrying
    eliminates the silent-drop-on-flake annoyance without making the build
    significantly slower in the common case.
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return _fetch_apstar(_APSTAR_CGI, prog, matno)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _normalize_compound_name(name: str) -> str:
    """NIST uses both 'WATER_LIQUID' and 'A-150 TISSUE-EQUIVALENT PLASTIC' style.

    Normalize to SHOUTY_UNDERSCORE_CASE: uppercase + collapse runs of
    non-alnum to single underscore + strip leading/trailing underscores.
    This matches the convention used by strata em/density_effect.parquet
    (G4_<UPPERCASE_NAME>) modulo the G4_ prefix.
    """
    s = name.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return s.strip("_")


def build(data_dir: Path | None = None) -> None:
    """Fetch NIST compound tables for PSTAR and ASTAR; write per-program parquets."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    stopping_dir = data_dir / "stopping" / "compounds"
    stopping_dir.mkdir(parents=True, exist_ok=True)

    for prog, label in (("PSTAR", "PSTAR_compound"), ("ASTAR", "ASTAR_compound")):
        print(f"Fetching NIST {prog} compound tables (matno {_MATNO_MIN}-{_MATNO_MAX})…")

        sources: list[str] = []
        matnos: list[int] = []
        compounds: list[str] = []
        energies: list[float] = []
        dedxs: list[float] = []
        compound_count = 0

        for matno in range(_MATNO_MIN, _MATNO_MAX + 1):
            try:
                html = _fetch_with_retry(prog, matno)
            except Exception as exc:  # noqa: BLE001
                print(f"  matno={matno:3d}: FAILED after retries ({exc})")
                time.sleep(_THROTTLE_S)
                continue

            name = _extract_name(html)
            if name is None:
                time.sleep(_THROTTLE_S)
                continue

            rows = _parse_apstar_html(html)
            if not rows:
                print(f"  matno={matno:3d}: {name!r} — no data rows parsed")
                time.sleep(_THROTTLE_S)
                continue

            canonical = _normalize_compound_name(name)
            for energy, total in rows:
                sources.append(label)
                matnos.append(matno)
                compounds.append(canonical)
                energies.append(energy)
                dedxs.append(total)
            compound_count += 1
            print(f"  matno={matno:3d}: {canonical} ({len(rows)} points)")
            time.sleep(_THROTTLE_S)

        if not energies:
            print(f"  No {prog} compound data fetched.")
            continue

        df = pl.DataFrame(
            {
                "source": pl.Series(sources, dtype=pl.Utf8),
                "matno": pl.Series(matnos, dtype=pl.Int32),
                "compound": pl.Series(compounds, dtype=pl.Utf8),
                "energy_MeV": pl.Series(energies, dtype=pl.Float64),
                "dedx": pl.Series(dedxs, dtype=pl.Float64),
            }
        ).sort("matno", "energy_MeV")

        out_path = stopping_dir / f"{prog}_compounds.parquet"
        df.write_parquet(out_path, compression="zstd")
        print(f"  Wrote {len(df):,} rows ({compound_count} compounds) → {out_path}\n")


if __name__ == "__main__":
    build()
