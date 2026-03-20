"""Fetch NIST ESTAR electron stopping power data and save as Parquet.

Fetches total mass stopping power [MeV cm2/g] for all 92 elements (Z=1-92)
from the NIST ESTAR database (https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html).

Output: stopping/stopping.parquet  (appends ESTAR rows; preserves PSTAR/ASTAR)

Usage:
    uv run python -m nucl_parquet.build_stopping
"""

from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

from .download import data_dir as _resolve_data_dir

_ESTAR_CGI = "https://physics.nist.gov/cgi-bin/Star/e_table-t.pl"


def _fetch_estar(matno: int) -> str:
    """POST to NIST ESTAR CGI and return the HTML response."""
    data = urllib.parse.urlencode({"matno": f"{matno:03d}", "ShowDefault": "on"}).encode()
    req = Request(
        _ESTAR_CGI,
        data=data,
        headers={
            "User-Agent": "nucl-parquet/0.3 (nuclear data research)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _parse_estar_html(html: str) -> list[tuple[float, float]]:
    """Extract (energy_MeV, total_dedx) from NIST ESTAR HTML response.

    The table has 7 columns (inside a <pre> block):
        KE(MeV)  Collision  Radiative  Total  CSDA_Range  Rad_Yield  Density_Effect
    We want column 1 (energy) and column 4 (total stopping power [MeV cm2/g]).
    """
    rows: list[tuple[float, float]] = []
    # NIST serves the <pre> block with <br> instead of newlines — normalize first
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    sci = r"\d+\.\d+E[+-]\d+"
    pattern = re.compile(
        rf"^\s+({sci})\s+({sci})\s+({sci})\s+({sci})",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(text):
        energy = float(m.group(1))
        total = float(m.group(4))
        rows.append((energy, total))
    return rows


def build(data_dir: Path | None = None) -> None:
    """Fetch ESTAR data and append to stopping/stopping.parquet."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    out_path = data_dir / "stopping" / "stopping.parquet"

    # Load existing stopping data to preserve PSTAR/ASTAR rows
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        # Drop any stale ESTAR rows so a re-run is idempotent
        existing = existing.filter(pl.col("source") != "ESTAR")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pl.DataFrame(
            {"source": [], "target_Z": [], "energy_MeV": [], "dedx": []},
            schema={
                "source": pl.Utf8,
                "target_Z": pl.Int32,
                "energy_MeV": pl.Float64,
                "dedx": pl.Float64,
            },
        )

    # --- Fetch ESTAR for Z=1..92 ---
    # NIST matno matches atomic number directly (001=H, 002=He, ..., 092=U)
    print("Fetching NIST ESTAR electron stopping power data...")
    sources, target_zs, energies, dedxs = [], [], [], []

    for z in range(1, 93):
        try:
            html = _fetch_estar(z)
            rows = _parse_estar_html(html)
            if not rows:
                print(f"  Z={z:3d}: no data parsed")
                continue
            for energy, total in rows:
                sources.append("ESTAR")
                target_zs.append(z)
                energies.append(energy)
                dedxs.append(total)
            print(f"  Z={z:3d}: {len(rows)} points")
        except Exception as exc:
            print(f"  Z={z:3d}: FAILED ({exc})")
        time.sleep(0.2)

    if not energies:
        print("No ESTAR data fetched — check network/URLs.")
        return

    estar_df = pl.DataFrame(
        {
            "source": pl.Series(sources, dtype=pl.Utf8),
            "target_Z": pl.Series(target_zs, dtype=pl.Int32),
            "energy_MeV": pl.Series(energies, dtype=pl.Float64),
            "dedx": pl.Series(dedxs, dtype=pl.Float64),
        }
    )

    combined = pl.concat([existing, estar_df]).sort("source", "target_Z", "energy_MeV")
    combined.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {len(combined)} rows ({len(estar_df)} ESTAR) to {out_path}")


if __name__ == "__main__":
    build()
