"""Fetch NIST ESTAR + PSTAR + ASTAR stopping power data and save as Parquet.

Fetches *total* mass stopping power [MeV cm²/g] for all 92 elements (Z=1-92)
from the NIST stopping-power CGIs:

- ESTAR (electrons): https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html
- PSTAR (protons):   https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- ASTAR (alphas):    https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html

All three programs serve ICRU-49 (PSTAR/ASTAR) or ICRU-37 (ESTAR) recommended
values. Output schema is identical across programs:

    source     Utf8       'ESTAR' | 'PSTAR' | 'ASTAR'
    target_Z   Int32      1..92
    energy_MeV Float64    total kinetic energy (NOT MeV/u)
    dedx       Float64    total mass stopping power [MeV cm²/g]

ESTAR uses an old urlencoded POST that returns a <pre>-formatted table.
PSTAR and ASTAR use a newer multipart/form-data POST that returns an HTML
table with <td> cells (seven cells per row). Both formats are parsed here.

Output: stopping/ESTAR.parquet, stopping/PSTAR.parquet, stopping/ASTAR.parquet
        (one file per source; idempotent — run any subset by passing programs=)

Usage:
    uv run python -m nucl_parquet.build_stopping
    uv run python -m nucl_parquet.build_stopping --programs astar
"""

from __future__ import annotations

import argparse
import re
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from .download import writable_data_dir as _resolve_data_dir

_ESTAR_CGI = "https://physics.nist.gov/cgi-bin/Star/e_table-t.pl"
# PSTAR and ASTAR share /cgi-bin/Star/ap_table.pl — distinguished by the
# `prog` form field (PSTAR vs ASTAR). NIST does not publish a separate
# /p_table.pl endpoint.
_APSTAR_CGI = "https://physics.nist.gov/cgi-bin/Star/ap_table.pl"

_USER_AGENT = "nucl-parquet/0.14 (nuclear data research)"

Program = Literal["estar", "pstar", "astar"]


# --- ESTAR fetch/parse: urlencoded POST + <pre>-block parser ----------------


def _fetch_estar(matno: int) -> str:
    data = urllib.parse.urlencode({"matno": f"{matno:03d}", "ShowDefault": "on"}).encode()
    req = Request(
        _ESTAR_CGI,
        data=data,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _parse_estar_html(html: str) -> list[tuple[float, float]]:
    """ESTAR <pre> table: KE Coll Rad Total CSDA Yield DensCorr → (KE, Total)."""
    rows: list[tuple[float, float]] = []
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


# --- PSTAR/ASTAR fetch/parse: multipart POST + <td>-cell parser -------------


def _multipart_body(fields: dict[str, str]) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data — NIST's CGI rejects urlencoded for these."""
    boundary = f"----nucl-parquet-{secrets.token_hex(8)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _fetch_apstar(cgi_url: str, prog: str, matno: int) -> str:
    fields = {
        "prog": prog,
        "matno": f"{matno:03d}",
        "GraphType": "StopPwr",
        "Graph3": "on",
        "ShowDefault": "on",
    }
    body, boundary = _multipart_body(fields)
    req = Request(
        cgi_url,
        data=body,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _parse_apstar_html(html: str) -> list[tuple[float, float]]:
    """PSTAR/ASTAR HTML table: 7 cells per row → (KE, Total).

    Columns: KE | Electronic | Nuclear | Total | CSDA | Projected | Detour.
    """
    cells = re.findall(r"<td[^>]*>\s*([-\d.E+]+)\s*</td>", html)
    rows: list[tuple[float, float]] = []
    for i in range(0, len(cells) - 6, 7):
        try:
            ke = float(cells[i])
            total = float(cells[i + 3])
        except ValueError:
            continue
        if ke > 0 and total > 0:
            rows.append((ke, total))
    return rows


# --- Driver -----------------------------------------------------------------


_PROGRAMS: dict[Program, tuple[str, str]] = {
    "estar": ("ESTAR", "electrons"),
    "pstar": ("PSTAR", "protons"),
    "astar": ("ASTAR", "α particles"),
}


def _fetch_program(prog: Program, matno: int) -> list[tuple[float, float]]:
    if prog == "estar":
        return _parse_estar_html(_fetch_estar(matno))
    if prog == "pstar":
        return _parse_apstar_html(_fetch_apstar(_APSTAR_CGI, "PSTAR", matno))
    return _parse_apstar_html(_fetch_apstar(_APSTAR_CGI, "ASTAR", matno))


def build(
    data_dir: Path | None = None,
    programs: tuple[Program, ...] = ("estar", "pstar", "astar"),
) -> None:
    """Fetch the requested NIST stopping programs and write per-source parquets.

    NIST matno matches atomic number for elemental targets (001=H, …, 092=U).
    Some Z values may have no published data — those are skipped without error.
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    stopping_dir = data_dir / "stopping"
    stopping_dir.mkdir(parents=True, exist_ok=True)

    for prog in programs:
        label, projectile_label = _PROGRAMS[prog]
        out_path = stopping_dir / f"{label}.parquet"
        print(f"Fetching NIST {label} ({projectile_label})…")

        sources: list[str] = []
        target_zs: list[int] = []
        energies: list[float] = []
        dedxs: list[float] = []

        for z in range(1, 93):
            try:
                rows = _fetch_program(prog, z)
            except Exception as exc:  # noqa: BLE001
                print(f"  Z={z:3d}: FAILED ({exc})")
                time.sleep(0.2)
                continue
            if not rows:
                print(f"  Z={z:3d}: no data parsed")
                time.sleep(0.2)
                continue
            for energy, total in rows:
                sources.append(label)
                target_zs.append(z)
                energies.append(energy)
                dedxs.append(total)
            print(f"  Z={z:3d}: {len(rows):3d} points")
            time.sleep(0.2)

        if not energies:
            print(f"  No {label} data fetched — check network/URLs.")
            continue

        df = pl.DataFrame(
            {
                "source": pl.Series(sources, dtype=pl.Utf8),
                "target_Z": pl.Series(target_zs, dtype=pl.Int32),
                "energy_MeV": pl.Series(energies, dtype=pl.Float64),
                "dedx": pl.Series(dedxs, dtype=pl.Float64),
            }
        ).sort("target_Z", "energy_MeV")
        df.write_parquet(out_path, compression="zstd")
        print(f"  Wrote {len(df):,} rows → {out_path}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--programs",
        nargs="+",
        choices=sorted(_PROGRAMS),
        default=sorted(_PROGRAMS),
        help="Subset of programs to fetch (default: all three).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(programs=tuple(args.programs))
