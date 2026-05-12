"""Fetch AME2020 atomic mass evaluation and convert to Parquet.

Source: AMDC (Atomic Mass Data Center, hosted at IAEA-NDS), the canonical
2020 atomic-mass evaluation by Huang/Wang/Kondev/Audi/Naimi (Chinese Physics
C45, 030002, March 2021). Plain-text fixed-width Fortran-formatted file,
~3500 nuclides with mass excess, binding energy, beta-decay energy, atomic
mass — and uncertainties on each.

Plugs the gap that G4 data files leave: G4ENSDFSTATE has level energies
and lifetimes but not atomic masses or Q-values relative to the nucleon
ground state. AME2020 fills both columns of `ground_states.parquet`.

Output: data/auxiliary/ame2020.parquet (gitignored — regenerate via this script).

Usage:
    python scripts/fetch_ame2020.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SRC_URL = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
RAW_PATH = ROOT / "data" / "g4_raw" / "ame2020" / "mass_1.mas20.txt"
OUT_PATH = ROOT / "data" / "auxiliary" / "ame2020.parquet"


def _fetch() -> Path:
    """Download mass_1.mas20.txt if not cached. Returns path to cached file."""
    if RAW_PATH.exists():
        logger.info("AME2020 cached at %s (%d bytes)", RAW_PATH, RAW_PATH.stat().st_size)
        return RAW_PATH
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", SRC_URL)
    resp = requests.get(SRC_URL, timeout=30)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)
    logger.info("Wrote %d bytes to %s", len(resp.content), RAW_PATH)
    return RAW_PATH


def _parse_value(s: str) -> tuple[float | None, bool]:
    """Parse an AME numeric field. Returns (value, is_estimated).

    `#` after digits marks an estimated (non-experimental) value.
    `*` means not calculable — returns (None, False).
    """
    s = s.strip()
    if not s or "*" in s:
        return None, False
    estimated = "#" in s
    s = s.replace("#", "").strip()
    if not s:
        return None, estimated
    try:
        return float(s), estimated
    except ValueError:
        return None, estimated


def _parse(path: Path) -> pl.DataFrame:
    """Parse the AME2020 fixed-width file → DataFrame.

    Per the file header:
        format: a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6
        cols:   cc NZ  N  Z  A    el  o     mass  unc binding unc      B  beta  unc    atomic_mass   unc

    The file starts with ~36 header lines; the data section begins after the
    line containing 'MASS LIST'. We skip everything until that anchor.

    Atomic mass is split into integer micro-u + fractional micro-u (last 2 cols);
    we recombine into a single float64 atomic_mass_u.
    """
    rows: list[dict] = []
    in_data = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not in_data:
            if line.lstrip().startswith("(keV)") or "MASS EXCESS" in line:
                in_data = True
            continue

        # The header continues for one more line ("(keV)..."). Skip blank/short lines.
        if len(line) < 100 or not line[1:5].strip().lstrip("-").isdigit() and not line[5:9].strip().isdigit():
            continue

        # Field positions per Fortran format:
        # cc[0:1] NZ[1:4] N[4:9] Z[9:14] A[14:19] _[19] el[20:23] o[23:27]
        # _[27] mass[28:42] unc[42:54]
        # binding[54:67] unc[67:78]
        # _[78] B[79:81] beta[81:94] unc[94:105]
        # _[105] atomic_mass_int[106:109] _[109] atomic_mass_frac[110:123] unc[123:135]
        try:
            n_field = line[4:9].strip()
            z_field = line[9:14].strip()
            a_field = line[14:19].strip()
            if not (n_field and z_field and a_field):
                continue
            n = int(n_field)
            z = int(z_field)
            a = int(a_field)
            symbol = line[20:23].strip()

            mass_excess_keV, mass_excess_est = _parse_value(line[28:42])
            mass_excess_unc_keV, _ = _parse_value(line[42:54])
            binding_per_a_keV, binding_est = _parse_value(line[54:67])
            binding_per_a_unc_keV, _ = _parse_value(line[67:78])
            beta_decay_keV, beta_est = _parse_value(line[81:94])
            beta_decay_unc_keV, _ = _parse_value(line[94:105])

            # Atomic mass in micro-u: `i3,1x,f13.6` — integer μ-u part + fractional μ-u part.
            atomic_mass_int = line[105:109].strip()
            atomic_mass_frac, atomic_mass_est = _parse_value(line[109:123])
            atomic_mass_unc_microu, _ = _parse_value(line[123:135])

            atomic_mass_microu = None
            if atomic_mass_int and atomic_mass_frac is not None:
                try:
                    atomic_mass_microu = int(atomic_mass_int) * 1_000_000.0 + atomic_mass_frac
                except ValueError:
                    pass

            rows.append(
                {
                    "Z": z,
                    "A": a,
                    "N": n,
                    "symbol": symbol,
                    "mass_excess_keV": mass_excess_keV,
                    "mass_excess_unc_keV": mass_excess_unc_keV,
                    "binding_energy_per_A_keV": binding_per_a_keV,
                    "binding_energy_per_A_unc_keV": binding_per_a_unc_keV,
                    "beta_decay_energy_keV": beta_decay_keV,
                    "beta_decay_energy_unc_keV": beta_decay_unc_keV,
                    "atomic_mass_microu": atomic_mass_microu,
                    "atomic_mass_unc_microu": atomic_mass_unc_microu,
                    "is_estimated": mass_excess_est or binding_est or beta_est or atomic_mass_est,
                }
            )
        except (ValueError, IndexError) as e:
            logger.debug("Skipped malformed row: %r (%s)", line[:60], e)
            continue

    return pl.DataFrame(rows).sort("Z", "A")


def main() -> None:
    raw = _fetch()
    df = _parse(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH, compression="zstd")
    logger.info("Wrote %d rows to %s", len(df), OUT_PATH)
    n_est = df["is_estimated"].sum()
    logger.info("  experimental: %d  estimated: %d", len(df) - n_est, n_est)


if __name__ == "__main__":
    main()
