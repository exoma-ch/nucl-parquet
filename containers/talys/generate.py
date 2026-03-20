"""Generate heavy-ion reaction cross-section tables using TALYS.

Runs TALYS for each (projectile, target) pair and parses residual production
cross-sections into the nucl-parquet XS schema.

Usage inside container:
    python3 /opt/generate.py [--out /data/hi-xs] [--proj c12,o16] [--zmax 92]

Output: {out}/{proj}_{target_symbol}.parquet
Schema: target_A, residual_Z, residual_A, state, energy_MeV, xs_mb
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]
SYMBOL_TO_Z = {s.lower(): i + 1 for i, s in enumerate(ELEMENT_SYMBOLS)}
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}

# Projectile definitions: label -> (TALYS symbol, proj_A, proj_Z)
PROJECTILES: dict[str, tuple[str, int, int]] = {
    "c12":  ("c",  12,  6),
    "o16":  ("o",  16,  8),
    "ne20": ("ne", 20, 10),
    "si28": ("si", 28, 14),
    "ar40": ("ar", 40, 18),
    "fe56": ("fe", 56, 26),
}

# Energy grid: 1–200 MeV/u, 40 log-spaced points, stored as total MeV
ENERGIES_PER_U = np.geomspace(1.0, 200.0, 40)

# Most abundant / longest-lived isotope per element for target mass
# (0 = natural / most abundant, which TALYS handles automatically)
# We run with mass=0 (natural) and get all residual channels
TARGET_MASS = 0  # natural — TALYS picks the most abundant stable isotope

# ---------------------------------------------------------------------------
# TALYS runner
# ---------------------------------------------------------------------------

_RP_PATTERN = re.compile(r"rp(\d{3})(\d{3})([gm]?)\.tot$", re.IGNORECASE)


def _write_energy_file(workdir: Path, energies_MeV: np.ndarray) -> Path:
    """Write TALYS energy input file."""
    epath = workdir / "energies.dat"
    epath.write_text("\n".join(f"{e:.6f}" for e in energies_MeV) + "\n")
    return epath


def _write_input(workdir: Path, proj_sym: str, proj_A: int, target_sym: str, energy_file: Path) -> Path:
    """Write TALYS input file for one (projectile, target) pair."""
    inp = workdir / "input"
    inp.write_text(
        f"projectile  {proj_sym}\n"
        f"Aproj       {proj_A}\n"
        f"element     {target_sym}\n"
        f"mass        {TARGET_MASS}\n"
        f"energyfile  {energy_file}\n"
        f"outfy       y\n"       # write residual production files
        f"outspectra  n\n"
        f"outdiscrete n\n"
    )
    return inp


def _run_talys(workdir: Path) -> bool:
    """Run TALYS in workdir. Returns True on success."""
    env = os.environ.copy()
    talys_struct = os.environ.get("TALYS_STRUCTURE", "/opt/talys-structure")
    env["PATH"] = f"/usr/local/bin:{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            ["talys"],
            stdin=open(workdir / "input"),
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"    TALYS stderr: {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    TALYS timed out")
        return False
    except FileNotFoundError:
        print("    talys binary not found — is it installed?")
        return False


def _parse_rp_files(workdir: Path, energies_MeV: np.ndarray) -> list[dict]:
    """Parse all rp*.tot files in workdir into row dicts."""
    rows = []
    for f in sorted(workdir.glob("rp*.tot")):
        m = _RP_PATTERN.match(f.name)
        if not m:
            continue
        res_Z = int(m.group(1))
        res_A = int(m.group(2))
        state = m.group(3).lower()

        try:
            lines = [ln for ln in f.read_text().splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
            xs_vals = [float(ln.split()[1]) for ln in lines if len(ln.split()) >= 2]
        except Exception:
            continue

        # Align with energy grid (TALYS may emit fewer lines if threshold not reached)
        for i, xs in enumerate(xs_vals):
            if i >= len(energies_MeV):
                break
            if xs > 0:
                rows.append({
                    "energy_MeV": float(energies_MeV[i]),
                    "residual_Z": res_Z,
                    "residual_A": res_A,
                    "state": state,
                    "xs_mb": xs,
                })
    return rows


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(out_dir: Path, proj_keys: list[str], z_max: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for proj_key in proj_keys:
        if proj_key not in PROJECTILES:
            print(f"Unknown projectile {proj_key!r}, skipping")
            continue

        proj_sym, proj_A, proj_Z = PROJECTILES[proj_key]
        energies_MeV = ENERGIES_PER_U * proj_A
        print(f"\n=== {proj_key.upper()} (Z={proj_Z}, A={proj_A}) ===")

        all_rows: list[dict] = []

        for target_Z in range(1, z_max + 1):
            target_sym = Z_TO_SYMBOL[target_Z]
            print(f"  {proj_key} + {target_sym} (Z={target_Z})...", end=" ", flush=True)

            with tempfile.TemporaryDirectory() as td:
                workdir = Path(td)
                efile = _write_energy_file(workdir, energies_MeV)
                _write_input(workdir, proj_sym, proj_A, target_sym, efile)

                if not _run_talys(workdir):
                    print("FAILED")
                    continue

                rows = _parse_rp_files(workdir, energies_MeV)
                for r in rows:
                    r["target_Z"] = target_Z
                    r["target_A"] = TARGET_MASS
                all_rows.extend(rows)
                print(f"{len(rows)} rows ({len(set((r['residual_Z'], r['residual_A']) for r in rows))} residuals)")

        if not all_rows:
            print(f"  No data for {proj_key}, skipping parquet write")
            continue

        df = pl.DataFrame({
            "target_Z":   pl.Series([r["target_Z"]   for r in all_rows], dtype=pl.Int32),
            "target_A":   pl.Series([r["target_A"]   for r in all_rows], dtype=pl.Int32),
            "residual_Z": pl.Series([r["residual_Z"] for r in all_rows], dtype=pl.Int32),
            "residual_A": pl.Series([r["residual_A"] for r in all_rows], dtype=pl.Int32),
            "state":      pl.Series([r["state"]      for r in all_rows], dtype=pl.Utf8),
            "energy_MeV": pl.Series([r["energy_MeV"] for r in all_rows], dtype=pl.Float64),
            "xs_mb":      pl.Series([r["xs_mb"]      for r in all_rows], dtype=pl.Float64),
        }).sort("target_Z", "target_A", "residual_Z", "residual_A", "energy_MeV")

        out_path = out_dir / f"{proj_key}.parquet"
        df.write_parquet(out_path, compression="zstd")
        print(f"  → {out_path} ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HI cross-sections with TALYS")
    parser.add_argument("--out",  default="/data/hi-xs",         help="Output directory (mounted volume)")
    parser.add_argument("--proj", default="c12,o16,ne20,si28,ar40,fe56", help="Comma-separated projectile keys")
    parser.add_argument("--zmax", type=int, default=92,           help="Max target Z (default 92)")
    args = parser.parse_args()

    run(
        out_dir=Path(args.out),
        proj_keys=args.proj.split(","),
        z_max=args.zmax,
    )
