"""Build photon interaction cross-sections from EPDL97 (LLNL/IAEA).

Converts the Evaluated Photon Data Library (EPDL97) from ENDF/B-6 format
to per-element Parquet files with per-process cross-section breakdown.

Data source: https://www-nds.iaea.org/epdl97/
Reference: D.E. Cullen et al., UCRL-50400 Vol.6 Rev.5 (1997)

Output files:
  meta/epdl97/photon_xs/     — Per-element photon cross-sections (barns/atom)
  meta/epdl97/form_factors/  — Coherent (Rayleigh) atomic form factors
  meta/epdl97/scattering_fn/ — Incoherent (Compton) scattering functions
  meta/epdl97/anomalous/     — Anomalous scattering factors (real + imaginary)
  meta/epdl97/subshell_pe/   — Subshell photoelectric cross-sections

Also processes EADL (atomic relaxation / fluorescence data):
  meta/eadl/                 — Radiative transition probabilities, fluorescence yields

Usage:
    uv run python -m nucl_parquet.build_epdl [--data-dir PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .download import data_dir as _resolve_data_dir

# ENDF MF/MT definitions for EPDL97
_XS_MTS = {
    501: "total",
    502: "coherent",       # Rayleigh
    504: "incoherent",     # Compton
    515: "pair_electron",  # pair production, electron field
    516: "pair_total",     # pair production, total
    517: "pair_nuclear",   # pair production, nuclear field
    522: "photoelectric",  # total photoionization
}

# Subshell designations (MT 534-572)
_SUBSHELL_NAMES = {
    534: "K",     535: "L1",    536: "L2",    537: "L3",
    538: "M1",    539: "M2",    540: "M3",    541: "M4",    542: "M5",
    543: "N1",    544: "N2",    545: "N3",    546: "N4",    547: "N5",
    548: "N6",    549: "N7",    550: "O1",    551: "O2",    552: "O3",
    553: "O4",    554: "O5",    555: "O6",    556: "O7",    557: "O8",
    558: "O9",    559: "P1",    560: "P2",    561: "P3",    562: "P4",
    563: "P5",    564: "P6",    565: "P7",    566: "P8",    567: "P9",
    568: "P10",   569: "P11",   570: "Q1",    571: "Q2",    572: "Q3",
}

_ELEMENTS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
]


def _download_epdl(dest: Path) -> Path:
    """Download EPDL97 from IAEA if not cached."""
    path = dest / "epdl97.all"
    if path.exists():
        return path
    dest.mkdir(parents=True, exist_ok=True)

    from urllib.request import Request, urlopen

    url = "https://www-nds.iaea.org/epdl97/data/endfb6/epdl97/epdl97.all"
    print(f"Downloading EPDL97 from {url}...")
    req = Request(url, headers={"User-Agent": "nucl-parquet/0.2 (nuclear data research)"})
    with urlopen(req) as resp:  # noqa: S310
        path.write_bytes(resp.read())
    print(f"  Saved {path.stat().st_size / 1e6:.1f} MB to {path}")
    return path


def _download_eadl(dest: Path) -> Path:
    """Download EADL from IAEA if not cached."""
    path = dest / "eadl.all"
    if path.exists():
        return path
    dest.mkdir(parents=True, exist_ok=True)

    from urllib.request import Request, urlopen

    url = "https://www-nds.iaea.org/epdl97/data/endfb6/eadl/eadl.all"
    print(f"Downloading EADL from {url}...")
    req = Request(url, headers={"User-Agent": "nucl-parquet/0.2 (nuclear data research)"})
    with urlopen(req) as resp:  # noqa: S310
        path.write_bytes(resp.read())
    print(f"  Saved {path.stat().st_size / 1e6:.1f} MB to {path}")
    return path


def _download_eedl(dest: Path) -> Path:
    """Download EEDL from IAEA if not cached."""
    path = dest / "eedl.all"
    if path.exists():
        return path
    dest.mkdir(parents=True, exist_ok=True)

    from urllib.request import Request, urlopen

    url = "https://www-nds.iaea.org/epdl97/data/endfb6/eedl/eedl.all"
    print(f"Downloading EEDL from {url}...")
    req = Request(url, headers={"User-Agent": "nucl-parquet/0.2 (nuclear data research)"})
    with urlopen(req) as resp:  # noqa: S310
        path.write_bytes(resp.read())
    print(f"  Saved {path.stat().st_size / 1e6:.1f} MB to {path}")
    return path


def build(data_dir: Path | None = None, cache_dir: Path | None = None) -> None:
    """Build EPDL97 + EADL Parquet files."""
    import endf
    import polars as pl

    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    if cache_dir is None:
        cache_dir = Path("/tmp/epdl97")

    # --- Download source data ---
    epdl_path = _download_epdl(cache_dir)
    eadl_path = _download_eadl(cache_dir)
    eedl_path = _download_eedl(cache_dir)

    # --- Parse EPDL97 ---
    print("Parsing EPDL97...")
    mats = endf.get_materials(str(epdl_path))
    print(f"  {len(mats)} elements loaded")

    # --- 1. Per-process photon cross-sections ---
    _build_photon_xs(mats, data_dir, pl)

    # --- 2. Form factors ---
    _build_form_factors(mats, data_dir, pl)

    # --- 3. Scattering functions ---
    _build_scattering_functions(mats, data_dir, pl)

    # --- 4. Anomalous scattering factors ---
    _build_anomalous(mats, data_dir, pl)

    # --- 5. Subshell photoelectric ---
    _build_subshell_pe(mats, data_dir, pl)

    # --- 6. EADL (atomic relaxation) ---
    print("\nParsing EADL...")
    eadl_mats = endf.get_materials(str(eadl_path))
    _build_eadl(eadl_mats, data_dir, pl)

    # --- 7. EEDL (electron interaction data) ---
    print("\nParsing EEDL...")
    eedl_mats = endf.get_materials(str(eedl_path))
    _build_eedl(eedl_mats, data_dir, pl)

    print("\nDone!")


def _build_photon_xs(mats, data_dir: Path, pl) -> None:
    """Build per-element photon cross-section Parquet files."""
    out_dir = data_dir / "meta" / "epdl97" / "photon_xs"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for mat in mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        rows_Z = []
        rows_E = []
        rows_process = []
        rows_xs = []

        for mt, process_name in _XS_MTS.items():
            sec = mat.section_data.get((23, mt))
            if sec is None:
                continue
            sigma = sec["sigma"]
            E_eV = np.array(sigma.x)
            xs_barns = np.array(sigma.y)

            for i in range(len(E_eV)):
                rows_Z.append(Z)
                rows_E.append(E_eV[i] / 1e6)  # eV -> MeV
                rows_process.append(process_name)
                rows_xs.append(xs_barns[i])

        if not rows_Z:
            continue

        df = pl.DataFrame({
            "Z": pl.Series(rows_Z, dtype=pl.Int32),
            "energy_MeV": pl.Series(rows_E, dtype=pl.Float64),
            "process": pl.Series(rows_process, dtype=pl.Utf8),
            "xs_barns": pl.Series(rows_xs, dtype=pl.Float64),
        }).sort("process", "energy_MeV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  Photon XS: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_form_factors(mats, data_dir: Path, pl) -> None:
    """Build coherent (Rayleigh) atomic form factor files."""
    out_dir = data_dir / "meta" / "epdl97" / "form_factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for mat in mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        sec = mat.section_data.get((27, 502))
        if sec is None:
            continue

        H = sec["H"]
        # x = momentum transfer (1/cm), y = form factor (dimensionless)
        df = pl.DataFrame({
            "Z": pl.Series([Z] * len(H.x), dtype=pl.Int32),
            "momentum_transfer": pl.Series(np.array(H.x), dtype=pl.Float64),
            "form_factor": pl.Series(np.array(H.y), dtype=pl.Float64),
        })

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  Form factors: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_scattering_functions(mats, data_dir: Path, pl) -> None:
    """Build incoherent (Compton) scattering function files."""
    out_dir = data_dir / "meta" / "epdl97" / "scattering_fn"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for mat in mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        sec = mat.section_data.get((27, 504))
        if sec is None:
            continue

        H = sec["H"]
        df = pl.DataFrame({
            "Z": pl.Series([Z] * len(H.x), dtype=pl.Int32),
            "momentum_transfer": pl.Series(np.array(H.x), dtype=pl.Float64),
            "scattering_fn": pl.Series(np.array(H.y), dtype=pl.Float64),
        })

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  Scattering functions: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_anomalous(mats, data_dir: Path, pl) -> None:
    """Build anomalous scattering factor files (real + imaginary)."""
    out_dir = data_dir / "meta" / "epdl97" / "anomalous"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for mat in mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        sec_real = mat.section_data.get((27, 506))
        sec_imag = mat.section_data.get((27, 505))

        if sec_real is None and sec_imag is None:
            continue

        rows = []
        if sec_imag is not None:
            H = sec_imag["H"]
            for i in range(len(H.x)):
                rows.append((Z, H.x[i] / 1e6, "imaginary", H.y[i]))

        if sec_real is not None:
            H = sec_real["H"]
            for i in range(len(H.x)):
                rows.append((Z, H.x[i] / 1e6, "real", H.y[i]))

        if not rows:
            continue

        df = pl.DataFrame({
            "Z": pl.Series([r[0] for r in rows], dtype=pl.Int32),
            "energy_MeV": pl.Series([r[1] for r in rows], dtype=pl.Float64),
            "component": pl.Series([r[2] for r in rows], dtype=pl.Utf8),
            "factor": pl.Series([r[3] for r in rows], dtype=pl.Float64),
        }).sort("component", "energy_MeV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  Anomalous factors: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_subshell_pe(mats, data_dir: Path, pl) -> None:
    """Build subshell photoelectric cross-section files."""
    out_dir = data_dir / "meta" / "epdl97" / "subshell_pe"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for mat in mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        rows_Z = []
        rows_E = []
        rows_shell = []
        rows_xs = []
        rows_edge = []
        rows_fluor = []

        for mt in range(534, 573):
            sec = mat.section_data.get((23, mt))
            if sec is None:
                continue
            shell_name = _SUBSHELL_NAMES.get(mt, f"S{mt-533}")
            sigma = sec["sigma"]
            E_eV = np.array(sigma.x)
            xs_barns = np.array(sigma.y)

            # Edge energy and fluorescence yield from section header
            edge_eV = sec.get("EPE", 0.0)
            fluor_yield = sec.get("EFL", 0.0)

            for i in range(len(E_eV)):
                rows_Z.append(Z)
                rows_E.append(E_eV[i] / 1e6)
                rows_shell.append(shell_name)
                rows_xs.append(xs_barns[i])
                rows_edge.append(edge_eV / 1e6 if edge_eV else 0.0)
                rows_fluor.append(fluor_yield)

        if not rows_Z:
            continue

        df = pl.DataFrame({
            "Z": pl.Series(rows_Z, dtype=pl.Int32),
            "energy_MeV": pl.Series(rows_E, dtype=pl.Float64),
            "subshell": pl.Series(rows_shell, dtype=pl.Utf8),
            "xs_barns": pl.Series(rows_xs, dtype=pl.Float64),
            "edge_MeV": pl.Series(rows_edge, dtype=pl.Float64),
            "fluorescence_yield_eV": pl.Series(rows_fluor, dtype=pl.Float64),
        }).sort("subshell", "energy_MeV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  Subshell PE: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_eadl(eadl_mats, data_dir: Path, pl) -> None:
    """Build EADL atomic relaxation / fluorescence data.

    EADL provides radiative and non-radiative transition data per subshell.
    Each shell has transitions with:
      - SUBI: initial vacancy subshell index
      - SUBJ: primary filling subshell
      - SUBK: secondary vacancy subshell (0 = radiative, >0 = Auger)
      - ETR: transition energy (eV)
      - FTR: transition probability (fractional)
    """
    out_dir = data_dir / "meta" / "eadl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subshell index -> name mapping (ENDF convention)
    _shell_idx = {
        1: "K", 2: "L1", 3: "L2", 4: "L3",
        5: "M1", 6: "M2", 7: "M3", 8: "M4", 9: "M5",
        10: "N1", 11: "N2", 12: "N3", 13: "N4", 14: "N5",
        15: "N6", 16: "N7", 17: "O1", 18: "O2", 19: "O3",
        20: "O4", 21: "O5", 22: "O6", 23: "O7",
        24: "P1", 25: "P2", 26: "P3",
    }

    total_rows = 0
    for mat in eadl_mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        sec = mat.section_data.get((28, 533))
        if sec is None or "shells" not in sec:
            continue

        rows_Z = []
        rows_shell = []
        rows_dest = []
        rows_type = []
        rows_energy = []
        rows_prob = []
        rows_edge = []

        for sh in sec["shells"]:
            subi = int(sh["SUBI"])
            shell_name = _shell_idx.get(subi, f"S{subi}")
            edge_eV = float(sh.get("EBI", 0.0))
            n_tr = int(sh["NTR"])

            for i in range(n_tr):
                subj = int(sh["SUBJ"][i])
                subk = int(sh["SUBK"][i])
                dest_name = _shell_idx.get(subj, f"S{subj}")

                # SUBK=0 means radiative (X-ray), SUBK>0 means Auger
                if subk == 0:
                    tr_type = "radiative"
                else:
                    tr_type = "auger"

                rows_Z.append(Z)
                rows_shell.append(shell_name)
                rows_dest.append(dest_name)
                rows_type.append(tr_type)
                rows_energy.append(float(sh["ETR"][i]) / 1e3)  # eV -> keV
                rows_prob.append(float(sh["FTR"][i]))
                rows_edge.append(edge_eV / 1e3)  # eV -> keV

        if not rows_Z:
            continue

        df = pl.DataFrame({
            "Z": pl.Series(rows_Z, dtype=pl.Int32),
            "vacancy_shell": pl.Series(rows_shell, dtype=pl.Utf8),
            "filling_shell": pl.Series(rows_dest, dtype=pl.Utf8),
            "transition_type": pl.Series(rows_type, dtype=pl.Utf8),
            "energy_keV": pl.Series(rows_energy, dtype=pl.Float64),
            "probability": pl.Series(rows_prob, dtype=pl.Float64),
            "edge_keV": pl.Series(rows_edge, dtype=pl.Float64),
        }).sort("vacancy_shell", "energy_keV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  EADL: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


def _build_eedl(eedl_mats, data_dir: Path, pl) -> None:
    """Build EEDL electron interaction cross-section files.

    EEDL MF23 MTs:
      526: Elastic scattering (large-angle)
      527: Bremsstrahlung
      528: Excitation
      534-572: Subshell ionization (same numbering as EPDL)
    """
    out_dir = data_dir / "meta" / "eedl"
    out_dir.mkdir(parents=True, exist_ok=True)

    _EEDL_MTS = {
        526: "elastic",
        527: "bremsstrahlung",
        528: "excitation",
    }

    total_rows = 0
    for mat in eedl_mats:
        Z = mat.MAT // 100
        if Z < 1 or Z > 100:
            continue
        sym = _ELEMENTS[Z]

        rows_Z = []
        rows_E = []
        rows_process = []
        rows_xs = []

        for (mf, mt), sec in mat.section_data.items():
            if mf != 23 or not isinstance(sec, dict) or "sigma" not in sec:
                continue

            if mt in _EEDL_MTS:
                process_name = _EEDL_MTS[mt]
            elif 534 <= mt <= 572:
                shell_name = _SUBSHELL_NAMES.get(mt, f"S{mt-533}")
                process_name = f"ionization_{shell_name}"
            else:
                continue

            sigma = sec["sigma"]
            E_eV = np.array(sigma.x)
            xs_barns = np.array(sigma.y)

            for i in range(len(E_eV)):
                rows_Z.append(Z)
                rows_E.append(E_eV[i] / 1e6)  # eV -> MeV
                rows_process.append(process_name)
                rows_xs.append(xs_barns[i])

        if not rows_Z:
            continue

        df = pl.DataFrame({
            "Z": pl.Series(rows_Z, dtype=pl.Int32),
            "energy_MeV": pl.Series(rows_E, dtype=pl.Float64),
            "process": pl.Series(rows_process, dtype=pl.Utf8),
            "xs_barns": pl.Series(rows_xs, dtype=pl.Float64),
        }).sort("process", "energy_MeV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_rows += len(df)

    print(f"  EEDL: {total_rows} rows across {len(list(out_dir.glob('*.parquet')))} elements")


if __name__ == "__main__":
    data_dir = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--data-dir" and i + 1 < len(sys.argv) - 1:
            data_dir = Path(sys.argv[i + 2])
    build(data_dir=data_dir)
