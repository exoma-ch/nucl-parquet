"""Download and parse evaluated nuclear data libraries into Parquet.

Fetches ENDF-6 format files from the IAEA NDS mirror, parses cross-sections
using the `endf` package, and converts to nucl-parquet Parquet format.

Supports all major evaluated libraries: ENDF/B-VIII.1, JEFF-4.0, JENDL-5,
TENDL-2025, CENDL-3.2, BROND-3.1, FENDL-3.2, EAF-2010.

Usage:
    # Fetch a single library (neutron sub-library):
    python scripts/fetch_endf_libs.py --library endfb-8.1 --sublibrary n

    # Fetch all neutron libraries:
    python scripts/fetch_endf_libs.py --sublibrary n --all

    # Fetch proton sub-library for a specific library:
    python scripts/fetch_endf_libs.py --library jendl-5 --sublibrary p

    # List available libraries:
    python scripts/fetch_endf_libs.py --list
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
IAEA_MIRROR = "https://nds.iaea.org/public/download-endf"
COMPRESSION = "zstd"

# ---------------------------------------------------------------------------
# Library registry
# ---------------------------------------------------------------------------

@dataclass
class LibraryDef:
    """Definition of an evaluated nuclear data library."""
    key: str                    # Our short identifier (used in directory names)
    name: str                   # Display name
    iaea_path: str              # Path on IAEA mirror
    description: str
    source_url: str
    sublibraries: dict[str, str]  # sublibrary code -> IAEA subdirectory name

LIBRARIES: dict[str, LibraryDef] = {
    "endfb-8.1": LibraryDef(
        key="endfb-8.1",
        name="ENDF/B-VIII.1",
        iaea_path="ENDF-B-VIII.1",
        description="US Evaluated Nuclear Data File (NNDC/BNL)",
        source_url="https://www.nndc.bnl.gov/endf-b8.1/",
        sublibraries={"n": "n", "p": "p", "d": "d", "t": "t", "h": "he3", "a": "he4"},
    ),
    "jeff-4.0": LibraryDef(
        key="jeff-4.0",
        name="JEFF-4.0",
        iaea_path="JEFF-4.0",
        description="Joint Evaluated Fission and Fusion File (NEA)",
        source_url="https://www.oecd-nea.org/dbdata/jeff/",
        sublibraries={"n": "n", "p": "p"},
    ),
    "jendl-5": LibraryDef(
        key="jendl-5",
        name="JENDL-5",
        iaea_path="JENDL-5",
        description="Japanese Evaluated Nuclear Data Library (JAEA)",
        source_url="https://wwwndc.jaea.go.jp/jendl/j5/j5.html",
        sublibraries={"n": "n", "p": "p", "d": "d", "a": "he4"},
    ),
    "tendl-2025": LibraryDef(
        key="tendl-2025",
        name="TENDL-2025",
        iaea_path="TENDL-2025",
        description="TALYS Evaluated Nuclear Data Library (PSI)",
        source_url="https://tendl.web.psi.ch/",
        sublibraries={"n": "n", "p": "p", "d": "d", "t": "t", "h": "he3", "a": "he4"},
    ),
    "cendl-3.2": LibraryDef(
        key="cendl-3.2",
        name="CENDL-3.2",
        iaea_path="CENDL-3.2",
        description="Chinese Evaluated Nuclear Data Library (CIAE)",
        source_url="http://www.nuclear.csdb.cn/",
        sublibraries={"n": "n"},
    ),
    "brond-3.1": LibraryDef(
        key="brond-3.1",
        name="BROND-3.1",
        iaea_path="BROND-3.1",
        description="Russian Evaluated Nuclear Data Library (IPPE)",
        source_url="https://vant.ippe.ru/",
        sublibraries={"n": "n"},
    ),
    "fendl-3.2": LibraryDef(
        key="fendl-3.2",
        name="FENDL-3.2",
        iaea_path="FENDL-3.2c",
        description="Fusion Evaluated Nuclear Data Library (IAEA)",
        source_url="https://www-nds.iaea.org/fendl/",
        sublibraries={"n": "n"},
    ),
    "eaf-2010": LibraryDef(
        key="eaf-2010",
        name="EAF-2010",
        iaea_path="EAF-2010",
        description="European Activation File (CCFE)",
        source_url="https://fispact.ukaea.uk/",
        sublibraries={"n": "n"},
    ),
    "irdff-2": LibraryDef(
        key="irdff-2",
        name="IRDFF-II",
        iaea_path="IRDFF-II",
        description="International Reactor Dosimetry and Fusion File (IAEA)",
        source_url="https://www-nds.iaea.org/IRDFF/",
        sublibraries={"n": "n"},
    ),
    "iaea-medical": LibraryDef(
        key="iaea-medical",
        name="IAEA-Medical",
        iaea_path="IAEA-Medical",
        description="Medical isotope production cross-sections (IAEA)",
        source_url="https://www-nds.iaea.org/medical/",
        sublibraries={"p": "p", "d": "d", "h": "he3", "a": "he4"},
    ),
}


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
    93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es",
}

# Projectile (Z, A) for mass/charge balance
PROJECTILE_ZA: dict[str, tuple[int, int]] = {
    "n": (0, 1),
    "p": (1, 1),
    "d": (1, 2),
    "t": (1, 3),
    "h": (2, 3),
    "a": (2, 4),
}


# ---------------------------------------------------------------------------
# MT number -> residual product mapping
# ---------------------------------------------------------------------------

# MT -> (delta_Z, delta_A, emitted_particles_description)
# delta is: target + projectile - residual
# residual_Z = target_Z + proj_Z - emitted_Z
# residual_A = target_A + proj_A - emitted_A

MT_TO_EMISSION: dict[int, tuple[int, int]] = {
    # (emitted_Z, emitted_A) — what leaves besides the residual
    2:   (0, 0),     # elastic: nothing emitted, residual = compound
    4:   (0, 1),     # (x,n') inelastic: 1 neutron
    16:  (0, 2),     # (x,2n)
    17:  (0, 3),     # (x,3n)
    18:  (0, 0),     # fission — skip (no single residual)
    22:  (2, 5),     # (x,nα): n + α
    23:  (2, 7),     # (x,n3α): n + 3α — rare
    24:  (2, 6),     # (x,2nα): 2n + α
    25:  (0, 4),     # (x,4n) — rare, added for completeness
    28:  (1, 2),     # (x,np): n + p
    29:  (2, 8),     # (x,n2α): n + 2α — rare
    32:  (1, 3),     # (x,nd): n + d
    33:  (1, 4),     # (x,nt): n + t
    34:  (2, 4),     # (x,n³He): n + ³He
    35:  (2, 5),     # (x,nd2α) — skip, too complex
    36:  (2, 6),     # (x,nt2α) — skip
    37:  (0, 5),     # (x,5n) — rare, added for completeness
    41:  (1, 3),     # (x,2np): 2n + p
    42:  (1, 4),     # (x,3np): 3n + p
    44:  (2, 6),     # (x,n2p): n + 2p
    45:  (2, 9),     # (x,npα): n + p + α
    102: (0, 0),     # (x,γ): capture, no particles emitted
    103: (1, 1),     # (x,p)
    104: (1, 2),     # (x,d)
    105: (1, 3),     # (x,t)
    106: (2, 3),     # (x,³He)
    107: (2, 4),     # (x,α)
    108: (4, 8),     # (x,2α)
    109: (4, 11),    # (x,3α)
    111: (1, 2),     # (x,2p)
    112: (3, 5),     # (x,pα)
    113: (3, 8),     # (x,t2α)
    115: (2, 5),     # (x,pd)
    116: (2, 6),     # (x,pt)
    117: (3, 7),     # (x,dα)
}

# MT ranges for discrete inelastic levels
# MT 51-91: (x,n') to specific levels — all emit 1 neutron
# MT 600-649: (x,p) to specific levels — all emit 1 proton
# MT 650-699: (x,d) levels
# MT 700-749: (x,t) levels
# MT 750-799: (x,³He) levels
# MT 800-849: (x,α) levels
# MT 875-891: (x,2n) to specific levels

LEVEL_RANGES: dict[tuple[int, int], tuple[int, int]] = {
    (51, 91):    (0, 1),    # n emission
    (600, 649):  (1, 1),    # p emission
    (650, 699):  (1, 2),    # d emission
    (700, 749):  (1, 3),    # t emission
    (750, 799):  (2, 3),    # ³He emission
    (800, 849):  (2, 4),    # α emission
    (875, 891):  (0, 2),    # 2n emission
}


def mt_to_residual(
    mt: int, target_z: int, target_a: int, proj_z: int, proj_a: int,
) -> tuple[int, int] | None:
    """Compute residual (Z, A) from MT number and target+projectile.

    Returns None for reactions that don't produce a single residual (fission, etc).
    """
    # Check explicit MT table first
    if mt in MT_TO_EMISSION:
        if mt == 18:  # fission
            return None
        emit_z, emit_a = MT_TO_EMISSION[mt]
        res_z = target_z + proj_z - emit_z
        res_a = target_a + proj_a - emit_a
        if res_z > 0 and res_a > 0:
            return (res_z, res_a)
        return None

    # Check level ranges
    for (mt_lo, mt_hi), (emit_z, emit_a) in LEVEL_RANGES.items():
        if mt_lo <= mt <= mt_hi:
            res_z = target_z + proj_z - emit_z
            res_a = target_a + proj_a - emit_a
            if res_z > 0 and res_a > 0:
                return (res_z, res_a)
            return None

    return None


# ---------------------------------------------------------------------------
# ENDF-6 file parsing
# ---------------------------------------------------------------------------

# Filename pattern: n_029-Cu-63_2925.zip or similar
# Standard: n_029-Cu-63_2925.zip  |  No zero-pad: n_95-Am-241_9543.zip
# Reversed (BROND): n_2925_29-Cu-63.zip
# Metastable: n_095-Am-244M_9553.zip
FILENAME_RE = re.compile(
    r"[a-z0-9]+_(\d{2,3})-([A-Za-z]+)-(\d+)[A-Za-z]*_(\d+)\.zip"
)
FILENAME_RE_ALT = re.compile(
    r"[a-z0-9]+_(\d+)_(\d{1,3})-([A-Za-z]+)-(\d+)[A-Za-z]*\.zip"
)


def _parse_endf_filename(filename: str) -> tuple[int, int] | None:
    """Extract (target_Z, target_A) from ENDF filename.

    Handles both standard and reversed naming conventions.
    """
    m = FILENAME_RE.match(filename)
    if m:
        return int(m.group(1)), int(m.group(3))
    m = FILENAME_RE_ALT.match(filename)
    if m:
        return int(m.group(2)), int(m.group(4))
    return None


def parse_endf_file(
    endf_text: str,
    target_z: int,
    target_a: int,
    projectile: str,
) -> list[dict]:
    """Parse an ENDF-6 format text file and extract cross-section data.

    Returns list of row dicts with keys matching the Parquet schema.
    """
    import endf

    proj_z, proj_a = PROJECTILE_ZA[projectile]

    try:
        material = endf.Material(io.StringIO(endf_text))
    except Exception as e:
        logger.warning("Failed to parse ENDF material Z=%d A=%d: %s", target_z, target_a, e)
        return []

    rows: list[dict] = []

    # Extract MF=3 (cross-section) data
    for (mf, mt), section in material.section_data.items():
        if mf != 3:
            continue

        residual = mt_to_residual(mt, target_z, target_a, proj_z, proj_a)
        if residual is None:
            continue

        res_z, res_a = residual

        try:
            # section for MF=3 is a dict with 'sigma' tabulated function
            tab = section.get("sigma")
            if tab is None:
                continue

            # endf.Tabulated1D has x (energy in eV) and y (cross-section in barns)
            energies_ev = tab.x
            xs_barns = tab.y

            for e_ev, xs_b in zip(energies_ev, xs_barns):
                if xs_b <= 0:
                    continue
                rows.append({
                    "target_A": target_a,
                    "residual_Z": res_z,
                    "residual_A": res_a,
                    "state": "",  # MF=3 doesn't distinguish isomers
                    "energy_MeV": e_ev * 1e-6,
                    "xs_mb": xs_b * 1e3,
                })
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("  Skipping MF=%d MT=%d: %s", mf, mt, e)
            continue

    # Extract MF=10 (isomeric production cross-sections)
    for (mf, mt), section in material.section_data.items():
        if mf != 10:
            continue

        try:
            # MF=10 sections contain cross-sections for specific product nuclides
            # Each subsection has ZA (product Z*1000+A), LFS (isomeric state), and TAB1
            subsections = section.get("subsections", [])
            if not subsections:
                continue

            for sub in subsections:
                za_product = sub.get("ZAPS", 0)
                lfs = sub.get("LFS", 0)
                tab = sub.get("sigma")
                if tab is None or za_product == 0:
                    continue

                res_z_10 = za_product // 1000
                res_a_10 = za_product % 1000
                state = "m" if lfs > 0 else ""

                for e_ev, xs_b in zip(tab.x, tab.y):
                    if xs_b <= 0:
                        continue
                    rows.append({
                        "target_A": target_a,
                        "residual_Z": res_z_10,
                        "residual_A": res_a_10,
                        "state": state,
                        "energy_MeV": e_ev * 1e-6,
                        "xs_mb": xs_b * 1e3,
                    })
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("  Skipping MF=10 MT=%d: %s", mt, e)
            continue

    return rows


# ---------------------------------------------------------------------------
# Download + process
# ---------------------------------------------------------------------------


def list_endf_files(
    lib: LibraryDef,
    sublib_code: str,
    session: requests.Session,
) -> list[str]:
    """Get list of zip filenames from the IAEA mirror directory."""
    sublib_dir = lib.sublibraries.get(sublib_code)
    if sublib_dir is None:
        return []

    url = f"{IAEA_MIRROR}/{lib.iaea_path}/{sublib_dir}/"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to list %s: %s", url, e)
        return []

    # Parse HTML directory listing for .zip filenames
    filenames = re.findall(r'href="([^"]+\.zip)"', resp.text)
    return filenames


def download_and_parse(
    lib: LibraryDef,
    sublib_code: str,
    filename: str,
    session: requests.Session,
) -> list[dict]:
    """Download a single ENDF zip file and parse it."""
    sublib_dir = lib.sublibraries[sublib_code]
    url = f"{IAEA_MIRROR}/{lib.iaea_path}/{sublib_dir}/{filename}"

    # Parse target Z, A from filename
    parsed = _parse_endf_filename(filename)
    if parsed is None:
        logger.warning("Cannot parse filename: %s", filename)
        return []

    target_z, target_a = parsed

    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Download failed %s: %s", filename, e)
        return []

    # Extract ENDF text from zip
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                return []
            endf_text = zf.read(names[0]).decode("ascii", errors="replace")
    except (zipfile.BadZipFile, KeyError) as e:
        logger.warning("Bad zip %s: %s", filename, e)
        return []

    return parse_endf_file(endf_text, target_z, target_a, sublib_code)


def fetch_library(
    lib_key: str,
    sublib_code: str,
    output_dir: Path,
    session: requests.Session,
) -> None:
    """Fetch and convert an entire sub-library to Parquet."""
    lib = LIBRARIES[lib_key]

    if sublib_code not in lib.sublibraries:
        logger.error("%s does not have sub-library '%s'", lib.name, sublib_code)
        return

    logger.info("Fetching %s / %s ...", lib.name, sublib_code)

    filenames = list_endf_files(lib, sublib_code, session)
    if not filenames:
        logger.warning("No files found for %s/%s", lib.name, sublib_code)
        return

    logger.info("  Found %d ENDF files", len(filenames))

    # Group rows by element
    element_rows: dict[str, list[dict]] = {}
    total_files = 0
    total_rows = 0

    for i, fname in enumerate(filenames):
        if (i + 1) % 50 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(filenames))

        rows = download_and_parse(lib, sublib_code, fname, session)
        if not rows:
            continue

        parsed = _parse_endf_filename(fname)
        if parsed is None:
            continue

        target_z = parsed[0]
        elem = _ELEMENT_SYMBOLS.get(target_z, f"Z{target_z}")
        element_rows.setdefault(elem, []).extend(rows)
        total_files += 1
        total_rows += len(rows)

    if not element_rows:
        logger.warning("No data extracted for %s/%s", lib.name, sublib_code)
        return

    # Write Parquet files per element
    # For neutron data: lib_key/xs/n_Fe.parquet
    # For charged particles: lib_key/xs/p_Fe.parquet (same as TENDL layout)
    xs_dir = output_dir / lib_key / "xs"
    xs_dir.mkdir(parents=True, exist_ok=True)

    for elem, rows in element_rows.items():
        df = pl.DataFrame(
            rows,
            schema={
                "target_A": pl.Int32,
                "residual_Z": pl.Int32,
                "residual_A": pl.Int32,
                "state": pl.Utf8,
                "energy_MeV": pl.Float64,
                "xs_mb": pl.Float64,
            },
        )
        # Deduplicate and sort
        df = df.unique(
            subset=["target_A", "residual_Z", "residual_A", "state", "energy_MeV"],
        ).sort("target_A", "residual_Z", "residual_A", "energy_MeV")

        out_path = xs_dir / f"{sublib_code}_{elem}.parquet"
        df.write_parquet(out_path, compression=COMPRESSION)

    # Write manifest
    manifest = {
        "library": lib_key,
        "sublibrary": sublib_code,
        "files": len(element_rows),
        "total_rows": total_rows,
        "source_files": total_files,
        "projectiles": [sublib_code],
        "elements": sorted(element_rows.keys()),
    }
    manifest_path = output_dir / lib_key / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "  Done: %d elements, %d source files, %d total rows → %s/",
        len(element_rows), total_files, total_rows, lib_key,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch evaluated nuclear data libraries and convert to Parquet.",
    )
    parser.add_argument(
        "--library", choices=list(LIBRARIES.keys()),
        help="Library to fetch",
    )
    parser.add_argument(
        "--sublibrary", default="n",
        choices=["n", "p", "d", "t", "h", "a"],
        help="Sub-library / projectile type (default: n)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all libraries for the specified sub-library",
    )
    parser.add_argument(
        "--all-sublibs", action="store_true",
        help="Fetch all sub-libraries for the specified library(ies)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT,
        help="Output directory (default: repo root)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available libraries and their sub-libraries",
    )
    args = parser.parse_args()

    if args.list:
        for key, lib in LIBRARIES.items():
            sublibs = ", ".join(sorted(lib.sublibraries.keys()))
            print(f"  {key:20s}  {lib.name:20s}  sub-libs: {sublibs}")
        return

    if not args.library and not args.all:
        parser.error("Specify --library or --all")

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    libs = list(LIBRARIES.keys()) if args.all else [args.library]

    for lib_key in libs:
        lib = LIBRARIES[lib_key]
        if args.all_sublibs:
            sublibs = sorted(lib.sublibraries.keys())
        else:
            sublibs = [args.sublibrary]

        for sublib in sublibs:
            if sublib not in lib.sublibraries:
                logger.info("Skipping %s/%s (not available)", lib.name, sublib)
                continue
            fetch_library(lib_key, sublib, args.output, session)


if __name__ == "__main__":
    main()
