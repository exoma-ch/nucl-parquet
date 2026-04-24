"""Extract neutron total and elastic cross-sections from ENDF/B-VIII.1.

Downloads ENDF-6 format files from the IAEA NDS mirror, parses MF=3 MT=1
(total) and MF=3 MT=2 (elastic) cross-sections, and writes per-element
Parquet files for use in flux attenuation calculations:

    Phi(x) = Phi_0 * exp(-Sigma_tot * x)

Output: data/meta/neutron_total/{Element}.parquet
Schema: Z int32, A int32, energy_MeV f64, xs_total_mb f64, xs_elastic_mb f64

Usage:
    uv run python scripts/build_neutron_total.py
    uv run python scripts/build_neutron_total.py --library endfb-8.1
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
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

# Filename pattern: n_029-Cu-63_2925.zip
FILENAME_RE = re.compile(r"[a-z]+_(\d{3})-([A-Za-z]+)-(\d+)_(\d+)\.zip")

# Library key -> IAEA path mapping (only neutron sub-library needed)
LIBRARY_PATHS: dict[str, str] = {
    "endfb-8.1": "ENDF-B-VIII.1/n",
}


def _list_zip_files(base_url: str, session: requests.Session) -> list[str]:
    """Fetch IAEA directory listing and return .zip filenames."""
    resp = session.get(base_url + "/", timeout=30)
    resp.raise_for_status()
    return re.findall(r'href="([^"]+\.zip)"', resp.text)


def _parse_total_elastic(
    endf_text: str,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Parse MF=3 MT=1 (total) and MT=2 (elastic) from ENDF text.

    Returns (E_total_eV, xs_total_b, E_elastic_eV, xs_elastic_b).
    Arrays are None when the section is missing.
    """
    import endf

    material = endf.Material(io.StringIO(endf_text))

    E_tot, xs_tot, E_el, xs_el = None, None, None, None

    for (mf, mt), section in material.section_data.items():
        if mf != 3:
            continue
        if mt not in (1, 2):
            continue

        tab = section.get("sigma")
        if tab is None:
            continue

        if mt == 1:
            E_tot = np.asarray(tab.x, dtype=np.float64)
            xs_tot = np.asarray(tab.y, dtype=np.float64)
        elif mt == 2:
            E_el = np.asarray(tab.x, dtype=np.float64)
            xs_el = np.asarray(tab.y, dtype=np.float64)

    return E_tot, xs_tot, E_el, xs_el


def build(
    data_dir: Path | None = None,
    library: str = "endfb-8.1",
) -> None:
    if data_dir is None:
        data_dir = ROOT / "data"

    iaea_path = LIBRARY_PATHS.get(library)
    if iaea_path is None:
        logger.error("Unknown library %s. Known: %s", library, list(LIBRARY_PATHS))
        return

    out_dir = data_dir / "meta" / "neutron_total"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"{IAEA_MIRROR}/{iaea_path}"
    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    filenames = _list_zip_files(base_url, session)
    logger.info("Found %d ENDF zip files at %s", len(filenames), base_url)

    # Group DataFrames by element symbol
    element_dfs: dict[str, list[pl.DataFrame]] = {}
    processed = 0

    for i, fname in enumerate(filenames):
        m = FILENAME_RE.match(fname)
        if not m:
            continue

        target_z = int(m.group(1))
        sym = m.group(2)
        target_a = int(m.group(3))

        if (i + 1) % 50 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(filenames))

        url = f"{base_url}/{fname}"
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Download failed %s: %s", fname, e)
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                if not names:
                    continue
                endf_text = zf.read(names[0]).decode("ascii", errors="replace")
        except (zipfile.BadZipFile, KeyError) as e:
            logger.warning("Bad zip %s: %s", fname, e)
            continue

        try:
            E_tot, xs_tot, E_el, xs_el = _parse_total_elastic(endf_text)
        except Exception as e:
            logger.warning("Parse failed %s: %s", fname, e)
            continue

        if E_tot is None or xs_tot is None:
            logger.debug("  No MT=1 (total) for %s-%d, skipping", sym, target_a)
            continue

        # Filter sentinels from total XS
        valid = xs_tot < 1e30
        E_tot = E_tot[valid]
        xs_tot = xs_tot[valid]

        if len(E_tot) == 0:
            continue

        # Interpolate elastic onto total's energy grid (or fill zeros)
        if E_el is not None and xs_el is not None:
            el_valid = xs_el < 1e30
            E_el = E_el[el_valid]
            xs_el = xs_el[el_valid]
            xs_el_interp = np.interp(E_tot, E_el, xs_el, left=0.0, right=0.0)
        else:
            xs_el_interp = np.zeros_like(E_tot)

        # Convert units: eV -> MeV, barns -> millibarns
        n = len(E_tot)
        isotope_df = pl.DataFrame(
            {
                "Z": pl.Series(np.full(n, target_z, dtype=np.int32), dtype=pl.Int32),
                "A": pl.Series(np.full(n, target_a, dtype=np.int32), dtype=pl.Int32),
                "energy_MeV": pl.Series(E_tot * 1e-6, dtype=pl.Float64),
                "xs_total_mb": pl.Series(xs_tot * 1e3, dtype=pl.Float64),
                "xs_elastic_mb": pl.Series(xs_el_interp * 1e3, dtype=pl.Float64),
            }
        )

        element_dfs.setdefault(sym, []).append(isotope_df)
        processed += 1

    if not element_dfs:
        logger.warning("No neutron total XS data extracted")
        return

    total_rows = 0
    for sym, dfs in sorted(element_dfs.items()):
        df = pl.concat(dfs).sort("A", "energy_MeV")

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression=COMPRESSION)
        total_rows += len(df)
        logger.info("  %s: %d rows", sym, len(df))

    logger.info(
        "Done: %d elements, %d isotopes, %d total rows",
        len(element_dfs),
        processed,
        total_rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract neutron total/elastic XS from ENDF libraries",
    )
    parser.add_argument("--library", default="endfb-8.1", help="Library to use")
    parser.add_argument("--data-dir", type=Path, help="Data directory (default: data/)")
    args = parser.parse_args()

    build(data_dir=args.data_dir, library=args.library)
