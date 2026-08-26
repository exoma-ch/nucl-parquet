"""Build the neutron *transport channel* library — total, elastic, inelastic,
nonelastic, fission and capture cross sections in long form.

Why this exists
---------------
Every evaluated library shipped today (`data/<lib>/xs/`) is **residual-production**
only: each row names the nuclide that comes *out*. That is the right shape for
activation and transmutation, and it is what `build_neutron_njoy.py` produces — it
deliberately drops elastic (2), inelastic (4, 51-91) and fission (18) because those
channels do not transmute the nucleus.

But attenuation, shielding and self-shielding need exactly the channels that build
discards, and the products are irrelevant:

    Phi(x) = Phi_0 * exp(-Sigma_tot * x),    Sigma_tot = N * sigma_tot

`data/meta/neutron_total/` was a first pass at this. It is limited to ENDF/B-VIII.1
MF=3 MT=1/2, capped at 20 MeV, and uses a *wide* schema (`xs_total_mb`,
`xs_elastic_mb`) that cannot grow a third channel without growing a third column.
Worse, raw MF=3 omits the resolved-resonance region entirely — precisely the region
that dominates shielding.

This build instead reuses the NJOY/OpenMC-processed pointwise data that
`build_neutron_njoy.py` already fetches: resonance-resolved, Doppler-broadened to
294 K, on the same grid OpenMC/MCNP/Serpent transport on, out to 150 MeV.

Output
------
    data/endfb-8.0/channels/n_{Element}.parquet
    Z int32, A int32, MT int32, reaction str, energy_MeV f64, xs_mb f64

Long form, so a new channel is new rows rather than a new column, and so EXFOR's
measured channels (issue #279) can be concatenated into the same view.

Note on MT=1
------------
The processed HDF5 carries no MT=1 — OpenMC sums the total on the fly. We
reconstruct it by summing every non-redundant reaction on the shared grid, which
reproduces Fe-56 to the published values (14.8 b total / 2.60 b capture at
0.0253 eV; 2.59 b total at 14 MeV).

Usage
-----
    nix develop -c .venv/bin/python scripts/build_channels.py --nuclides Fe56 Pb208
    nix develop -c .venv/bin/python scripts/build_channels.py          # whole library
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR  # noqa: E402

# Reuse the VIII.0 fetch/thin/inventory machinery rather than restating it.
from build_neutron_njoy import (  # noqa: E402
    COMPRESSION,
    TEMPERATURE,
    _ensure_native_libs,
    fetch_h5,
    list_nuclides,
    parse_nuclide_stem,
    thin_pointwise,
)
from fetch_endf_libs import mt_to_residual  # noqa: E402

from nucl_parquet._schemas import CANONICAL_XS_SCHEMA  # noqa: E402
from nucl_parquet.builder_stamp import manifest_path_for, write_builder_stamp  # noqa: E402
from nucl_parquet.state_vocabulary import SUM  # noqa: E402

LIBRARY = "endfb-8.0-channels"

# Transport channels, in ENDF MT numbering. Deliberately excludes the
# definitionally murky MT=27/101 "absorption": nonelastic (3) already carries the
# removal term for shielding, and is unambiguous (total - elastic).
CHANNELS: dict[int, str] = {
    1: "n,tot",
    2: "n,el",
    3: "n,nonel",
    4: "n,inl",
    18: "n,f",
    102: "n,g",
}

# Inelastic is stored as discrete levels when the summed MT=4 is absent.
_INELASTIC_LEVELS = range(51, 92)
# Fission is stored as its partials when the summed MT=18 is absent.
_FISSION_PARTIALS = (19, 20, 21, 38)


def _channel_curves(h5_bytes: bytes, stem: str) -> tuple:
    """Return (energy_eV, {MT: sigma_barns}) for the transport channels.

    Every curve is returned dense on the nuclide's shared energy grid, so the
    channels stay directly comparable before thinning.
    """
    import h5py
    import numpy as np

    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        if stem not in f:
            return None, {}
        g = f[stem]
        if TEMPERATURE not in g["energy"]:
            return None, {}
        energy = g["energy"][TEMPERATURE][:]
        reactions = g["reactions"]

        def dense(mt: int):
            """Expand one reaction onto the shared grid, honouring threshold_idx."""
            key = f"reaction_{mt:03d}"
            if key not in reactions:
                return None
            node = reactions[key]
            if TEMPERATURE not in node:
                return None
            data = node[TEMPERATURE]["xs"]
            start = int(data.attrs.get("threshold_idx", 0))
            out = np.zeros_like(energy)
            values = data[:]
            out[start : start + values.shape[0]] = values
            return out

        def dense_sum(mts) -> np.ndarray | None:
            acc = None
            for mt in mts:
                v = dense(mt)
                if v is None:
                    continue
                acc = v if acc is None else acc + v
            return acc

        # MT=1 is absent from the processed files; sum the fundamental reactions.
        total = np.zeros_like(energy)
        for key in reactions:
            node = reactions[key]
            if int(node.attrs.get("redundant", 0)):
                continue
            v = dense(int(node.attrs.get("mt", key.split("_")[1])))
            if v is not None:
                total += v

        elastic = dense(2)
        inelastic = dense(4)
        if inelastic is None:
            inelastic = dense_sum(_INELASTIC_LEVELS)
        fission = dense(18)
        if fission is None:
            fission = dense_sum(_FISSION_PARTIALS)
        capture = dense(102)

        curves = {1: total, 2: elastic, 4: inelastic, 18: fission, 102: capture}
        if elastic is not None:
            # Nonelastic is the removal term: everything that is not elastic scatter.
            curves[3] = np.clip(total - elastic, 0.0, None)
        return energy, {mt: c for mt, c in curves.items() if c is not None}


def extract_channels(h5_bytes: bytes, stem: str, z: int, a: int, tol: float) -> list[dict]:
    """Thin each transport channel and flatten it to long-form rows."""
    import numpy as np

    energy, curves = _channel_curves(h5_bytes, stem)
    if energy is None:
        return []

    rows: list[dict] = []
    for mt, sigma in sorted(curves.items()):
        positive = sigma > 0
        if not positive.any():
            continue
        e_pos = energy[positive]
        s_pos = sigma[positive]
        e_thin, s_thin = thin_pointwise(e_pos, s_pos, tol=tol)
        # Where MT determines a residual (capture, (n,p), (n,a) …) carry it, so the
        # row also answers "what produces Fe-57". Total/elastic/inelastic/fission
        # name none, and mt_to_residual returns None — which stays null, not 0.
        residual = mt_to_residual(mt, z, a, 0, 1)
        rz, ra = residual if residual else (None, None)
        rows.extend(
            {
                "library": LIBRARY,
                "kind": "channel",
                "projectile": "n",
                "proj_Z": 0,
                "proj_A": 1,
                "target_Z": z,
                "target_A": a,
                "MT": mt,
                "residual_Z": rz,
                "residual_A": ra,
                "state": SUM,
                "energy_MeV": float(e) * 1e-6,
                "xs_mb": float(s) * 1e3,
                "energy_err_MeV": None,
                "xs_err_mb": None,
                "source_entry": None,
                "author": None,
                "year": None,
            }
            for e, s in zip(np.asarray(e_thin), np.asarray(s_thin))
        )
    return rows


def write_element(out_dir: Path, sym: str, rows: list[dict]) -> Path:
    import polars as pl

    df = pl.DataFrame(
        rows,
        schema={c: getattr(pl, t) for c, t in CANONICAL_XS_SCHEMA.items()},
    ).sort("target_A", "MT", "energy_MeV")
    out_path = out_dir / f"n_{sym}.parquet"
    df.write_parquet(out_path, compression=COMPRESSION)
    return out_path


def build(
    out_dir: Path,
    nuclides: list[str] | None,
    tol: float,
    cache_dir: Path | None = None,
) -> None:
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet-build/1.0"

    if nuclides:
        stems = nuclides
    else:
        logger.info("Listing VIII.0 neutron inventory…")
        stems = list_nuclides(session)
        logger.info("Found %d nuclide files", len(stems))

    by_element: dict[str, list[dict]] = {}
    skipped = 0
    for n, stem in enumerate(stems, 1):
        parsed = parse_nuclide_stem(stem)
        if parsed is None:
            logger.info("skip (metastable/unknown): %s", stem)
            skipped += 1
            continue
        z, a, sym = parsed
        try:
            blob = fetch_h5(session, stem, cache_dir=cache_dir)
            rows = extract_channels(blob, stem, z, a, tol)
        except Exception as e:  # one bad nuclide must not sink the build
            logger.warning("  %s failed: %s", stem, e)
            skipped += 1
            continue
        if not rows:
            skipped += 1
            continue
        by_element.setdefault(sym, []).extend(rows)
        logger.info("  [%d/%d] %s: %d rows", n, len(stems), stem, len(rows))

    total_rows = 0
    for sym, rows in sorted(by_element.items()):
        path = write_element(out_dir, sym, rows)
        total_rows += len(rows)
        logger.info("wrote %s (%d rows)", path.name, len(rows))

    # Record which builder produced this, so a fix here that never reached the
    # data is detectable without re-downloading anything (#342).
    stamp_path = manifest_path_for(out_dir, LIBRARY)
    write_builder_stamp(stamp_path, Path(__file__), files_written=len(by_element))
    logger.info(
        "Done: %d elements, %d rows, %d skipped -> %s (stamped %s)",
        len(by_element),
        total_rows,
        skipped,
        out_dir,
        stamp_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "endfb-8.0" / "channels",
        help="output directory (default: data/endfb-8.0/channels)",
    )
    parser.add_argument("--nuclides", nargs="+", help="nuclide stems, e.g. Fe56 Pb208")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="thinning tolerance, relative (default 0.01)",
    )
    parser.add_argument("--cache-dir", type=Path, help="cache raw .h5 downloads here")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    _ensure_native_libs()
    build(args.out_dir, args.nuclides, args.tol, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
