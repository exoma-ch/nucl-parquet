"""Ingest NJOY-processed pointwise neutron cross sections into per-isotope Parquet.

Replaces the MF=3-only ENDF ingestion (`fetch_endf_libs.py --sublibrary n`) for
the neutron channels with the *authoritative NJOY-processed pointwise data* — the
exact data OpenMC / MCNP / Serpent transport on. This carries the resolved- and
unresolved-resonance regions (Doppler-broadened at 294 K) and the thermal point,
which the raw-MF=3 tabulation omits entirely: e.g. Nd-143(n,γ) in the old build
had zero points below 0.225 MeV, so the resonance region was silently missing and
downstream consumers had to reconstruct it themselves (see #263, hyrr#513).

Source
------
OpenMC-processed ENDF/B-VIII.0 HDF5, one file per nuclide, fetched lazily from the
GitHub mirror (no ~GB Box tarball, no auth):

    https://raw.githubusercontent.com/openmc-data-storage/ENDF-B-VIII.0-NNDC/main/h5_files/neutron/<Nuc>.h5

Each file exposes a shared energy grid per temperature at ``/<Nuc>/energy/<T>`` (eV)
and reaction cross sections at ``/<Nuc>/reactions/reaction_<MT>/<T>/xs`` (barns),
each starting at ``threshold_idx`` into the shared grid. We take **294 K**.

Channels
--------
Transmutation channels only — every MT whose residual ``(Z, A)`` differs from the
target: capture (102) plus threshold reactions (16/17/22/24/28/103/107…). Elastic
(2), inelastic-to-levels (4, 51–91) and fission (18) are excluded — they don't
transmute the nucleus (elastic/inelastic residual == target) and #260 already
removed the elastic-contamination class. ``redundant`` summed reactions are skipped.

Isomeric note
-------------
The processed transport HDF5 carries only the *total* channel per MT; the residual
metastable split (e.g. In-116m) lives in ENDF MF=8/9/10 and is **not** present here
(nor in the previous neutron build, which shipped zero ``state='m'`` rows). The
``state`` column is retained (always '' for now) so an MF=10 merge can be layered in
later without a schema change. Activation/dosimetry isomeric data is already served
by the IRDFF-2 and IAEA-Medical libraries.

Metastable *targets* (Ag110_m1, …) are skipped in this pass — the per-isotope shard
name ``n_<Sym><A>`` has no target-isomer degree of freedom, matching the existing
per-element schema which carries no target-state column.

Output
------
One per-element file: ``data/endfb-8.0/xs/n_<Sym>.parquet`` (e.g. ``n_Cu.parquet``),
holding every isotope of that element — identical layout, naming, and 6-column schema
(``target_A, residual_Z, residual_A, state, energy_MeV, xs_mb``) to the other 15
evaluated xs libraries. Per-element (not per-isotope) so the file matches the
``<proj>_<Symbol>`` convention every client uses to derive the target Z (rust/go parse
it from the filename); this makes it work across py/rs/ts/go with zero client code.
The per-isotope shards are also mirrored on Hugging Face (gerchowl/nucl-parquet-data)
for lazy browser fetches.

Usage
-----
    # Build a few nuclides into a staging dir (for testing):
    uv run python scripts/build_neutron_njoy.py --out /tmp/neutron --nuclides Nd143,In115,Fe56

    # Full in-repo build (all ~530 nuclides):
    uv run python scripts/build_neutron_njoy.py

On NixOS h5py/numpy need libz + libstdc++ on LD_LIBRARY_PATH; the module sets a
sensible default via `nix-build` if they are missing (see `_ensure_native_libs`).
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_neutron_njoy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

# Reuse the single source of truth for MT -> residual mapping.
from fetch_endf_libs import mt_to_residual  # noqa: E402

from nucl_parquet.builder_stamp import manifest_path_for, write_builder_stamp  # noqa: E402
from nucl_parquet.state_vocabulary import SUM  # noqa: E402

VIII0_RAW = "https://raw.githubusercontent.com/openmc-data-storage/ENDF-B-VIII.0-NNDC/main/h5_files/neutron"
# Large files (heavy actinides, ~100 MB) are Git-LFS-backed; raw.githubusercontent
# returns the pointer, so those must be fetched from the media endpoint instead.
VIII0_MEDIA = "https://media.githubusercontent.com/media/openmc-data-storage/ENDF-B-VIII.0-NNDC/main/h5_files/neutron"
VIII0_API = "https://api.github.com/repos/openmc-data-storage/ENDF-B-VIII.0-NNDC/contents/h5_files/neutron"
_LFS_POINTER = b"version https://git-lfs.github.com"
TEMPERATURE = "294K"
COMPRESSION = "zstd"

# Channels that do not transmute the nucleus (residual == target) or lack a single
# residual — excluded from a transmutation cross-section library.
_SKIP_MT = {2, 18}  # elastic, fission
# Inelastic-to-continuum (4) and discrete levels (51-91) keep the target nuclide.
_INELASTIC_RANGE = range(51, 92)

# Element symbols Z=1..118 (index 0 == Z=1).
_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds "
    "Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_ELEMENT_SYMBOLS)}
# Nuclide filename like "Nd143" or "Ag110_m1" (metastable target — skipped).
_NUC_RE = re.compile(r"^([A-Z][a-z]?)(\d+)(_m\d+)?$")


def _ensure_native_libs() -> None:
    """On NixOS, put libz/libstdc++ on LD_LIBRARY_PATH so h5py/numpy import.

    No-op if the imports already work (e.g. glibc distros, or the caller already
    exported LD_LIBRARY_PATH). Uses `nix-build '<nixpkgs>' -A zlib/gcc.cc.lib`.
    """
    import os

    try:
        import h5py  # noqa: F401
        import numpy  # noqa: F401

        return
    except ImportError:
        pass

    import subprocess

    extra: list[str] = []
    for attr, lib in (("zlib", "libz.so.1"), ("gcc.cc.lib", "libstdc++.so.6")):
        try:
            out = subprocess.run(
                ["nix-build", "<nixpkgs>", "-A", attr, "--no-out-link"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            store = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
            if store and (Path(store) / "lib" / lib).exists():
                extra.append(str(Path(store) / "lib"))
        except Exception:  # nix-build absent / offline — leave it to the caller
            pass
    if extra:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(extra + [os.environ.get("LD_LIBRARY_PATH", "")])
        # Re-exec once so the dynamic linker picks up the new path.
        if not os.environ.get("_NJOY_RELAUNCHED"):
            os.environ["_NJOY_RELAUNCHED"] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])


def _github_token() -> str | None:
    """A GitHub token for the API listing (env, else `gh auth token`).

    The unauthenticated Contents API allows only 60 requests/hour; an authenticated
    request gets 5000/hour. Only the *listing* is authenticated — the per-file raw/
    media fetches hit a public CDN and need no token.
    """
    import os
    import subprocess

    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
        tok = out.stdout.strip()
        return tok or None
    except Exception:
        return None


def list_nuclides(session) -> list[str]:
    """Return every neutron nuclide stem in the VIII.0 mirror (e.g. 'Nd143').

    The GitHub Contents API returns the whole directory (≤1000 entries) in one
    response — no pagination — so a single authenticated call lists all ~556.
    """
    headers = {}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = session.get(VIII0_API, headers=headers, timeout=30)
    resp.raise_for_status()
    batch = resp.json()
    if not isinstance(batch, list):
        raise RuntimeError(f"Unexpected listing response: {str(batch)[:200]}")
    return sorted(e["name"][:-3] for e in batch if e["name"].endswith(".h5"))


def parse_nuclide_stem(stem: str) -> tuple[int, int, str] | None:
    """('Nd143') -> (Z=60, A=143, sym='Nd'); metastable targets ('Ag110_m1') -> None."""
    m = _NUC_RE.match(stem)
    if not m:
        return None
    sym, a, meta = m.group(1), int(m.group(2)), m.group(3)
    if meta:  # metastable target — no shard slot in the n_<Sym><A> scheme
        return None
    z = _SYMBOL_TO_Z.get(sym)
    if z is None:
        return None
    return z, a, sym


def stem_skip_reason(stem: str) -> str:
    """Why `parse_nuclide_stem` returned None — and whether that is acceptable.

    It returns None for three unrelated reasons, and only one of them is a
    correct outcome. Collapsing them is the #329 mistake one level down: a
    metastable target genuinely has no shard slot, but a stem whose element
    symbol is unknown means `_SYMBOL_TO_Z` has fallen behind the upstream
    inventory, and a stem that does not match `_NUC_RE` at all means the naming
    convention moved. Those two are data loss wearing a skip's clothing.

    Returns 'metastable', 'unknown-symbol' or 'unparseable'. Callers treat only
    the first as expected.

    All 22 stems the VIII.0 inventory currently skips are metastable, so the
    other two branches are latent today — which is exactly when to separate
    them, rather than after an inventory refresh quietly drops an element.
    """
    m = _NUC_RE.match(stem)
    if m is None:
        return "unparseable"
    if m.group(3):
        return "metastable"
    if _SYMBOL_TO_Z.get(m.group(1)) is None:
        return "unknown-symbol"
    return "metastable"  # pragma: no cover — parse_nuclide_stem would have succeeded


def thin_pointwise(energy, xs, tol: float = 0.01):
    """Thin a pointwise σ(E) curve, keeping ≤ ``tol`` relative error under *both*
    lin-lin and log-log interpolation.

    Top-down Ramer–Douglas–Peucker: for each segment, score every interior point
    by the *larger* of its two reconstruction errors — linear-in-(E, σ) and
    log-log — and split at the worst point if it exceeds ``tol``. Recursing until
    both errors are within ``tol`` everywhere means the shipped grid reproduces
    the curve to within ``tol`` no matter which interpolation law the consumer
    picks: the natural ``np.interp`` linear default *and* ENDF INT=5 log-log both
    land within tolerance.

    Guarding lin-lin (not just log-log, as an earlier version did) is what keeps
    the sparse 1/v thermal region correct for linear consumers: a pure-log-log
    thin drops every interior thermal point (1/v is a straight log-log line), so a
    linear reader over-predicts the flux-averaged thermal σ by ~36× (see #271).
    A lin-lin-accurate 1/v grid needs ~9 points/decade, which this restores.
    Resonance peaks survive either way (they are the max-error points). Vectorized
    per segment → O(n log n) typical. Requires ``xs > 0`` (caller filters).
    """
    import numpy as np

    n = len(energy)
    if n <= 2:
        return energy, xs
    e = np.asarray(energy, dtype=float)
    s = np.asarray(xs, dtype=float)
    le = np.log(e)
    ls = np.log(s)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        if le[hi] <= le[lo]:  # degenerate (duplicate energies) — keep interior verbatim
            keep[lo + 1 : hi] = True
            continue
        j = np.arange(lo + 1, hi)
        # log-log reconstruction error at each interior point
        t_log = (le[j] - le[lo]) / (le[hi] - le[lo])
        interp_ls = ls[lo] + t_log * (ls[hi] - ls[lo])
        rel_log = np.abs(np.expm1(interp_ls - ls[j]))  # |σ_loglog/σ - 1|
        # lin-lin reconstruction error (linear in E, linear in σ)
        t_lin = (e[j] - e[lo]) / (e[hi] - e[lo])
        interp_s = s[lo] + t_lin * (s[hi] - s[lo])
        rel_lin = np.abs(interp_s / s[j] - 1.0)  # |σ_linlin/σ - 1|
        rel = np.maximum(rel_log, rel_lin)
        k = int(np.argmax(rel))
        if rel[k] > tol:
            split = lo + 1 + k
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    idx = np.where(keep)[0]
    return e[idx], s[idx]


def extract_nuclide(h5_bytes: bytes, stem: str, z: int, a: int, tol: float) -> list[dict]:
    """Extract 294 K transmutation channels from one nuclide HDF5, thinned.

    A residual nuclide is produced by *every* channel that leaves it behind, and
    several MTs can share one residual: the discrete-level partials of a reaction
    (e.g. MT800 (n,α₀) and MT801 (n,α₁) both leave Li-7), and different reactions
    that reach the same nucleus (e.g. (n,np)/MT28 and (n,d)/MT104 both leave
    Z-1,A-1). We keep the *non-redundant* (fundamental) channels — skipping the
    redundant summary MTs that would double-count — and **sum** every fundamental
    contribution to a given residual on the shared union energy grid before
    thinning. Summing (not concatenating) is essential: appending the partials
    produced interleaved duplicate-energy points that no interpolation can read
    (e.g. B-10(n,α) read ~12000 b / ~240 b at thermal vs the true 3844 b).
    """
    import h5py
    import numpy as np

    rows: list[dict] = []
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        nuc = f[stem]
        if int(nuc.attrs.get("metastable", 0)) != 0:
            return rows
        if TEMPERATURE not in nuc["energy"]:
            logger.warning("%s: no %s energy grid", stem, TEMPERATURE)
            return rows
        egrid = nuc["energy"][TEMPERATURE][()]  # eV
        # Accumulate every fundamental channel's σ(E) onto the full union grid,
        # summed per residual nuclide.
        by_residual: dict[tuple[int, int], np.ndarray] = {}
        for key, rx in nuc["reactions"].items():
            mt = int(rx.attrs.get("mt", key.split("_")[1]))
            if mt in _SKIP_MT or mt == 4 or mt in _INELASTIC_RANGE:
                continue
            if int(rx.attrs.get("redundant", 0)) != 0:  # summary MT — its partials carry the data
                continue
            residual = mt_to_residual(mt, z, a, 0, 1)
            if residual is None:
                continue
            res_z, res_a = residual
            if (res_z, res_a) == (z, a):  # non-transmuting (shouldn't happen post-filter)
                continue
            if TEMPERATURE not in rx:
                continue
            xsd = rx[TEMPERATURE]["xs"]
            xs_b = np.asarray(xsd[()], dtype=float)
            xs_b[~np.isfinite(xs_b) | (xs_b >= 1e30) | (xs_b < 0)] = 0.0  # drop sentinels
            thr = int(xsd.attrs.get("threshold_idx", 0))
            acc = by_residual.get((res_z, res_a))
            if acc is None:
                acc = np.zeros(len(egrid))
                by_residual[(res_z, res_a)] = acc
            acc[thr : thr + len(xs_b)] += xs_b
        for (res_z, res_a), total in sorted(by_residual.items()):
            good = total > 0  # keep the reaction's support (contiguous above threshold)
            if good.sum() < 2:
                continue
            e_ev, xs_b = egrid[good], total[good]
            e_ev, xs_b = thin_pointwise(e_ev, xs_b, tol)
            for ee, ss in zip(e_ev, xs_b):
                rows.append(
                    {
                        # Schema matches the other evaluated xs libraries exactly, so
                        # the shard auto-unions into the loader's `xs` view. Target
                        # element is implied by residual_Z + channel (and the filename),
                        # so no target_Z column — same as jendl/tendl/endfb-8.1.
                        "target_A": a,
                        "residual_Z": res_z,
                        "residual_A": res_a,
                        # The processed transport HDF5 carries only the
                        # total per MT, so every row here is summed over states.
                        "state": SUM,
                        "energy_MeV": float(ee) * 1e-6,
                        "xs_mb": float(ss) * 1e3,
                    }
                )
    return rows


def write_element(out_dir: Path, sym: str, rows: list[dict]) -> Path:
    """Write one per-element file ``n_<Sym>.parquet`` holding every isotope of that
    element. Per-element (not per-isotope) so the file matches the ``<proj>_<Symbol>``
    convention all clients use to derive the target Z (rust/go parse it from the
    filename), identical to the other 15 evaluated xs libraries."""
    import polars as pl

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
    ).sort("target_A", "residual_Z", "residual_A", "energy_MeV")
    out_path = out_dir / f"n_{sym}.parquet"
    df.write_parquet(out_path, compression=COMPRESSION)
    return out_path


def _get_with_retry(session, url: str, timeout: int, max_retries: int = 6) -> bytes:
    """GET ``url`` with exponential backoff on 429/5xx (raw.githubusercontent throttles
    bursts of requests). Honors ``Retry-After`` when present."""
    for attempt in range(max_retries):
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = int(resp.headers.get("Retry-After", 0)) or min(2**attempt, 60)
            logger.info(
                "  %s on %s — backing off %ds (attempt %d)", resp.status_code, url.rsplit("/", 1)[-1], wait, attempt + 1
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.content
    resp.raise_for_status()  # exhausted retries — surface the last error
    return resp.content


def fetch_h5(session, stem: str, cache_dir: Path | None = None) -> bytes:
    """Fetch one nuclide HDF5, transparently following Git-LFS to the media endpoint.

    With ``cache_dir`` the raw bytes are cached on disk, so a re-thin/rebuild reuses
    the ~1 GB of upstream HDF5 instead of re-hammering raw.githubusercontent (which
    throttles bursts with 429). The cache key is the nuclide stem — VIII.0 is a
    frozen release, so cached bytes never go stale."""
    if cache_dir is not None:
        cached = cache_dir / f"{stem}.h5"
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()
    content = _get_with_retry(session, f"{VIII0_RAW}/{stem}.h5", timeout=180)
    if content[: len(_LFS_POINTER)] == _LFS_POINTER:  # LFS pointer -> real content
        content = _get_with_retry(session, f"{VIII0_MEDIA}/{stem}.h5", timeout=600)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{stem}.h5").write_bytes(content)
    return content


class LostNuclides(RuntimeError):
    """A build was asked for nuclides it did not produce. Raised, not logged."""


@dataclasses.dataclass
class NuclideLedger:
    """What a nuclide-by-nuclide build asked for, and what it actually got.

    Both VIII.0 builders walk the same ~556-stem inventory, and both used to
    collapse every non-outcome into one integer::

        skipped += 1     # metastable target - correct
        skipped += 1     # fetch raised      - a defect
        skipped += 1     # produced no rows  - a defect

    That integer is how ``endfb-8.0/channels/`` came to ship without plutonium
    (#329). A throttle window during the #280 build hit fifteen *consecutive*
    stems - Pu236-Pu246 and Ra223-Ra226, indices 356-370 of the build order,
    with Pt198 present before and Rb85 present after. ``_get_with_retry``
    exhausted its six attempts, the exception was caught and downgraded to a
    warning, and the run logged ``"37 skipped"`` and exited 0. Twenty-two of
    those thirty-seven were correct - a metastable target has no ``n_<Sym>``
    slot - and fifteen were the defect. Nothing downstream could tell them
    apart, so a *transient* failure became a *permanent* gap that outlived every
    release since.

    So the counter is replaced by a ledger that keeps the distinction the count
    destroyed - the same "expected absence vs unexplained absence" split #383
    made for provenance and #386 for band attribution:

    * ``expected_skips`` - asked, correctly produced nothing.
    * ``failed`` - asked, raised. Usually transient, and transient is exactly
      the case that must not pass silently, because nothing retries it later.
    * ``empty`` - asked, parsed, produced no rows. Either a real upstream gap or
      a parser gone quiet; both need a human.

    ``failed`` and ``empty`` are *losses*. A build with any of them that still
    writes a stamped, apparently-complete library is the #334 defect wearing the
    words "one bad nuclide must not sink the build".
    """

    #: Stems that legitimately produce nothing, with the reason.
    expected_skips: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    #: Stems whose fetch or parse raised, with the error text.
    failed: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    #: Stems that parsed cleanly and yielded no rows.
    empty: list[str] = dataclasses.field(default_factory=list)
    #: Stems that produced rows.
    built: list[str] = dataclasses.field(default_factory=list)

    def skip(self, stem: str, why: str) -> None:
        self.expected_skips.append((stem, why))
        logger.info("  skip (%s): %s", why, stem)

    def fail(self, stem: str, error: object) -> None:
        self.failed.append((stem, str(error)[:200]))
        logger.error("  %s FAILED: %s", stem, error)

    def nothing(self, stem: str) -> None:
        self.empty.append(stem)
        logger.warning("  %s produced no rows", stem)

    def ok(self, stem: str) -> None:
        self.built.append(stem)

    @property
    def lost(self) -> list[str]:
        """Every stem that was asked for and did not arrive, by name."""
        return sorted([s for s, _ in self.failed] + self.empty)

    def summary(self) -> str:
        return (
            f"{len(self.built)} built, {len(self.expected_skips)} skipped as expected, "
            f"{len(self.failed)} failed, {len(self.empty)} empty"
        )

    def check(self, library: str, allow_missing: bool) -> None:
        """Report the run, then refuse it if anything asked-for went missing.

        ``allow_missing`` surveys the whole run instead of stopping at the first
        loss - and the caller still exits non-zero, exactly as
        ``migrate_xs_schema.py --skip-unmigratable`` does. Surveying the damage
        is not the same as accepting it.
        """
        logger.info("%s: %s", library, self.summary())
        if not self.lost:
            return
        detail = "\n  ".join(
            [f"{stem}: {err}" for stem, err in sorted(self.failed)]
            + [f"{stem}: produced no rows" for stem in sorted(self.empty)]
        )
        message = (
            f"{library}: {len(self.lost)} nuclide(s) were requested and did not arrive:\n  {detail}\n\n"
            "Named rather than counted, because a count is what hid #329 for months: fifteen "
            "throttled downloads and twenty-two legitimate metastable skips were identical "
            "inside one 'skipped' integer, and endfb-8.0/channels shipped without plutonium. "
            "A transient failure is fine; shipping a library that silently lacks an element is "
            "not. Re-run - the --cache-dir makes a retry cheap - or, if the absence is real, "
            "record it before shipping."
        )
        if allow_missing:
            logger.error("%s", message)
            return
        raise LostNuclides(message)


def build(
    out_dir: Path,
    nuclides: list[str] | None,
    tol: float,
    cache_dir: Path | None = None,
    allow_missing: bool = False,
) -> NuclideLedger:
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
    ledger = NuclideLedger()
    for stem in stems:
        parsed = parse_nuclide_stem(stem)
        if parsed is None:
            reason = stem_skip_reason(stem)
            if reason == "metastable":
                ledger.skip(stem, "metastable target — no n_<Sym> shard slot")
            else:
                # Not a skip. The inventory offered a nuclide this build cannot
                # name, which means the symbol table or the upstream convention
                # has moved and an element is about to go missing quietly (#329).
                ledger.fail(stem, f"stem not understood ({reason}); _SYMBOL_TO_Z or _NUC_RE is behind the inventory")
            continue
        z, a, sym = parsed
        cache_hit = cache_dir is not None and (cache_dir / f"{stem}.h5").exists()
        # Gentle inter-request pacing keeps us under raw.githubusercontent's burst limit
        # (skip it on cache hits — no network round-trip).
        if not cache_hit:
            time.sleep(0.2)
        try:
            content = fetch_h5(session, stem, cache_dir)
            rows = extract_nuclide(content, stem, z, a, tol)
        except Exception as e:  # noqa: BLE001
            ledger.fail(stem, e)
            continue
        if not rows:
            ledger.nothing(stem)
            continue
        by_element.setdefault(sym, []).extend(rows)
        ledger.ok(stem)
        logger.info("%-8s: %d rows", stem, len(rows))

    # Before anything is written. A stamped library that silently lacks an
    # element is worse than no library, because the stamp says it is complete
    # (#329) — so the refusal has to come first.
    ledger.check("endfb-8.0", allow_missing)

    written = 0
    for sym, rows in sorted(by_element.items()):
        path = write_element(out_dir, sym, rows)
        written += 1
        logger.info("-> %s (%d rows, %.0f KB)", path.name, len(rows), path.stat().st_size / 1024)
    # Record which builder produced this, so a fix here that never reached the
    # data is detectable without re-downloading anything (#342).
    stamp_path = manifest_path_for(out_dir, "endfb-8.0")
    write_builder_stamp(stamp_path, Path(__file__), files_written=written)
    logger.info(
        "Done: %d per-element files written, %s, out=%s (stamped %s)",
        written,
        ledger.summary(),
        out_dir,
        stamp_path,
    )
    return ledger


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "endfb-8.0" / "xs",
        help="Output directory for shards (default: data/endfb-8.0/xs/)",
    )
    ap.add_argument(
        "--nuclides",
        type=str,
        default=None,
        help="Comma-separated stems to build (e.g. Nd143,In115). Omit for the full inventory.",
    )
    ap.add_argument("--tol", type=float, default=0.01, help="Thinning tolerance, lin-lin & log-log (default 1%%)")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache raw HDF5 here to avoid re-fetching on a re-thin/rebuild (VIII.0 is frozen).",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="survey the whole run instead of refusing at the end. The library is still written and the process still exits non-zero — this reports the damage, it does not accept it (same contract as migrate_xs_schema.py --skip-unmigratable).",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    _ensure_native_libs()
    nuclides = [s.strip() for s in args.nuclides.split(",")] if args.nuclides else None
    try:
        ledger = build(args.out, nuclides, args.tol, args.cache_dir, allow_missing=args.allow_missing)
    except LostNuclides:
        logger.exception("refusing to report success on an incomplete library")
        return 1
    # Reached only under --allow-missing, which writes the library anyway. The
    # exit code is what stops a survey being mistaken for a clean build (#329).
    return 1 if ledger.lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
