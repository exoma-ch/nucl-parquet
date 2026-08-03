"""Ingest EXFOR Master Files (.x4) into the `exfor/` and `exfor-channels/` layouts.

Replaces the `dataexplorer/api/reactions/xs` ingestion in `scripts/fetch_exfor.py`,
which is hardwired to residual-production mode — the endpoint echoes back
`"mt": "10", "reaction": "<proj>,x"` regardless of what you send, so it structurally
cannot return the MT-coded reaction data that is the bulk of neutron EXFOR (#279).
That path yields 12 points for 27Al(n,a)24Na and zero for 18O(p,n)18F; this one
yields 543 and 535.

Source: https://github.com/IAEA-NDS/exfor_master (CC-BY-4.0), pinned by release tag.
That is the provenance `data/licenses.toml::[libraries.exfor]` already declares —
the current API path contradicts it ("not the live retrieval system").

Two products, split on *residual absence*:

    exfor/           residual-production (unchanged schema; existing queries work)
    exfor-channels/  transport channels — (n,tot), (n,el), (n,inl), (n,f) — which
                     name no outgoing nuclide and so cannot share a table with
                     production data without all collapsing to residual (0, 0).
                     Concatenates with `data/endfb-8.0/channels/` (build_channels.py)
                     to compare measurement against evaluation.

Fetch the source first (259 MB, no clone needed):

    curl -sL https://codeload.github.com/IAEA-NDS/exfor_master/legacy.zip/refs/tags/Backup-2026-06-29 -o exfor_master.zip
    python -c "import zipfile; zipfile.ZipFile('exfor_master.zip').extractall('x4src')"

Usage:
    python scripts/fetch_exfor_master.py x4src/*/exforall data/ [--limit N]

Full corpus: 27,070 entries -> ~4.5M rows in ~40 s.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import re
import sys

# ---------------------------------------------------------------- X4 constants

FIELD_W = 11
FIELDS_PER_LINE = 6

# SF8/SF9 modifiers that mean "not an absolute point cross-section vs energy".
# Counts from a full-corpus survey are in comments.
_REJECT_MODIFIERS = {
    "MXW",  # 4515 Maxwellian-averaged
    "SPA",  # 3974 spectrum-averaged
    "FIS",  # 1500 fission-spectrum-averaged
    "RTH",  # 2468 thermal-spectrum
    "AV",  # 2187 averaged over an energy range
    "TT",  # 1632 thick-target yield
    "REL",  # 4263 relative, not absolute
    "DERIV",  # 3764 derived, not measured
    "EVAL",  # 1106 evaluated, not experimental
    "RECOM",  # 1586 recommended
    "RAW",  # 560  uncorrected
    "SDT",  # single differential
    "MSC",  # 982  miscellaneous
}

# Energy units -> MeV
_E_SCALE = {
    "MEV": 1.0,
    "KEV": 1e-3,
    "EV": 1e-6,
    "GEV": 1e3,
    "MILLI-EV": 1e-9,
    "MILLIEV": 1e-9,
}

# Cross-section units -> mb
_XS_SCALE = {
    "MB": 1.0,
    "B": 1e3,
    "BARNS": 1e3,
    "MICRO-B": 1e-3,
    "MU-B": 1e-3,
    "MUB": 1e-3,
    "NO-DIM": None,  # dimensionless -> reject
    "NB": 1e-6,
    "PER-CENT": None,
}

_ELEMENTS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()
Z_BY_SYMBOL = {s.upper(): i for i, s in enumerate(_ELEMENTS)}

# Incident-projectile code (SF2 head) -> nucl-parquet shorthand.
PROJECTILE = {"N": "n", "P": "p", "D": "d", "T": "t", "HE3": "h", "A": "a"}

# X4 outgoing-particle code (SF3) -> ENDF MT number.
#
# Needed because 82% of EXFOR SIG rows describe reactions with no residual nuclide
# ((n,tot), (n,f), (n,el), (n,inl)). Without this they all collapse to
# residual_Z = residual_A = 0 and become mutually indistinguishable.
MT_BY_PROCESS = {
    "TOT": 1,
    "EL": 2,
    "NON": 3,
    "INL": 4,
    "X": 5,
    "2N": 16,
    "3N": 17,
    "F": 18,
    "4N": 37,
    "N+A": 22,
    "N+P": 28,
    "2N+P": 41,
    "ABS": 27,
    "G": 102,
    "P": 103,
    "D": 104,
    "T": 105,
    "HE3": 106,
    "A": 107,
    "2P": 111,
    "2A": 108,
    "N+2A": 24,
    "3N+F": 21,
}

# Channels that name no residual nuclide — they belong in the transport-channel
# product (see scripts/build_channels.py), not in residual-production tables.
TRANSPORT_MT = {1, 2, 3, 4, 18, 102}

# X4 numbers may use ENDF-style implicit exponents: "1.23+5" == 1.23e5.
_IMPLICIT_EXP = re.compile(r"^([+-]?[\d.]+)([+-]\d+)$")


def parse_number(tok: str) -> float | None:
    tok = tok.strip()
    if not tok:
        return None
    try:
        return float(tok)
    except ValueError:
        pass
    m = _IMPLICIT_EXP.match(tok)
    if m:
        try:
            return float(f"{m.group(1)}e{m.group(2)}")
        except ValueError:
            return None
    return None


def split_fields(line: str, n: int) -> list[str]:
    """Split an X4 record line into its fixed-width 11-char fields."""
    return [line[i * FIELD_W : (i + 1) * FIELD_W] for i in range(n)]


def split_top(s: str) -> list[str]:
    """Split a REACTION body on commas at paren-depth 0, stopping at the closer."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


_NUCLIDE = re.compile(r"^(\d+)-([A-Z]{1,3})-(\d+)(?:-([A-Z0-9]+))?$")


def parse_nuclide(tok: str) -> tuple[int, int, str] | None:
    """'11-NA-24-L' -> (11, 24, 'l');  '13-AL-27' -> (13, 27, '')."""
    m = _NUCLIDE.match(tok.strip().upper())
    if not m:
        return None
    z, a = int(m.group(1)), int(m.group(3))
    state = (m.group(4) or "").lower()
    # X4 uses G/M/M1/M2 for ground/metastable; L means "level" (isomer unresolved).
    if state not in ("", "g", "m", "m1", "m2", "l"):
        state = "m" if state.startswith("M") else ""
    return z, a, state


# ------------------------------------------------------------ section parsing


class Section:
    """A parsed COMMON or DATA block: per-field name, pointer, unit, and column values."""

    __slots__ = ("names", "pointers", "units", "rows")

    def __init__(self, names, pointers, units, rows):
        self.names = names
        self.pointers = pointers
        self.units = units
        self.rows = rows

    def column(self, name: str, pointer: str) -> tuple[int, str] | None:
        """Resolve a field by name, preferring an exact pointer match over a blank one."""
        fallback = None
        for i, (nm, pt) in enumerate(zip(self.names, self.pointers)):
            if nm != name:
                continue
            if pt == pointer:
                return i, self.units[i]
            if pt == "":
                fallback = (i, self.units[i])
        return fallback


def parse_section(lines: list[str], start: int, nfields: int, end_kw: str) -> tuple[Section, int]:
    """Parse a COMMON/DATA block starting at the line after its header."""
    per_rec = max(1, math.ceil(nfields / FIELDS_PER_LINE))

    def gather(idx: int) -> tuple[list[str], int]:
        buf: list[str] = []
        for k in range(per_rec):
            ln = lines[idx + k] if idx + k < len(lines) else ""
            buf.extend(split_fields(ln, FIELDS_PER_LINE))
        return buf[:nfields], idx + per_rec

    head_raw, i = gather(start)
    unit_raw, i = gather(i)

    names = [h[:10].strip() for h in head_raw]
    pointers = [h[10:11].strip() for h in head_raw]
    units = [u.strip().upper() for u in unit_raw]

    rows: list[list[str]] = []
    while i < len(lines) and not lines[i].startswith(end_kw):
        rec, i = gather(i)
        if any(c.strip() for c in rec):
            rows.append(rec)
    return Section(names, pointers, units, rows), i


_SECTION_HDR = re.compile(r"^(COMMON|DATA)\s+(\d+)\s+(\d+)\s*$")
_BIB_KW = re.compile(r"^([A-Z][A-Z0-9-]*)\s*(.)(.*)$")


def parse_subentry(lines: list[str]) -> dict:
    """Extract REACTION codes (by pointer), AUTHOR, YEAR, COMMON and DATA from a subentry."""
    out: dict = {"reactions": {}, "author": None, "year": None, "common": None, "data": None}
    i = 0
    last_kw: str | None = None
    while i < len(lines):
        ln = lines[i]
        m = _SECTION_HDR.match(ln)
        if m:
            kind, nfields = m.group(1), int(m.group(2))
            sec, i = parse_section(lines, i + 1, nfields, "END" + kind)
            out["common" if kind == "COMMON" else "data"] = sec
            continue

        if ln[:11].strip() and not ln.startswith(("ENDBIB", "NOCOMMON", "ENDSUBENT", "ENDDATA")):
            kw = ln[:10].strip()
            pointer = ln[10:11].strip()
            body = ln[11:]
            if kw:
                keyword = kw
            elif pointer and last_kw:
                # A pointered entry continuing the previous keyword: the keyword
                # field is blank and only the pointer column is set. Dropping
                # these loses every REACTION after the first in a pointered subentry.
                keyword = last_kw
            else:
                i += 1
                continue
            last_kw = keyword

            j = i + 1
            while j < len(lines) and lines[j][:11].strip() == "":
                body += " " + lines[j][11:].strip()
                j += 1
            if keyword == "REACTION":
                out["reactions"][pointer] = body.strip()
            elif keyword == "AUTHOR":
                out["author"] = body.strip().lstrip("(").rstrip(")")
            elif keyword == "REFERENCE" and out["year"] is None:
                y = re.search(r"\b(19\d{2}|20\d{2})\b", body)
                if y:
                    out["year"] = int(y.group(1))
            i = j
            continue
        i += 1
    return out


def iter_subentries(text: str):
    lines = text.splitlines()
    cur: list[str] | None = None
    sub_id = None
    for ln in lines:
        if ln.startswith("SUBENT"):
            cur = []
            sub_id = ln[14:22].strip()
            continue
        if ln.startswith("ENDSUBENT"):
            if cur is not None and sub_id:
                yield sub_id, cur
            cur = None
            continue
        if cur is not None:
            cur.append(ln)


# ------------------------------------------------------------------ conversion


class Stats(collections.Counter):
    pass


def convert_entry(path: pathlib.Path, stats: Stats) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stats["file_unreadable"] += 1
        return []

    subs = list(iter_subentries(text))
    if not subs:
        stats["no_subentries"] += 1
        return []

    entry_id = path.stem
    header = parse_subentry(subs[0][1])
    entry_author = header["author"]
    entry_year = header["year"]
    entry_common = header["common"]

    rows: list[dict] = []
    for sub_id, sub_lines in subs[1:]:
        s = parse_subentry(sub_lines)
        if not s["reactions"]:
            continue
        data = s["data"]
        if data is None or not data.rows:
            stats["subent_no_data"] += 1
            continue

        author = s["author"] or entry_author
        year = s["year"] or entry_year

        for pointer, code in s["reactions"].items():
            body = code[1:] if code.startswith("(") else code
            parts = split_top(body)
            if len(parts) < 3:
                stats["rx_unparseable"] += 1
                continue
            if parts[2].strip() != "SIG":
                stats["rx_not_sig"] += 1
                continue
            mods = {p.strip() for p in parts[3:] if p.strip()}
            if mods & _REJECT_MODIFIERS:
                stats["rx_rejected_modifier"] += 1
                continue

            head = parts[0].strip()
            m = re.match(r"^([\dA-Z-]+)\(([^,]+),([^)]*)\)(.*)$", head)
            if not m:
                stats["rx_head_unparseable"] += 1
                continue
            target = parse_nuclide(m.group(1))
            if target is None:
                stats["target_unparseable"] += 1
                continue
            proj = PROJECTILE.get(m.group(2).strip().upper())
            if proj is None:
                stats["projectile_skipped"] += 1
                continue
            residual_token = m.group(4).strip()
            residual = parse_nuclide(residual_token) if residual_token else None

            # 'ELEM/MASS' means the residual is not fixed for the subentry: it varies
            # per row via ELEMENT (Z) and MASS (A) data columns.
            elem_mass = residual is None and "ELEM" in residual_token.upper()

            tz, ta, _ = target
            if residual is not None:
                rz, ra, state = residual
            else:
                rz = ra = 0
                state = ""

            process = m.group(3).strip().upper()
            emitted = _emit_rows(
                data,
                entry_common,
                s["common"],
                pointer,
                entry_id,
                sub_id,
                proj,
                process,
                tz,
                ta,
                rz,
                ra,
                state,
                author,
                year,
                rows,
                stats,
                elem_mass,
            )
            if not emitted:
                stats["rx_no_usable_points"] += 1
    return rows


def _to_mb(value: float | None, unit: str, reference_mb: float) -> float | None:
    """Convert an uncertainty to mb. PER-CENT is relative to the datum it annotates."""
    if value is None:
        return None
    if unit == "PER-CENT":
        return abs(reference_mb * value / 100.0)
    scale = _XS_SCALE.get(unit)
    return None if scale is None else abs(value * scale)


def _lookup_scalar(sections: list[Section | None], name: str, pointer: str) -> tuple[float, str] | None:
    """Find a single-valued field in COMMON sections (subentry first, then entry)."""
    for sec in sections:
        if sec is None or not sec.rows:
            continue
        col = sec.column(name, pointer)
        if col is None:
            continue
        idx, unit = col
        val = parse_number(sec.rows[0][idx])
        if val is not None:
            return val, unit
    return None


def _emit_rows(
    data,
    entry_common,
    sub_common,
    pointer,
    entry_id,
    sub_id,
    proj,
    process,
    tz,
    ta,
    rz,
    ra,
    state,
    author,
    year,
    out,
    stats,
    elem_mass=False,
) -> bool:
    commons = [sub_common, entry_common]

    # Per-row residual columns, used only for ELEM/MASS subentries.
    em_z = data.column("ELEMENT", pointer) if elem_mass else None
    em_a = data.column("MASS", pointer) if elem_mass else None
    em_iso = data.column("ISOMER", pointer) if elem_mass else None
    if elem_mass and (em_z is None or em_a is None):
        stats["elem_mass_no_columns"] += 1
        return False

    xs_col = data.column("DATA", pointer)
    if xs_col is None:
        stats["no_data_column"] += 1
        return False
    xs_idx, xs_unit = xs_col
    xs_scale = _XS_SCALE.get(xs_unit)
    if xs_scale is None:
        stats["bad_xs_unit"] += 1
        return False

    en_col = data.column("EN", pointer)
    en_common = None if en_col else _lookup_scalar(commons, "EN", pointer)
    if en_col is None and en_common is None:
        stats["no_energy"] += 1
        return False

    # Uncertainty: prefer the explicit total, then generic, then statistical.
    err_idx = err_unit = None
    for cand in ("DATA-ERR", "ERR-T", "ERR-S"):
        c = data.column(cand, pointer)
        if c is not None:
            err_idx, err_unit = c
            break
    err_common = None
    if err_idx is None:
        for cand in ("DATA-ERR", "ERR-T"):
            err_common = _lookup_scalar(commons, cand, pointer)
            if err_common is not None:
                break

    een_idx = een_unit = None
    c = data.column("EN-ERR", pointer)
    if c is not None:
        een_idx, een_unit = c

    n_before = len(out)
    for rec in data.rows:
        # --- residual: fixed for the subentry, or per-row for ELEM/MASS
        if elem_mass:
            z_val = parse_number(rec[em_z[0]])
            a_val = parse_number(rec[em_a[0]])
            if z_val is None or a_val is None:
                continue
            row_rz, row_ra = int(z_val), int(a_val)
            row_state = ""
            if em_iso is not None:
                iso = parse_number(rec[em_iso[0]])
                if iso is not None:
                    row_state = {0: "g", 1: "m", 2: "m2"}.get(int(iso), "")
        else:
            row_rz, row_ra, row_state = rz, ra, state

        # --- energy
        if en_col is not None:
            e_raw = parse_number(rec[en_col[0]])
            e_unit = en_col[1]
        else:
            e_raw, e_unit = en_common
        if e_raw is None:
            continue
        e_scale = _E_SCALE.get(e_unit)
        if e_scale is None:
            stats["bad_energy_unit"] += 1
            continue
        energy = e_raw * e_scale

        # --- cross-section
        xs_raw = parse_number(rec[xs_idx])
        if xs_raw is None:
            continue
        xs = xs_raw * xs_scale
        if xs <= 0:
            continue

        # --- uncertainties (PER-CENT is relative to the value)
        if err_idx is not None:
            xs_err = _to_mb(parse_number(rec[err_idx]), err_unit, xs)
        elif err_common is not None:
            xs_err = _to_mb(err_common[0], err_common[1], xs)
        else:
            xs_err = None

        e_err = None
        if een_idx is not None:
            v = parse_number(rec[een_idx])
            scale = _E_SCALE.get(een_unit)
            if v is not None and scale is not None:
                e_err = v * scale

        out.append(
            {
                "projectile": proj,
                # Match the existing entry-subentry-pointer convention, e.g. "10186-002-0".
                "exfor_entry": f"{sub_id[:5]}-{sub_id[5:8]}-{pointer or '0'}",
                "reaction": f"{proj.upper() if proj != 'h' else 'HE3'},{process}",
                "MT": MT_BY_PROCESS.get(process),
                "target_Z": tz,
                "target_A": ta,
                "residual_Z": row_rz,
                "residual_A": row_ra,
                "state": row_state,
                "energy_MeV": energy,
                "energy_err_MeV": e_err,
                "xs_mb": xs,
                "xs_err_mb": xs_err,
                "author": author,
                "year": year,
            }
        )
    return len(out) > n_before


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path, help="exforall/ directory of exfor_master")
    ap.add_argument("out", type=pathlib.Path, help="output directory (writes <out>/exfor/)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import polars as pl

    stats = Stats()
    all_rows: list[dict] = []
    files = sorted(args.src.rglob("*.x4"))
    if args.limit:
        files = files[: args.limit]

    for n, f in enumerate(files, 1):
        all_rows.extend(convert_entry(f, stats))
        stats["files"] += 1
        if n % 2000 == 0:
            print(f"  {n}/{len(files)} files, {len(all_rows):,} rows", file=sys.stderr)

    print(f"\nparsed {stats['files']:,} entries -> {len(all_rows):,} rows")
    print("skip reasons:")
    for k, v in stats.most_common():
        if k != "files":
            print(f"  {v:>9,}  {k}")

    if not all_rows:
        return

    df = pl.DataFrame(
        all_rows,
        schema={
            "projectile": pl.Utf8,
            "exfor_entry": pl.Utf8,
            "reaction": pl.Utf8,
            "MT": pl.Int32,
            "target_Z": pl.Int32,
            "target_A": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "energy_MeV": pl.Float64,
            "energy_err_MeV": pl.Float64,
            "xs_mb": pl.Float64,
            "xs_err_mb": pl.Float64,
            "author": pl.Utf8,
            "year": pl.Int32,
        },
    )
    df = df.unique(
        subset=[
            "exfor_entry",
            "reaction",
            "target_A",
            "residual_Z",
            "residual_A",
            "state",
            "energy_MeV",
        ]
    )

    # Split into the two products. The criterion is *residual absence*, not MT:
    # a transport channel names no outgoing nuclide. Routing on MT alone
    # misfiles fission-product production — (92-U-235(N,F)36-KR-88,PAR,SIG) is
    # MT=18 but is a ~10 mb yield for one fragment, not the ~1.5 b fission
    # channel, and mixing the two makes the fission channel look 30x low.
    is_channel = pl.col("MT").is_in(list(TRANSPORT_MT)) & (pl.col("residual_Z") == 0) & (pl.col("residual_A") == 0)
    products = {
        "exfor": df.filter(~is_channel),
        "exfor-channels": df.filter(is_channel),
    }

    for name, part in products.items():
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = (
            [
                "exfor_entry",
                "reaction",
                "MT",
                "target_Z",
                "target_A",
                "energy_MeV",
                "energy_err_MeV",
                "xs_mb",
                "xs_err_mb",
                "author",
                "year",
            ]
            if name == "exfor-channels"
            else None
        )
        written = 0
        for (proj, tz), grp in part.group_by(["projectile", "target_Z"]):
            sym = _ELEMENTS[tz] if 0 < tz < len(_ELEMENTS) else f"Z{tz}"
            grp = grp.drop("projectile")
            if cols:
                grp = grp.select(cols).sort("MT", "energy_MeV")
            else:
                grp = grp.sort("residual_Z", "residual_A", "energy_MeV")
            grp.write_parquet(out_dir / f"{proj}_{sym}.parquet", compression="zstd")
            written += 1
        print(f"wrote {written:>4} files, {part.height:>9,} rows -> {out_dir}")


if __name__ == "__main__":
    main()
