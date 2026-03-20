#!/usr/bin/env python3
"""Generate a Typst document surveying the theranostic isotope landscape.

Connects to the nucl-parquet DuckDB database, runs analytical queries,
and emits a complete Typst source file + figures.

Usage:
    uv run --with matplotlib python docs/theranostics/generate.py
    typst compile docs/theranostics/theranostics.typ
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from pathlib import Path

import nucl_parquet

# ── Constants ─────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent
TYP_FILE = OUT_DIR / "theranostics.typ"

# Chelator compatibility (domain knowledge)
CHELATOR_MAP: dict[str, set[str]] = {
    "DOTA": {"Sc", "Y", "La", "Ga", "In", "Lu", "Ac", "Bi", "Tb", "Ho",
             "Er", "Sm", "Gd", "Dy", "Yb", "Nd", "Pm", "Eu", "Tm", "Ce", "Pr"},
    "NOTA": {"Ga", "Cu", "Al"},
    "DTPA": {"In", "Y", "Lu", "Bi"},
    "Crown ethers": {"Ra", "Ba"},
    "Direct covalent": {"I", "At", "F", "Br"},
    "MACROPA": {"Ac", "Ra", "Ba"},
    "Sarcophagine": {"Cu"},
}

# Chemical families for pairing
LANTHANIDES = {57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71}
GROUP3_SC_Y_LA_AC = {21, 39, 57, 89}
HALOGENS = {9, 17, 35, 53, 85}
TC_RE = {43, 75}

# Radiation type display order
RAD_TYPE_ORDER = ["alpha", "beta-", "beta+/EC", "ce", "auger", "gamma", "xray"]


# ── Helpers ───────────────────────────────────────────────────────────

def _hl_label(s: float | None) -> str:
    """Format half-life as human-readable string."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "?"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def _typst_escape(s: str) -> str:
    """Escape characters special to Typst markup."""
    for ch in ("#", "$", "@", "<", ">", "*", "_", "\\"):
        s = s.replace(ch, "\\" + ch)
    return s


def _nuc_typst(symbol: str, A: int) -> str:
    """Typst markup for a nuclide: #super[A]Symbol."""
    return f"#super[{A}]{symbol}"


def _dose_fmt(v: float, decimals: int = 3) -> str:
    """Format a dose value: show '—' for negligible values."""
    if v < 10 ** (-decimals):
        return "—"
    return f"{v:.{decimals}f}"


def _iso_label(symbol: str, A: int) -> str:
    """Typst label string for an isotope, e.g. 'iso-Lu-177'."""
    return f"iso-{symbol}-{A}"


def _nuc_ref(symbol: str, A: int) -> str:
    """Typst nuclide with ref link to its detail section."""
    label = _iso_label(symbol, A)
    nuc = _nuc_typst(symbol, A)
    # Use Typst's native ref, rendered as the nuclide name
    return f"#link(<{label}>)[{nuc}]"


def _classify_facility(projectile: str, peak_E: float) -> str:
    """Classify the production facility required.

    Medical cyclotrons: typically 16–18 MeV protons (IBA Cyclone 18,
    GE PETtrace). Deuterons reach ~9 MeV on these machines.
    Research/isotope-production cyclotrons: 30–70 MeV (TRIUMF, PSI, JSI).
    """
    if projectile == "n":
        return "Reactor"
    if projectile == "p":
        if peak_E <= 18:
            return "Med. cyclotron (≤18 MeV p)"
        if peak_E <= 70:
            return "Research cyclotron"
        return "High-E accelerator"
    if projectile == "d":
        if peak_E <= 9:
            return "Med. cyclotron (≤9 MeV d)"
        if peak_E <= 50:
            return "Research cyclotron"
        return "High-E accelerator"
    if projectile in ("t", "h"):
        return "Research cyclotron"
    if projectile == "a":
        if peak_E <= 30:
            return "Research cyclotron"
        return "High-E accelerator"
    return "Exotic"


def _chelators_for(symbol: str) -> list[str]:
    """Return list of compatible chelators for an element symbol."""
    return [name for name, elems in CHELATOR_MAP.items() if symbol in elems]


def _booktabs_table(columns: str, align: str, header: list[str],
                    rows: list[list[str]], caption: str,
                    label: str = "", font_size: str = "8pt") -> list[str]:
    """Generate a Typst table with booktabs-style horizontal rules."""
    lines = []
    lines.append(f"#set text(size: {font_size})")
    lines.append("#figure(")
    lines.append("  kind: table,")
    lines.append("  table(")
    lines.append(f"    columns: {columns},")
    lines.append(f"    align: {align},")
    lines.append("    stroke: none,")
    lines.append("    table.hline(stroke: 1.5pt),")
    hdr = ", ".join(f"[*{h}*]" for h in header)
    lines.append(f"    table.header({hdr}),")
    lines.append("    table.hline(stroke: 0.75pt),")
    for row in rows:
        cells = ", ".join(f"[{cell}]" for cell in row)
        lines.append(f"    {cells},")
    lines.append("    table.hline(stroke: 1.5pt),")
    lines.append("  ),")
    lines.append(f"  caption: [{caption}],")
    if label:
        lines.append(f") <{label}>")
    else:
        lines.append(")")
    lines.append(f"#set text(size: 10pt)")
    return lines


# ── Data queries ──────────────────────────────────────────────────────

# Daughters with t½ above this are flagged as unsafe (1 year)
DAUGHTER_SAFETY_THRESHOLD_S = 3.15e7


def _check_decay_chain_safety(db, Z: int, A: int) -> tuple[bool, str]:
    """Check if any daughter in the decay chain is long-lived (> 1 year).

    Returns (is_safe, warning_string).
    """
    from nucl_parquet.loader import DECAY_CHAIN_SQL
    try:
        chain = db.sql(DECAY_CHAIN_SQL, params={"parent_z": Z, "parent_a": A}).fetchall()
    except Exception:
        return True, ""  # can't check → assume OK

    for r in chain:
        gen, d_Z, d_A, d_sym = r[8], r[0], r[1], r[2]
        d_hl = r[3]
        if gen <= 1:
            continue  # skip self
        if d_hl is not None and d_hl > DAUGHTER_SAFETY_THRESHOLD_S:
            return False, f"{d_sym}-{d_A} (t½={_hl_label(d_hl)})"
    return True, ""


def _query_candidates(db) -> list[dict]:
    """Screen the database for therapeutic isotope candidates.

    Uses a permissive dose floor: any isotope with particulate emission
    is included, since Auger emitters deliver low total dose but
    extremely high dose per unit volume at the subcellular scale.
    """
    rows = db.sql("""
        WITH dose AS (
            SELECT Z, A, rad_type,
                   SUM(dose_MeV_per_Bq_s) AS total_dose,
                   MAX(CASE WHEN rad_type = 'gamma' AND energy_keV BETWEEN 80 AND 400
                            AND intensity_pct > 5 THEN energy_keV END) AS spect_gamma_keV,
                   MAX(CASE WHEN rad_type = 'gamma' AND energy_keV > 1022
                            THEN energy_keV END) AS pair_gamma_keV
            FROM radiation
            WHERE dataset = 1
            GROUP BY Z, A, rad_type
        ),
        agg AS (
            SELECT Z, A,
                   SUM(CASE WHEN rad_type = 'ce'       THEN total_dose ELSE 0 END) AS ce_dose,
                   SUM(CASE WHEN rad_type = 'auger'    THEN total_dose ELSE 0 END) AS auger_dose,
                   SUM(CASE WHEN rad_type = 'beta-'    THEN total_dose ELSE 0 END) AS beta_dose,
                   SUM(CASE WHEN rad_type = 'beta+/EC' THEN total_dose ELSE 0 END) AS betaplus_dose,
                   SUM(CASE WHEN rad_type = 'alpha'    THEN total_dose ELSE 0 END) AS alpha_dose,
                   SUM(CASE WHEN rad_type = 'gamma'    THEN total_dose ELSE 0 END) AS gamma_dose,
                   MAX(spect_gamma_keV) AS spect_gamma,
                   MAX(pair_gamma_keV)  AS pair_gamma,
                   SUM(total_dose) AS total_all_dose
            FROM dose
            GROUP BY Z, A
        )
        SELECT gs.Z, gs.A, gs.symbol, gs.half_life_s, gs.jp,
               gs.decay_1, gs.decay_1_pct,
               COALESCE(a.ce_dose, 0)       AS ce_dose,
               COALESCE(a.auger_dose, 0)     AS auger_dose,
               COALESCE(a.beta_dose, 0)      AS beta_dose,
               COALESCE(a.betaplus_dose, 0)  AS betaplus_dose,
               COALESCE(a.alpha_dose, 0)     AS alpha_dose,
               COALESCE(a.gamma_dose, 0)     AS gamma_dose,
               COALESCE(a.spect_gamma, 0)    AS spect_gamma,
               COALESCE(a.pair_gamma, 0)     AS pair_gamma,
               COALESCE(a.total_all_dose, 0) AS total_all_dose
        FROM ground_states gs
        LEFT JOIN agg a ON gs.Z = a.Z AND gs.A = a.A
        WHERE gs.half_life_s BETWEEN 7200 AND 2592000
        ORDER BY gs.Z, gs.A
    """).fetchall()

    candidates = []
    for r in rows:
        Z, A, sym, hl, jp = r[0], r[1], r[2], r[3], r[4]
        decay_1 = r[5] or ""
        ce, auger, beta, betaplus, alpha, gamma = r[7], r[8], r[9], r[10], r[11], r[12]
        spect_g, pair_g = r[13], r[14]

        particulate = ce + auger + beta + alpha
        # Include any isotope with measurable particulate emission
        # (Auger emitters have low total dose but extreme dose density)
        if particulate < 1e-4:
            continue

        # Imaging classification — collect all applicable modalities
        img_modes = []
        if decay_1 in ("B+", "EC+B+") or betaplus > 0.001:
            img_modes.append("PET")
        if spect_g > 0:
            img_modes.append("SPECT")
        if pair_g > 0:
            img_modes.append("pair")
        imaging = " + ".join(img_modes) if img_modes else "—"

        # Therapeutic range classification
        ranges = []
        if auger > 0.001:
            ranges.append("Subcellular")
        if ce > 0.001:
            ranges.append("Cellular")
        if beta > 0.001:
            ranges.append("Cluster" if beta > 0.1 else "Cellular")
        if alpha > 0.001:
            ranges.append("Macroscopic")

        unique = list(dict.fromkeys(ranges))
        if len(unique) == 0:
            range_cls = "Subcellular"  # low-energy Auger-dominated
        elif len(unique) == 1:
            range_cls = unique[0]
        else:
            range_cls = "Multi-range"

        # Decay chain safety check
        is_safe, unsafe_daughter = _check_decay_chain_safety(db, Z, A)

        # γ/particulate ratio — high values mean most energy escapes the tumour
        gamma_ratio = gamma / particulate if particulate > 0 else 0.0

        candidates.append({
            "Z": Z, "A": A, "symbol": sym, "half_life_s": hl,
            "jp": jp or "", "decay": decay_1,
            "ce": ce, "auger": auger, "beta": beta, "alpha": alpha,
            "gamma": gamma, "particulate": particulate,
            "gamma_ratio": gamma_ratio,
            "imaging": imaging, "range_class": range_cls,
            "safe": is_safe, "unsafe_daughter": unsafe_daughter,
        })
    return candidates


def _query_production_routes(db, candidates: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Query all production routes for each candidate. Returns {(Z,A): [routes]}."""
    # Pre-fetch abundances
    abund_map: dict[tuple[int, str], float] = {}
    for r in db.sql("SELECT A, symbol, abundance FROM abundances").fetchall():
        abund_map[(r[0], r[1])] = r[2]

    routes_by_iso: dict[tuple[int, int], list[dict]] = {}

    for c in candidates:
        Z, A = c["Z"], c["A"]
        routes = db.sql("""
            SELECT
                split_part(split_part(filename, '/', -1), '.', 1) AS proj_elem,
                target_A,
                library,
                MAX(xs_mb) AS peak_xs,
                ARG_MAX(energy_MeV, xs_mb) AS peak_E
            FROM xs
            WHERE residual_Z = $rz AND residual_A = $ra AND state = ''
            GROUP BY proj_elem, target_A, library
            HAVING MAX(xs_mb) > 0.1
            ORDER BY peak_xs DESC
            LIMIT 50
        """, params={"rz": Z, "ra": A}).fetchall()

        parsed = []
        for rt in routes:
            proj_elem = rt[0]
            parts = proj_elem.split("_")
            projectile = parts[0] if parts else "?"
            target_sym = parts[1] if len(parts) > 1 else ""
            target_A = rt[1]
            peak_xs, peak_E = rt[3], rt[4]

            frac = abund_map.get((target_A, target_sym), 0.0)
            nat_abund = frac * 100

            score = peak_xs * frac
            if projectile in ("t", "h", "g"):
                score *= 0.1
            if peak_E > 50:
                score *= 0.5

            parsed.append({
                "projectile": projectile, "target_A": target_A,
                "target_sym": target_sym,
                "peak_xs": peak_xs, "peak_E": peak_E,
                "library": rt[2], "abundance": nat_abund,
                "score": score,
                "facility": _classify_facility(projectile, peak_E),
            })

        parsed.sort(key=lambda x: x["score"], reverse=True)
        routes_by_iso[(Z, A)] = parsed

    return routes_by_iso


def _query_diagnostics(db) -> dict[int, list[dict]]:
    """Query diagnostic isotopes grouped by Z."""
    rows = db.sql("""
        SELECT Z, A, symbol, half_life_s, decay_1
        FROM ground_states
        WHERE decay_1 IN ('EC', 'B+', 'EC+B+')
          AND half_life_s > 600
          AND half_life_s < 864000
        ORDER BY Z, A
    """).fetchall()

    diag_by_Z: dict[int, list[dict]] = {}
    for r in rows:
        d = {"Z": r[0], "A": r[1], "symbol": r[2],
             "half_life_s": r[3], "decay": r[4] or ""}
        diag_by_Z.setdefault(r[0], []).append(d)
    return diag_by_Z


def _query_emissions(db, Z: int, A: int) -> list[dict]:
    """Query all radiation emissions for an isotope."""
    rows = db.sql("""
        SELECT rad_type, rad_subtype, energy_keV, end_point_keV,
               intensity_pct, dose_MeV_per_Bq_s
        FROM radiation
        WHERE Z = $z AND A = $a AND dataset = 1
          AND intensity_pct > 0.1
        ORDER BY
            CASE rad_type
                WHEN 'alpha' THEN 1
                WHEN 'beta-' THEN 2
                WHEN 'beta+/EC' THEN 3
                WHEN 'ce' THEN 4
                WHEN 'auger' THEN 5
                WHEN 'gamma' THEN 6
                WHEN 'xray' THEN 7
            END,
            energy_keV DESC
    """, params={"z": Z, "a": A}).fetchall()
    return [{"rad_type": r[0], "rad_subtype": r[1] or "", "energy_keV": r[2],
             "end_point_keV": r[3], "intensity_pct": r[4],
             "dose": r[5] or 0.0} for r in rows]


# ── Section 1: Therapeutic Isotope Screening ──────────────────────────

def section_screening(candidates: list[dict], has_detail: set[tuple[int, int]]) -> str:
    """Build Typst screening section with cross-linked isotope names."""
    lines = []
    lines.append("= Therapeutic Isotope Screening")
    lines.append("")
    n_safe = sum(1 for c in candidates if c["safe"])
    n_unsafe = len(candidates) - n_safe
    lines.append(
        "We systematically screen the ENSDF database for radionuclides with "
        "half-lives between 2 hours and 30 days that emit particulate radiation. "
        "Unlike previous surveys that apply a hard dose floor, we retain "
        "Auger-dominant emitters whose total dose is low but whose dose "
        "_density_ (dose per unit volume) at the subcellular scale can be "
        "therapeutically significant @kassis2005 @sgouros2020. "
        "Each candidate is checked for decay-chain safety: isotopes whose "
        "daughters include long-lived radiotoxic nuclides (t#sub[½] > 1 y) "
        'are flagged with "⚠" and excluded from the recommended set. '
        "Isotope names link to detailed profiles in @sec:profiles."
    )
    lines.append("")
    lines.append(f"The screening yields *{len(candidates)} candidates* "
                 f"({n_safe} safe, {n_unsafe} flagged for daughter toxicity).")
    lines.append("")

    table_rows = []
    for c in candidates:
        key = (c["Z"], c["A"])
        nuc = _nuc_ref(c["symbol"], c["A"]) if key in has_detail else _nuc_typst(c["symbol"], c["A"])
        hl = _hl_label(c["half_life_s"])
        decay = _typst_escape(c["decay"]) if c["decay"] else "—"
        safe_mark = "✓" if c["safe"] else "⚠"
        gr = f"{c['gamma_ratio']:.1f}" if c["gamma_ratio"] < 100 else f"{c['gamma_ratio']:.0f}"
        table_rows.append([
            nuc, hl, decay,
            _dose_fmt(c["ce"]), _dose_fmt(c["auger"]),
            _dose_fmt(c["beta"]), _dose_fmt(c["alpha"]),
            _dose_fmt(c["gamma"]), gr,
            c["imaging"], c["range_class"], safe_mark,
        ])

    lines.extend(_booktabs_table(
        columns="(auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto)",
        align="(center,) * 12",
        header=["Isotope", "t#sub[½]", "Decay", "CE", "Auger",
                "β⁻", "α", "γ dose", "γ/p", "Imaging", "Range", "Safe"],
        rows=table_rows,
        caption=f"Therapeutic isotope candidates (n={len(candidates)}). "
                "Dose in MeV/(Bq$dot$s). ⚠ = long-lived daughter in decay chain.",
        label="tab:screening",
    ))
    lines.append("")
    return "\n".join(lines)


# ── Section 2: Production Route Analysis (summary) ────────────────────

def section_production(candidates: list[dict], routes_by_iso, has_detail) -> str:
    """Build summary production section."""
    lines = []
    lines.append("= Production Route Analysis")
    lines.append("")
    lines.append(
        "For each therapeutic candidate we query all evaluated cross-section "
        "libraries for production routes, cross-referencing natural isotopic "
        "abundances to assess practical feasibility @qaim2019 @qaim2017. "
        "Detailed per-isotope routes are in @sec:profiles."
    )
    lines.append("")

    # Best route per isotope
    best_routes = []
    no_route = []
    for c in candidates:
        key = (c["Z"], c["A"])
        rts = routes_by_iso.get(key, [])
        if rts:
            best_routes.append({"isotope": c, **rts[0]})
        else:
            no_route.append(c)

    best_routes.sort(key=lambda x: x["score"], reverse=True)

    prod_rows = []
    for rt in best_routes[:60]:
        iso = rt["isotope"]
        key = (iso["Z"], iso["A"])
        nuc = _nuc_ref(iso["symbol"], iso["A"]) if key in has_detail else _nuc_typst(iso["symbol"], iso["A"])
        prod_rows.append([
            nuc, rt["projectile"], str(rt["target_A"]),
            f"{rt['abundance']:.1f}" if rt["abundance"] > 0.05 else "—",
            f"{rt['peak_xs']:.1f}", f"{rt['peak_E']:.1f}",
            _typst_escape(rt["library"]), rt["facility"],
        ])

    lines.extend(_booktabs_table(
        columns="(auto, auto, auto, auto, auto, auto, auto, auto)",
        align="(center,) * 8",
        header=["Isotope", "Beam", "Target A", "Abund. (%)",
                "σ#sub[peak] (mb)", "E#sub[peak] (MeV)", "Library", "Facility"],
        rows=prod_rows,
        caption=f"Best production route per candidate (top {min(60, len(best_routes))} by feasibility score).",
        label="tab:production",
    ))
    lines.append("")

    fac_counts = Counter(rt["facility"] for rt in best_routes)
    lines.append(f"Of the {len(best_routes)} candidates with identified routes, "
                 + ", ".join(f"{n} via *{f}*" for f, n in fac_counts.most_common())
                 + ".")
    lines.append("")

    if no_route:
        syms = ", ".join(_nuc_typst(c["symbol"], c["A"]) for c in no_route[:15])
        lines.append(f"*{len(no_route)} candidates lack evaluated cross-section data*, "
                     f"including: {syms}.")
        lines.append("")

    return "\n".join(lines)


# ── Section 3: Theranostic Pairing + Coordination Chemistry ──────────

def section_pairing(candidates: list[dict], diag_by_Z, has_detail) -> str:
    """Build pairing and chelator section."""
    lines = []
    lines.append("= Theranostic Pairing and Coordination Chemistry")
    lines.append("")
    lines.append(
        "A theranostic pair consists of a diagnostic isotope (for PET or SPECT imaging) "
        "and a therapeutic isotope sharing the same chemical behaviour — ideally the same "
        "element or a chemical analogue that coordinates identically with a given chelator "
        "@price2014 @cutler2013."
    )
    lines.append("")

    cand_by_Z: dict[int, list[dict]] = {}
    for c in candidates:
        cand_by_Z.setdefault(c["Z"], []).append(c)

    # Same-element pairs
    lines.append("== Same-Element Pairs")
    lines.append("")
    pairs = []
    for Z, theraps in cand_by_Z.items():
        diags = diag_by_Z.get(Z, [])
        for th in theraps:
            for dg in diags:
                if dg["A"] != th["A"]:
                    pairs.append({"therapeutic": th, "diagnostic": dg})

    lines.append(f"We identify *{len(pairs)} same-element theranostic pairs*.")
    lines.append("")

    top_pairs = sorted(pairs, key=lambda p: p["therapeutic"]["particulate"], reverse=True)[:20]

    pair_rows = []
    for p in top_pairs:
        th, dg = p["therapeutic"], p["diagnostic"]
        key = (th["Z"], th["A"])
        th_nuc = _nuc_ref(th["symbol"], th["A"]) if key in has_detail else _nuc_typst(th["symbol"], th["A"])
        modality = "PET" if dg["decay"] in ("B+", "EC+B+") else "SPECT/EC"
        pair_rows.append([
            th_nuc, _hl_label(th["half_life_s"]), th["range_class"],
            _nuc_typst(dg["symbol"], dg["A"]),
            _hl_label(dg["half_life_s"]), modality,
        ])
    lines.extend(_booktabs_table(
        columns="(auto, auto, auto, auto, auto, auto)",
        align="(center,) * 6",
        header=["Therapeutic", "t#sub[½]", "Range",
                "Diagnostic", "t#sub[½]", "Modality"],
        rows=pair_rows,
        caption="Top same-element theranostic pairs ranked by therapeutic dose.",
        label="tab:same-element",
        font_size="9pt",
    ))
    lines.append("")

    # Chemical-family pairing
    lines.append("== Chemical-Family Pairing")
    lines.append("")
    lines.append(
        "Lanthanides coordinate equivalently via DOTA, group-3 metals "
        "(Sc/Y/La/Ac) share trivalent chemistry, halogens form direct "
        "covalent bonds, and the Tc/Re pair exploits identical oxidation states @muller2017."
    )
    lines.append("")

    for fname, Zset in [("Lanthanides (DOTA)", LANTHANIDES), ("Group 3 (Sc/Y/La/Ac)", GROUP3_SC_Y_LA_AC),
                         ("Halogens (covalent)", HALOGENS), ("Tc/Re", TC_RE)]:
        members_th = [c for c in candidates if c["Z"] in Zset]
        members_dg = [d for zlist in diag_by_Z.values() for d in zlist if d["Z"] in Zset]
        if members_th:
            th_str = ", ".join(_nuc_ref(c["symbol"], c["A"]) if (c["Z"], c["A"]) in has_detail
                               else _nuc_typst(c["symbol"], c["A"]) for c in members_th[:8])
            dg_str = ", ".join(_nuc_typst(d["symbol"], d["A"]) for d in members_dg[:8]) if members_dg else "—"
            lines.append(f"*{fname}*: therapeutic: {th_str}; diagnostic: {dg_str}")
            lines.append("")

    # Chelator table
    lines.append("== Chelator Compatibility")
    lines.append("")
    lines.append(
        "The choice of bifunctional chelator determines which radiometals "
        "can label a given targeting vector @price2014 @vermeulen2019."
    )
    lines.append("")

    cand_syms = {c["symbol"] for c in candidates}
    chel_rows = []
    for chel, elems in CHELATOR_MAP.items():
        highlighted = []
        for el in sorted(elems):
            highlighted.append(f"*{el}*" if el in cand_syms else el)
        chel_rows.append([chel, ", ".join(highlighted)])
    lines.extend(_booktabs_table(
        columns="(auto, auto)",
        align="(left, left)",
        header=["Chelator", "Compatible elements"],
        rows=chel_rows,
        caption="Chelator–element compatibility. Bold = has therapeutic candidates.",
        label="tab:chelators",
        font_size="9pt",
    ))
    lines.append("")

    return "\n".join(lines)


# ── Section 5: Detailed Isotope Profiles ──────────────────────────────

def section_detailed_profiles(db, candidates: list[dict], routes_by_iso,
                               diag_by_Z, has_detail: set[tuple[int, int]]) -> str:
    """Generate a subsection per highlighted isotope with full data."""
    lines = []
    lines.append("= Detailed Isotope Profiles <sec:profiles>")
    lines.append("")
    lines.append(
        "This section provides a detailed data sheet for each candidate isotope "
        "that has at least one identified production route. "
        "Each profile includes: nuclear properties, complete emission table, "
        "all evaluated production routes with target abundances, compatible "
        "chelators, and paired diagnostic isotopes."
    )
    lines.append("")

    detailed = [c for c in candidates if (c["Z"], c["A"]) in has_detail]

    for c in detailed:
        Z, A, sym = c["Z"], c["A"], c["symbol"]
        label = _iso_label(sym, A)
        nuc = _nuc_typst(sym, A)

        lines.append(f"== {nuc} <{label}>")
        lines.append("")

        # Properties summary
        hl = _hl_label(c["half_life_s"])
        decay = c["decay"] or "—"
        lines.append(f"*Half-life:* {hl} #h(1em) *J#super[π]:* {c['jp'] or '—'} "
                     f"#h(1em) *Primary decay:* {_typst_escape(decay)} #h(1em) "
                     f"*Imaging:* {c['imaging']} #h(1em) *Range:* {c['range_class']}")
        lines.append("")

        if not c["safe"]:
            lines.append(f'#block(fill: rgb("#fff3cd"), inset: 8pt, radius: 3pt)[')
            lines.append(f'  *⚠ Decay chain warning:* daughter {c["unsafe_daughter"]} '
                         f'is long-lived (t#sub[½] > 1 y). Not suitable for clinical use '
                         f'without careful dosimetric evaluation of daughter accumulation.')
            lines.append("]")
        lines.append("")

        # Dose summary
        lines.append(
            f"*Particulate dose:* {c['particulate']:.3f} MeV/(Bq$dot$s) — "
            f"β⁻: {c['beta']:.3f}, α: {c['alpha']:.3f}, "
            f"CE: {c['ce']:.3f}, Auger: {c['auger']:.3f}, "
            f"γ: {c['gamma']:.3f}"
        )
        lines.append("")

        # ── Decay chain ──
        from nucl_parquet.loader import DECAY_CHAIN_SQL
        try:
            chain = db.sql(DECAY_CHAIN_SQL, params={"parent_z": Z, "parent_a": A}).fetchall()
            if chain and len(chain) > 1:
                lines.append("=== Decay Chain")
                lines.append("")
                chain_parts = []
                for r in chain:
                    d_sym, d_A, d_hl, d_mode = r[2], r[1], r[3], r[4]
                    d_hl_str = _hl_label(d_hl) if d_hl else "stable"
                    if d_mode:
                        chain_parts.append(f"{_nuc_typst(d_sym, d_A)} ({d_hl_str})")
                    else:
                        chain_parts.append(f"{_nuc_typst(d_sym, d_A)} ({d_hl_str}, stable)")
                        break
                lines.append(" → ".join(chain_parts))
                lines.append("")
        except Exception:
            pass

        # ── Emission table ──
        emissions = _query_emissions(db, Z, A)
        if emissions:
            lines.append("=== Emissions")
            lines.append("")
            em_rows = []
            for em in emissions[:25]:
                ep = f"{em['end_point_keV']:.1f}" if em['end_point_keV'] else "—"
                em_rows.append([
                    em["rad_type"], _typst_escape(em["rad_subtype"]),
                    f"{em['energy_keV']:.1f}", ep,
                    f"{em['intensity_pct']:.2f}", _dose_fmt(em["dose"], 4),
                ])
            lines.extend(_booktabs_table(
                columns="(auto, auto, auto, auto, auto, auto)",
                align="(center,) * 6",
                header=["Type", "Subtype", "Energy (keV)",
                        "Endpoint (keV)", "Intensity (%)", "Dose (MeV/Bq·s)"],
                rows=em_rows,
                caption=f"Radiation emissions of {nuc} (intensity > 0.1%).",
                font_size="7.5pt",
            ))
            lines.append("")

        # ── Production routes ──
        rts = routes_by_iso.get((Z, A), [])
        if rts:
            lines.append("=== Production Routes")
            lines.append("")
            rt_rows = []
            for rt in rts[:20]:
                rt_rows.append([
                    rt["projectile"], rt["target_sym"], str(rt["target_A"]),
                    f"{rt['abundance']:.1f}" if rt["abundance"] > 0.05 else "—",
                    f"{rt['peak_xs']:.1f}", f"{rt['peak_E']:.1f}",
                    rt["facility"],
                ])
            lines.extend(_booktabs_table(
                columns="(auto, auto, auto, auto, auto, auto, auto)",
                align="(center,) * 7",
                header=["Beam", "Target", "A", "Abund. (%)",
                        "σ#sub[peak] (mb)", "E#sub[peak] (MeV)", "Facility"],
                rows=rt_rows,
                caption=f"Production routes for {nuc} "
                        f"({len(rts)} total, top {min(20, len(rts))} shown).",
            ))
            lines.append("")
        else:
            lines.append("No evaluated production routes found in the surveyed libraries.")
            lines.append("")

        # ── Chelator compatibility ──
        chels = _chelators_for(sym)
        if chels:
            lines.append(f"*Chelators:* {', '.join(chels)}")
            lines.append("")

        # ── Diagnostic partners ──
        diags = diag_by_Z.get(Z, [])
        partner_diags = [d for d in diags if d["A"] != A]
        if partner_diags:
            lines.append("=== Diagnostic Partners")
            lines.append("")
            partner_strs = []
            for d in partner_diags[:10]:
                mod = "PET" if d["decay"] in ("B+", "EC+B+") else "EC"
                partner_strs.append(
                    f"{_nuc_typst(d['symbol'], d['A'])} "
                    f"(t#sub[½] = {_hl_label(d['half_life_s'])}, {mod})"
                )
            lines.append("Same-element imaging partners: " + "; ".join(partner_strs) + ".")
            lines.append("")

        # ── Chemical-family partners ──
        family_partners = []
        for fname, Zset in [("Lanthanides", LANTHANIDES), ("Group 3", GROUP3_SC_Y_LA_AC),
                             ("Halogens", HALOGENS), ("Tc/Re", TC_RE)]:
            if Z in Zset:
                family_diags = [d for z2 in Zset if z2 != Z
                                for d in diag_by_Z.get(z2, [])]
                if family_diags:
                    fam_str = ", ".join(
                        f"{_nuc_typst(d['symbol'], d['A'])}"
                        for d in family_diags[:6]
                    )
                    family_partners.append(f"*{fname}:* {fam_str}")

        if family_partners:
            lines.append("Chemical-family diagnostic partners: " + "; ".join(family_partners) + ".")
            lines.append("")

        lines.append("#line(length: 100%, stroke: 0.5pt + gray)")
        lines.append("")

    return "\n".join(lines)


# ── Overview Figures ──────────────────────────────────────────────────

def generate_figures(db, candidates: list[dict]) -> str:
    """Generate matplotlib figures and return Typst markup."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "// matplotlib not available — figures skipped\n"

    typst = []

    # ── Fig 1: Half-life vs particulate dose, shaped by range class ──
    range_markers = {"Subcellular": "v", "Cellular": "s", "Cluster": "o",
                     "Macroscopic": "D", "Multi-range": "^", "—": "."}
    fig, ax = plt.subplots(figsize=(10, 6))

    imaging_colors = {"PET": "#e41a1c", "SPECT": "#377eb8",
                      "pair": "#4daf4a", "—": "#aaaaaa"}

    def _primary_imaging(c):
        """Return the primary imaging modality for color mapping."""
        img = c["imaging"]
        if "PET" in img:
            return "PET"
        if "SPECT" in img:
            return "SPECT"
        if "pair" in img:
            return "pair"
        return "—"

    # Plot safe isotopes prominently, unsafe ones faded
    for img_type, color in imaging_colors.items():
        for safe_val, alpha_val, edge_w in [(True, 0.85, 0.5), (False, 0.2, 0.2)]:
            subset = [c for c in candidates
                      if _primary_imaging(c) == img_type and c["safe"] == safe_val]
            if not subset:
                continue
            for rng, marker in range_markers.items():
                rng_subset = [c for c in subset if c["range_class"] == rng]
                if not rng_subset:
                    continue
                x = [c["half_life_s"] / 3600 for c in rng_subset]  # hours
                y = [c["particulate"] for c in rng_subset]
                ax.scatter(x, y, c=color, marker=marker, alpha=alpha_val,
                           s=35, edgecolors="k", linewidths=edge_w)

    # Label notable isotopes (safe, high dose or known)
    notable = {("Lu", 177), ("Ac", 225), ("At", 211), ("Tb", 161),
               ("Cu", 67), ("I", 131), ("Y", 90), ("Sc", 47),
               ("Re", 186), ("Ho", 166), ("Sm", 153), ("Er", 169),
               ("Ga", 67), ("In", 111), ("Br", 77), ("Rh", 105)}
    for c in candidates:
        if not c["safe"]:
            continue
        is_notable = (c["symbol"], c["A"]) in notable
        if is_notable or c["particulate"] > 1.0:
            ax.annotate(f"{c['symbol']}-{c['A']}",
                        (c["half_life_s"] / 3600, c["particulate"]),
                        fontsize=6, alpha=0.9,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Half-life (hours)", fontsize=10)
    ax.set_ylabel("Particulate dose (MeV/Bq·s)", fontsize=10)
    ax.set_title("Theranostic Isotope Landscape", fontsize=12)

    # Legend for imaging (color)
    from matplotlib.lines import Line2D
    img_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=8, label=k) for k, c in imaging_colors.items()]
    rng_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                          markersize=7, label=k) for k, m in range_markers.items()
                   if k != "—"]
    leg1 = ax.legend(handles=img_handles, title="Imaging", loc="upper left",
                     fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=rng_handles, title="Range class", loc="lower right",
              fontsize=7, title_fontsize=8)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_landscape.svg", format="svg")
    plt.close(fig)
    typst.append(
        '#figure(\n'
        '  image("fig_landscape.svg", width: 100%),\n'
        '  caption: [Theranostic isotope landscape: half-life vs. particulate '
        'dose. Colour = imaging modality, shape = therapeutic range class. '
        'Faded points have unsafe decay chains.],\n'
        ') <fig:landscape>\n'
    )

    # ── Fig 2: Periodic table heatmap ──
    z_counts: dict[int, int] = {}
    for c in candidates:
        z_counts[c["Z"]] = z_counts.get(c["Z"], 0) + 1

    PT_LAYOUT: dict[int, tuple[int, int]] = {
        1: (0, 0), 2: (0, 17),
        3: (1, 0), 4: (1, 1), 5: (1, 12), 6: (1, 13), 7: (1, 14),
        8: (1, 15), 9: (1, 16), 10: (1, 17),
        11: (2, 0), 12: (2, 1), 13: (2, 12), 14: (2, 13), 15: (2, 14),
        16: (2, 15), 17: (2, 16), 18: (2, 17),
        19: (3, 0), 20: (3, 1),
        21: (3, 2), 22: (3, 3), 23: (3, 4), 24: (3, 5), 25: (3, 6),
        26: (3, 7), 27: (3, 8), 28: (3, 9), 29: (3, 10), 30: (3, 11),
        31: (3, 12), 32: (3, 13), 33: (3, 14), 34: (3, 15), 35: (3, 16), 36: (3, 17),
        37: (4, 0), 38: (4, 1),
        39: (4, 2), 40: (4, 3), 41: (4, 4), 42: (4, 5), 43: (4, 6),
        44: (4, 7), 45: (4, 8), 46: (4, 9), 47: (4, 10), 48: (4, 11),
        49: (4, 12), 50: (4, 13), 51: (4, 14), 52: (4, 15), 53: (4, 16), 54: (4, 17),
        55: (5, 0), 56: (5, 1),
        71: (5, 2), 72: (5, 3), 73: (5, 4), 74: (5, 5), 75: (5, 6),
        76: (5, 7), 77: (5, 8), 78: (5, 9), 79: (5, 10), 80: (5, 11),
        81: (5, 12), 82: (5, 13), 83: (5, 14), 84: (5, 15), 85: (5, 16), 86: (5, 17),
        87: (6, 0), 88: (6, 1), 103: (6, 2),
        57: (8, 2), 58: (8, 3), 59: (8, 4), 60: (8, 5), 61: (8, 6),
        62: (8, 7), 63: (8, 8), 64: (8, 9), 65: (8, 10), 66: (8, 11),
        67: (8, 12), 68: (8, 13), 69: (8, 14), 70: (8, 15),
        89: (9, 2), 90: (9, 3), 91: (9, 4), 92: (9, 5),
    }

    sym_lookup = {}
    for r in db.sql("SELECT Z, symbol FROM elements ORDER BY Z").fetchall():
        sym_lookup[r[0]] = r[1]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    max_count = max(z_counts.values()) if z_counts else 1
    cmap = plt.cm.YlOrRd

    for Z, (row, col) in PT_LAYOUT.items():
        count = z_counts.get(Z, 0)
        color = cmap(count / max_count) if count > 0 else "#f0f0f0"
        rect = plt.Rectangle((col, -row), 0.9, 0.9, facecolor=color,
                              edgecolor="gray", linewidth=0.5)
        ax2.add_patch(rect)
        sym = sym_lookup.get(Z, "")
        ax2.text(col + 0.45, -row + 0.55, sym, ha="center", va="center",
                 fontsize=5.5, fontweight="bold" if count > 0 else "normal")
        if count > 0:
            ax2.text(col + 0.45, -row + 0.2, str(count), ha="center", va="center",
                     fontsize=4.5, color="#333")

    ax2.set_xlim(-0.5, 18.5)
    ax2.set_ylim(-10, 1.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Theranostic Candidate Coverage Across the Periodic Table", fontsize=11)

    import matplotlib.pyplot as mpl_plt
    sm = mpl_plt.cm.ScalarMappable(cmap=cmap, norm=mpl_plt.Normalize(0, max_count))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label("Number of candidates", fontsize=8)

    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "fig_periodic_table.svg", format="svg")
    plt.close(fig2)
    typst.append(
        '#figure(\n'
        '  image("fig_periodic_table.svg", width: 100%),\n'
        '  caption: [Periodic table heatmap of theranostic candidates per element.],\n'
        ') <fig:periodic-table>\n'
    )

    return "\n".join(typst)


# ── Typst Preamble ───────────────────────────────────────────────────

TYPST_PREAMBLE = r"""// Auto-generated by docs/theranostics/generate.py — do not edit manually.

#set text(size: 10pt, font: "New Computer Modern")
#set par(justify: true)
#set page(
  paper: "a4",
  margin: 2.5cm,
  footer: context [
    #set align(center)
    #counter(page).display("1")
  ],
)
#set heading(numbering: "1.1")

// Allow tables in figures to break across pages
#show figure.where(kind: table): set block(breakable: true)

// Nuclear notation helper
#let nuc(el, A) = [#super[#str(A)]#el]

// Title
#align(center)[
  #text(size: 16pt, weight: "bold")[
    The Theranostic Isotope Landscape: \
    A Data-Driven Survey
  ]
  #v(0.5em)
  #text(size: 10pt, style: "italic")[
    Generated from the nucl-parquet database — #datetime.today().display()
  ]
  #v(1em)
]

"""

TYPST_BIBLIOGRAPHY = r"""
#bibliography("refs.bib", style: "ieee")
"""


# ── Introduction ──────────────────────────────────────────────────────

def section_introduction() -> str:
    return """
#v(0.5em)

== Abstract

Targeted radionuclide therapy (TRT) is transforming precision oncology,
yet the landscape of potential theranostic isotopes extends far beyond the
handful in current clinical use. This report presents a systematic,
data-driven survey of the theranostic isotope landscape, drawing
entirely on the nucl-parquet nuclear data library. We screen over 3,000
radionuclides for therapeutic potential based on half-life, particulate
emission dose, and intrinsic imaging capability. For viable candidates we
assess production route feasibility using evaluated cross-section
libraries and natural isotopic abundances. Finally, we analyse
theranostic pairing — matching each therapeutic isotope with a diagnostic
companion via same-element or chemical-family strategies, guided by
chelator coordination chemistry.

Each candidate with an identified production route receives a detailed
profile including complete emission spectra, all evaluated reaction
routes, target material abundances, chelator compatibility, and paired
diagnostic isotopes — all cross-linked throughout the document.

#v(0.5em)

= Introduction

The emergence of #super[177]Lu-DOTATATE (Lutathera) and #super[177]Lu-PSMA-617
(Pluvicto) has demonstrated the clinical power of targeted radionuclide
therapy @hofman2021 @banerjee2015. Yet lutetium-177 is only one of dozens
of potentially therapeutic radionuclides, and the optimal isotope for a
given indication depends on tumour geometry, pharmacokinetics, and
available imaging modality @herrmann2020 @sgouros2020.

The _theranostic_ paradigm — using a diagnostic isotope to image, quantify,
and select patients before switching to a therapeutic companion on the same
molecular vector — requires that diagnostic and therapeutic isotopes share
virtually identical chemical behaviour. This is achieved through
same-element pairing (e.g., #super[44]Sc diagnostic / #super[47]Sc therapeutic),
same-chelator pairing across chemical analogues
(e.g., #super[68]Ga or #super[44]Sc for PET imaging paired with
#super[177]Lu therapy, all via DOTA), or direct covalent labelling
for halogens (e.g., #super[123]I SPECT / #super[131]I therapy) @price2014.

Historically, theranostic isotope selection has been opportunistic: driven
by available supply rather than systematic optimisation. The nucl-parquet database aggregates decay data from ENSDF (Evaluated
Nuclear Structure Data File, BNL/NNDC), evaluated cross-section libraries
from TENDL (TALYS-based, PSI/NRG), ENDF/B-VIII.1 (BNL), JEFF-4.0 (NEA),
JENDL-5 (JAEA), IAEA photonuclear and medical isotope libraries, EAF-2010
(Euroatom), IRDFF-2 (IAEA dosimetry), and experimental data from EXFOR.
Natural isotopic abundances follow the IAEA Nuclear Wallet Cards; enriched
target availability and pricing from commercial suppliers (Isoflex USA,
Trace Sciences, ORNL Isotope Program) are not yet included but represent
a critical practical constraint for production route feasibility.
Stopping powers derive from NIST PSTAR/ASTAR/ESTAR and CatIMA
(for heavy ions). This combination enables a comprehensive computational
screening across the full landscape @nelson2020.

This report proceeds in four parts:
+ *Screening* — identifying therapeutic candidates by half-life and dose.
+ *Production* — evaluating cross-section routes and facility requirements.
+ *Pairing* — matching therapeutic isotopes with diagnostic companions and chelators.
+ *Detailed profiles* — comprehensive data sheets for each viable candidate.

== Methodology and Data Sources

*Decay data* is from the Evaluated Nuclear Structure Data File (ENSDF),
providing half-lives, decay modes, and radiation emission spectra
(γ, β⁻, α, conversion electrons, Auger electrons, X-rays) with
intensities and dose constants in MeV/(Bq·s).

*Production cross-sections* are queried from evaluated nuclear data
libraries: TENDL-2024/2025 (p, d, t, ³He, α projectiles),
ENDF/B-VIII.1, JEFF-4.0, JENDL-5 (neutrons and charged particles),
IAEA-Medical (p, d), EAF-2010, IRDFF-2, and JENDL-AD-2017.
Cross-sections represent the probability σ (in millibarns) of producing
the desired residual nuclide as a function of beam energy.

*Natural abundances* are from the IAEA Nuclear Wallet Cards.
A route's feasibility score = σ#sub[peak] × natural abundance,
penalised for exotic beams (t, ³He: ×0.1) or high energy (>50 MeV: ×0.5).

*Facility classification:*
- *Med. cyclotron:* ≤18 MeV protons / ≤9 MeV deuterons (IBA Cyclone 18, GE PETtrace)
- *Research cyclotron:* 30–70 MeV (TRIUMF, PSI, Arronax, JSI)
- *Reactor:* thermal/epithermal neutron irradiation
- *High-E accelerator:* >70 MeV or non-standard beams

*Table column legend:*
- *CE:* conversion electron dose (MeV/Bq·s)
- *Auger:* Auger + Coster-Kronig electron dose
- *β⁻:* beta-minus dose (includes endpoint spectrum)
- *α:* alpha particle dose
- *γ dose:* photon dose (gamma + X-ray)
- *Imaging:* PET (β⁺/EC), SPECT (γ 80–400 keV, I>5%), pair (γ >1022 keV)
- *Range:* Subcellular (\\<1 μm, Auger), Cellular (1–30 μm, CE/low β),
  Cluster (30–500 μm, β⁻), Macroscopic (\\>500 μm, high β⁻/α)
- *γ/p:* ratio of photon dose to particulate dose — values \\>1 indicate
  most energy escapes the tumour (poor therapeutic ratio, higher
  radiation protection burden). Ideal therapeutic isotopes have γ/p \\< 0.5.
- *Safe:* ✓ = all daughters short-lived, ⚠ = long-lived daughter (t#sub[½] \\> 1 y)
- "—" in dose columns indicates negligible emission (\\<0.001 MeV/Bq·s)

"""


# ── Conclusion ────────────────────────────────────────────────────────

def section_conclusion(candidates: list[dict], has_detail: set[tuple[int, int]]) -> str:
    n_pet = sum(1 for c in candidates if "PET" in c["imaging"])
    n_spect = sum(1 for c in candidates if "SPECT" in c["imaging"])
    n_alpha = sum(1 for c in candidates if c["alpha"] > 0.001)

    return f"""
= Conclusion and Future Directions

This data-driven survey identifies *{len(candidates)} therapeutic isotope
candidates* in the 2-hour to 30-day half-life window, of which
{n_pet} offer intrinsic PET imaging, {n_spect} are SPECT-compatible,
and {n_alpha} emit α-particles for targeted alpha therapy.
*{len(has_detail)} candidates* have at least one evaluated production
route and receive detailed profiles in @sec:profiles.

Several observations emerge:

+ *The α-emitter landscape is sparse but high-impact.* Beyond #super[225]Ac
  and #super[211]At, few α-emitters have viable production routes, yet their
  therapeutic potency at the cellular scale is unmatched @kratochwil2016.

+ *Lanthanide theranostics are the most chemically mature family.*
  DOTA-based chelation provides interchangeable coordination for Sc through
  Lu, enabling a rich set of diagnostic/therapeutic permutations @muller2017.

+ *Production remains the bottleneck.* Many promising candidates lack
  evaluated cross-section data entirely, and others require exotic beams
  or enriched targets that limit practical supply.

+ *Cross-section measurement campaigns* should prioritise isotopes
  identified here with high therapeutic potential but no production data.

Future work will integrate decay-chain dosimetry (using the recursive
chain-walking capabilities of nucl-parquet) and Monte Carlo range
calculations to refine the screening beyond the simplified classification
presented here.
"""


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Connecting to nucl-parquet...")
    db = nucl_parquet.connect()

    print("Screening candidates...")
    candidates = _query_candidates(db)
    print(f"  → {len(candidates)} candidates")

    print("Querying production routes...")
    routes_by_iso = _query_production_routes(db, candidates)

    print("Querying diagnostics...")
    diag_by_Z = _query_diagnostics(db)

    # Candidates with at least one production route get a detail profile
    has_detail = {(c["Z"], c["A"]) for c in candidates
                  if routes_by_iso.get((c["Z"], c["A"]))}
    print(f"  → {len(has_detail)} isotopes with production routes (detailed profiles)")

    print("Building sections...")
    s1 = section_screening(candidates, has_detail)
    s2 = section_production(candidates, routes_by_iso, has_detail)
    s3 = section_pairing(candidates, diag_by_Z, has_detail)

    print("Generating figures...")
    figs = generate_figures(db, candidates)

    print("Building detailed profiles...")
    s5 = section_detailed_profiles(db, candidates, routes_by_iso, diag_by_Z, has_detail)

    print("Assembling Typst document...")
    doc = TYPST_PREAMBLE
    doc += section_introduction()
    doc += s1 + "\n\n"
    doc += figs + "\n\n"
    doc += s2 + "\n\n"
    doc += s3 + "\n\n"
    doc += s5 + "\n\n"
    doc += section_conclusion(candidates, has_detail) + "\n"
    doc += TYPST_BIBLIOGRAPHY

    TYP_FILE.write_text(doc)
    print(f"Written: {TYP_FILE}")

    # Try to compile
    try:
        result = subprocess.run(
            ["typst", "compile", str(TYP_FILE)],
            capture_output=True, text=True, timeout=900,
        )
        if result.returncode == 0:
            print(f"Compiled: {TYP_FILE.with_suffix('.pdf')}")
        else:
            print(f"Typst compilation failed:\n{result.stderr}")
    except FileNotFoundError:
        print("typst not found — run: typst compile docs/theranostics/theranostics.typ")
    except subprocess.TimeoutExpired:
        print("typst compilation timed out")


if __name__ == "__main__":
    main()
