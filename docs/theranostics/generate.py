#!/usr/bin/env python3
"""Generate a Typst document surveying the theranostic isotope landscape.

Connects to the nucl-parquet DuckDB database, runs analytical queries,
and emits a complete Typst source file + figures.

Usage:
    uv run python docs/theranostics/generate.py
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


# ── Section 1: Therapeutic Isotope Screening ──────────────────────────

def section_screening(db) -> tuple[str, list[dict]]:
    """Screen the database for therapeutic isotope candidates.

    Returns (typst_section_string, candidates_list).
    """
    # Query: ground states with t½ 2h-30d, joined with aggregated radiation
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

        # Total particulate dose (therapeutic)
        particulate = ce + auger + beta + alpha
        if particulate < 0.005:
            continue

        # Imaging classification
        imaging = "None"
        if decay_1 in ("B+", "EC+B+") or betaplus > 0.001:
            imaging = "PET"
        elif spect_g > 0:
            imaging = "SPECT"
        elif pair_g > 0:
            imaging = "Pair"

        # Therapeutic range classification
        ranges = []
        if auger > 0.001:
            ranges.append("Subcellular")
        if ce > 0.001:
            ranges.append("Cellular")
        if beta > 0.001:
            if beta > 0.1:
                ranges.append("Cluster")
            else:
                ranges.append("Cellular")
        if alpha > 0.001:
            ranges.append("Macroscopic")

        if len(ranges) == 0:
            range_cls = "—"
        elif len(ranges) == 1:
            range_cls = ranges[0]
        else:
            # deduplicate
            unique = list(dict.fromkeys(ranges))
            range_cls = "Multi-range" if len(unique) > 1 else unique[0]

        candidates.append({
            "Z": Z, "A": A, "symbol": sym, "half_life_s": hl,
            "jp": jp or "", "decay": decay_1,
            "ce": ce, "auger": auger, "beta": beta, "alpha": alpha,
            "gamma": gamma, "particulate": particulate,
            "imaging": imaging, "range_class": range_cls,
        })

    # Build Typst table
    lines = []
    lines.append("= Therapeutic Isotope Screening")
    lines.append("")
    lines.append(
        "We systematically screen the ENSDF database for radionuclides with "
        "half-lives between 2 hours and 30 days whose particulate radiation dose "
        "exceeds 0.005 MeV/(Bq$dot$s). "
        "This identifies candidates suitable for targeted radionuclide therapy, "
        "classified by their dominant emission range and intrinsic imaging capability "
        "@kassis2005 @sgouros2020."
    )
    lines.append("")
    lines.append(f"The screening yields *{len(candidates)} candidates* from {len(rows)} "
                 f"isotopes in the 2 h – 30 d half-life window.")
    lines.append("")

    # Table
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),")
    lines.append("    align: (center,) * 10,")
    lines.append("    stroke: 0.5pt,")
    lines.append("    table.header(")
    lines.append('      [*Isotope*], [*t#sub[½]*], [*Decay*], [*CE*], [*Auger*],')
    lines.append('      [*β⁻*], [*α*], [*γ dose*], [*Imaging*], [*Range*],')
    lines.append("    ),")

    for c in candidates:
        nuc = _nuc_typst(c["symbol"], c["A"])
        hl = _hl_label(c["half_life_s"])
        decay = _typst_escape(c["decay"]) if c["decay"] else "—"
        lines.append(
            f"    [{nuc}], [{hl}], [{decay}], "
            f"[{c['ce']:.3f}], [{c['auger']:.3f}], "
            f"[{c['beta']:.3f}], [{c['alpha']:.3f}], "
            f"[{c['gamma']:.3f}], [{c['imaging']}], [{c['range_class']}],"
        )

    lines.append("  ),")
    lines.append(f'  caption: [Therapeutic isotope candidates (n={len(candidates)}). '
                 'Dose values in MeV/(Bq$dot$s). '
                 'Imaging: intrinsic capability (PET, SPECT, pair-production, or none).],')
    lines.append(") <tab:screening>")
    lines.append("")

    return "\n".join(lines), candidates


# ── Section 2: Production Route Analysis ──────────────────────────────

def section_production(db, candidates: list[dict]) -> str:
    """Analyse production routes for each candidate isotope."""
    lines = []
    lines.append("= Production Route Analysis")
    lines.append("")
    lines.append(
        "For each therapeutic candidate we query all evaluated cross-section "
        "libraries for production routes, cross-referencing natural isotopic "
        "abundances to assess practical feasibility @qaim2019 @qaim2017. "
        "A route score is computed as peak cross-section (mb) multiplied by "
        "target natural abundance, penalised for exotic beams or high threshold energy."
    )
    lines.append("")

    # Facility classification by beam + energy
    def classify_facility(projectile: str, peak_E: float) -> str:
        if projectile in ("n",):
            return "Reactor"
        if projectile in ("p", "d"):
            if peak_E <= 25:
                return "Med. cyclotron"
            return "High-E cyclotron"
        if projectile in ("t", "h"):
            return "High-E cyclotron"
        if projectile == "a":
            if peak_E <= 40:
                return "Med. cyclotron"
            return "High-E cyclotron"
        return "Exotic"

    # Pre-fetch all abundances: (A, symbol) → fractional abundance
    abund_map: dict[tuple[int, str], float] = {}
    for r in db.sql("SELECT A, symbol, abundance FROM abundances").fetchall():
        abund_map[(r[0], r[1])] = r[2]

    route_table = []
    no_route = []

    for c in candidates:
        Z, A, sym = c["Z"], c["A"], c["symbol"]

        # Extract projectile from filename pattern: .../projectile_Element.parquet
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

        if not routes:
            no_route.append(c)
            continue

        best_score = 0
        best_route = None
        for rt in routes:
            proj_elem = rt[0]  # e.g. "p_Cu"
            parts = proj_elem.split("_")
            projectile = parts[0] if parts else "?"
            target_sym = parts[1] if len(parts) > 1 else ""
            target_A = rt[1]
            library = rt[2]
            peak_xs = rt[3]
            peak_E = rt[4]

            # Look up target natural abundance
            frac = abund_map.get((target_A, target_sym), 0.0)
            nat_abund = frac * 100

            score = peak_xs * (nat_abund / 100.0)
            # Penalise exotic beams
            if projectile in ("t", "h", "g"):
                score *= 0.1
            # Penalise very high energy
            if peak_E > 50:
                score *= 0.5

            if score > best_score:
                best_score = score
                best_route = {
                    "projectile": projectile, "target_A": target_A,
                    "peak_xs": peak_xs, "peak_E": peak_E,
                    "library": library, "abundance": nat_abund,
                    "score": score,
                    "facility": classify_facility(projectile, peak_E),
                }

        if best_route:
            route_table.append({"isotope": c, **best_route})

    # Sort by score descending
    route_table.sort(key=lambda x: x["score"], reverse=True)

    # Output table
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: (auto, auto, auto, auto, auto, auto, auto, auto),")
    lines.append("    align: (center,) * 8,")
    lines.append("    stroke: 0.5pt,")
    lines.append("    table.header(")
    lines.append('      [*Isotope*], [*Beam*], [*Target A*], [*Abund. (%)*], '
                 '[*σ#sub[peak] (mb)*], [*E#sub[peak] (MeV)*], [*Library*], [*Facility*],')
    lines.append("    ),")

    for rt in route_table[:60]:
        iso = rt["isotope"]
        nuc = _nuc_typst(iso["symbol"], iso["A"])
        lines.append(
            f"    [{nuc}], [{rt['projectile']}], [{rt['target_A']}], "
            f"[{rt['abundance']:.1f}], [{rt['peak_xs']:.1f}], "
            f"[{rt['peak_E']:.1f}], [{_typst_escape(rt['library'])}], [{rt['facility']}],"
        )

    lines.append("  ),")
    lines.append(f'  caption: [Best production route per candidate (top {min(60, len(route_table))} by feasibility score). '
                 "Peak cross-section σ from evaluated nuclear data libraries.],")
    lines.append(") <tab:production>")
    lines.append("")

    # Facility summary
    fac_counts = Counter(rt["facility"] for rt in route_table)
    lines.append(f"Of the {len(route_table)} candidates with identified routes, "
                 + ", ".join(f"{n} are accessible via *{f}*" for f, n in fac_counts.most_common())
                 + ".")
    lines.append("")

    if no_route:
        syms = ", ".join(_nuc_typst(c["symbol"], c["A"]) for c in no_route[:15])
        lines.append(f"*{len(no_route)} candidates lack any evaluated cross-section data* "
                     f"in the surveyed libraries, including: {syms}. "
                     "These represent gaps where experimental measurements or dedicated "
                     "TALYS calculations are needed.")
        lines.append("")

    return "\n".join(lines)


# ── Section 3: Theranostic Pairing + Coordination Chemistry ──────────

def section_pairing(db, candidates: list[dict]) -> str:
    """Analyse theranostic diagnostic pairing and chelator compatibility."""
    lines = []
    lines.append("= Theranostic Pairing and Coordination Chemistry")
    lines.append("")
    lines.append(
        "A theranostic pair consists of a diagnostic isotope (for PET or SPECT imaging) "
        "and a therapeutic isotope sharing the same chemical behaviour — ideally the same "
        "element or a chemical analogue that coordinates identically with a given chelator "
        "@price2014 @cutler2013. We analyse same-element and chemical-family pairings "
        "for all screened candidates."
    )
    lines.append("")

    cand_by_Z: dict[int, list[dict]] = {}
    for c in candidates:
        cand_by_Z.setdefault(c["Z"], []).append(c)

    # Find diagnostic isotopes: EC or β⁺ decay, suitable half-life (>10 min)
    diag_rows = db.sql("""
        SELECT Z, A, symbol, half_life_s, decay_1
        FROM ground_states
        WHERE decay_1 IN ('EC', 'B+', 'EC+B+')
          AND half_life_s > 600
          AND half_life_s < 864000
        ORDER BY Z, A
    """).fetchall()

    diag_by_Z: dict[int, list[dict]] = {}
    for r in diag_rows:
        d = {"Z": r[0], "A": r[1], "symbol": r[2],
             "half_life_s": r[3], "decay": r[4] or ""}
        diag_by_Z.setdefault(r[0], []).append(d)

    # == Same-element pairs ==
    lines.append("== Same-Element Pairs")
    lines.append("")

    pairs = []
    for Z, theraps in cand_by_Z.items():
        diags = diag_by_Z.get(Z, [])
        for th in theraps:
            for dg in diags:
                if dg["A"] == th["A"]:
                    continue
                pairs.append({"therapeutic": th, "diagnostic": dg})

    lines.append(f"We identify *{len(pairs)} same-element theranostic pairs* "
                 "across the candidate set.")
    lines.append("")

    # Top 15 pairs by therapeutic particulate dose
    top_pairs = sorted(pairs, key=lambda p: p["therapeutic"]["particulate"], reverse=True)[:15]

    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: (auto, auto, auto, auto, auto, auto),")
    lines.append("    align: (center,) * 6,")
    lines.append("    stroke: 0.5pt,")
    lines.append("    table.header(")
    lines.append('      [*Therapeutic*], [*t#sub[½]*], [*Range*], '
                 '[*Diagnostic*], [*t#sub[½]*], [*Modality*],')
    lines.append("    ),")

    for p in top_pairs:
        th = p["therapeutic"]
        dg = p["diagnostic"]
        modality = "PET" if dg["decay"] in ("B+", "EC+B+") else "SPECT/EC"
        lines.append(
            f"    [{_nuc_typst(th['symbol'], th['A'])}], "
            f"[{_hl_label(th['half_life_s'])}], [{th['range_class']}], "
            f"[{_nuc_typst(dg['symbol'], dg['A'])}], "
            f"[{_hl_label(dg['half_life_s'])}], [{modality}],"
        )

    lines.append("  ),")
    lines.append('  caption: [Top same-element theranostic pairs ranked by therapeutic dose.],')
    lines.append(") <tab:same-element>")
    lines.append("")

    # == Chemical-family pairs ==
    lines.append("== Chemical-Family Pairing")
    lines.append("")
    lines.append(
        "Beyond same-element pairs, chemical analogues within established "
        "chelator families enable cross-element theranostics. "
        "Lanthanides coordinate equivalently via DOTA, group-3 metals "
        "(Sc/Y/La/Ac) share trivalent chemistry, halogens form direct "
        "covalent bonds, and the Tc/Re pair exploits identical oxidation states @muller2017."
    )
    lines.append("")

    family_groups = [
        ("Lanthanides (DOTA)", LANTHANIDES),
        ("Group 3 (Sc/Y/La/Ac)", GROUP3_SC_Y_LA_AC),
        ("Halogens (covalent)", HALOGENS),
        ("Tc/Re", TC_RE),
    ]

    for fname, Zset in family_groups:
        members_th = [c for c in candidates if c["Z"] in Zset]
        members_dg = [d for zlist in diag_by_Z.values()
                      for d in zlist if d["Z"] in Zset]
        if members_th:
            th_str = ", ".join(_nuc_typst(c["symbol"], c["A"]) for c in members_th[:8])
            dg_str = ", ".join(_nuc_typst(d["symbol"], d["A"]) for d in members_dg[:8]) if members_dg else "—"
            lines.append(f"*{fname}*: therapeutic: {th_str}; diagnostic: {dg_str}")
            lines.append("")

    # == Chelator compatibility ==
    lines.append("== Chelator Compatibility")
    lines.append("")
    lines.append(
        "The choice of bifunctional chelator determines which radiometals "
        "can label a given targeting vector. We map each candidate's element "
        "to established chelator families @price2014 @vermeulen2019."
    )
    lines.append("")

    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: (auto, auto),")
    lines.append("    align: (left, left),")
    lines.append("    stroke: 0.5pt,")
    lines.append("    table.header([*Chelator*], [*Compatible elements*]),")
    cand_syms = {c["symbol"] for c in candidates}
    for chel, elems in CHELATOR_MAP.items():
        highlighted = []
        for el in sorted(elems):
            if el in cand_syms:
                highlighted.append(f"*{el}*")
            else:
                highlighted.append(el)
        lines.append(f"    [{chel}], [{', '.join(highlighted)}],")
    lines.append("  ),")
    lines.append('  caption: [Chelator–element compatibility. '
                 'Bold elements have therapeutic candidates in the screened set.],')
    lines.append(") <tab:chelators>")
    lines.append("")

    return "\n".join(lines)


# ── Figures ───────────────────────────────────────────────────────────

def generate_figures(db, candidates: list[dict]) -> str:
    """Generate matplotlib figures and return Typst markup to include them."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "// matplotlib not available — figures skipped\n"

    typst = []

    # ── Fig 1: Dose vs range scatter ──
    range_order = {"Subcellular": 0.5, "Cellular": 15, "Cluster": 200,
                   "Macroscopic": 1000, "Multi-range": 100, "—": 50}
    imaging_colors = {"PET": "#e41a1c", "SPECT": "#377eb8",
                      "Pair": "#4daf4a", "None": "#999999"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for img_type, color in imaging_colors.items():
        subset = [c for c in candidates if c["imaging"] == img_type]
        if not subset:
            continue
        x = [range_order.get(c["range_class"], 50) for c in subset]
        y = [c["particulate"] for c in subset]
        labels = [f"{c['symbol']}-{c['A']}" for c in subset]
        ax.scatter(x, y, c=color, label=img_type, alpha=0.7, s=40, edgecolors="k", linewidths=0.3)
        for xi, yi, lab in zip(x, y, labels):
            if yi > 0.05 or img_type != "None":
                ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.8,
                            xytext=(2, 2), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Approximate range class (μm)", fontsize=10)
    ax.set_ylabel("Particulate dose (MeV/Bq·s)", fontsize=10)
    ax.set_title("Therapeutic Isotope Landscape", fontsize=12)
    ax.legend(title="Imaging", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1_path = OUT_DIR / "fig_dose_range.svg"
    fig.savefig(fig1_path, format="svg")
    plt.close(fig)
    typst.append(
        '#figure(\n'
        '  image("fig_dose_range.svg", width: 100%),\n'
        '  caption: [Therapeutic isotope landscape: particulate dose vs. '
        'approximate range class, coloured by intrinsic imaging capability.],\n'
        ') <fig:dose-range>\n'
    )

    # ── Fig 2: Periodic table heatmap ──
    # Count candidates per element
    z_counts: dict[int, int] = {}
    for c in candidates:
        z_counts[c["Z"]] = z_counts.get(c["Z"], 0) + 1

    # Standard periodic table layout: (row, col) for each Z
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
        87: (6, 0), 88: (6, 1),
        103: (6, 2),
        # Lanthanides
        57: (8, 2), 58: (8, 3), 59: (8, 4), 60: (8, 5), 61: (8, 6),
        62: (8, 7), 63: (8, 8), 64: (8, 9), 65: (8, 10), 66: (8, 11),
        67: (8, 12), 68: (8, 13), 69: (8, 14), 70: (8, 15),
        # Actinides
        89: (9, 2), 90: (9, 3), 91: (9, 4), 92: (9, 5),
    }

    # Z → symbol lookup from elements table
    sym_lookup = {}
    all_elems = db.sql("SELECT Z, symbol FROM elements ORDER BY Z").fetchall()
    for r in all_elems:
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

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label("Number of candidates", fontsize=8)

    fig2.tight_layout()
    fig2_path = OUT_DIR / "fig_periodic_table.svg"
    fig2.savefig(fig2_path, format="svg")
    plt.close(fig2)
    typst.append(
        '#figure(\n'
        '  image("fig_periodic_table.svg", width: 100%),\n'
        '  caption: [Periodic table heatmap showing the number of theranostic '
        'candidates per element. Darker shading indicates more isotopes in the '
        '2 h – 30 d therapeutic window.],\n'
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
same-element pairing (e.g., #super[68]Ga / #super[177]Lu via DOTA) or
chemical-family substitution within a chelator scaffold @price2014.

Historically, theranostic isotope selection has been opportunistic: driven
by available supply rather than systematic optimisation. The nucl-parquet
database, aggregating ENSDF decay data, TENDL/ENDF evaluated cross-sections,
natural abundances, and stopping powers, enables a comprehensive
computational screening @nelson2020.

This report proceeds in three parts:
+ *Screening* — identifying therapeutic candidates by half-life and dose.
+ *Production* — evaluating cross-section routes and facility requirements.
+ *Pairing* — matching therapeutic isotopes with diagnostic companions and chelators.

"""


# ── Conclusion ────────────────────────────────────────────────────────

def section_conclusion(candidates: list[dict]) -> str:
    n_pet = sum(1 for c in candidates if c["imaging"] == "PET")
    n_spect = sum(1 for c in candidates if c["imaging"] == "SPECT")
    n_alpha = sum(1 for c in candidates if c["alpha"] > 0.001)

    return f"""
= Conclusion and Future Directions

This data-driven survey identifies *{len(candidates)} therapeutic isotope
candidates* in the 2-hour to 30-day half-life window, of which
{n_pet} offer intrinsic PET imaging, {n_spect} are SPECT-compatible,
and {n_alpha} emit α-particles for targeted alpha therapy.

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

    print("Section 1: Screening...")
    s1, candidates = section_screening(db)

    print(f"  → {len(candidates)} candidates")

    print("Section 2: Production routes...")
    s2 = section_production(db, candidates)

    print("Section 3: Pairing + coordination chemistry...")
    s3 = section_pairing(db, candidates)

    print("Generating figures...")
    figs = generate_figures(db, candidates)

    print("Assembling Typst document...")
    doc = TYPST_PREAMBLE
    doc += section_introduction()
    doc += s1 + "\n\n"
    doc += figs + "\n\n"
    doc += s2 + "\n\n"
    doc += s3 + "\n\n"
    doc += section_conclusion(candidates) + "\n"
    doc += TYPST_BIBLIOGRAPHY

    TYP_FILE.write_text(doc)
    print(f"Written: {TYP_FILE}")

    # Try to compile
    try:
        result = subprocess.run(
            ["typst", "compile", str(TYP_FILE)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"Compiled: {TYP_FILE.with_suffix('.pdf')}")
        else:
            print(f"Typst compilation failed:\n{result.stderr}")
    except FileNotFoundError:
        print("typst not found — skipping compilation. Run: typst compile docs/theranostics/theranostics.typ")
    except subprocess.TimeoutExpired:
        print("typst compilation timed out")


if __name__ == "__main__":
    main()
