"""Build a single self-contained HTML file for therapeutic isotope production routes.

Connects via loader.py, queries ground_states and radiation tables for therapy
candidates, then fetches excitation functions from the xs view.

Categories:
  β⁻  — targeted radionuclide therapy (e.g. Lu-177, Y-90, I-131)
  Auger — ultra-local EC/conversion-electron therapy (e.g. In-111, I-125)
  Alpha — targeted alpha therapy (e.g. At-211, Ac-225, Bi-213)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import loader
from target_feasibility import load_abundance_map, load_element_z, assess_target, wikipedia_url

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBRARY_PRIORITY = ["iaea-medical", "tendl-2025", "tendl-2024"]
ALLOWED_LIBRARIES = ("iaea-medical", "tendl-2025", "tendl-2024")

# Maximum beam energy for each projectile
MAX_E = {"p": 18.0, "d": 9.0}

# Half-life windows (seconds)
HL_BETA_MIN, HL_BETA_MAX = 21_600, 2_592_000       # 6 h – 30 d
HL_AUGER_MIN, HL_AUGER_MAX = 3_600, 2_592_000       # 1 h – 30 d
HL_ALPHA_MIN, HL_ALPHA_MAX = 3_600, 2_592_000       # 1 h – 30 d

# Minimum peak cross-section to consider a route (mb)
MIN_PEAK_XS = 10.0

# Bogus xs cap (filters inf/overflow values in some libraries)
XS_CAP = 1e10


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def format_half_life(half_life_s: float) -> str:
    """Format half-life into a human-readable string."""
    if half_life_s < 60:
        return f"{half_life_s:.1f} s"
    elif half_life_s < 3600:
        return f"{half_life_s / 60:.1f} min"
    elif half_life_s < 86400:
        return f"{half_life_s / 3600:.2f} h"
    else:
        return f"{half_life_s / 86400:.1f} d"


def superscript_mass(text: str) -> str:
    """'68Zn' -> '<sup>68</sup>Zn'."""
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    if i == 0:
        return text
    return f"<sup>{text[:i]}</sup>{text[i:]}"


def reaction_notation(projectile: str, target: str, residual: str) -> str:
    """Build '⁶⁸Zn(p,x)⁶⁸Ga' with HTML superscripts."""
    return f"{superscript_mass(target)}({projectile},x){superscript_mass(residual)}"


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------


def query_therapy_isotopes(db) -> list[dict]:
    """Return all therapy isotopes across the three categories.

    Returns a list of dicts with keys: Z, A, symbol, half_life_s, category.
    """
    rows = db.execute(f"""
        WITH bm AS (
            SELECT Z, A, symbol, half_life_s, 'beta-' AS category
            FROM ground_states
            WHERE (decay_1 = 'B-' OR decay_2 = 'B-' OR decay_3 = 'B-')
              AND half_life_s BETWEEN {HL_BETA_MIN} AND {HL_BETA_MAX}
        ),
        auger AS (
            SELECT Z, A, symbol, half_life_s, 'auger' AS category
            FROM ground_states
            WHERE decay_1 IN ('EC', 'EC+B+', 'B+')
              AND half_life_s BETWEEN {HL_AUGER_MIN} AND {HL_AUGER_MAX}
        ),
        alpha AS (
            SELECT Z, A, symbol, half_life_s, 'alpha' AS category
            FROM ground_states
            WHERE decay_1 = 'A'
              AND half_life_s BETWEEN {HL_ALPHA_MIN} AND {HL_ALPHA_MAX}
        )
        SELECT * FROM bm
        UNION ALL
        SELECT * FROM auger
        UNION ALL
        SELECT * FROM alpha
        ORDER BY Z, A, category
    """).fetchall()

    seen: set[tuple[int, int, str]] = set()
    result = []
    for Z, A, symbol, half_life_s, category in rows:
        key = (Z, A, category)
        if key not in seen:
            seen.add(key)
            result.append({
                "Z": int(Z),
                "A": int(A),
                "symbol": symbol,
                "half_life_s": float(half_life_s) if half_life_s else 0.0,
                "category": category,
                "name": f"{A}{symbol}",
            })
    return result


def query_all_routes(db) -> dict[tuple[int, int, str], list[dict]]:
    """Fetch best-library routes for all p/d reactions, keyed by (Z, A, category).

    Returns: {(residual_Z, residual_A): [route_dicts]}  — category included separately.

    We use a two-step approach:
    1. Get one best-library row per (projectile, target_el, target_A, residual_Z, residual_A).
    2. Caller will match by (residual_Z, residual_A).
    """
    # Extract all routes: per unique (library, proj, target_el, target_A, resZ, resA)
    print("  Querying all p/d routes from xs...", flush=True)
    lib_filter = ",".join(f"'{l}'" for l in ALLOWED_LIBRARIES)
    rows = db.execute(f"""
        SELECT
            xs.library,
            regexp_extract(xs.filename, '/([pd])_', 1) AS projectile,
            regexp_extract(xs.filename, '/[pd]_([A-Za-z]+)', 1) AS target_el,
            xs.target_A,
            xs.residual_Z,
            xs.residual_A,
            MAX(xs.xs_mb) AS peak_xs,
            FIRST(xs.energy_MeV ORDER BY xs.xs_mb DESC) AS peak_E
        FROM xs
        WHERE (xs.filename LIKE '%/p_%' OR xs.filename LIKE '%/d_%')
          AND xs.library IN ({lib_filter})
          AND xs.energy_MeV <= CASE
                WHEN xs.filename LIKE '%/p_%' THEN {MAX_E['p']}
                ELSE {MAX_E['d']}
              END
          AND xs.xs_mb < {XS_CAP}
          AND xs.xs_mb > 0
        GROUP BY xs.library, projectile, target_el, xs.target_A, xs.residual_Z, xs.residual_A
        HAVING MAX(xs.xs_mb) > {MIN_PEAK_XS}
    """).fetchall()

    print(f"  Raw route candidates: {len(rows)}", flush=True)

    # Group by (proj, target_el, target_A, resZ, resA) and pick best library
    by_route: dict[tuple, dict] = {}
    for lib, proj, target_el, target_A, resZ, resA, peak_xs, peak_E in rows:
        if not target_el or not proj:
            continue
        key = (proj, target_el, int(target_A), int(resZ), int(resA))
        if key not in by_route:
            by_route[key] = {
                "library": lib,
                "projectile": proj,
                "target_el": target_el,
                "target_A": int(target_A),
                "residual_Z": int(resZ),
                "residual_A": int(resA),
                "peak_xs_mb": float(peak_xs),
                "peak_E_MeV": float(peak_E) if peak_E is not None else 0.0,
            }
        else:
            # Prefer higher-priority library
            existing_lib = by_route[key]["library"]
            if _lib_priority(lib) < _lib_priority(existing_lib):
                by_route[key]["library"] = lib
                by_route[key]["peak_xs_mb"] = float(peak_xs)
                by_route[key]["peak_E_MeV"] = float(peak_E) if peak_E is not None else 0.0

    # Fetch elements table for identity exclusion
    el_z: dict[str, int] = {
        sym: Z
        for Z, sym in db.execute("SELECT Z, symbol FROM elements").fetchall()
    }

    # Group by residual (Z, A)
    by_residual: dict[tuple[int, int], list[dict]] = {}
    for route in by_route.values():
        resZ = route["residual_Z"]
        resA = route["residual_A"]
        target_el = route["target_el"]
        target_A = route["target_A"]
        target_Z = el_z.get(target_el, -1)
        # Exclude identity reactions
        if target_Z == resZ and target_A == resA:
            continue
        # Add target nuclide name and reaction notation
        target_name = f"{target_A}{target_el}"
        route["target"] = target_name
        route["residual"] = f"{resA}{route.get('residual_symbol', '')}"
        key = (resZ, resA)
        by_residual.setdefault(key, []).append(route)

    return by_residual


def _lib_priority(lib: str) -> int:
    """Lower is better."""
    try:
        return LIBRARY_PRIORITY.index(lib)
    except ValueError:
        return len(LIBRARY_PRIORITY)


def query_radiation_summary(db, Z: int, A: int) -> dict:
    """Compute therapeutic dose, gamma burden, imaging channel, and key emissions."""
    rows = db.execute("""
        SELECT rad_type, rad_subtype, energy_keV, intensity_pct, dose_MeV_per_Bq_s
        FROM radiation
        WHERE Z = ? AND A = ?
          AND parent_level_keV = 0
    """, [Z, A]).fetchall()

    therapeutic_dose = 0.0
    gamma_dose = 0.0
    beta_emissions: list[dict] = []
    electron_emissions: list[dict] = []
    alpha_emissions: list[dict] = []
    gamma_lines: list[dict] = []

    for rad_type, rad_subtype, energy_keV, intensity_pct, dose in rows:
        if dose is None:
            dose = 0.0
        if intensity_pct is None:
            intensity_pct = 0.0
        if energy_keV is None:
            energy_keV = 0.0

        if rad_type == "beta-":
            therapeutic_dose += dose
            beta_emissions.append({
                "type": "β⁻",
                "energy_keV": round(energy_keV, 1),
                "intensity_pct": round(intensity_pct, 2),
                "dose": round(dose, 6),
            })
        elif rad_type in ("auger", "ce"):
            electron_emissions.append({
                "type": rad_subtype or rad_type,
                "energy_keV": round(energy_keV, 1),
                "intensity_pct": round(intensity_pct, 2),
                "dose": round(dose, 6),
            })
        elif rad_type == "alpha":
            alpha_emissions.append({
                "type": "α",
                "energy_keV": round(energy_keV, 1),
                "intensity_pct": round(intensity_pct, 2),
                "dose": round(dose, 6),
            })
        elif rad_type == "gamma":
            gamma_dose += dose
            gamma_lines.append({
                "energy_keV": round(energy_keV, 1),
                "intensity_pct": round(intensity_pct, 2),
                "dose": round(dose, 6),
            })

    # Determine imaging channel
    imaging = "none"
    has_pet = any(
        abs(g["energy_keV"] - 511) < 5 and g["intensity_pct"] >= 1.0
        for g in gamma_lines
    )
    has_spect = any(
        50 <= g["energy_keV"] <= 300 and g["intensity_pct"] >= 5.0
        for g in gamma_lines
    )
    if has_pet and has_spect:
        imaging = "SPECT+PET"
    elif has_pet:
        imaging = "PET"
    elif has_spect:
        imaging = "SPECT"

    # Sort and truncate to top 3
    beta_emissions.sort(key=lambda x: x["intensity_pct"], reverse=True)
    electron_emissions.sort(key=lambda x: x["intensity_pct"], reverse=True)
    alpha_emissions.sort(key=lambda x: x["intensity_pct"], reverse=True)
    gamma_lines.sort(key=lambda x: x["intensity_pct"], reverse=True)

    return {
        "beta_dose": round(therapeutic_dose, 6),
        "electron_dose": round(sum(e["dose"] for e in electron_emissions), 6),
        "alpha_dose": round(sum(e["dose"] for e in alpha_emissions), 6),
        "gamma_dose": round(gamma_dose, 6),
        "imaging": imaging,
        "beta_emissions": beta_emissions[:3],
        "electron_emissions": electron_emissions[:3],
        "alpha_emissions": alpha_emissions[:3],
        "gamma_lines": gamma_lines[:3],
    }


def best_excitation_function(
    db,
    projectile: str,
    target_el: str,
    target_A: int,
    residual_Z: int,
    residual_A: int,
) -> list[dict]:
    """Fetch excitation function, preferring best library."""
    max_e = MAX_E[projectile]
    lib_filter_sql = ",".join(f"'{l}'" for l in ALLOWED_LIBRARIES)
    rows = db.execute(f"""
        SELECT library, energy_MeV, xs_mb
        FROM xs
        WHERE filename LIKE '%' || ? || '_' || ? || '%'
          AND target_A = ?
          AND residual_Z = ?
          AND residual_A = ?
          AND energy_MeV <= ?
          AND xs_mb > 0
          AND xs_mb < ?
          AND library IN ({lib_filter_sql})
        ORDER BY energy_MeV
    """, [projectile, target_el, target_A, residual_Z, residual_A, max_e, XS_CAP]).fetchall()

    if not rows:
        return [], 0.0, 0.0

    by_lib: dict[str, list[tuple[float, float]]] = {}
    for lib, e, xs in rows:
        by_lib.setdefault(lib, []).append((e, xs))

    chosen_lib = None
    for lib in LIBRARY_PRIORITY:
        if lib in by_lib:
            chosen_lib = lib
            break
    if chosen_lib is None:
        chosen_lib = max(by_lib, key=lambda k: len(by_lib[k]))

    pts = by_lib[chosen_lib]
    xs_data = [{"energy_MeV": round(e, 4), "xs_mb": round(xs, 4)} for e, xs in pts]
    peak_xs = max(xs for _, xs in pts) if pts else 0.0
    peak_E = max(pts, key=lambda p: p[1])[0] if pts else 0.0
    return xs_data, round(peak_xs, 4), round(peak_E, 4)


# ---------------------------------------------------------------------------
# Main data assembly
# ---------------------------------------------------------------------------


def build_data(db) -> tuple[list[dict], dict]:
    """Build full data for all three therapeutic categories."""
    print("Querying therapy isotopes...", flush=True)
    isotopes = query_therapy_isotopes(db)

    print("Querying all p/d routes...", flush=True)
    routes_by_residual = query_all_routes(db)

    print("Loading abundance/element maps...", flush=True)
    abundance_map = load_abundance_map(db)
    el_z_map_feas = load_element_z(db)

    # Attach symbol to routes (needed for reaction notation)
    symbol_map: dict[tuple[int, int], str] = {
        (iso["Z"], iso["A"]): iso["symbol"] for iso in isotopes
    }

    # Filter to isotopes that have at least one producible route
    result: list[dict] = []
    stats = {"beta-": 0, "auger": 0, "alpha": 0, "total_routes": 0}

    total = len(isotopes)
    print(f"Processing {total} candidate isotopes...", flush=True)

    for idx, iso in enumerate(isotopes, 1):
        Z, A, symbol = iso["Z"], iso["A"], iso["symbol"]
        category = iso["category"]
        name = iso["name"]

        routes = routes_by_residual.get((Z, A), [])
        if not routes:
            continue

        # Compute radiation summary
        rad = query_radiation_summary(db, Z, A)

        # Determine therapeutic dose for this category
        if category == "beta-":
            therapeutic_dose = rad["beta_dose"]
        elif category == "auger":
            therapeutic_dose = rad["electron_dose"]
        else:  # alpha
            therapeutic_dose = rad["alpha_dose"]

        # Sort routes by peak xs descending
        routes = sorted(routes, key=lambda r: r["peak_xs_mb"], reverse=True)

        # Add reaction notation and fetch excitation functions
        for route in routes:
            route["residual"] = name
            route["reaction"] = reaction_notation(
                route["projectile"], route["target"], name
            )
            # Target feasibility
            tgt_Z = el_z_map_feas.get(route["target_el"], -1)
            tgt_A = route["target_A"]
            ab = abundance_map.get((tgt_Z, tgt_A))
            route["target_info"] = assess_target(tgt_Z, tgt_A, ab, route["target_el"])
            route["wiki_url"] = wikipedia_url(tgt_Z)
            print(
                f"  [{idx}/{total}] {name} ({category}) | "
                f"{route['target']}({route['projectile']},x){name} ...",
                end=" ",
                flush=True,
            )
            xs_data, lib_peak_xs, lib_peak_E = best_excitation_function(
                db,
                projectile=route["projectile"],
                target_el=route["target_el"],
                target_A=route["target_A"],
                residual_Z=Z,
                residual_A=A,
            )
            route["xs_data"] = xs_data
            route["peak_xs_mb"] = lib_peak_xs
            route["peak_E_MeV"] = lib_peak_E
            print(f"{len(xs_data)} pts", flush=True)

        # Re-sort after library-consistent peak values
        routes = sorted(routes, key=lambda r: r["peak_xs_mb"], reverse=True)

        iso_data = {
            "name": name,
            "Z": Z,
            "A": A,
            "symbol": symbol,
            "half_life_s": iso["half_life_s"],
            "half_life_str": format_half_life(iso["half_life_s"]),
            "category": category,
            "therapeutic_dose": round(therapeutic_dose, 6),
            "gamma_dose": rad["gamma_dose"],
            "imaging": rad["imaging"],
            "beta_emissions": rad["beta_emissions"],
            "electron_emissions": rad["electron_emissions"],
            "alpha_emissions": rad["alpha_emissions"],
            "gamma_lines": rad["gamma_lines"],
            "best_peak_xs_mb": routes[0]["peak_xs_mb"] if routes else 0.0,
            "routes": routes,
        }
        result.append(iso_data)
        stats[category] += 1
        stats["total_routes"] += len(routes)

    # Sort by therapeutic dose descending within each category
    result.sort(key=lambda x: x["therapeutic_dose"], reverse=True)
    return result, stats


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Therapeutic Isotope Production Routes</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222536;
    --surface3: #2a2e42;
    --border: #333750;
    --text: #e2e8f0;
    --text-muted: #8892a4;
    --accent: #6c8ef7;
    --proton-color: #6c8ef7;
    --deuteron-color: #fb923c;
    --beta-color: #60a5fa;
    --auger-color: #c084fc;
    --alpha-color: #f87171;
    --green: #4ade80;
    --amber: #fbbf24;
    --grey: #64748b;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
  }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  header h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
  header p { color: var(--text-muted); font-size: 13px; margin-top: 2px; }

  .controls {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    align-items: center;
    flex-wrap: wrap;
  }

  .search-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 12px;
    color: var(--text);
    font-size: 13px;
    width: 200px;
    outline: none;
    transition: border-color 0.15s;
  }
  .search-box:focus { border-color: var(--accent); }
  .search-box::placeholder { color: var(--text-muted); }

  .filter-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 14px;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
  }
  .filter-btn:hover:not(.active) { border-color: var(--accent); color: var(--text); }
  .filter-btn.active[data-filter="all"]   { background: var(--accent); border-color: var(--accent); color: #fff; }
  .filter-btn.active[data-filter="beta-"] { background: var(--beta-color); border-color: var(--beta-color); color: #0f1117; }
  .filter-btn.active[data-filter="auger"] { background: var(--auger-color); border-color: var(--auger-color); color: #0f1117; }
  .filter-btn.active[data-filter="alpha"] { background: var(--alpha-color); border-color: var(--alpha-color); color: #0f1117; }

  .sort-select {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    color: var(--text);
    font-size: 13px;
    outline: none;
    cursor: pointer;
  }
  .sort-select:focus { border-color: var(--accent); }

  .toggle-label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    transition: all 0.15s;
  }
  .toggle-label:has(input:checked) { border-color: var(--accent); color: var(--text); }
  .toggle-label input { accent-color: var(--accent); cursor: pointer; }

  .slider-group {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
  }
  .slider-group label { white-space: nowrap; }
  .slider {
    -webkit-appearance: none;
    width: 80px;
    height: 4px;
    border-radius: 2px;
    background: var(--surface3);
    outline: none;
    cursor: pointer;
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }

  .stats { margin-left: auto; color: var(--text-muted); font-size: 12px; }

  main {
    max-width: 1120px;
    margin: 0 auto;
    padding: 20px 16px;
  }

  /* ---- Isotope card ---- */
  .iso-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 8px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .iso-card:hover { border-color: #445; }
  .iso-card.open { border-color: var(--cat-color, var(--accent)); }

  .iso-card[data-cat="beta-"]  { --cat-color: var(--beta-color); }
  .iso-card[data-cat="auger"]  { --cat-color: var(--auger-color); }
  .iso-card[data-cat="alpha"]  { --cat-color: var(--alpha-color); }

  .iso-header {
    display: grid;
    grid-template-columns: 90px 120px 130px 130px 110px 36px;
    align-items: center;
    padding: 13px 16px;
    cursor: pointer;
    gap: 10px;
    user-select: none;
  }
  .iso-header:hover { background: var(--surface2); }

  .iso-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--cat-color, var(--accent));
    letter-spacing: -0.2px;
  }
  .iso-half-life { color: var(--text-muted); font-size: 12px; margin-top: 2px; }

  .cat-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
  }
  .cat-beta  { background: rgba(96,165,250,0.15); border: 1px solid rgba(96,165,250,0.3); color: var(--beta-color); }
  .cat-auger { background: rgba(192,132,252,0.15); border: 1px solid rgba(192,132,252,0.3); color: var(--auger-color); }
  .cat-alpha { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.3); color: var(--alpha-color); }

  .dose-col { text-align: right; }
  .dose-col .value { font-weight: 700; font-size: 14px; font-variant-numeric: tabular-nums; }
  .dose-col .label { color: var(--text-muted); font-size: 11px; }

  .imaging-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
  }
  .img-spect     { background: rgba(74,222,128,0.15); border: 1px solid rgba(74,222,128,0.3); color: var(--green); }
  .img-pet       { background: rgba(74,222,128,0.15); border: 1px solid rgba(74,222,128,0.3); color: var(--green); }
  .img-spect-pet { background: rgba(74,222,128,0.25); border: 1px solid rgba(74,222,128,0.5); color: var(--green); }
  .img-none      { background: rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); color: var(--grey); }

  .route-count { text-align: center; color: var(--text-muted); font-size: 12px; }
  .route-count strong { color: var(--text); font-size: 14px; display: block; }

  .expand-icon {
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 16px;
    transition: transform 0.2s, color 0.15s;
  }
  .open .expand-icon { transform: rotate(90deg); color: var(--cat-color, var(--accent)); }

  /* ---- Routes panel ---- */
  .routes-panel {
    display: none;
    border-top: 1px solid var(--border);
    padding: 0 16px 16px;
  }
  .iso-card.open .routes-panel { display: block; }

  /* Emissions summary */
  .emissions-grid {
    display: flex;
    gap: 16px;
    margin-top: 14px;
    flex-wrap: wrap;
  }
  .emissions-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1;
    min-width: 180px;
  }
  .emissions-block h4 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 6px;
    font-weight: 600;
  }
  .emission-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 2px 0;
    border-bottom: 1px solid rgba(51,55,80,0.4);
  }
  .emission-row:last-child { border-bottom: none; }
  .emission-type { color: var(--text-muted); }
  .emission-vals { font-variant-numeric: tabular-nums; }

  /* Routes table */
  .routes-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
    font-size: 13px;
  }
  .routes-table th {
    text-align: left;
    color: var(--text-muted);
    font-weight: 500;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .routes-table td {
    padding: 9px 12px;
    border-bottom: 1px solid rgba(51,55,80,0.5);
    vertical-align: middle;
  }
  .routes-table tr:last-child td { border-bottom: none; }
  .routes-table tr.route-row { cursor: pointer; transition: background 0.12s; }
  .routes-table tr.route-row:hover { background: var(--surface2); }
  .routes-table tr.route-row.active-route { background: var(--surface3); }

  .proj-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }
  .proj-p { background: rgba(108,142,247,0.2); color: var(--proton-color); }
  .proj-d { background: rgba(251,146,60,0.2); color: var(--deuteron-color); }

  .wiki-link { color: inherit; text-decoration: none; border-bottom: 1px dotted var(--text-muted); }
  .wiki-link:hover { color: var(--accent); border-bottom-color: var(--accent); }

  .xs-value { font-variant-numeric: tabular-nums; font-weight: 600; }
  .abundance-cell { font-variant-numeric: tabular-nums; font-size: 12px; color: var(--text-muted); }

  .feas-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
    cursor: help;
  }
  .feas-excellent { background: rgba(74,222,128,0.15); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; }
  .feas-good { background: rgba(96,165,250,0.15); border: 1px solid rgba(96,165,250,0.3); color: #60a5fa; }
  .feas-moderate { background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
  .feas-bad { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.3); color: #f87171; }

  .xs-bar-cell { width: 80px; }
  .xs-bar-wrap { background: var(--surface3); border-radius: 3px; height: 6px; overflow: hidden; }
  .xs-bar { height: 6px; border-radius: 3px; transition: width 0.3s; }
  .bar-p { background: var(--proton-color); }
  .bar-d { background: var(--deuteron-color); }

  /* Plot panel */
  .plot-panel {
    display: none;
    margin-top: 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  .plot-panel.visible { display: block; }
  .plot-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .plot-title { font-size: 13px; font-weight: 600; color: var(--text); }
  .plot-subtitle { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .plot-close {
    background: none; border: none; color: var(--text-muted);
    font-size: 18px; cursor: pointer; padding: 2px 6px; border-radius: 4px;
  }
  .plot-close:hover { color: var(--text); background: var(--surface3); }
  .chart-container { position: relative; height: 260px; }
  .no-data { text-align: center; color: var(--text-muted); padding: 40px; font-size: 13px; }

  @media (max-width: 700px) {
    .iso-header { grid-template-columns: 1fr auto 36px; }
    .iso-header .cat-badge,
    .iso-header .dose-col,
    .iso-header .route-count { display: none; }
    .search-box { width: 140px; }
  }
</style>
</head>
<body>

<header>
  <h1>Therapeutic Isotope Production Routes</h1>
  <p>18 MeV proton / 9 MeV deuteron cyclotron — TENDL-2025 / TENDL-2024 / IAEA-Medical — <span id="vis-count"></span> isotopes</p>
  <div class="controls">
    <input class="search-box" type="search" placeholder="Search isotope…" id="search" autocomplete="off">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="beta-">β⁻</button>
    <button class="filter-btn" data-filter="auger">Auger</button>
    <button class="filter-btn" data-filter="alpha">Alpha</button>
    <select class="sort-select" id="sort-select">
      <option value="dose-desc">Sort: Therapeutic dose ↓</option>
      <option value="dose-asc">Sort: Therapeutic dose ↑</option>
      <option value="xs-desc">Sort: Peak σ ↓</option>
      <option value="hl-asc">Sort: Half-life ↑</option>
      <option value="hl-desc">Sort: Half-life ↓</option>
      <option value="routes-desc">Sort: Routes ↓</option>
    </select>
    <span class="stats" id="stats-label"></span>
  </div>
  <div class="controls" style="margin-top:8px;">
    <label class="toggle-label"><input type="checkbox" id="f-no-radio" checked> <span>Hide radioactive targets</span></label>
    <label class="toggle-label"><input type="checkbox" id="f-no-gas"> <span>Hide gas targets</span></label>
    <label class="toggle-label"><input type="checkbox" id="f-natural-only"> <span>Enrichable targets only (&gt;0.05%)</span></label>
    <label class="toggle-label"><input type="checkbox" id="f-has-imaging"> <span>Has imaging (SPECT/PET)</span></label>
    <span class="slider-group">
      <label>Min σ: <span id="xs-val">10</span> mb</label>
      <input type="range" id="f-min-xs" min="0" max="500" step="10" value="10" class="slider">
    </span>
  </div>
</header>

<main>
  <div id="iso-list"></div>
</main>

<script>
const DATA = __DATA_JSON__;

let activeChart = null;
let activeRouteEl = null;
let activePlotPanel = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtXs(v) {
  if (v >= 1000) return (v / 1000).toFixed(2) + ' b';
  if (v >= 10) return v.toFixed(1) + ' mb';
  return v.toFixed(3) + ' mb';
}

function fmtAbundance(info) {
  if (!info || info.abundance_pct === null) return '<span style="color:var(--alpha-color)">—</span>';
  const a = info.abundance_pct;
  if (a >= 50) return a.toFixed(1) + '%';
  if (a >= 1) return a.toFixed(2) + '%';
  if (a >= 0.01) return a.toFixed(3) + '%';
  return a.toFixed(4) + '%';
}

function feasibilityBadge(info) {
  if (!info) return '';
  const f = info.feasibility;
  const cls = f === 'excellent' ? 'feas-excellent'
            : f === 'good' ? 'feas-good'
            : f === 'moderate' ? 'feas-moderate'
            : 'feas-bad';
  return `<span class="feas-badge ${cls}" title="${info.note}">${info.availability}</span>`;
}

function fmtDose(v) {
  if (v === 0) return '—';
  if (v >= 1) return v.toFixed(3) + ' MeV/Bq·s';
  if (v >= 0.001) return (v * 1000).toFixed(2) + ' meV/Bq·s';
  return v.toExponential(2) + ' MeV/Bq·s';
}

function imagingClass(img) {
  if (img === 'SPECT+PET') return 'img-spect-pet';
  if (img === 'SPECT') return 'img-spect';
  if (img === 'PET') return 'img-pet';
  return 'img-none';
}

function catClass(cat) {
  if (cat === 'beta-') return 'cat-beta';
  if (cat === 'auger') return 'cat-auger';
  return 'cat-alpha';
}

function catLabel(cat) {
  if (cat === 'beta-') return 'β⁻';
  if (cat === 'auger') return 'Auger';
  return 'Alpha';
}

function safeId(name) {
  return name.replace(/[^a-zA-Z0-9]/g, '_');
}

// ---------------------------------------------------------------------------
// Build DOM
// ---------------------------------------------------------------------------

function buildEmissions(iso) {
  const blocks = [];

  // Therapeutic emissions
  let therList = [];
  if (iso.category === 'beta-') {
    therList = iso.beta_emissions || [];
  } else if (iso.category === 'auger') {
    therList = iso.electron_emissions || [];
  } else {
    therList = iso.alpha_emissions || [];
  }

  if (therList.length > 0) {
    const rows = therList.map(e =>
      `<div class="emission-row">
        <span class="emission-type">${e.type}</span>
        <span class="emission-vals">${e.energy_keV} keV &nbsp;${e.intensity_pct.toFixed(1)}%</span>
      </div>`
    ).join('');
    const label = iso.category === 'beta-' ? 'β⁻ Emissions'
                : iso.category === 'auger' ? 'e⁻ Emissions'
                : 'α Emissions';
    blocks.push(`<div class="emissions-block"><h4>${label}</h4>${rows}</div>`);
  }

  // Gamma lines
  const gammas = iso.gamma_lines || [];
  if (gammas.length > 0) {
    const rows = gammas.map(g =>
      `<div class="emission-row">
        <span class="emission-type">γ ${g.energy_keV} keV</span>
        <span class="emission-vals">${g.intensity_pct.toFixed(1)}%</span>
      </div>`
    ).join('');
    blocks.push(`<div class="emissions-block"><h4>Gamma Lines</h4>${rows}</div>`);
  }

  if (blocks.length === 0) return '';
  return `<div class="emissions-grid">${blocks.join('')}</div>`;
}

function buildRoutesPanel(iso) {
  const maxXs = iso.routes.reduce((m, r) => Math.max(m, r.peak_xs_mb), 0);
  const sid = safeId(iso.name);

  const rows = iso.routes.map(route => {
    const barPct = maxXs > 0 ? (route.peak_xs_mb / maxXs * 100).toFixed(1) : 0;
    const barClass = route.projectile === 'p' ? 'bar-p' : 'bar-d';
    const routeJson = JSON.stringify(route).replace(/"/g, '&quot;');
    const info = route.target_info || {};
    return `
      <tr class="route-row" onclick="showPlot(this, '${iso.name}', ${routeJson})">
        <td><span class="proj-badge proj-${route.projectile}">${route.projectile}</span></td>
        <td>${route.wiki_url ? '<a href="'+route.wiki_url+'" target="_blank" rel="noopener" class="wiki-link" onclick="event.stopPropagation()" title="Wikipedia: target element">'+route.reaction+'</a>' : route.reaction}</td>
        <td class="xs-value">${fmtXs(route.peak_xs_mb)}</td>
        <td class="xs-value">${route.peak_E_MeV.toFixed(1)} MeV</td>
        <td class="abundance-cell">${fmtAbundance(info)}</td>
        <td>${feasibilityBadge(info)}</td>
        <td class="xs-bar-cell">
          <div class="xs-bar-wrap">
            <div class="xs-bar ${barClass}" style="width:${barPct}%"></div>
          </div>
        </td>
      </tr>`;
  }).join('');

  return `
    ${buildEmissions(iso)}
    <table class="routes-table">
      <thead>
        <tr>
          <th>Beam</th>
          <th>Reaction</th>
          <th>Peak σ</th>
          <th>E<sub>peak</sub></th>
          <th>Abund.</th>
          <th>Target availability</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="plot-panel" id="plot-${sid}">
      <div class="plot-header">
        <div>
          <div class="plot-title" id="pt-title-${sid}"></div>
          <div class="plot-subtitle" id="pt-sub-${sid}"></div>
        </div>
        <button class="plot-close" onclick="closePlot('${sid}')">×</button>
      </div>
      <div class="chart-container">
        <canvas id="chart-${sid}"></canvas>
      </div>
    </div>`;
}

function buildIsotopes(data) {
  const list = document.getElementById('iso-list');
  list.innerHTML = '';

  data.forEach((iso, isoIdx) => {
    const card = document.createElement('div');
    card.className = 'iso-card';
    card.dataset.isoIdx = isoIdx;
    card.dataset.cat = iso.category;

    const imgClass = imagingClass(iso.imaging);
    const imgLabel = iso.imaging === 'none' ? 'No imaging' : iso.imaging;
    const therLabel = iso.category === 'beta-' ? 'β⁻ dose' : iso.category === 'auger' ? 'e⁻ dose' : 'α dose';

    card.innerHTML = `
      <div class="iso-header" onclick="toggleCard(this)">
        <div>
          <div class="iso-name">${iso.name}</div>
          <div class="iso-half-life">t<sub>½</sub> = ${iso.half_life_str}</div>
        </div>
        <div>
          <span class="cat-badge ${catClass(iso.category)}">${catLabel(iso.category)}</span>
        </div>
        <div class="dose-col">
          <div class="label">${therLabel}</div>
          <div class="value">${fmtDose(iso.therapeutic_dose)}</div>
        </div>
        <div class="dose-col">
          <div class="label">γ burden</div>
          <div class="value">${fmtDose(iso.gamma_dose)}</div>
        </div>
        <div>
          <div class="label" style="color:var(--text-muted);font-size:11px;margin-bottom:3px;">Imaging</div>
          <span class="imaging-badge ${imgClass}">${imgLabel}</span>
          <div class="route-count" style="margin-top:4px;"><strong>${iso.routes.length}</strong> route${iso.routes.length !== 1 ? 's' : ''}</div>
        </div>
        <div class="expand-icon">›</div>
      </div>
      <div class="routes-panel">
        ${buildRoutesPanel(iso)}
      </div>
    `;
    list.appendChild(card);
  });

  updateVisCount();
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

function toggleCard(headerEl) {
  const card = headerEl.closest('.iso-card');
  const wasOpen = card.classList.contains('open');
  card.classList.toggle('open', !wasOpen);
  if (wasOpen) {
    const isoName = card.querySelector('.iso-name').textContent;
    closePlot(safeId(isoName));
  }
}

function showPlot(rowEl, isoName, route) {
  const sid = safeId(isoName);
  const panel = document.getElementById('plot-' + sid);
  const titleEl = document.getElementById('pt-title-' + sid);
  const subEl = document.getElementById('pt-sub-' + sid);
  const canvas = document.getElementById('chart-' + sid);
  if (!panel || !canvas) return;

  if (activeRouteEl && activeRouteEl !== rowEl) activeRouteEl.classList.remove('active-route');
  activeRouteEl = rowEl;

  const isActive = rowEl.classList.contains('active-route');
  if (isActive) {
    rowEl.classList.remove('active-route');
    panel.classList.remove('visible');
    if (activeChart) { activeChart.destroy(); activeChart = null; }
    return;
  }

  rowEl.classList.add('active-route');
  panel.classList.add('visible');
  titleEl.innerHTML = route.reaction;
  subEl.textContent = route.xs_data.length > 0
    ? `Excitation function — ${route.xs_data.length} data points`
    : 'No data available';

  if (activeChart && activePlotPanel === sid) { activeChart.destroy(); activeChart = null; }
  activePlotPanel = sid;

  if (!route.xs_data || route.xs_data.length === 0) {
    canvas.style.display = 'none';
    if (!panel.querySelector('.no-data')) {
      const nd = document.createElement('div');
      nd.className = 'no-data';
      nd.textContent = 'No cross-section data available for this route.';
      panel.querySelector('.chart-container').appendChild(nd);
    }
    return;
  }

  canvas.style.display = '';
  const noData = panel.querySelector('.no-data');
  if (noData) noData.remove();

  const energies = route.xs_data.map(p => p.energy_MeV);
  const xsVals = route.xs_data.map(p => p.xs_mb);
  const lineColor = route.projectile === 'p' ? '#6c8ef7' : '#fb923c';
  const fillColor = route.projectile === 'p' ? 'rgba(108,142,247,0.12)' : 'rgba(251,146,60,0.12)';

  activeChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: energies,
      datasets: [{
        label: 'σ (mb)',
        data: xsVals,
        borderColor: lineColor,
        backgroundColor: fillColor,
        borderWidth: 2,
        pointRadius: energies.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
        pointBackgroundColor: lineColor,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#222536',
          borderColor: '#333750',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#8892a4',
          callbacks: {
            title: ctx => `E = ${ctx[0].label} MeV`,
            label: ctx => `σ = ${Number(ctx.raw).toFixed(3)} mb`,
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Energy (MeV)', color: '#8892a4', font: { size: 11 } },
          ticks: { color: '#8892a4', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          title: { display: true, text: 'Cross section (mb)', color: '#8892a4', font: { size: 11 } },
          ticks: { color: '#8892a4', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
          min: 0,
        },
      },
    },
  });
}

function closePlot(sid) {
  const panel = document.getElementById('plot-' + sid);
  if (panel) panel.classList.remove('visible');
  if (activeChart && activePlotPanel === sid) { activeChart.destroy(); activeChart = null; }
  if (activeRouteEl) { activeRouteEl.classList.remove('active-route'); activeRouteEl = null; }
}

// ---------------------------------------------------------------------------
// Filter + Search + Sort
// ---------------------------------------------------------------------------

let currentFilter = 'all';
let currentSearch = '';
let currentSort = 'dose-desc';

function getFilters() {
  return {
    cat: currentFilter,
    search: currentSearch,
    noRadio: document.getElementById('f-no-radio').checked,
    noGas: document.getElementById('f-no-gas').checked,
    naturalOnly: document.getElementById('f-natural-only').checked,
    hasImaging: document.getElementById('f-has-imaging').checked,
    minXs: parseFloat(document.getElementById('f-min-xs').value),
  };
}

function routePassesFilter(route, f) {
  const info = route.target_info || {};
  if (f.noRadio && info.abundance_pct === null) return false;
  if (f.noGas && info.note && info.note.includes('gas target')) return false;
  if (f.naturalOnly && (info.abundance_pct === null || info.abundance_pct < 0.05)) return false;
  if (route.peak_xs_mb < f.minXs) return false;
  return true;
}

function updateVisCount() {
  const cards = document.querySelectorAll('.iso-card');
  const vis = [...cards].filter(c => c.style.display !== 'none').length;
  const total = DATA.length;
  document.getElementById('vis-count').textContent = vis;
  document.getElementById('stats-label').textContent =
    vis < total ? `${vis} of ${total} shown` : `${total} total`;
}

function applySort(data) {
  const d = data.slice();
  switch (currentSort) {
    case 'dose-desc': return d.sort((a, b) => b.therapeutic_dose - a.therapeutic_dose);
    case 'dose-asc':  return d.sort((a, b) => a.therapeutic_dose - b.therapeutic_dose);
    case 'xs-desc':   return d.sort((a, b) => b.best_peak_xs_mb - a.best_peak_xs_mb);
    case 'hl-asc':    return d.sort((a, b) => a.half_life_s - b.half_life_s);
    case 'hl-desc':   return d.sort((a, b) => b.half_life_s - a.half_life_s);
    case 'routes-desc': return d.sort((a, b) => b.routes.length - a.routes.length);
    default: return d;
  }
}

function applyFilters() {
  const f = getFilters();
  let filtered = DATA.filter(iso => {
    const searchMatch = f.search === '' ||
      iso.name.toLowerCase().includes(f.search.toLowerCase()) ||
      iso.symbol.toLowerCase().includes(f.search.toLowerCase());
    const catMatch = f.cat === 'all' || iso.category === f.cat;
    if (!searchMatch || !catMatch) return false;
    if (f.hasImaging && iso.imaging === 'none') return false;
    // At least one route must pass
    return iso.routes.some(r => routePassesFilter(r, f));
  });
  filtered = applySort(filtered);
  buildIsotopes(filtered);

  // Dim filtered-out routes
  const f2 = f;
  document.querySelectorAll('.iso-card').forEach((card, i) => {
    if (i >= filtered.length) return;
    const iso = filtered[i];
    card.querySelectorAll('.route-row').forEach((row, ri) => {
      if (ri < iso.routes.length) {
        const passes = routePassesFilter(iso.routes[ri], f2);
        row.style.opacity = passes ? '1' : '0.25';
        row.style.pointerEvents = passes ? '' : 'none';
      }
    });
  });
}

document.getElementById('search').addEventListener('input', e => {
  currentSearch = e.target.value.trim();
  applyFilters();
});

document.getElementById('f-min-xs').addEventListener('input', e => {
  document.getElementById('xs-val').textContent = e.target.value;
  applyFilters();
});

['f-no-radio', 'f-no-gas', 'f-natural-only', 'f-has-imaging'].forEach(id => {
  document.getElementById(id).addEventListener('change', applyFilters);
});

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    applyFilters();
  });
});

document.getElementById('sort-select').addEventListener('change', e => {
  currentSort = e.target.value;
  applyFilters();
});

// Init
buildIsotopes(applySort(DATA));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Connecting to DuckDB...", flush=True)
    db = loader.connect()

    print("Building therapy isotope data...", flush=True)
    data, stats = build_data(db)

    print(f"\n--- Results ---", flush=True)
    print(f"  β⁻  isotopes : {stats['beta-']}", flush=True)
    print(f"  Auger isotopes: {stats['auger']}", flush=True)
    print(f"  Alpha isotopes: {stats['alpha']}", flush=True)
    print(f"  Total isotopes: {sum([stats['beta-'], stats['auger'], stats['alpha']])}", flush=True)
    print(f"  Total routes  : {stats['total_routes']}", flush=True)

    print(f"\nBuilding HTML with {len(data)} isotopes...", flush=True)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    out_path = Path(__file__).parent / "therapy_isotopes.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nOutput: {out_path}", flush=True)
    print(f"Size:   {size_kb:.1f} KB ({out_path.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
