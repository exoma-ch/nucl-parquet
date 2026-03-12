"""Build a single self-contained HTML file for PET isotope production routes.

Loads pet_candidates.parquet via loader.py, fetches excitation functions from
the xs view, and embeds everything as JSON in a dark-themed HTML/Chart.js page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import loader
from target_feasibility import load_abundance_map, load_element_z, assess_target, wikipedia_url

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

LIBRARY_PRIORITY = ["iaea-medical", "tendl-2025", "tendl-2024"]
ALLOWED_LIBRARIES = ("iaea-medical", "tendl-2025", "tendl-2024")


def best_excitation_function(
    db,
    projectile: str,
    target_el: str,
    target_A: int,
    residual_Z: int,
    residual_A: int,
) -> list[dict]:
    """Fetch the excitation function for a route, preferring the best library.

    Returns a list of {energy_MeV, xs_mb} dicts, or [] if no data found.
    Prefers iaea-medical > tendl-2025 > tendl-2024 > anything else.
    Uses the library with the most data points if multiple are available.
    """
    max_e = 18.0 if projectile == "p" else 9.0

    lib_filter = ",".join(f"'{l}'" for l in ALLOWED_LIBRARIES)
    rows = db.execute(
        f"""
        SELECT library, energy_MeV, xs_mb
        FROM xs
        WHERE filename LIKE '%' || ? || '_' || ? || '%'
          AND target_A = ?
          AND residual_Z = ?
          AND residual_A = ?
          AND energy_MeV <= ?
          AND xs_mb > 0
          AND library IN ({lib_filter})
        ORDER BY energy_MeV
        """,
        [projectile, target_el, target_A, residual_Z, residual_A, max_e],
    ).fetchall()

    if not rows:
        return [], 0.0, 0.0

    # Group by library
    by_lib: dict[str, list[tuple[float, float]]] = {}
    for lib, e, xs in rows:
        by_lib.setdefault(lib, []).append((e, xs))

    # Pick best library
    chosen_lib = None
    for lib in LIBRARY_PRIORITY:
        if lib in by_lib:
            chosen_lib = lib
            break
    if chosen_lib is None:
        # Fall back to whatever has the most points
        chosen_lib = max(by_lib, key=lambda k: len(by_lib[k]))

    pts = by_lib[chosen_lib]
    xs_data = [{"energy_MeV": round(e, 4), "xs_mb": round(xs, 4)} for e, xs in pts]
    # Also return peak from chosen library (avoids cross-library mismatch)
    peak_xs = max(xs for _, xs in pts) if pts else 0.0
    peak_E = max(pts, key=lambda p: p[1])[0] if pts else 0.0
    return xs_data, round(peak_xs, 4), round(peak_E, 4)


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
    """Convert leading digits in a nuclide string to HTML superscript.

    '68Zn' -> '<sup>68</sup>Zn'
    """
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    if i == 0:
        return text
    return f"<sup>{text[:i]}</sup>{text[i:]}"


def reaction_notation(projectile: str, target: str, residual: str) -> str:
    """Build reaction notation like ⁶⁸Zn(p,x)⁶⁸Ga using HTML superscripts."""
    tgt_html = superscript_mass(target)
    res_html = superscript_mass(residual)
    return f"{tgt_html}({projectile},x){res_html}"


def _lib_priority(lib: str) -> int:
    try:
        return LIBRARY_PRIORITY.index(lib)
    except ValueError:
        return len(LIBRARY_PRIORITY)


# ---------------------------------------------------------------------------
# Main data assembly
# ---------------------------------------------------------------------------

def build_data(db) -> list[dict]:
    """Build the full data structure for the HTML page."""
    lib_filter = ",".join(f"'{l}'" for l in ALLOWED_LIBRARIES)

    # Identify PET isotopes from radiation data
    pet_isotopes = db.execute("""
        SELECT r.Z, r.A, gs.symbol, gs.half_life_s,
               SUM(CASE WHEN r.rad_type = 'gamma' AND r.energy_keV BETWEEN 510 AND 512
                        THEN r.intensity_pct ELSE 0 END) / 2.0 AS beta_plus_pct
        FROM radiation r
        JOIN ground_states gs ON r.Z = gs.Z AND r.A = gs.A
        GROUP BY r.Z, r.A, gs.symbol, gs.half_life_s
        HAVING beta_plus_pct > 50
           AND gs.half_life_s BETWEEN 120 AND 86400
    """).fetchall()

    pet_set = {(int(Z), int(A)): (sym, float(hl), float(bp))
               for Z, A, sym, hl, bp in pet_isotopes}

    # Get all TENDL-only routes for these isotopes
    routes_raw = db.execute(f"""
        SELECT
            xs.library,
            CASE WHEN xs.filename LIKE '%/p_%' THEN 'p' ELSE 'd' END AS projectile,
            REPLACE(SPLIT_PART(SPLIT_PART(xs.filename, '/', -1), '_', 2), '.parquet', '') AS target_el,
            xs.target_A,
            xs.residual_Z,
            xs.residual_A,
            MAX(xs.xs_mb) AS peak_xs,
            ARG_MAX(xs.energy_MeV, xs.xs_mb) AS peak_E
        FROM xs
        JOIN elements e ON e.symbol = REPLACE(SPLIT_PART(SPLIT_PART(xs.filename, '/', -1), '_', 2), '.parquet', '')
        WHERE (xs.filename LIKE '%/p_%' OR xs.filename LIKE '%/d_%')
          AND library IN ({lib_filter})
          AND NOT (e.Z = xs.residual_Z AND xs.target_A = xs.residual_A)
          AND (
              (xs.filename LIKE '%/p_%' AND xs.energy_MeV BETWEEN 0 AND 18)
              OR
              (xs.filename LIKE '%/d_%' AND xs.energy_MeV BETWEEN 0 AND 9)
          )
          AND xs.xs_mb > 0
        GROUP BY xs.library, projectile, target_el, xs.target_A, xs.residual_Z, xs.residual_A
        HAVING peak_xs > 10
    """).fetchall()

    # De-duplicate routes across libraries (pick best priority)
    by_route: dict[tuple, dict] = {}
    for lib, proj, tel, tA, rZ, rA, pxs, pE in routes_raw:
        if not tel or not proj:
            continue
        key = (proj, tel, int(tA), int(rZ), int(rA))
        if key not in by_route or _lib_priority(lib) < _lib_priority(by_route[key]["library"]):
            by_route[key] = {
                "library": lib, "projectile": proj, "target_el": tel,
                "target_A": int(tA), "target": f"{int(tA)}{tel}",
                "residual_Z": int(rZ), "residual_A": int(rA),
                "peak_xs_mb": float(pxs), "peak_E_MeV": float(pE) if pE else 0.0,
            }

    # Group routes by isotope
    isotopes: dict[str, dict] = {}
    for route in by_route.values():
        rkey = (route["residual_Z"], route["residual_A"])
        if rkey not in pet_set:
            continue
        sym, hl, bp = pet_set[rkey]
        iso_name = f"{sym}-{route['residual_A']}"
        if iso_name not in isotopes:
            isotopes[iso_name] = {
                "name": iso_name, "Z": rkey[0], "A": rkey[1], "symbol": sym,
                "half_life_s": hl, "half_life_str": format_half_life(hl),
                "beta_plus_pct": bp, "routes": [],
            }
        route["residual"] = iso_name
        isotopes[iso_name]["routes"].append(route)

    # Load abundance and element maps for feasibility
    abundance_map = load_abundance_map(db)
    el_z_map = load_element_z(db)

    total_isotopes = len(isotopes)
    print(f"Processing {total_isotopes} isotopes...", flush=True)

    result = []
    for idx, (iso_name, iso) in enumerate(isotopes.items(), 1):
        # Add reaction notation, feasibility, and fetch excitation functions
        for route in iso["routes"]:
            route["reaction"] = reaction_notation(
                route["projectile"], route["target"], iso_name
            )
            # Target feasibility
            tgt_Z = el_z_map.get(route["target_el"], -1)
            tgt_A = route["target_A"]
            ab = abundance_map.get((tgt_Z, tgt_A))
            route["target_info"] = assess_target(tgt_Z, tgt_A, ab, route["target_el"])
            route["wiki_url"] = wikipedia_url(tgt_Z)
            print(
                f"  [{idx}/{total_isotopes}] {iso_name} | "
                f"{route['target']}({route['projectile']},x){iso_name} ...",
                end=" ",
                flush=True,
            )
            xs_data, lib_peak_xs, lib_peak_E = best_excitation_function(
                db,
                projectile=route["projectile"],
                target_el=route["target_el"],
                target_A=route["target_A"],
                residual_Z=route["residual_Z"],
                residual_A=route["residual_A"],
            )
            route["xs_data"] = xs_data
            # Override peak with value from the displayed library
            route["peak_xs_mb"] = lib_peak_xs
            route["peak_E_MeV"] = lib_peak_E
            print(f"{len(xs_data)} pts", flush=True)

        # Sort routes by (library-consistent) peak_xs_mb descending
        iso["routes"].sort(key=lambda r: r["peak_xs_mb"], reverse=True)
        iso["best_peak_xs_mb"] = iso["routes"][0]["peak_xs_mb"] if iso["routes"] else 0.0

        result.append(iso)

    # Sort isotopes by best peak xs descending
    result.sort(key=lambda x: x["best_peak_xs_mb"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PET Isotope Production Routes</title>
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
    --accent2: #a78bfa;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
    --proton-color: #6c8ef7;
    --deuteron-color: #fb923c;
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

  header h1 {
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.3px;
  }

  header p {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 2px;
  }

  .controls {
    display: flex;
    gap: 12px;
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
  }
  .filter-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .filter-btn:hover:not(.active) { border-color: var(--accent); color: var(--text); }

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

  .stats {
    margin-left: auto;
    color: var(--text-muted);
    font-size: 12px;
  }

  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px 16px;
  }

  /* Isotope card */
  .iso-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 8px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .iso-card:hover { border-color: #445; }
  .iso-card.open { border-color: var(--accent); }

  .iso-header {
    display: grid;
    grid-template-columns: 80px 1fr 120px 110px 90px 36px;
    align-items: center;
    padding: 14px 16px;
    cursor: pointer;
    gap: 12px;
    user-select: none;
  }
  .iso-header:hover { background: var(--surface2); }

  .iso-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.2px;
  }

  .iso-half-life { color: var(--text-muted); font-size: 12px; margin-top: 2px; }

  .beta-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(108, 142, 247, 0.15);
    border: 1px solid rgba(108, 142, 247, 0.3);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
  }

  .peak-xs {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .peak-xs .value { font-weight: 700; font-size: 14px; }
  .peak-xs .unit { color: var(--text-muted); font-size: 11px; margin-left: 2px; }

  .route-count {
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
  }
  .route-count strong { color: var(--text); font-size: 14px; display: block; }

  .expand-icon {
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 16px;
    transition: transform 0.2s, color 0.15s;
  }
  .open .expand-icon { transform: rotate(90deg); color: var(--accent); }

  /* Routes panel */
  .routes-panel {
    display: none;
    border-top: 1px solid var(--border);
    padding: 0 16px 16px;
  }
  .iso-card.open .routes-panel { display: block; }

  .routes-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
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
    border-bottom: 1px solid rgba(51, 55, 80, 0.5);
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
  .proj-p { background: rgba(108, 142, 247, 0.2); color: var(--proton-color); }
  .proj-d { background: rgba(251, 146, 60, 0.2); color: var(--deuteron-color); }

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

  .plot-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .plot-title { font-size: 13px; font-weight: 600; color: var(--text); }
  .plot-subtitle { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .plot-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 18px;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .plot-close:hover { color: var(--text); background: var(--surface3); }

  .chart-container { position: relative; height: 260px; }

  .no-data {
    text-align: center;
    color: var(--text-muted);
    padding: 40px;
    font-size: 13px;
  }

  .col-label {
    display: none;
  }

  @media (max-width: 640px) {
    .iso-header {
      grid-template-columns: 1fr auto auto 36px;
    }
    .iso-header .beta-badge { display: none; }
    .iso-header .route-count { display: none; }
    .search-box { width: 160px; }
  }
</style>
</head>
<body>

<header>
  <h1>PET Isotope Production Routes</h1>
  <p>Cross-section data from TENDL-2025 / TENDL-2024 / IAEA-Medical — <span id="vis-count"></span> isotopes</p>
  <div class="controls">
    <input class="search-box" type="search" placeholder="Search isotope…" id="search" autocomplete="off">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="p">Proton</button>
    <button class="filter-btn" data-filter="d">Deuteron</button>
    <span class="stats" id="stats-label"></span>
  </div>
  <div class="controls" style="margin-top:8px;">
    <label class="toggle-label"><input type="checkbox" id="f-no-radio" checked> <span>Hide radioactive targets</span></label>
    <label class="toggle-label"><input type="checkbox" id="f-no-gas"> <span>Hide gas targets</span></label>
    <label class="toggle-label"><input type="checkbox" id="f-natural-only"> <span>Enrichable targets only (&gt;0.05%)</span></label>
    <span class="slider-group">
      <label>Min σ: <span id="xs-val">10</span> mb</label>
      <input type="range" id="f-min-xs" min="0" max="500" step="10" value="10" class="slider">
    </span>
    <span class="slider-group">
      <label>Min β⁺: <span id="bp-val">50</span>%</label>
      <input type="range" id="f-min-bp" min="0" max="99" step="5" value="50" class="slider">
    </span>
  </div>
</header>

<main>
  <div id="iso-list"></div>
</main>

<script>
const DATA = __DATA_JSON__;

// Chart instance tracking
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

function projClass(p) {
  return p === 'p' ? 'proj-p' : 'proj-d';
}

function projLabel(p) {
  return p === 'p' ? 'p' : 'd';
}

// ---------------------------------------------------------------------------
// Build DOM
// ---------------------------------------------------------------------------

function buildIsotopes(data) {
  const list = document.getElementById('iso-list');
  list.innerHTML = '';

  data.forEach((iso, isoIdx) => {
    const card = document.createElement('div');
    card.className = 'iso-card';
    card.dataset.isoIdx = isoIdx;

    // Find best peak xs
    const bestRoute = iso.routes[0];
    const maxXs = iso.routes.reduce((m, r) => Math.max(m, r.peak_xs_mb), 0);

    card.innerHTML = `
      <div class="iso-header" onclick="toggleCard(this)">
        <div>
          <div class="iso-name">${iso.name}</div>
          <div class="iso-half-life">t<sub>½</sub> = ${iso.half_life_str}</div>
        </div>
        <div>
          <div style="color:var(--text-muted);font-size:11px;margin-bottom:2px;">β<sup>+</sup> branch</div>
          <span class="beta-badge">β<sup>+</sup> ${iso.beta_plus_pct.toFixed(1)}%</span>
        </div>
        <div class="peak-xs">
          <div style="color:var(--text-muted);font-size:11px;margin-bottom:2px;">Peak σ</div>
          <span class="value">${fmtXs(iso.best_peak_xs_mb)}</span>
        </div>
        <div class="route-count">
          <strong>${iso.routes.length}</strong>
          route${iso.routes.length !== 1 ? 's' : ''}
        </div>
        <div class="expand-icon">›</div>
      </div>
      <div class="routes-panel">
        ${buildRoutesPanel(iso, maxXs)}
      </div>
    `;
    list.appendChild(card);
  });

  updateVisCount();
}

function fmtAbundance(info) {
  if (!info || info.abundance_pct === null) return '<span style="color:var(--red)">—</span>';
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

function buildRoutesPanel(iso, maxXs) {
  const rows = iso.routes.map((route, ri) => {
    const barPct = maxXs > 0 ? (route.peak_xs_mb / maxXs * 100).toFixed(1) : 0;
    const barClass = route.projectile === 'p' ? 'bar-p' : 'bar-d';
    const info = route.target_info || {};
    return `
      <tr class="route-row" onclick="showPlot(this, '${iso.name}', ${JSON.stringify(route).replace(/"/g, '&quot;')})">
        <td><span class="proj-badge ${projClass(route.projectile)}">${projLabel(route.projectile)}</span></td>
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
    <div class="plot-panel" id="plot-${iso.name.replace(/[^a-zA-Z0-9]/g, '_')}">
      <div class="plot-header">
        <div>
          <div class="plot-title" id="pt-title-${iso.name.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
          <div class="plot-subtitle" id="pt-sub-${iso.name.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
        </div>
        <button class="plot-close" onclick="closePlot('${iso.name.replace(/[^a-zA-Z0-9]/g, '_')}')">×</button>
      </div>
      <div class="chart-container">
        <canvas id="chart-${iso.name.replace(/[^a-zA-Z0-9]/g, '_')}"></canvas>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

function toggleCard(headerEl) {
  const card = headerEl.closest('.iso-card');
  const wasOpen = card.classList.contains('open');

  // Close all cards if clicking a new one (optional: comment out for multi-open)
  // document.querySelectorAll('.iso-card.open').forEach(c => c.classList.remove('open'));

  card.classList.toggle('open', !wasOpen);

  if (wasOpen) {
    // Close any open plot in this card
    const isoName = card.querySelector('.iso-name').textContent;
    const safeId = isoName.replace(/[^a-zA-Z0-9]/g, '_');
    closePlot(safeId);
  }
}

function showPlot(rowEl, isoName, route) {
  const safeId = isoName.replace(/[^a-zA-Z0-9]/g, '_');
  const panel = document.getElementById('plot-' + safeId);
  const titleEl = document.getElementById('pt-title-' + safeId);
  const subEl = document.getElementById('pt-sub-' + safeId);
  const canvas = document.getElementById('chart-' + safeId);

  if (!panel || !canvas) return;

  // Deactivate previously active row
  if (activeRouteEl && activeRouteEl !== rowEl) {
    activeRouteEl.classList.remove('active-route');
  }
  activeRouteEl = rowEl;

  // Toggle: clicking active row again closes the plot
  const isAlreadyActive = rowEl.classList.contains('active-route');
  if (isAlreadyActive) {
    rowEl.classList.remove('active-route');
    panel.classList.remove('visible');
    if (activeChart) { activeChart.destroy(); activeChart = null; }
    return;
  }

  rowEl.classList.add('active-route');
  panel.classList.add('visible');

  titleEl.innerHTML = route.reaction;
  const libNote = route.xs_data.length > 0
    ? `${route.xs_data.length} data points`
    : 'No data available';
  subEl.textContent = `Excitation function — ${libNote}`;

  // Destroy old chart on this canvas
  if (activeChart && activePlotPanel === safeId) {
    activeChart.destroy();
    activeChart = null;
  }
  activePlotPanel = safeId;

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
  const fillColor = route.projectile === 'p'
    ? 'rgba(108,142,247,0.12)' : 'rgba(251,146,60,0.12)';

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
          title: {
            display: true,
            text: 'Energy (MeV)',
            color: '#8892a4',
            font: { size: 11 },
          },
          ticks: { color: '#8892a4', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          title: {
            display: true,
            text: 'Cross section (mb)',
            color: '#8892a4',
            font: { size: 11 },
          },
          ticks: { color: '#8892a4', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
          min: 0,
        },
      },
    },
  });
}

function closePlot(safeId) {
  const panel = document.getElementById('plot-' + safeId);
  if (panel) panel.classList.remove('visible');
  if (activeChart && activePlotPanel === safeId) {
    activeChart.destroy();
    activeChart = null;
  }
  if (activeRouteEl) {
    activeRouteEl.classList.remove('active-route');
    activeRouteEl = null;
  }
}

// ---------------------------------------------------------------------------
// Filter + Search
// ---------------------------------------------------------------------------

const NOBLE_GAS_Z = new Set([2,10,18,36,54,86]);
let currentFilter = 'all';
let currentSearch = '';

function getFilters() {
  return {
    proj: currentFilter,
    search: currentSearch,
    noRadio: document.getElementById('f-no-radio').checked,
    noGas: document.getElementById('f-no-gas').checked,
    naturalOnly: document.getElementById('f-natural-only').checked,
    minXs: parseFloat(document.getElementById('f-min-xs').value),
    minBp: parseFloat(document.getElementById('f-min-bp').value),
  };
}

function routePassesFilter(route, f) {
  if (f.proj === 'p' && route.projectile !== 'p') return false;
  if (f.proj === 'd' && route.projectile !== 'd') return false;
  const info = route.target_info || {};
  if (f.noRadio && info.abundance_pct === null) return false;
  if (f.noGas && info.abundance_pct !== null) {
    // Check if target element Z is noble gas — encoded in note
    if (info.note && info.note.includes('gas target')) return false;
  }
  if (f.naturalOnly && (info.abundance_pct === null || info.abundance_pct < 0.05)) return false;
  if (route.peak_xs_mb < f.minXs) return false;
  return true;
}

function updateVisCount() {
  const cards = document.querySelectorAll('.iso-card');
  const vis = [...cards].filter(c => c.style.display !== 'none').length;
  document.getElementById('vis-count').textContent = vis;
  document.getElementById('stats-label').textContent =
    vis < DATA.length ? `${vis} of ${DATA.length} shown` : `${DATA.length} total`;
}

function applyFilters() {
  const f = getFilters();
  const cards = document.querySelectorAll('.iso-card');
  cards.forEach((card, i) => {
    const iso = DATA[i];
    const searchMatch = f.search === '' ||
      iso.name.toLowerCase().includes(f.search.toLowerCase());
    if (!searchMatch || iso.beta_plus_pct < f.minBp) {
      card.style.display = 'none';
      return;
    }
    // Check if any route passes filters
    const hasRoute = iso.routes.some(r => routePassesFilter(r, f));
    card.style.display = hasRoute ? '' : 'none';

    // Also dim individual route rows that don't pass
    card.querySelectorAll('.route-row').forEach((row, ri) => {
      if (ri < iso.routes.length) {
        const passes = routePassesFilter(iso.routes[ri], f);
        row.style.opacity = passes ? '1' : '0.25';
        row.style.pointerEvents = passes ? '' : 'none';
      }
    });
  });
  updateVisCount();
}

// Slider value labels
document.getElementById('f-min-xs').addEventListener('input', e => {
  document.getElementById('xs-val').textContent = e.target.value;
  applyFilters();
});
document.getElementById('f-min-bp').addEventListener('input', e => {
  document.getElementById('bp-val').textContent = e.target.value;
  applyFilters();
});

document.getElementById('search').addEventListener('input', e => {
  currentSearch = e.target.value.trim();
  applyFilters();
});

// Checkbox filters
['f-no-radio', 'f-no-gas', 'f-natural-only'].forEach(id => {
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

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
buildIsotopes(DATA);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Connecting to DuckDB...")
    db = loader.connect()

    print("Loading PET candidates and fetching excitation functions...")
    data = build_data(db)

    print(f"\nBuilding HTML with {len(data)} isotopes...")
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    out_path = Path(__file__).parent / "pet_isotopes.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nOutput: {out_path}")
    print(f"Size:   {size_kb:.1f} KB ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
