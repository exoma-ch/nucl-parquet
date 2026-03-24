"""FastAPI backend for the Therapeutic Isotope Explorer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.interpolate import interp1d

import nucl_parquet
from nucl_parquet.loader import DECAY_CHAIN_SQL, elemental_dedx

app = FastAPI(title="Isotope Explorer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db = nucl_parquet.connect()

# ── Range computation (ESTAR / ASTAR / XCOM) ──────────────────────

_ESTAR = np.array([
    [1e-03,4.537e-06],[1.5e-03,8.655e-06],[2e-03,1.376e-05],[2.5e-03,1.975e-05],
    [3e-03,2.655e-05],[4e-03,4.228e-05],[5e-03,6.043e-05],[6e-03,8.080e-05],
    [7e-03,1.032e-04],[8e-03,1.275e-04],[9e-03,1.536e-04],[1e-02,1.814e-04],
    [1.25e-02,2.568e-04],[1.5e-02,3.424e-04],[1.75e-02,4.372e-04],
    [2e-02,5.406e-04],[2.5e-02,7.714e-04],[3e-02,1.032e-03],
    [3.5e-02,1.320e-03],[4e-02,1.632e-03],[4.5e-02,1.966e-03],
    [5e-02,2.321e-03],[5.5e-02,2.696e-03],[6e-02,3.089e-03],
    [7e-02,3.924e-03],[8e-02,4.822e-03],[9e-02,5.776e-03],
    [1e-01,6.781e-03],[1.25e-01,9.444e-03],[1.5e-01,1.226e-02],
    [1.75e-01,1.520e-02],[2e-01,1.823e-02],[2.5e-01,2.450e-02],
    [3e-01,3.100e-02],[3.5e-01,3.767e-02],[4e-01,4.445e-02],
    [4.5e-01,5.133e-02],[5e-01,5.828e-02],[5.5e-01,6.528e-02],
    [6e-01,7.233e-02],[7e-01,8.651e-02],[8e-01,1.008e-01],
    [9e-01,1.151e-01],[1.0,1.294e-01],[1.25,1.653e-01],
    [1.5,2.012e-01],[2.0,2.728e-01],[2.5,3.440e-01],
    [3.0,4.148e-01],[4.0,5.553e-01],[5.0,6.943e-01],
])
_estar_f = interp1d(np.log10(_ESTAR[:,0]), np.log10(_ESTAR[:,1]), fill_value="extrapolate")

_astar_raw = db.sql(
    "SELECT energy_MeV, dedx FROM stopping WHERE source='ASTAR' AND target_Z=276 ORDER BY energy_MeV"
).fetchall()
_aE = np.array([r[0] for r in _astar_raw])
_aS = np.array([r[1] for r in _astar_raw])
_aR = np.zeros_like(_aE)
_inv = 1.0 / _aS
for i in range(1, len(_aE)):
    _aR[i] = _aR[i-1] + 0.5*(_inv[i]+_inv[i-1])*(_aE[i]-_aE[i-1])
_aR_f = interp1d(np.log10(_aE), np.log10(np.clip(_aR, 1e-20, None)), fill_value="extrapolate")

_xcom_raw = db.sql(
    "SELECT energy_MeV, mu_rho_cm2_g FROM xcom_compounds WHERE material='water' ORDER BY energy_MeV"
).fetchall()
_xE = np.array([r[0] for r in _xcom_raw])
_xM = np.array([r[1] for r in _xcom_raw])
_xM_f = interp1d(np.log10(_xE), np.log10(_xM), fill_value="extrapolate")


def _range_um(rad_type: str, energy_keV: float) -> float | None:
    if rad_type in ("auger", "ce", "beta-", "beta+/EC"):
        e = energy_keV / 1000.0
        if e < 1e-4:
            return float(_ESTAR[0,1] * (e/_ESTAR[0,0])**1.7 * 1e4)
        return float(10**_estar_f(np.log10(e)) * 1e4)
    elif rad_type == "alpha":
        e_per_u = energy_keV / 1000.0 / 4.0
        if e_per_u < _aE[0]:
            return float(4.0 * 10**_aR_f(np.log10(_aE[0])) * (e_per_u/_aE[0])**1.5 * 1e4)
        return float(4.0 * 10**_aR_f(np.log10(e_per_u)) * 1e4)
    elif rad_type in ("gamma", "xray"):
        mu = 10**_xM_f(np.log10(energy_keV/1000.0))
        return float(1.0/(mu*1.0)*1e4)
    return None


def _hl_label(s: float | None) -> str:
    if s is None or math.isnan(s):
        return "?"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    if s < 86400:
        return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"


# ── Precompute master dataset ──────────────────────────────────────

def _build_emissions():
    rows = db.sql("""
        SELECT r.Z, r.A, e.symbol,
               gs.half_life_s, gs.decay_1 AS primary_decay,
               r.rad_type, r.rad_subtype,
               r.energy_keV, r.end_point_keV,
               r.intensity_pct, r.dose_MeV_per_Bq_s
        FROM radiation r
        JOIN ground_states gs ON r.Z = gs.Z AND r.A = gs.A
        JOIN elements e ON r.Z = e.Z
        WHERE r.dataset = 1
          AND gs.half_life_s BETWEEN 60 AND 90 * 86400
          AND r.dose_MeV_per_Bq_s > 0.0002
        ORDER BY gs.Z, gs.A, r.dose_MeV_per_Bq_s DESC
    """).fetchall()

    emissions = []
    for row in rows:
        z, a, sym, hl, decay, rt, rsub, ek, ep, inten, dose = row
        r = _range_um(rt, ek)
        if r is None:
            continue
        emissions.append({
            "Z": z, "A": a, "symbol": sym,
            "isotope": f"{sym}-{a}",
            "half_life_s": hl,
            "hl_label": _hl_label(hl),
            "primary_decay": decay or "",
            "rad_type": rt, "rad_subtype": rsub or "",
            "energy_keV": round(ek, 2),
            "endpoint_keV": round(ep, 1) if ep else None,
            "intensity_pct": round(inten, 3),
            "dose": round(dose, 6),
            "range_um": round(r, 3),
        })
    return emissions

_ALL_EMISSIONS = _build_emissions()
print(f"Precomputed {len(_ALL_EMISSIONS)} emission lines")


# ── API routes ─────────────────────────────────────────────────────

@app.get("/api/emissions")
def get_emissions(
    hl_min_h: float = Query(1.0),
    hl_max_h: float = Query(720.0),
    dose_min: float = Query(0.001),
    range_min: float = Query(0.01),
    range_max: float = Query(1e6),
    rad_types: str = Query("auger,ce,beta-,alpha,gamma"),
    decay_modes: str = Query("B-,EC,B+,EC+B+,A,IT"),
):
    hl_min_s = hl_min_h * 3600
    hl_max_s = hl_max_h * 3600
    allowed_rt = set(rad_types.split(","))
    allowed_dm = set(decay_modes.split(","))

    return [
        e for e in _ALL_EMISSIONS
        if hl_min_s <= e["half_life_s"] <= hl_max_s
        and e["rad_type"] in allowed_rt
        and e["dose"] >= dose_min
        and range_min <= e["range_um"] <= range_max
        and e["primary_decay"] in allowed_dm
    ]


@app.get("/api/isotopes")
def get_isotopes(
    hl_min_h: float = Query(1.0),
    hl_max_h: float = Query(720.0),
    decay_modes: str = Query("B-,EC,B+,EC+B+,A,IT"),
):
    """Aggregated per-isotope summary."""
    hl_min_s = hl_min_h * 3600
    hl_max_s = hl_max_h * 3600
    allowed_dm = set(decay_modes.split(","))

    by_iso: dict[str, dict] = {}
    for e in _ALL_EMISSIONS:
        if not (hl_min_s <= e["half_life_s"] <= hl_max_s):
            continue
        if e["primary_decay"] not in allowed_dm:
            continue
        key = f"{e['Z']}_{e['A']}"
        if key not in by_iso:
            by_iso[key] = {
                "Z": e["Z"], "A": e["A"], "isotope": e["isotope"],
                "symbol": e["symbol"],
                "half_life_s": e["half_life_s"], "hl_label": e["hl_label"],
                "primary_decay": e["primary_decay"],
                "ce": 0, "auger": 0, "beta": 0, "alpha": 0, "gamma": 0, "total": 0,
            }
        d = by_iso[key]
        d["total"] += e["dose"]
        if e["rad_type"] == "ce":
            d["ce"] += e["dose"]
        elif e["rad_type"] == "auger":
            d["auger"] += e["dose"]
        elif e["rad_type"] == "beta-":
            d["beta"] += e["dose"]
        elif e["rad_type"] == "alpha":
            d["alpha"] += e["dose"]
        elif e["rad_type"] == "gamma":
            d["gamma"] += e["dose"]

    result = []
    for d in by_iso.values():
        d["electron_pct"] = round((d["ce"] + d["auger"]) / max(d["total"], 1e-10) * 100, 1)
        for k in ("ce", "auger", "beta", "alpha", "gamma", "total"):
            d[k] = round(d[k], 5)
        result.append(d)

    result.sort(key=lambda x: x["electron_pct"], reverse=True)
    return result


@app.get("/api/isotope/{z}/{a}")
def get_isotope_detail(z: int, a: int):
    """Full detail for a single isotope."""

    # Ground state info
    gs = db.sql("SELECT * FROM ground_states WHERE Z=$1 AND A=$2", params=[z, a]).fetchone()
    gs_dict = {}
    if gs:
        cols = [d[0] for d in db.sql("SELECT * FROM ground_states LIMIT 0").description]
        gs_dict = dict(zip(cols, gs))
        gs_dict["hl_label"] = _hl_label(gs_dict.get("half_life_s"))

    # Target abundances
    targets = db.sql(f"""
        SELECT Z, A, symbol, abundance, atomic_mass FROM abundances
        WHERE (Z={z} OR Z={z-1} OR Z={z+1} OR Z={z-2})
          AND A BETWEEN {a-4} AND {a+2} AND abundance > 0
        ORDER BY abundance DESC
    """).fetchall()
    target_list = [
        {"Z": t[0], "A": t[1], "symbol": t[2], "abundance": round(t[3], 4), "mass": round(t[4], 4)}
        for t in targets
    ]

    # Production routes
    routes = db.sql(f"""
        SELECT library,
               REGEXP_EXTRACT(filename, '([a-z]+)_[A-Z]', 1) AS projectile,
               REGEXP_EXTRACT(filename, '[a-z]+_([A-Za-z]+)\\.parquet', 1) AS target_el,
               target_A,
               MAX(xs_mb) AS peak_xs,
               (ARRAY_AGG(energy_MeV ORDER BY xs_mb DESC))[1] AS peak_E,
               MIN(energy_MeV) AS E_min, MAX(energy_MeV) AS E_max,
               COUNT(*) AS n_points
        FROM xs
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
        GROUP BY library, projectile, target_el, target_A
        HAVING MAX(xs_mb) > 0.1
        ORDER BY peak_xs DESC LIMIT 30
    """).fetchall()
    route_list = [
        {"library": r[0], "projectile": r[1] or "?", "target_el": r[2] or "?",
         "target_A": r[3], "peak_xs": round(r[4], 2), "peak_E": round(r[5], 2),
         "E_min": round(r[6], 2), "E_max": round(r[7], 2), "n_points": r[8]}
        for r in routes
    ]

    # EXFOR
    exfor = db.sql(f"""
        SELECT REGEXP_EXTRACT(filename, '([a-z]+)_[A-Z]', 1) AS projectile,
               REGEXP_EXTRACT(filename, '[a-z]+_([A-Za-z]+)\\.parquet', 1) AS target_el,
               target_A,
               COUNT(DISTINCT exfor_entry) AS n_entries, COUNT(*) AS n_points,
               MAX(xs_mb) AS peak_xs,
               STRING_AGG(DISTINCT author || ' (' || year || ')', ', ' ORDER BY author) AS refs
        FROM exfor
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
        GROUP BY projectile, target_el, target_A
        HAVING MAX(xs_mb) > 0.01
        ORDER BY n_entries DESC LIMIT 20
    """).fetchall()
    exfor_list = [
        {"projectile": e[0], "target_el": e[1], "target_A": e[2],
         "n_entries": e[3], "n_points": e[4], "peak_xs": round(e[5], 2),
         "refs": e[6][:200] if e[6] else ""}
        for e in exfor
    ]

    # Emissions
    emissions = db.sql(f"""
        SELECT rad_type, rad_subtype, energy_keV, end_point_keV,
               intensity_pct, dose_MeV_per_Bq_s
        FROM radiation
        WHERE Z={z} AND A={a} AND dataset=1 AND intensity_pct > 0.1
        ORDER BY rad_type, dose_MeV_per_Bq_s DESC
    """).fetchall()
    emission_list = [
        {"rad_type": e[0], "subtype": e[1] or "", "energy_keV": round(e[2], 2),
         "endpoint_keV": round(e[3], 1) if e[3] else None,
         "intensity_pct": round(e[4], 3), "dose": round(e[5], 5)}
        for e in emissions[:40]
    ]

    # Theranostic partners
    el = db.sql(f"SELECT symbol FROM elements WHERE Z={z}").fetchone()
    symbol = el[0] if el else "?"

    partners = db.sql(f"""
        SELECT gs.A, e.symbol, gs.half_life_s, gs.decay_1, gs.decay_1_pct
        FROM ground_states gs JOIN elements e ON gs.Z = e.Z
        WHERE gs.Z={z} AND gs.half_life_s BETWEEN 60 AND 90*86400 AND gs.A != {a}
        ORDER BY gs.half_life_s
    """).fetchall()
    partner_list = [
        {"A": p[0], "symbol": p[1], "isotope": f"{p[1]}-{p[0]}",
         "half_life_s": p[2], "hl_label": _hl_label(p[2]),
         "decay": p[3] or "?", "branching": round(p[4], 1) if p[4] else None,
         "role": (
            "PET diagnostic" if p[3] in ("B+", "EC", "EC+B+") else
            "β⁻ therapy" if p[3] == "B-" else
            "α therapy" if p[3] == "A" else
            "Generator / CE" if p[3] == "IT" else ""
         )}
        for p in partners
    ]

    return {
        "Z": z, "A": a, "symbol": symbol, "isotope": f"{symbol}-{a}",
        "ground_state": gs_dict,
        "targets": target_list,
        "routes": route_list,
        "exfor": exfor_list,
        "emissions": emission_list,
        "partners": partner_list,
    }


@app.get("/api/excitation/{z}/{a}")
def get_excitation(
    z: int, a: int,
    projectile: str = Query("p"),
    target_a: int | None = Query(None),
):
    """Cross-section data for excitation function plot."""
    ta_filter = f" AND target_A={target_a}" if target_a is not None else ""
    evaluated = db.sql(f"""
        SELECT library, energy_MeV, xs_mb FROM xs
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
          AND filename LIKE '%{projectile}_%'{ta_filter}
        ORDER BY library, energy_MeV
    """).fetchall()

    by_lib: dict[str, list] = {}
    for lib, e, xs in evaluated:
        by_lib.setdefault(lib, []).append({"E": round(e, 4), "xs": round(xs, 4)})

    exfor = db.sql(f"""
        SELECT energy_MeV, xs_mb, xs_err_mb, author, year FROM exfor
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
          AND filename LIKE '%{projectile}_%'{ta_filter}
        ORDER BY energy_MeV
    """).fetchall()
    exfor_pts = [
        {"E": round(e[0], 4), "xs": round(e[1], 4),
         "err": round(e[2], 4) if e[2] else None,
         "ref": f"{e[3]} ({e[4]})"}
        for e in exfor
    ]

    return {"evaluated": by_lib, "exfor": exfor_pts}


@app.get("/api/gamma-identify")
def gamma_identify(energy: float = Query(1157.0), tolerance: float = Query(2.0)):
    rows = db.sql(
        "SELECT r.Z, r.A, gs.symbol, r.energy_keV, r.intensity_pct, "
        "gs.half_life_s, gs.decay_1, ABS(r.energy_keV - $1) AS delta "
        "FROM radiation r JOIN ground_states gs ON r.Z=gs.Z AND r.A=gs.A "
        "WHERE r.rad_type='gamma' AND r.energy_keV BETWEEN ($1-$2) AND ($1+$2) "
        "AND r.intensity_pct > 0.1 ORDER BY delta, r.intensity_pct DESC LIMIT 20",
        params=[energy, tolerance],
    ).fetchall()
    return [
        {"Z": r[0], "A": r[1], "symbol": r[2], "isotope": f"{r[2]}-{r[1]}",
         "energy_keV": round(r[3], 2), "intensity_pct": round(r[4], 2),
         "half_life_s": r[5], "hl_label": _hl_label(r[5]),
         "decay": r[6], "delta_keV": round(r[7], 2)}
        for r in rows
    ]


# ── Decay Chain endpoint ──────────────────────────────────────────

@app.get("/api/decay-chain/{z}/{a}")
def get_decay_chain(z: int, a: int, max_gen: int = Query(10)):
    """Full decay chain with per-generation dose breakdown."""
    rows = db.sql(DECAY_CHAIN_SQL, params={"parent_z": z, "parent_a": a}).fetchall()

    nodes = []
    cum_br = 1.0
    prev_hl = None
    total_dose = 0.0

    for row in rows:
        rz, ra, sym, hl, decay_mode, br_pct, dz, da, gen = row
        if gen > max_gen:
            break

        # Dose by type for this nuclide
        rad_rows = db.sql(f"""
            SELECT rad_type, SUM(dose_MeV_per_Bq_s) AS dose
            FROM radiation
            WHERE Z={rz} AND A={ra} AND dataset=1 AND intensity_pct > 0.1
            GROUP BY rad_type
        """).fetchall()
        dose_by_type = {}
        for rt, d in rad_rows:
            key = rt if rt in ("alpha", "gamma", "xray") else (
                "ce" if rt == "ce" else "auger" if rt == "auger" else "beta-"
            )
            dose_by_type[key] = dose_by_type.get(key, 0) + (d or 0)

        # Cumulative branching
        if gen > 1 and br_pct:
            cum_br *= (br_pct / 100.0)

        # Equilibrium classification
        eq = None
        if prev_hl is not None and hl and hl > 0:
            ratio = prev_hl / hl if hl > 0 else 0
            if ratio > 100:
                eq = "secular"
            elif ratio > 1.5:
                eq = "transient"
        prev_hl = hl

        node_dose = sum(dose_by_type.values())
        total_dose += node_dose * cum_br

        nodes.append({
            "Z": rz, "A": ra, "symbol": sym or "?",
            "half_life_s": hl, "decay_mode": decay_mode or "",
            "branching_pct": br_pct, "cum_branching": round(cum_br, 6),
            "generation": gen, "equilibrium": eq,
            "dose_by_type": {k: round(v, 6) for k, v in dose_by_type.items()},
        })

    return {"nodes": nodes, "cumulative_dose": round(total_dose, 5)}


# ── Cost Estimator endpoint ──────────────────────────────────────

@app.get("/api/cost-estimate/{z}/{a}")
def get_cost_estimate(z: int, a: int):
    """Relative production cost/feasibility score (0-100)."""
    # Target abundance
    abund = db.sql(f"""
        SELECT MAX(abundance) FROM abundances
        WHERE (Z={z} OR Z={z-1} OR Z={z+1} OR Z={z-2})
          AND A BETWEEN {a-4} AND {a+2} AND abundance > 0
    """).fetchone()
    best_abund = abund[0] if abund and abund[0] else 0
    abund_score = min(best_abund / 100 * 100, 100)

    # Peak cross-section
    peak = db.sql(f"""
        SELECT MAX(xs_mb) FROM xs
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
    """).fetchone()
    peak_xs = peak[0] if peak and peak[0] else 0
    xs_score = min(math.log10(max(peak_xs, 0.01)) / 3.0 * 100, 100)
    xs_score = max(xs_score, 0)

    # Route type
    routes = db.sql(f"""
        SELECT DISTINCT REGEXP_EXTRACT(filename, '([a-z]+)_[A-Z]', 1) AS projectile
        FROM xs WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
    """).fetchall()
    proj_set = {r[0] for r in routes if r[0]}
    if proj_set & {"p", "d"}:
        route_score = 90  # cyclotron — most accessible
    elif proj_set & {"n"}:
        route_score = 70  # reactor
    elif proj_set & {"a", "h", "he3"}:
        route_score = 40  # exotic beam
    else:
        route_score = 20

    # Half-life shipping penalty
    gs = db.sql(f"SELECT half_life_s FROM ground_states WHERE Z={z} AND A={a}").fetchone()
    hl = gs[0] if gs and gs[0] else 0
    if hl > 6 * 3600:
        hl_score = 90  # >6h, shippable
    elif hl > 3600:
        hl_score = 60
    elif hl > 600:
        hl_score = 30
    else:
        hl_score = 10  # too short for transport

    # Route flexibility
    n_routes = db.sql(f"""
        SELECT COUNT(DISTINCT library || '_' || filename || '_' || target_A) FROM xs
        WHERE residual_Z={z} AND residual_A={a} AND state IN ('','g')
    """).fetchone()
    n = n_routes[0] if n_routes and n_routes[0] else 0
    flex_score = min(n / 20 * 100, 100)

    # Weighted composite
    score = (
        0.35 * abund_score
        + 0.25 * xs_score
        + 0.15 * route_score
        + 0.15 * hl_score
        + 0.10 * flex_score
    )

    notes = []
    if best_abund < 5:
        notes.append("Very low natural abundance — high enrichment cost")
    if peak_xs < 1:
        notes.append("Low peak cross-section — low production efficiency")
    if hl < 3600:
        notes.append("Short half-life — local production required")

    return {
        "score": round(score, 1),
        "factors": {
            "target_abundance": round(abund_score, 1),
            "peak_cross_section": round(xs_score, 1),
            "route_type": round(route_score, 1),
            "half_life": round(hl_score, 1),
            "route_flexibility": round(flex_score, 1),
        },
        "notes": "; ".join(notes) if notes else None,
    }


# ── Dose Kernel endpoint ─────────────────────────────────────────

@app.get("/api/dose-kernel/{z}/{a}")
def get_dose_kernel(z: int, a: int):
    """Radial dose distribution D(r) in water."""
    emissions = db.sql(f"""
        SELECT rad_type, energy_keV, intensity_pct
        FROM radiation
        WHERE Z={z} AND A={a} AND dataset=1 AND intensity_pct > 0.5
        ORDER BY dose_MeV_per_Bq_s DESC
        LIMIT 40
    """).fetchall()

    if not emissions:
        return {"r_um": [], "total_Gy_per_decay": [], "by_type": {}, "r90_um": None}

    r = np.geomspace(0.001, 1e5, 200)

    electron_dose = np.zeros_like(r)
    alpha_dose = np.zeros_like(r)
    photon_dose = np.zeros_like(r)

    for rt, ek, inten in emissions:
        frac = inten / 100.0
        e_MeV = ek / 1000.0

        if rt in ("auger", "ce", "beta-", "beta+/EC"):
            # Electron: Cole-type kernel
            R0 = _range_um(rt, ek)
            if R0 is None or R0 < 0.001:
                continue
            mask = r < R0
            if mask.any():
                # D ∝ (1 - r/R0)^1.67 / (4πr²) — normalized so integral = E
                kernel = np.zeros_like(r)
                kernel[mask] = (1.0 - r[mask] / R0) ** 1.67
                # Normalize: total deposited energy = E_MeV * frac
                integral = np.trapezoid(kernel * 4 * np.pi * (r * 1e-4) ** 2, r * 1e-4)
                if integral > 0:
                    kernel *= (e_MeV * frac * 1.602e-13) / integral  # J per decay
                    # Convert to Gy: J / (ρ * dV) — for a shell at r
                    # D(r) = kernel / ρ where ρ=1 g/cm³ for water
                    electron_dose += kernel

        elif rt == "alpha":
            R0 = _range_um("alpha", ek)
            if R0 is None or R0 < 0.001:
                continue
            mask = r < R0
            if mask.any():
                # Bragg peak approximation
                kernel = np.zeros_like(r)
                x = r[mask] / R0
                kernel[mask] = 1.0 / ((1.0 - x) ** 0.5 + 0.01)
                integral = np.trapezoid(kernel * 4 * np.pi * (r * 1e-4) ** 2, r * 1e-4)
                if integral > 0:
                    kernel *= (e_MeV * frac * 1.602e-13) / integral
                    alpha_dose += kernel

        elif rt in ("gamma", "xray"):
            if ek < 1:
                continue
            mu_rho = 10 ** _xM_f(np.log10(e_MeV))  # cm²/g
            mu = mu_rho * 1.0  # 1/cm for water (ρ=1)
            r_cm = r * 1e-4
            # D ∝ μ_en * exp(-μr) / (4πr²)
            with np.errstate(divide="ignore", invalid="ignore"):
                kernel = mu * np.exp(-mu * r_cm) / (4 * np.pi * r_cm ** 2 + 1e-30)
            kernel *= e_MeV * frac * 1.602e-13
            photon_dose += kernel

    total = electron_dose + alpha_dose + photon_dose

    # Convert D(r) to shell-energy form: 4πr²ρ·D(r) [MeV/μm/decay]
    # D(r) is currently in J/cm³/decay internally; shell area = 4πr²
    # Shell energy = D(r) · 4πr² · dr  (energy in shell of thickness dr)
    # We want dE/dr in MeV/μm, so: 4πr²·D(r) with r in cm, then convert
    r_cm = r * 1e-4
    rho = 1.0  # g/cm³ water

    def to_shell_energy(d):
        """Convert volumetric D(r) to 4πr²ρ·D(r) in MeV/μm/decay."""
        # d is in J/cm³/decay, multiply by 4πr² (cm²) → J/cm/decay
        # then convert J→MeV (÷1.602e-13) and cm→μm (×1e-4, since 1cm=1e4μm)
        shell = d * 4 * np.pi * r_cm**2
        return shell / 1.602e-13 * 1e-4  # MeV/μm/decay

    shell_electron = to_shell_energy(electron_dose)
    shell_alpha = to_shell_energy(alpha_dose)
    shell_photon = to_shell_energy(photon_dose)
    shell_total = shell_electron + shell_alpha + shell_photon

    # Cumulative F(r): fraction of energy absorbed within radius r
    dr = np.gradient(r)  # μm
    # shell_total is MeV/μm/decay, so integrating over dr gives MeV/decay
    cum_energy = np.cumsum(shell_total * dr)
    total_energy = cum_energy[-1] if cum_energy[-1] > 0 else 1.0
    F_r = cum_energy / total_energy

    # R50, R90
    r50 = float(r[np.searchsorted(F_r, 0.5)]) if np.any(F_r >= 0.5) else None
    r90 = float(r[np.searchsorted(F_r, 0.9)]) if np.any(F_r >= 0.9) else None

    # Build per-type shell-energy arrays
    shell_by_type = {}
    if shell_electron.sum() > 0:
        shell_by_type["electron"] = [round(float(v), 8) for v in shell_electron]
    if shell_alpha.sum() > 0:
        shell_by_type["alpha"] = [round(float(v), 8) for v in shell_alpha]
    if shell_photon.sum() > 0:
        shell_by_type["photon"] = [round(float(v), 8) for v in shell_photon]

    return {
        "r_um": [round(float(v), 4) for v in r],
        "shell_MeV_per_um": [round(float(v), 8) for v in shell_total],
        "cumulative_F": [round(float(v), 6) for v in F_r],
        "by_type": shell_by_type,
        "r50_um": round(r50, 2) if r50 else None,
        "r90_um": round(r90, 2) if r90 else None,
        "total_energy_MeV": round(float(total_energy), 5),
    }


# ── Yield Estimator endpoint ─────────────────────────────────────

class YieldRequest(BaseModel):
    mode: str  # "cyclotron" or "reactor"
    z: int
    a: int
    projectile: str = "p"
    target_Z: int | None = None
    target_A: int | None = None
    beam_current_uA: float = 10.0
    energy_in_MeV: float = 30.0
    energy_out_MeV: float = 5.0
    enrichment: float = 0.99
    irrad_time_h: float = 4.0
    target_mass_mg: float = 100.0
    flux_n_cm2_s: float = 1e14


@app.post("/api/yield-estimate")
def estimate_yield(req: YieldRequest):
    """Thick-target yield for cyclotron or reactor production."""
    # Get half-life
    gs = db.sql(
        "SELECT half_life_s, symbol FROM ground_states gs JOIN elements e ON gs.Z=e.Z "
        "WHERE gs.Z=$1 AND gs.A=$2",
        params=[req.z, req.a],
    ).fetchone()
    if not gs or not gs[0]:
        return {"activity_Bq": None, "warning": "Unknown isotope or stable"}
    hl_s = gs[0]
    lam = math.log(2) / hl_s  # decay constant [1/s]

    if req.mode == "cyclotron":
        target_Z = req.target_Z or req.z
        target_A = req.target_A or (req.a - 1)

        # Get atomic mass
        mass_row = db.sql(
            f"SELECT atomic_mass FROM abundances WHERE Z={target_Z} AND A={target_A}"
        ).fetchone()
        M = mass_row[0] if mass_row and mass_row[0] else target_A

        # Build energy grid for integration
        e_grid = np.linspace(req.energy_out_MeV, req.energy_in_MeV, 200)

        # Cross-section interpolation (best library)
        xs_rows = db.sql(f"""
            SELECT energy_MeV, xs_mb FROM xs
            WHERE residual_Z={req.z} AND residual_A={req.a}
              AND target_A={target_A} AND state IN ('','g')
              AND filename LIKE '%{req.projectile}_%'
            ORDER BY energy_MeV
        """).fetchall()

        if not xs_rows:
            return {"activity_Bq": None, "warning": "No cross-section data for this route"}

        xs_E = np.array([r[0] for r in xs_rows])
        xs_S = np.array([r[1] for r in xs_rows])
        sigma_interp = interp1d(xs_E, xs_S, bounds_error=False, fill_value=0.0)
        sigma = sigma_interp(e_grid)  # mb

        # Stopping power
        try:
            S = elemental_dedx(db, req.projectile, target_Z, e_grid)  # MeV cm²/g
        except (KeyError, Exception):
            return {"activity_Bq": None, "warning": f"No stopping power for {req.projectile} in Z={target_Z}"}

        # Integrand: σ(E)/S(E) — convert σ from mb to cm²
        sigma_cm2 = sigma * 1e-27
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(S > 0, sigma_cm2 / S, 0.0)

        # Y = (N_A / M) * enrichment * I_beam * ∫ σ/S dE
        N_A = 6.02214076e23
        I_beam = req.beam_current_uA * 1e-6 / 1.602e-19  # particles/s
        integral = np.trapezoid(integrand, e_grid)
        Y = (N_A / M) * req.enrichment * I_beam * integral  # atoms/s

        # A_EOB = Y * (1 - exp(-λt))
        t_s = req.irrad_time_h * 3600
        sat_factor = 1.0 - math.exp(-lam * t_s)
        A_EOB = Y * sat_factor  # Bq

        # Specific activity
        n_atoms = Y * t_s  # approximate total atoms
        sp_act = A_EOB / (n_atoms / N_A * 1e6) if n_atoms > 0 else None  # GBq/μmol

        return {
            "activity_Bq": round(A_EOB, 1),
            "yield_atoms": round(float(Y * t_s), 1),
            "specific_activity_GBq_umol": round(sp_act / 1e9, 4) if sp_act else None,
            "integrand": {
                "E": [round(float(v), 3) for v in e_grid],
                "sigma": [round(float(v), 4) for v in sigma],
                "S": [round(float(v), 4) for v in S],
                "integrand": [round(float(v), 12) for v in integrand],
            },
        }

    else:  # reactor
        target_Z = req.target_Z or req.z
        target_A = req.target_A or (req.a - 1)
        mass_row = db.sql(
            f"SELECT atomic_mass FROM abundances WHERE Z={target_Z} AND A={target_A}"
        ).fetchone()
        M = mass_row[0] if mass_row and mass_row[0] else target_A

        # Thermal cross-section at ~0.025 eV
        xs_thermal = db.sql(f"""
            SELECT xs_mb FROM xs
            WHERE residual_Z={req.z} AND residual_A={req.a}
              AND target_A={target_A} AND state IN ('','g')
              AND filename LIKE '%n_%'
              AND energy_MeV < 0.001
            ORDER BY ABS(energy_MeV - 0.0000253) LIMIT 1
        """).fetchone()

        if not xs_thermal:
            return {"activity_Bq": None, "warning": "No thermal neutron cross-section found"}

        sigma_b = xs_thermal[0] * 1e-3  # mb to barns
        sigma_cm2 = sigma_b * 1e-24

        N_A = 6.02214076e23
        N_target = (req.target_mass_mg * 1e-3 / M) * N_A * req.enrichment
        t_s = req.irrad_time_h * 3600
        sat_factor = 1.0 - math.exp(-lam * t_s)
        A_EOB = N_target * sigma_cm2 * req.flux_n_cm2_s * sat_factor

        return {
            "activity_Bq": round(A_EOB, 1),
            "yield_atoms": round(float(N_target * sigma_cm2 * req.flux_n_cm2_s * t_s), 1),
            "specific_activity_GBq_umol": None,
            "sigma_thermal_b": round(sigma_b, 3),
        }


# ── Suppliers endpoint ────────────────────────────────────────────

_suppliers_path = Path(__file__).parent.parent.parent / "data" / "suppliers.json"


@app.get("/api/suppliers")
def get_suppliers():
    """Enriched isotope and radioisotope suppliers."""
    if _suppliers_path.exists():
        return json.loads(_suppliers_path.read_text())
    return []


# Serve static frontend if built
_static = Path(__file__).parent.parent / "frontend" / "dist"
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="frontend")
