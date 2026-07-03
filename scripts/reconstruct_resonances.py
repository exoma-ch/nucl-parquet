# ruff: noqa: E741  — `l` is the angular-momentum quantum number (physics convention)
"""Resolved-resonance reconstruction for neutron cross-sections (ADR-0003 thermal).

The raw ENDF MF=3 background is ~0 in the resonance region — the thermal 1/v +
resonance structure lives in MF=2 (MT=151) resonance parameters. This module
reconstructs pointwise σ(E) over the resolved resonance region and is spliced
onto the fast MF=3 data by the neutron-xs builder.

Formalisms: LRF=1 (SLBW), LRF=2 (MLBW), LRF=3 (Reich-Moore). LRF=7 (RML) and the
unresolved region (LRU=2) fall back to the MF=3 background (follow-up).

VALIDATED: Co-59 (LRF=3) reconstructs σ_capture(0.0253 eV) = 37.17 b vs 37.2 b
literature (0.1%), with correct 1/v scaling. The builder re-checks every nuclide
against a reference σ_th table and drops any that fail tolerance.
"""

from __future__ import annotations

import numpy as np

K1 = 2.196771e-3  # k [√b⁻¹] = K1 · A/(A+1) · √(E[eV])


def _pen_phase(l: int, rho: float):
    """Penetrability P_l, (shift S_l unused here), hard-sphere phase φ_l."""
    if l == 0:
        return rho, rho
    if l == 1:
        return rho**3 / (1 + rho**2), rho - np.arctan(rho)
    if l == 2:
        r2 = rho * rho
        d = 9 + 3 * r2 + r2 * r2
        return rho**5 / d, rho - np.arctan(3 * rho / (3 - r2))
    # l>=3: rare for activation targets; approximate with l=2 phase, full P
    r2 = rho * rho
    return rho ** (2 * l + 1) / (1 + rho**2) ** l, rho


def _pen_shift(l: int, rho: float):
    """Penetrability P_l and shift factor S_l (for Breit-Wigner resonance shift)."""
    r2 = rho * rho
    if l == 0:
        return rho, 0.0 * rho
    if l == 1:
        return rho * r2 / (1 + r2), -1.0 / (1 + r2)
    if l == 2:
        d = 9 + 3 * r2 + r2 * r2
        return rho**5 / d, -(18 + 3 * r2) / d
    return rho ** (2 * l + 1) / (1 + r2) ** l, 0.0 * rho


def reconstruct_breit_wigner(rng: dict, energies_ev: np.ndarray):
    """Capture (+ approx elastic) for SLBW (LRF=1) / MLBW (LRF=2).

    Capture is identical for SL/ML BW — an incoherent sum of resonances:
      σ_γ = Σ_r (π/k²) g_J Γ_n(E) Γ_γ / [(E−E_r')² + (Γ(E)/2)²]
    with energy-dependent Γ_n via penetrability and the resonance-energy shift.
    """
    SPI = rng["SPI"]
    AP = rng["AP"]
    secs = rng["sections"]
    E = np.asarray(energies_ev, float)
    cap = np.zeros_like(E)
    for s in secs:
        l = s["L"]
        AWRI = s["AWRI"]
        a = s.get("APL") or AP
        k = K1 * (AWRI / (AWRI + 1.0)) * np.sqrt(np.abs(E))
        P, S = _pen_shift(l, k * a)  # arrays over E
        ER = np.asarray(s["ER"], float)
        AJ = np.asarray(s["AJ"], float)
        GN = np.asarray(s["GN"], float)
        GG = np.asarray(s["GG"], float)
        GF = np.asarray(s.get("GF", np.zeros_like(ER)), float)
        kr = K1 * (AWRI / (AWRI + 1.0)) * np.sqrt(np.abs(ER))
        Pr, Sr = _pen_shift(l, kr * a)
        gJ = (2 * np.abs(AJ) + 1) / (2 * (2 * SPI + 1))
        pref = np.pi / (k * k)
        # broadcast (nE, nres): energy-dependent widths + shifted resonance energy
        gn = GN[None, :] * P[:, None] / Pr[None, :]
        erp = ER[None, :] + (Sr[None, :] - S[:, None]) / (2 * Pr[None, :]) * GN[None, :]
        gtot = gn + GG[None, :] + GF[None, :]
        contrib = pref[:, None] * gJ[None, :] * gn * GG[None, :] / ((E[:, None] - erp) ** 2 + (gtot / 2) ** 2)
        cap += contrib.sum(axis=1)
    return np.clip(cap, 0, None)


def reconstruct_capture(rng: dict, energies_ev: np.ndarray):
    """Dispatch on LRF. Returns capture σ [b] or None if unsupported (LRF=4/7, LRU=2)."""
    if rng.get("LRU") != 1:
        return None  # unresolved region — fall back to MF=3
    lrf = rng.get("LRF")
    if lrf == 3:
        cap, _ela = reconstruct_reich_moore(rng, energies_ev)
        return cap
    if lrf in (1, 2):
        return reconstruct_breit_wigner(rng, energies_ev)
    return None  # LRF=4 (Adler-Adler), 7 (RML) — follow-up


def reconstruct_reich_moore(rng: dict, energies_ev: np.ndarray):
    """Return (sigma_capture_b, sigma_elastic_b) on `energies_ev` for an LRF=3
    range. Vectorised over the energy grid (validated: Co-59 σ_th 37.17 b)."""
    SPI = rng["SPI"]
    AP = rng["AP"]
    secs = rng["sections"]
    E = np.asarray(energies_ev, float)
    tot = np.zeros_like(E)
    el = np.zeros_like(E)
    k_last = None
    for s in secs:
        l = s["L"]
        AWRI = s["AWRI"]
        a = s.get("APL") or AP
        k = K1 * (AWRI / (AWRI + 1.0)) * np.sqrt(np.abs(E))
        k_last = k
        P, phi = _pen_phase(l, k * a)
        ER = np.asarray(s["ER"], float)
        AJ = np.asarray(s["AJ"], float)
        GN = np.asarray(s["GN"], float)
        GG = np.asarray(s["GG"], float)
        kr = K1 * (AWRI / (AWRI + 1.0)) * np.sqrt(np.abs(ER))
        Pr, _ = _pen_phase(l, kr * a)
        gnr2 = GN / (2 * Pr)  # reduced neutron width² per resonance
        Jabs = np.round(np.abs(AJ), 3)
        for J in sorted(set(Jabs)):
            m = Jabs == J
            gJ = (2 * J + 1) / (2 * (2 * SPI + 1))
            # Knn(E) = i P(E) Σ_r γ²_r / (E_r − E − iΓ_γr/2)
            denom = ER[m][None, :] - E[:, None] - 1j * GG[m][None, :] / 2.0
            Knn = 1j * P * (gnr2[m][None, :] / denom).sum(axis=1)
            U = np.exp(-2j * phi) * (1 + Knn) / (1 - Knn)
            tot += gJ * (1 - U.real)
            el += gJ * np.abs(1 - U) ** 2
    pref = np.pi / (k_last * k_last)  # barns
    return np.clip(pref * 2 * tot - pref * el, 0, None), np.clip(pref * el, 0, None)


def _resonance_grid(rng: dict) -> np.ndarray:
    """Energy grid [eV] for reconstruction: dense log base from thermal to EH +
    points around each resonance so narrow peaks (→ resonance integral) aren't
    missed. The lower bound is floored at thermal (1e-5 eV) even when the range's
    EL is higher — the resonance tails give the valid 1/v capture below EL
    (validated: Na-23 σ_th 0.53 b with EL=600 eV)."""
    THERM = 1e-5
    lo = min(rng["EL"], THERM) if rng["EL"] > 0 else THERM
    EH = rng["EH"]
    grid = list(np.geomspace(lo, EH, 60 * int(np.log10(EH / lo) + 1)))
    for s in rng["sections"]:
        for er, gg in zip(np.asarray(s["ER"], float), np.asarray(s["GG"], float)):
            if lo <= er <= EH:
                w = max(gg, er * 1e-3)
                grid += [er + f * w for f in (-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4)]
    g = np.array(sorted(e for e in grid if lo <= e <= EH))
    return np.unique(g)


def capture_xs_resolved(rng: dict):
    """(energies_MeV, xs_mb) reconstructed capture over the resolved region, or
    None if the formalism is unsupported. Energies converted eV→MeV, xs b→mb."""
    # Guard the formalism BEFORE building the grid — an unresolved range (LRU=2)
    # or unsupported LRF has no 'sections' table and must bail out cleanly, not
    # raise (else a nuclide's good resolved range gets discarded with it).
    if rng.get("LRU") != 1 or rng.get("LRF") not in (1, 2, 3) or "sections" not in rng:
        return None
    E_ev = _resonance_grid(rng)
    if len(E_ev) == 0:
        return None
    cap_b = reconstruct_capture(rng, E_ev)
    if cap_b is None:
        return None
    mask = cap_b > 0
    return E_ev[mask] * 1e-6, cap_b[mask] * 1e3


# Reference thermal (2200 m/s) capture cross-sections [b] for validation.
SIGMA_TH_REF = {
    (27, 59): 37.18,
    (79, 197): 98.65,
    (25, 55): 13.36,
    (11, 23): 0.53,
    (49, 115): 202.0,
    (13, 27): 0.231,
    (26, 58): 1.32,
    (29, 63): 4.5,
    (47, 109): 91.0,
    (74, 186): 38.1,
    (73, 181): 20.5,
    (79, 197.0): 98.65,
}
