"""Expected Parquet column schemas for validation."""

from __future__ import annotations

XS_SCHEMA = {
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "state": "Utf8",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}

# ---------------------------------------------------------------------------
# Canonical cross-section schema
# ---------------------------------------------------------------------------
# One shape for every sigma(E) table, so that evaluated production data,
# transport channels and EXFOR measurements union without special cases.
#
# It exists because the legacy 6-column XS_SCHEMA above encodes reaction
# identity *in the file path* rather than in the data. Two shipped bugs came
# straight out of that:
#
#   #273  the unified `xs` view interleaved isobaric targets, because target_Z
#         is not a column — the loader now regexes the filename and joins a
#         symbol->Z table at query time to recover it.
#   ---   the same view still silently merges FIVE projectiles: a query for
#         Cu-63 -> Zn-63 returns (p,n), (d,2n), (a,x), (h,x) and (t,x) rows
#         superposed, because `projectile` is likewise only in the filename.
#
# Identity therefore lives in columns. Nulls mean "not applicable" — never 0,
# which collides with real Z=0 products and made (n,tot)/(n,el)/(n,f)
# indistinguishable in the 82% of EXFOR rows that name no residual.
CANONICAL_XS_SCHEMA = {
    # --- provenance of the evaluation / measurement
    "library": "Utf8",
    # 'production' — summed over every channel reaching this residual (MT null)
    # 'channel'    — one ENDF reaction channel (MT set; residual may be null)
    "kind": "Utf8",
    # --- reaction identity
    "projectile": "Utf8",  # n p d t h a g, or a heavy ion such as 'ar40'
    "proj_Z": "Int32",
    "proj_A": "Int32",
    "target_Z": "Int32",
    "target_A": "Int32",  # 0 = natural element (ENDF convention)
    "MT": "Int32",  # null for production sums
    "residual_Z": "Int32",  # null when the channel names no residual
    "residual_A": "Int32",
    "state": "Utf8",  # residual isomeric state: '' g m m1 m2
    # --- the datum
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
    "energy_err_MeV": "Float64",
    "xs_err_mb": "Float64",
    # --- experimental provenance (null for evaluations)
    "source_entry": "Utf8",
    "author": "Utf8",
    "year": "Int32",
}

# Columns that must never be null in a canonical table — the identity spine.
CANONICAL_XS_REQUIRED = (
    "library",
    "kind",
    "projectile",
    "proj_Z",
    "proj_A",
    "target_Z",
    "target_A",
    "energy_MeV",
    "xs_mb",
)

# Heavy-ion production XS (hi-xs-prod): carries projectile identity
HI_XS_PROD_SCHEMA = {
    "proj_Z": "Int32",
    "proj_A": "Int32",
    "target_Z": "Int32",
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}

# Heavy-ion total reaction XS (hi-xs): projectile-target pair only
HI_XS_SCHEMA = {
    "target_Z": "Int32",
    "target_A": "Int32",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}


STOPPING_SCHEMA = {
    "source": "Utf8",
    "target_Z": "Int32",
    "energy_MeV": "Float64",
    "dedx": "Float64",
}

ABUNDANCES_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "symbol": "Utf8",
    "abundance": "Float64",
    "atomic_mass": "Float64",
}

DECAY_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "half_life_s": "Float64",
    "decay_mode": "Utf8",
    "daughter_Z": "Int32",
    "daughter_A": "Int32",
    "daughter_state": "Utf8",
    "branching": "Float64",
}

ELEMENTS_SCHEMA = {
    "Z": "Int32",
    "symbol": "Utf8",
}

DOSE_CONSTANTS_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "k_uSv_m2_MBq_h": "Float64",
    "dominant_gamma_keV": "Float64",
    "n_photon_lines": "Int32",
    "source": "Utf8",
}

XCOM_ELEMENTS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "mu_rho_cm2_g": "Float64",
    "mu_en_rho_cm2_g": "Float64",
}

XCOM_COMPOUNDS_SCHEMA = {
    "material": "Utf8",
    "energy_MeV": "Float64",
    "mu_rho_cm2_g": "Float64",
    "mu_en_rho_cm2_g": "Float64",
}

EPDL_PHOTON_XS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "process": "Utf8",
    "xs_barns": "Float64",
}

EPDL_FORM_FACTORS_SCHEMA = {
    "Z": "Int32",
    "momentum_transfer": "Float64",
    "form_factor": "Float64",
}

EPDL_SCATTERING_FN_SCHEMA = {
    "Z": "Int32",
    "momentum_transfer": "Float64",
    "scattering_fn": "Float64",
}

EPDL_ANOMALOUS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "component": "Utf8",
    "factor": "Float64",
}

EPDL_SUBSHELL_PE_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "subshell": "Utf8",
    "xs_barns": "Float64",
    "edge_MeV": "Float64",
    "fluorescence_yield_eV": "Float64",
}

EADL_TRANSITIONS_SCHEMA = {
    "Z": "Int32",
    "vacancy_shell": "Utf8",
    "filling_shell": "Utf8",
    "transition_type": "Utf8",
    "energy_keV": "Float64",
    "probability": "Float64",
    "edge_keV": "Float64",
}

EEDL_ELECTRON_XS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "process": "Utf8",
    "xs_barns": "Float64",
}

NUCLIDES_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "symbol": "Utf8",
    "jp": "Utf8",
    "half_life_s": "Float64",
    "level_keV": "Float64",
    "decay_1": "Utf8",
    "decay_1_pct": "Float64",
    "decay_2": "Utf8",
    "decay_2_pct": "Float64",
}

# EXFOR now shares CANONICAL_XS_SCHEMA like every other cross-section table;
# `exfor_entry` became the source-agnostic `source_entry`. Kept as an alias so
# external callers importing it keep working.
EXFOR_SCHEMA = CANONICAL_XS_SCHEMA
