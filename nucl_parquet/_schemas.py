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

EXFOR_SCHEMA = {
    "exfor_entry": "Utf8",
    "target_Z": "Int32",
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "state": "Utf8",
    "energy_MeV": "Float64",
    "energy_err_MeV": "Float64",
    "xs_mb": "Float64",
    "xs_err_mb": "Float64",
    "author": "Utf8",
    "year": "Int32",
}
