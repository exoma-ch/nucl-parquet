"""Ship NIST XCOM compound elemental compositions (#113).

`xcom_compounds.parquet` ships integrated µ/ρ for 33 standard NIST materials
(water, air, tissue, bone, …) but doesn't store composition — so consumers
can't break a compound's photon attenuation into per-process σ (PE / Compton
/ Rayleigh / pair) via Bragg additivity.

This builds `data/meta/compound_compositions.parquet` from NIST XCOM's
published compound list (https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html),
with one row per (material, Z, weight_fraction).

The data is hardcoded here (33 materials × 3-9 elements each, ~150 entries)
because it's small, stable physics reference data; vendoring is cleaner than
HTML-scraping a NIST page that hasn't changed since 1995.

Output schema:
    material         str    (matches xcom_compounds.material slug, e.g. "water")
    Z                int32  (constituent element)
    weight_fraction  float64 (Bragg-additivity weight; Σ = 1.0 ± 1e-6 per material)

Issue: https://github.com/exoma-ch/nucl-parquet/issues/113
NIST source: NIST XCOM 1.5 (Berger et al.) — Table 4 compounds list.

Usage:
    uv run python -m nucl_parquet.build_compound_compositions
"""

from __future__ import annotations

from pathlib import Path

from .download import data_dir as _resolve_data_dir

# Compositions per NIST XCOM Table 4 (mass fractions). All entries sum to 1.0
# within ICRU/NIST rounding (typically ±5e-5). Keys match xcom_compounds.material.
#
# Source: NIST XCOM compound database, mirrored from NIST PML pages:
# - https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# - https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html
# - ICRU-44 (1989) for tissue/organ compositions
_COMPOSITIONS: dict[str, dict[int, float]] = {
    "water": {1: 0.111894, 8: 0.888106},
    "air": {6: 0.000124, 7: 0.755268, 8: 0.231781, 18: 0.012827},
    "polyethylene": {1: 0.143711, 6: 0.856289},
    "pmma": {1: 0.080538, 6: 0.599848, 8: 0.319614},
    "polystyrene": {1: 0.077418, 6: 0.922582},
    "teflon": {6: 0.240183, 9: 0.759817},  # PTFE
    "polyvinyl": {1: 0.048380, 6: 0.384360, 17: 0.567260},  # PVC
    "a150": {  # A-150 Tissue-Equivalent Plastic
        1: 0.101327,
        6: 0.775501,
        7: 0.035057,
        8: 0.052316,
        9: 0.017422,
        20: 0.018378,
    },
    "b100": {  # B-100 Bone-Equivalent Plastic
        1: 0.065471,
        6: 0.536945,
        7: 0.021500,
        8: 0.032085,
        9: 0.167411,
        20: 0.176589,
    },
    "c552": {  # C-552 Air-Equivalent Plastic
        1: 0.024680,
        6: 0.501610,
        8: 0.004527,
        9: 0.465209,
        14: 0.003973,
    },
    "lithiumflu": {3: 0.267585, 9: 0.732415},  # LiF
    "lithium": {3: 0.082085, 5: 0.255680, 8: 0.662235},  # Li2B4O7 (Lithium tetraborate)
    "fluoride": {9: 0.486659, 20: 0.513341},  # CaF2
    "calcium": {8: 0.470095, 16: 0.235534, 20: 0.294371},  # CaSO4
    "concrete": {  # Ordinary concrete (NIST std)
        1: 0.010,
        6: 0.001,
        8: 0.529107,
        11: 0.016000,
        12: 0.002,
        13: 0.033872,
        14: 0.337021,
        19: 0.013000,
        20: 0.044000,
        26: 0.014000,
    },
    "concreteba": {  # Barite concrete
        1: 0.003585,
        8: 0.311622,
        12: 0.001195,
        13: 0.004183,
        14: 0.010457,
        16: 0.107858,
        20: 0.050194,
        26: 0.047505,
        56: 0.463400,
    },
    "pyrex": {  # Borosilicate glass
        5: 0.040064,
        8: 0.539562,
        11: 0.028191,
        13: 0.011644,
        14: 0.377220,
        19: 0.003321,
    },
    "glass": {  # Lead glass
        8: 0.156453,
        14: 0.080866,
        22: 0.008092,
        33: 0.002651,
        82: 0.751938,
    },
    "cesium": {53: 0.488452, 55: 0.511548},  # CsI
    "telluride": {48: 0.468355, 52: 0.531645},  # CdTe
    "gallium": {31: 0.482034, 33: 0.517966},  # GaAs
    "bone": {  # Cortical bone, ICRU-44
        1: 0.034000,
        6: 0.155000,
        7: 0.042000,
        8: 0.435000,
        11: 0.001000,
        12: 0.002000,
        15: 0.103000,
        16: 0.003000,
        20: 0.225000,
    },
    "tissue": {  # Soft tissue, ICRU-44
        1: 0.102000,
        6: 0.143000,
        7: 0.034000,
        8: 0.708000,
        11: 0.002000,
        15: 0.003000,
        16: 0.003000,
        17: 0.002000,
        19: 0.003000,
    },
    "tissue4": {  # Soft tissue, ICRU 4-component
        1: 0.101172,
        6: 0.111000,
        7: 0.026000,
        8: 0.761828,
    },
    "muscle": {  # Skeletal muscle, ICRU-44
        1: 0.102000,
        6: 0.143000,
        7: 0.034000,
        8: 0.710000,
        11: 0.001000,
        15: 0.002000,
        16: 0.003000,
        17: 0.001000,
        19: 0.004000,
    },
    "adipose": {  # Adipose tissue, ICRU-44
        1: 0.114000,
        6: 0.598000,
        7: 0.007000,
        8: 0.278000,
        11: 0.001000,
        16: 0.001000,
        17: 0.001000,
    },
    "blood": {  # Whole blood, ICRU-44
        1: 0.102000,
        6: 0.110000,
        7: 0.033000,
        8: 0.745000,
        11: 0.001000,
        15: 0.001000,
        16: 0.002000,
        17: 0.003000,
        19: 0.002000,
        26: 0.001000,
    },
    "brain": {  # Grey/white matter, ICRU-44
        1: 0.107000,
        6: 0.145000,
        7: 0.022000,
        8: 0.712000,
        11: 0.002000,
        15: 0.004000,
        16: 0.002000,
        17: 0.003000,
        19: 0.003000,
    },
    "breast": {  # Breast tissue, ICRU-44
        1: 0.106000,
        6: 0.332000,
        7: 0.030000,
        8: 0.527000,
        11: 0.001000,
        15: 0.001000,
        16: 0.002000,
        17: 0.001000,
    },
    "lung": {  # Lung tissue, ICRU-44
        1: 0.103000,
        6: 0.105000,
        7: 0.031000,
        8: 0.749000,
        11: 0.002000,
        15: 0.002000,
        16: 0.003000,
        17: 0.003000,
        19: 0.002000,
    },
    "ovary": {  # Ovary, ICRU-44
        1: 0.105000,
        6: 0.093000,
        7: 0.024000,
        8: 0.768000,
        11: 0.002000,
        15: 0.002000,
        16: 0.002000,
        17: 0.002000,
        19: 0.002000,
    },
    "testis": {  # Testis, ICRU-44
        1: 0.106000,
        6: 0.099000,
        7: 0.020000,
        8: 0.766000,
        11: 0.002000,
        15: 0.001000,
        16: 0.002000,
        17: 0.002000,
        19: 0.002000,
    },
    "eye": {  # Eye lens, ICRU-44
        1: 0.099600,
        6: 0.193700,
        7: 0.053700,
        8: 0.653000,
    },
}


def build(data_dir: Path | None = None) -> None:
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    rows_material: list[str] = []
    rows_z: list[int] = []
    rows_w: list[float] = []
    for material in sorted(_COMPOSITIONS):
        weights = _COMPOSITIONS[material]
        total = sum(weights.values())
        if abs(total - 1.0) > 5e-3:
            raise ValueError(f"{material} weight fractions sum to {total}, expected 1.0 ± 5e-3")
        # Renormalize to remove the documented rounding noise (≤5e-4 typically)
        for z, w in sorted(weights.items()):
            rows_material.append(material)
            rows_z.append(int(z))
            rows_w.append(float(w) / total)

    df = pl.DataFrame(
        {
            "material": pl.Series(rows_material, dtype=pl.Utf8),
            "Z": pl.Series(rows_z, dtype=pl.Int32),
            "weight_fraction": pl.Series(rows_w, dtype=pl.Float64),
        }
    ).sort("material", "Z")

    out_path = data_dir / "meta" / "compound_compositions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    n_materials = df["material"].n_unique()
    print(f"  compound_compositions.parquet: {len(df):,} rows ({n_materials} materials) → {out_path}")


if __name__ == "__main__":
    build()
