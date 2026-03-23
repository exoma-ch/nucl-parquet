# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "numpy",
# ]
# ///
"""
PET scanner cost analysis: crystal depth vs axial FOV.
Scanner: PET4PETs for 44Sc imaging, clinical bore R_in=350mm.
Dual-ended readout, 8x8 SiPM array modules, LYSO crystals.
Pricing from supplier quotes (EPIC/EBO, Hamamatsu), March 2026.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Scanner geometry ---
R_IN_MM = 350.0
CRYSTAL_DEPTHS_MM = [25, 40]
FOV_VALUES_MM = [3, 6, 12, 24, 48, 96, 192]
CRYSTAL_PITCH_MM = 3.0
MODULE_SIZE = 8  # 8x8 crystals per SiPM array module

# Crystals around the full circumference — fixed for all configs
N_CRYSTALS_PHI = int(np.floor(2 * np.pi * R_IN_MM / CRYSTAL_PITCH_MM))
N_MODULES_PHI = int(np.floor(N_CRYSTALS_PHI / MODULE_SIZE))

# --- Pricing ---
# LYSO crystal cost [CHF/mm³]
# Source: EPIC Q260055 (Feb 2026, qty~1000), EBO Q260042 (qty~few)
LYSO_COST_PER_MM3 = {
    25: 0.032,
    40: 0.033,
}

# Hamamatsu S14161-3050HS-08 (8x8 array, 3x3mm pixels, 3.2mm pitch)
# Source: Hamamatsu H25706 (Nov 2025), H20829
SIPM_PRICE_TIERS = [
    (100, 409),
    (500, 230),
    (1000, 200),
    (2000, 164),
]


def sipm_price_per_array(n_arrays: int) -> float:
    """Quantity-tiered price per SiPM array [CHF]."""
    price = 926.0  # list price for qty < 100
    for min_qty, unit_price in SIPM_PRICE_TIERS:
        if n_arrays >= min_qty:
            price = unit_price
    return float(price)


def compute_scanner_costs(dr_mm: int, fov_mm: float) -> dict:
    """Compute all relevant quantities for a scanner config."""
    n_rings = max(1, int(np.floor(fov_mm / CRYSTAL_PITCH_MM)))

    n_crystals_total = N_CRYSTALS_PHI * n_rings

    n_modules_axial = int(np.ceil(n_rings / MODULE_SIZE))
    n_modules_total = n_modules_axial * N_MODULES_PHI

    n_sipm_arrays = n_modules_total * 2
    n_channels_total = n_crystals_total * 2

    r_out = R_IN_MM + dr_mm
    actual_axial_length = n_rings * CRYSTAL_PITCH_MM
    v_lyso_mm3 = np.pi * (r_out**2 - R_IN_MM**2) * actual_axial_length

    cost_lyso = v_lyso_mm3 * LYSO_COST_PER_MM3[dr_mm]
    price_per_array = sipm_price_per_array(n_sipm_arrays)
    cost_sipm = n_sipm_arrays * price_per_array
    cost_total = cost_lyso + cost_sipm

    return {
        "dr_mm": dr_mm,
        "fov_mm": fov_mm,
        "n_rings": n_rings,
        "n_crystals_phi": N_CRYSTALS_PHI,
        "n_crystals_total": n_crystals_total,
        "n_modules_total": n_modules_total,
        "n_sipm_arrays": n_sipm_arrays,
        "n_channels_total": n_channels_total,
        "v_lyso_mm3": v_lyso_mm3,
        "cost_lyso_chf": cost_lyso,
        "cost_sipm_chf": cost_sipm,
        "cost_total_chf": cost_total,
        "sipm_price_per_array_chf": price_per_array,
    }


def build_dataframe() -> pd.DataFrame:
    rows = [
        compute_scanner_costs(dr, fov)
        for dr in CRYSTAL_DEPTHS_MM
        for fov in FOV_VALUES_MM
    ]
    return pd.DataFrame(rows)


# --- LYSO volume ratio for fixed-LYSO comparison ---
# V_LYSO(25, L) / L = pi * ((R+25)^2 - R^2) = pi * (700*25 + 625) = pi * 18125
# V_LYSO(40, L) / L = pi * ((R+40)^2 - R^2) = pi * (700*40 + 1600) = pi * 29600
_LYSO_SLOPE_25 = np.pi * (2 * R_IN_MM * 25 + 25**2)  # mm² per mm FOV
_LYSO_SLOPE_40 = np.pi * (2 * R_IN_MM * 40 + 40**2)
FOV_SCALE_25_TO_40 = _LYSO_SLOPE_25 / _LYSO_SLOPE_40  # ≈ 0.621


# --- Plotting helpers ---
def _annotate_pct_diff(ax: plt.Axes, x: np.ndarray, vals_a: np.ndarray, vals_b: np.ndarray) -> None:
    """Label each bar group with the % change from vals_a to vals_b."""
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        pct = 100 * (b - a) / a
        ax.text(i, max(a, b) * 1.03, f"{pct:+.0f}%", ha="center", fontsize=7, color="dimgray")


def _save(fig: plt.Figure, name: str, out_dir: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_channel_count(df: pd.DataFrame, out_dir: str) -> None:
    df25 = df[df.dr_mm == 25].sort_values("fov_mm")
    df40 = df[df.dr_mm == 40].sort_values("fov_mm")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    fovs = df25.fov_mm.values
    x = np.arange(len(fovs))
    w = 0.35

    ax1.bar(x - w / 2, df25.n_channels_total.values, w, label="25 mm depth", color="#4C72B0")
    ax1.bar(x + w / 2, df40.n_channels_total.values, w, label="40 mm depth", color="#DD8452")

    ax2.plot(x - w / 2, df25.n_modules_total.values, "s--", color="#4C72B0", alpha=0.6, label="modules 25 mm")
    ax2.plot(x + w / 2, df40.n_modules_total.values, "o--", color="#DD8452", alpha=0.6, label="modules 40 mm")

    _annotate_pct_diff(ax1, x, df25.n_channels_total.values, df40.n_channels_total.values)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:.0f}" for v in fovs])
    ax1.set_xlabel("Axial FOV [mm]")
    ax1.set_ylabel("Total readout channels")
    ax2.set_ylabel("Total SiPM modules")
    ax1.set_title("Readout channel count vs axial FOV\n(dual-ended readout, 8×8 modules)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    _save(fig, "cost_channel_count.png", out_dir)


def plot_cost_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, dr in zip(axes, CRYSTAL_DEPTHS_MM):
        sub = df[df.dr_mm == dr].sort_values("fov_mm")
        fovs = sub.fov_mm.values
        x = np.arange(len(fovs))
        lyso = sub.cost_lyso_chf.values / 1000
        sipm = sub.cost_sipm_chf.values / 1000

        ax.bar(x, lyso, label="LYSO crystals", color="#4C72B0")
        ax.bar(x, sipm, bottom=lyso, label="SiPM arrays", color="#DD8452")

        for i, (l, s) in enumerate(zip(lyso, sipm)):
            ax.text(i, l + s + 0.5, f"{l+s:.0f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.0f}" for v in fovs])
        ax.set_xlabel("Axial FOV [mm]")
        ax.set_ylabel("Cost [kCHF]")
        ax.set_title(f"Crystal depth {dr} mm")
        ax.legend(fontsize=8)

    fig.suptitle("Scanner cost breakdown: LYSO + SiPM arrays", fontsize=12)
    fig.tight_layout()
    _save(fig, "cost_breakdown.png", out_dir)


def plot_fixed_lyso_comparison(df: pd.DataFrame, out_dir: str) -> None:
    df25 = df[df.dr_mm == 25].sort_values("fov_mm").reset_index(drop=True)

    equiv_fovs = [fov * FOV_SCALE_25_TO_40 for fov in df25.fov_mm]
    rows_40_equiv = [compute_scanner_costs(40, fov) for fov in equiv_fovs]
    df40_equiv = pd.DataFrame(rows_40_equiv)

    fov_labels = [f"{v:.0f}" for v in df25.fov_mm]
    x = np.arange(len(fov_labels))
    w = 0.35

    fig, (ax_ch, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

    ax_ch.bar(x - w / 2, df25.n_channels_total.values, w, label="25 mm @ FOV", color="#4C72B0")
    ax_ch.bar(x + w / 2, df40_equiv.n_channels_total.values, w, label=f"40 mm @ {FOV_SCALE_25_TO_40:.3f}×FOV", color="#DD8452")
    ax_ch.set_xticks(x)
    ax_ch.set_xticklabels(fov_labels)
    ax_ch.set_xlabel("25 mm FOV [mm]")
    ax_ch.set_ylabel("Total readout channels")
    ax_ch.set_title("Channels (equal LYSO volume)")
    ax_ch.legend(fontsize=8)
    _annotate_pct_diff(ax_ch, x, df25.n_channels_total.values, df40_equiv.n_channels_total.values)

    cost25 = df25.cost_total_chf.values / 1000
    cost40 = df40_equiv.cost_total_chf.values / 1000
    ax_cost.bar(x - w / 2, cost25, w, label="25 mm @ FOV", color="#4C72B0")
    ax_cost.bar(x + w / 2, cost40, w, label=f"40 mm @ {FOV_SCALE_25_TO_40:.3f}×FOV", color="#DD8452")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(fov_labels)
    ax_cost.set_xlabel("25 mm FOV [mm]")
    ax_cost.set_ylabel("Total cost [kCHF]")
    ax_cost.set_title("Total cost (equal LYSO volume)")
    ax_cost.legend(fontsize=8)
    _annotate_pct_diff(ax_cost, x, cost25, cost40)

    fig.suptitle(
        f"Fixed LYSO volume comparison\n(40 mm uses {FOV_SCALE_25_TO_40:.3f}× FOV of 25 mm config)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "cost_fixed_lyso_comparison.png", out_dir)


def plot_sipm_fraction(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for dr, marker, color in [(25, "o-", "#4C72B0"), (40, "s-", "#DD8452")]:
        sub = df[df.dr_mm == dr].sort_values("fov_mm")
        frac = sub.cost_sipm_chf.values / sub.cost_total_chf.values
        ax.plot(sub.fov_mm.values, frac * 100, marker, label=f"{dr} mm depth", color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Axial FOV [mm]")
    ax.set_ylabel("SiPM cost fraction [%]")
    ax.set_title("SiPM cost fraction vs axial FOV\n(dual-ended readout)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax.set_xticks(FOV_VALUES_MM)
    ax.set_xticklabels([str(v) for v in FOV_VALUES_MM])

    fig.tight_layout()
    _save(fig, "cost_sipm_fraction.png", out_dir)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = script_dir
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    df = build_dataframe()

    csv_path = os.path.join(results_dir, "cost_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")
    print(f"  rows: {len(df)}")

    print("\nKey figures:")
    print(df[["dr_mm", "fov_mm", "n_channels_total", "cost_total_chf"]].to_string(index=False))

    print("\nGenerating plots...")
    plot_channel_count(df, png_dir)
    plot_cost_breakdown(df, png_dir)
    plot_fixed_lyso_comparison(df, png_dir)
    plot_sipm_fraction(df, png_dir)
    print("Done.")


if __name__ == "__main__":
    main()
