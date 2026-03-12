"""Generate SVG cross-section comparison plots (TENDL vs EXFOR).

Produces one SVG per reaction showing evaluated data as smooth curves
and experimental EXFOR data as scatter points with error bars.

Usage:
    # Plot a specific reaction:
    python scripts/plot_xs.py --projectile p --element Cu --target-a 63 \
        --residual-z 30 --residual-a 63

    # Plot all reactions for an element:
    python scripts/plot_xs.py --projectile p --element Cu

    # Plot all reactions for all elements:
    python scripts/plot_xs.py --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# Element symbols
_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
}
_SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in _ELEMENT_SYMBOLS.items()}

PROJECTILE_NAMES = {"p": "proton", "d": "deuteron", "t": "triton", "h": "³He", "a": "alpha"}

# Plot styling
EVAL_COLORS = {
    "tendl-2024": "#2563eb",
    "endfb-8.1": "#dc2626",
    "jendl-5": "#16a34a",
}
EXFOR_CMAP = plt.cm.Set2


def _sym(z: int) -> str:
    return _ELEMENT_SYMBOLS.get(z, f"Z{z}")


def _reaction_label(
    target_z: int, target_a: int, projectile: str,
    residual_z: int, residual_a: int, state: str,
) -> str:
    """Generate reaction notation like ⁶³Cu(p,n)⁶³Zn."""
    target_sym = _sym(target_z)
    res_sym = _sym(residual_z)
    state_suffix = f"{'m' if state == 'm' else 'g' if state == 'g' else ''}"
    proj_name = projectile
    return f"{target_a}{target_sym}({proj_name},x){residual_a}{res_sym}{state_suffix}"


def _interpolate_eval(energies: np.ndarray, xs: np.ndarray, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline interpolation on log-log scale for smooth evaluated curves."""
    if len(energies) < 4:
        return energies, xs

    # Filter to positive values for log-log
    mask = (energies > 0) & (xs > 0)
    e = energies[mask]
    x = xs[mask]
    if len(e) < 4:
        return energies, xs

    log_e = np.log10(e)
    log_x = np.log10(x)

    cs = CubicSpline(log_e, log_x, extrapolate=False)
    e_interp = np.logspace(log_e[0], log_e[-1], n_points)
    x_interp = 10 ** cs(np.log10(e_interp))

    # Remove NaN from extrapolation
    valid = np.isfinite(x_interp) & (x_interp > 0)
    return e_interp[valid], x_interp[valid]


def plot_reaction(
    projectile: str,
    element: str,
    target_z: int,
    target_a: int,
    residual_z: int,
    residual_a: int,
    state: str,
    output_dir: Path,
) -> Path | None:
    """Plot a single reaction comparison (TENDL vs EXFOR) as SVG."""
    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False

    # --- Evaluated libraries ---
    for lib_name, color in EVAL_COLORS.items():
        lib_path = ROOT / lib_name / "xs" / f"{projectile}_{element}.parquet"
        if not lib_path.exists():
            continue

        df = pl.read_parquet(lib_path).filter(
            (pl.col("target_A") == target_a)
            & (pl.col("residual_Z") == residual_z)
            & (pl.col("residual_A") == residual_a)
            & (pl.col("state") == state)
        ).sort("energy_MeV")

        if df.is_empty():
            continue

        energies = df["energy_MeV"].to_numpy()
        xs = df["xs_mb"].to_numpy()

        # Interpolate for smooth curve
        e_smooth, xs_smooth = _interpolate_eval(energies, xs)
        ax.plot(e_smooth, xs_smooth, color=color, linewidth=1.5, label=lib_name, zorder=3)
        has_data = True

    # --- EXFOR experimental data ---
    exfor_path = ROOT / "exfor" / f"{projectile}_{element}.parquet"
    if exfor_path.exists():
        exfor_df = pl.read_parquet(exfor_path).filter(
            (pl.col("target_A").is_in([target_a, 0]))  # Include nat. element data
            & (pl.col("residual_Z") == residual_z)
            & (pl.col("residual_A") == residual_a)
            & (pl.col("state") == state)
        ).sort("energy_MeV")

        if not exfor_df.is_empty():
            # Group by author+year for separate markers
            datasets = exfor_df.group_by(["author", "year"]).agg(pl.all())
            for i, row in enumerate(datasets.iter_rows(named=True)):
                author = row["author"]
                year = row["year"]
                e = np.array(row["energy_MeV"])
                xs = np.array(row["xs_mb"])
                xerr_raw = row["xs_err_mb"]
                xerr = np.array([v if v is not None else 0.0 for v in xerr_raw], dtype=np.float64)

                color = EXFOR_CMAP(i % 8)
                label = f"{author} ({year})"

                if np.any(xerr > 0):
                    ax.errorbar(
                        e, xs, yerr=xerr, fmt="o", color=color, markersize=3,
                        linewidth=0.5, capsize=1.5, label=label, zorder=2, alpha=0.8,
                    )
                else:
                    ax.scatter(
                        e, xs, color=color, s=12, label=label, zorder=2, alpha=0.8,
                    )
                has_data = True

    if not has_data:
        plt.close(fig)
        return None

    # --- Styling ---
    title = _reaction_label(target_z, target_a, projectile, residual_z, residual_a, state)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Energy (MeV)", fontsize=10)
    ax.set_ylabel("Cross-section (mb)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.1, linewidth=0.3)

    # Legend: limit to 10 entries, place outside if many
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 10:
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
    else:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1),
                  framealpha=0.8, ncol=1)
        fig.subplots_adjust(right=0.72)

    fig.tight_layout()

    # Save SVG
    res_sym = _sym(residual_z)
    state_suffix = f"-{state}" if state else ""
    filename = f"{target_a}{element}_{projectile}_{residual_z}-{res_sym}-{residual_a}{state_suffix}.svg"
    plot_dir = output_dir / "plots" / projectile / element / str(target_a)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / filename
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_element(
    projectile: str,
    element: str,
    output_dir: Path,
) -> list[Path]:
    """Plot all reactions for an element + projectile."""
    tendl_path = ROOT / "tendl-2024" / "xs" / f"{projectile}_{element}.parquet"
    if not tendl_path.exists():
        logger.warning("No TENDL data for %s_%s", projectile, element)
        return []

    df = pl.read_parquet(tendl_path)
    target_z_vals = df.select(pl.col("residual_Z")).unique()  # Just to get Z context

    # Get target Z from element
    target_z = _SYMBOL_TO_Z.get(element)
    if target_z is None:
        return []

    # Get unique reactions
    reactions = (
        df.select("target_A", "residual_Z", "residual_A", "state")
        .unique()
        .sort("target_A", "residual_Z", "residual_A", "state")
    )

    paths: list[Path] = []
    for row in reactions.iter_rows(named=True):
        path = plot_reaction(
            projectile=projectile,
            element=element,
            target_z=target_z,
            target_a=row["target_A"],
            residual_z=row["residual_Z"],
            residual_a=row["residual_A"],
            state=row["state"],
            output_dir=output_dir,
        )
        if path:
            paths.append(path)

    logger.info("  %s(%s,x): %d plots", element, projectile, len(paths))
    return paths


def get_tendl_elements(projectile: str) -> list[str]:
    """Get elements available in TENDL for a projectile."""
    xs_dir = ROOT / "tendl-2024" / "xs"
    return sorted(
        f.stem.split("_", 1)[1]
        for f in xs_dir.glob(f"{projectile}_*.parquet")
        if not f.stem.split("_", 1)[1].startswith("Z")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SVG cross-section comparison plots.",
    )
    parser.add_argument("--projectile", choices=["p", "d", "t", "h", "a"])
    parser.add_argument("--element", help="Element symbol")
    parser.add_argument("--target-a", type=int, help="Target mass number")
    parser.add_argument("--residual-z", type=int)
    parser.add_argument("--residual-a", type=int)
    parser.add_argument("--state", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()

    total = 0

    if args.target_a and args.residual_z and args.residual_a:
        # Single reaction
        from scripts.fetch_exfor import _SYMBOL_TO_Z
        target_z = _SYMBOL_TO_Z[args.element]
        path = plot_reaction(
            args.projectile, args.element, target_z, args.target_a,
            args.residual_z, args.residual_a, args.state, args.output,
        )
        if path:
            logger.info("Wrote %s", path)
            total = 1
    elif args.element and args.projectile:
        paths = plot_element(args.projectile, args.element, args.output)
        total = len(paths)
    elif args.all:
        projectiles = [args.projectile] if args.projectile else ["p", "d", "t", "h", "a"]
        for proj in projectiles:
            for elem in get_tendl_elements(proj):
                paths = plot_element(proj, elem, args.output)
                total += len(paths)
    else:
        parser.error("Specify --element + --projectile, or --all")

    logger.info("Done. %d plots generated.", total)


if __name__ == "__main__":
    main()
