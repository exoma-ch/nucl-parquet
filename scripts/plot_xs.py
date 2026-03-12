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

# Plot styling — colors for evaluated libraries (auto-discovered from catalog)
EVAL_PALETTE = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#854d0e",  # brown
    "#4338ca",  # indigo
    "#059669",  # emerald
    "#7c3aed",  # violet
    "#b91c1c",  # dark red
]
EXFOR_CMAP = plt.cm.Set2

# Line styles to further distinguish libraries
EVAL_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]


def _get_eval_libraries() -> list[tuple[str, str]]:
    """Discover evaluated libraries from catalog.json.

    Returns list of (lib_key, lib_name) for all non-experimental libraries.
    """
    catalog_path = ROOT / "catalog.json"
    if not catalog_path.exists():
        return [("tendl-2024", "TENDL-2024")]

    import json
    catalog = json.loads(catalog_path.read_text())
    libs = []
    for key, info in catalog.get("libraries", {}).items():
        if info.get("data_type") != "experimental_cross_sections":
            libs.append((key, info.get("name", key)))
    return libs


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

    # --- Evaluated libraries (auto-discovered) ---
    eval_libs = _get_eval_libraries()
    for i, (lib_key, lib_label) in enumerate(eval_libs):
        lib_path = ROOT / lib_key / "xs" / f"{projectile}_{element}.parquet"
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

        color = EVAL_PALETTE[i % len(EVAL_PALETTE)]
        ls = EVAL_LINESTYLES[i % len(EVAL_LINESTYLES)]

        # Interpolate for smooth curve
        e_smooth, xs_smooth = _interpolate_eval(energies, xs)
        ax.plot(e_smooth, xs_smooth, color=color, linestyle=ls,
                linewidth=1.5, label=lib_label, zorder=3)
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
    # Collect unique reactions across ALL libraries
    target_z = _SYMBOL_TO_Z.get(element)
    if target_z is None:
        return []

    all_dfs: list[pl.DataFrame] = []
    for lib_dir in ROOT.iterdir():
        xs_path = lib_dir / "xs" / f"{projectile}_{element}.parquet"
        if xs_path.exists():
            try:
                df = pl.read_parquet(xs_path)
                all_dfs.append(df.select("target_A", "residual_Z", "residual_A", "state"))
            except Exception:
                continue

    if not all_dfs:
        logger.warning("No data for %s_%s in any library", projectile, element)
        return []

    combined = pl.concat(all_dfs)
    reactions = (
        combined.unique()
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


def get_available_elements(projectile: str) -> list[str]:
    """Get elements available across all libraries for a projectile."""
    seen: set[str] = set()
    for lib_dir in ROOT.iterdir():
        xs_dir = lib_dir / "xs"
        if not xs_dir.is_dir():
            continue
        for f in xs_dir.glob(f"{projectile}_*.parquet"):
            elem = f.stem.split("_", 1)[1]
            if not elem.startswith("Z"):
                seen.add(elem)
    return sorted(seen)


# Backward compat alias
get_tendl_elements = get_available_elements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SVG cross-section comparison plots.",
    )
    parser.add_argument("--projectile", choices=["n", "p", "d", "t", "h", "a"])
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
        projectiles = [args.projectile] if args.projectile else ["n", "p", "d", "t", "h", "a"]
        for proj in projectiles:
            for elem in get_tendl_elements(proj):
                paths = plot_element(proj, elem, args.output)
                total += len(paths)
    else:
        parser.error("Specify --element + --projectile, or --all")

    logger.info("Done. %d plots generated.", total)


if __name__ == "__main__":
    main()
