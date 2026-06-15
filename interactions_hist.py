import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "Interaction_plots_hist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SYSTEMS AND RUNS
# ============================================================

SYSTEMS = {
    "0 mM KCl": "0KCl",
    "5 mM KCl": "5KCl",
    "100 mM KCl": "100KCl"
}

RUNS = ["run1", "run2", "run3"]

# ============================================================
# FILE SETTINGS
# ============================================================

HBOND_FILENAME = "hbnum_all.xvg"

# ============================================================
# READ XVG FILE
# ============================================================

def read_xvg(filepath):
    data = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                time = float(parts[0])
                value = float(parts[1])
                data.append([time, value])
            except ValueError:
                continue

    return pd.DataFrame(data, columns=["time", "hbonds"])

# ============================================================
# LOAD HYDROGEN BOND DATA
# ============================================================

def load_hbond_data(system_folder):
    all_hbonds = []

    for run in RUNS:
        filepath = os.path.join(
            system_folder,
            run,
            "Data",
            HBOND_FILENAME
        )

        if not os.path.exists(filepath):
            print(f"WARNING: missing file: {filepath}")
            continue

        df = read_xvg(filepath)

        if df.empty:
            print(f"WARNING: empty file: {filepath}")
            continue

        all_hbonds.extend(df["hbonds"].values)

    return np.array(all_hbonds)

# ============================================================
# PLOT HISTOGRAM FOR ONE SYSTEM
# ============================================================

def plot_hbond_histogram(system_name, system_folder):
    hbonds = load_hbond_data(system_folder)

    if len(hbonds) == 0:
        print(f"No hydrogen bond data found for {system_name}")
        return

    plt.figure(figsize=(8, 3))

    # Histogram as probability density
    plt.hist(
        hbonds,
        bins=30,
        density=True,
        alpha=0.45,
        edgecolor="white"
    )

    # Smooth KDE curve
    if len(np.unique(hbonds)) > 1:
        kde = gaussian_kde(hbonds)
        x_grid = np.linspace(hbonds.min(), hbonds.max(), 300)
        y_grid = kde(x_grid)

        plt.plot(
            x_grid,
            y_grid,
            linewidth=2.5
        )

    plt.xlabel("Number of hydrogen bonds", fontsize=13)
    plt.ylabel("Probability density", fontsize=13)
    plt.title(f"Hydrogen bonds - {system_name}", fontsize=14)

    plt.tight_layout()

    safe_system_name = (
        system_name
        .replace(" ", "_")
        .replace("+", "")
        .replace("/", "")
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"hydrogen_bonds_hist_{safe_system_name}.png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_file}")
# ============================================================
# COMBINED HISTOGRAM OF ALL SYSTEMS
# ============================================================

def plot_combined_hbond_histogram():

    plt.figure(figsize=(8,4))

    colors = {
        "0 mM KCl": "purple",
        "5 mM KCl": "cornflowerblue",
        "100 mM KCl": "deeppink"
    }

    for system_name, system_folder in SYSTEMS.items():

        hbonds = load_hbond_data(system_folder)

        # histogram
        plt.hist(
            hbonds,
            bins=25,
            density=True,
            alpha=0.18,
            color=colors[system_name]
        )

        # smooth density
        kde = gaussian_kde(hbonds)
        x = np.linspace(hbonds.min(), hbonds.max(), 400)
        y = kde(x)

        plt.plot(
            x,
            y,
            linewidth=2.5,
            color=colors[system_name],
            label=system_name
        )

    plt.xlabel("Number of hydrogen bonds", fontsize=13)
    plt.ylabel("Probability density", fontsize=13)

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Hydrogen_bonds_combined.png"
        ),
        dpi=300
    )

    plt.show()

# ============================================================
# MAKE PLOTS
# ============================================================

for system_name, system_folder in SYSTEMS.items():
    print(f"Plotting {system_name}")
    plot_hbond_histogram(system_name, system_folder)
plot_combined_hbond_histogram()
