import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Output directory
# ============================================================
output_dir = Path("RMSF_plots_runs")
output_dir.mkdir(exist_ok=True)

# ============================================================
# Function to read XVG files
# ============================================================
def read_xvg(filename):
    data = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue

            parts = line.split()

            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])

    return np.array(data)


# ============================================================
# RMSF files for the five situations
# ============================================================
situations = {
    # Unfolded TBA
    "TBA_unfolded_0KCl": [
        "simulation_TBA/0KCl/run1/Data/rmsf_tba.xvg",
        "simulation_TBA/0KCl/run2/Data/rmsf_tba.xvg",
        "simulation_TBA/0KCl/run3/Data/rmsf_tba.xvg",
    ],

    "TBA_unfolded_100KCl": [
        "simulation_TBA/100KCl/run1/Data/rmsf_tba.xvg",
        "simulation_TBA/100KCl/run2/Data/rmsf_tba.xvg",
        "simulation_TBA/100KCl/run3/Data/rmsf_tba.xvg",
    ],

    # Folded TBA
    "TBA_folded_0KCl": [
        "simulation_TBA_K_Thrombin/0KCl/run1/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/0KCl/run2/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/0KCl/run3/Data/rmsf_tba.xvg",
    ],

    "TBA_folded_5KCl": [
        "simulation_TBA_K_Thrombin/5KCl/run1/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/5KCl/run2/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/5KCl/run3/Data/rmsf_tba.xvg",
    ],

    "TBA_folded_100KCl": [
        "simulation_TBA_K_Thrombin/100KCl/run1/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/100KCl/run2/Data/rmsf_tba.xvg",
        "simulation_TBA_K_Thrombin/100KCl/run3/Data/rmsf_tba.xvg",
    ],
}

# ============================================================
# Groups
# ============================================================
unfolded_keys = [
    "TBA_unfolded_0KCl",
    "TBA_unfolded_100KCl"
]

folded_keys = [
    "TBA_folded_0KCl",
    "TBA_folded_5KCl",
    "TBA_folded_100KCl"
]

run_colors = ["red", "green", "blue"]

# ============================================================
# Read all data
# ============================================================
all_data = {}

for situation, files in situations.items():
    runs = []

    for file in files:
        file = Path(file)

        if not file.exists():
            print(f"Warning: file not found: {file}")
            continue

        data = read_xvg(file)

        if data.size == 0:
            continue

        runs.append(data)

    all_data[situation] = runs

# ============================================================
# Function to calculate axis limits for a group
# ============================================================
def get_group_axis_limits(keys, all_data):
    all_x = []
    all_y = []

    for key in keys:
        for run in all_data[key]:
            all_x.extend(run[:, 0])
            all_y.extend(run[:, 1])

    x_min = np.min(all_x)
    x_max = np.max(all_x)

    y_min = 0
    y_max = np.max(all_y) * 1.15

    return x_min, x_max, y_min, y_max


# Axis limits per group
unfolded_xmin, unfolded_xmax, unfolded_ymin, unfolded_ymax = get_group_axis_limits(
    unfolded_keys,
    all_data
)

folded_xmin, folded_xmax, folded_ymin, folded_ymax = get_group_axis_limits(
    folded_keys,
    all_data
)

print(f"Unfolded axes: x = {unfolded_xmin}-{unfolded_xmax}, y = {unfolded_ymin}-{unfolded_ymax}")
print(f"Folded axes: x = {folded_xmin}-{folded_xmax}, y = {folded_ymin}-{folded_ymax}")

# ============================================================
# Plot each situation separately
# ============================================================
for situation, runs in all_data.items():

    if len(runs) == 0:
        print(f"No data for {situation}")
        continue

    plt.figure(figsize=(8, 5))

    y_values = []

    # --------------------------------------------------------
    # Plot separate runs
    # --------------------------------------------------------
    for i, run in enumerate(runs):

        x = run[:, 0]
        y = run[:, 1]

        y_values.append(y)

        plt.plot(
            x,
            y,
            color=run_colors[i],
            linewidth=1.5,
            label=f"run{i+1}"
        )

    # --------------------------------------------------------
    # Mean and variance/std
    # --------------------------------------------------------
    y_values = np.array(y_values)

    mean_rmsf = np.mean(y_values, axis=0)
    std_rmsf = np.std(y_values, axis=0)

    plt.plot(
        x,
        mean_rmsf,
        color="black",
        linewidth=2.5,
        label="mean"
    )

    plt.fill_between(
        x,
        mean_rmsf - std_rmsf,
        mean_rmsf + std_rmsf,
        color="grey",
        alpha=0.4,
        label="std"
    )

    # --------------------------------------------------------
    # Apply same axes within folded/unfolded group
    # --------------------------------------------------------
    if situation in unfolded_keys:
        plt.xlim(unfolded_xmin, unfolded_xmax)
        plt.ylim(unfolded_ymin, unfolded_ymax)

    elif situation in folded_keys:
        plt.xlim(folded_xmin, folded_xmax)
        plt.ylim(folded_ymin, folded_ymax)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    plt.xlabel("Residue / atom index")
    plt.ylabel("RMSF Å")
    plt.title(situation)

    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------
    plt.savefig(
        output_dir / f"{situation}_RMSF_runs.png",
        dpi=300
    )

    plt.savefig(
        output_dir / f"{situation}_RMSF_runs.pdf"
    )

    plt.close()

    # --------------------------------------------------------
    # Save mean/std data
    # --------------------------------------------------------
    np.savetxt(
        output_dir / f"{situation}_RMSF_mean_std.txt",
        np.column_stack((x, mean_rmsf, std_rmsf)),
        header="Index Mean_RMSF Std_RMSF"
    )

    print(f"Saved RMSF plot for {situation}")

print(f"\nAll RMSF plots saved in: {output_dir}")
