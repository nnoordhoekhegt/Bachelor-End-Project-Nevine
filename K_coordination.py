import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Output directory
# ============================================================
output_dir = Path("Oxygen_coordination_histograms")
output_dir.mkdir(exist_ok=True)

# ============================================================
# Read XVG file
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
# Input files
# Adjust names if needed
# ============================================================
situations = {
    "TBA_unfolded_100KCl": [
        "B_100KCl_run1_K_coordination.xvg",
        "B_100KCl_run2_K_coordination.xvg",
        "B_100KCl_run3_K_coordination.xvg",
    ],

    "TBA_folded_100KCl": [
        "D_Thrombin_100KCl_run1_K_coordination.xvg",
        "D_Thrombin_100KCl_run2_K_coordination.xvg",
        "D_Thrombin_100KCl_run3_K_coordination.xvg",
    ],

    "TBA_folded_5KCl": [
        "E_Thrombin_5KCl_run1_K_coordination.xvg",
        "E_Thrombin_5KCl_run2_K_coordination.xvg",
        "E_Thrombin_5KCl_run3_K_coordination.xvg",
    ],
}

# ============================================================
# Read coordination data
# ============================================================
coord_data = {}

for situation, files in situations.items():

    all_coord = []

    for file in files:

        file = Path(file)

        if not file.exists():
            print(f"Warning: file not found: {file}")
            continue

        data = read_xvg(file)

        if data.size == 0:
            print(f"Warning: empty file: {file}")
            continue

        # Column 1 = number of coordinated oxygens / selected atoms
        coordinated_oxygens = data[:, 1]

        # Round to nearest integer, because this should be a count
        coordinated_oxygens = np.rint(coordinated_oxygens).astype(int)

        all_coord.extend(coordinated_oxygens)

    coord_data[situation] = np.array(all_coord)

# ============================================================
# Common axes
# ============================================================
all_values = np.concatenate([
    values for values in coord_data.values()
    if len(values) > 0
])

x_min = int(np.min(all_values))
x_max = int(np.max(all_values))

bins = np.arange(x_min - 0.5, x_max + 1.5, 1)

# Calculate common y-axis
y_max = 0

for situation, values in coord_data.items():

    if len(values) == 0:
        continue

    hist, _ = np.histogram(
        values,
        bins=bins,
        density=True
    )

    y_max = max(y_max, np.max(hist))

y_max *= 1.10

# ============================================================
# Plot each situation separately
# ============================================================
for situation, values in coord_data.items():

    if len(values) == 0:
        print(f"Skipping {situation}: no data")
        continue

    hist, bin_edges = np.histogram(
        values,
        bins=bins,
        density=True
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Save histogram data
    np.savetxt(
        output_dir / f"{situation}_oxygen_coordination_histogram.txt",
        np.column_stack((bin_centers, hist)),
        header="Number_of_coordinated_oxygens Probability_density"
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.7,
        edgecolor="black"
    )

    plt.xlim(x_min - 0.5, x_max + 0.5)
    plt.ylim(0, y_max)

    plt.xlabel("Number of coordinated oxygens")
    plt.ylabel("Probability density")
    plt.title(situation)

    plt.tight_layout()

    plt.savefig(output_dir / f"{situation}_oxygen_coordination_histogram.png", dpi=300)
    plt.savefig(output_dir / f"{situation}_oxygen_coordination_histogram.pdf")

    plt.close()

    print(f"Saved oxygen coordination histogram for {situation}")

print(f"\nAll coordination histograms saved in: {output_dir}")
