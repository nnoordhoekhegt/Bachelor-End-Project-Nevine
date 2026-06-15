import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("Hist_data2")
output_dir.mkdir(exist_ok=True)

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

situations = {
    "TBA_unfolded_0KCl": [
        "simulation_TBA/0KCl/run1/Data/rg_tba.xvg",
        "simulation_TBA/0KCl/run2/Data/rg_tba.xvg",
        "simulation_TBA/0KCl/run3/Data/rg_tba.xvg",
    ],
    "TBA_unfolded_100KCl": [
        "simulation_TBA/100KCl/run1/Data/rg_tba.xvg",
        "simulation_TBA/100KCl/run2/Data/rg_tba.xvg",
        "simulation_TBA/100KCl/run3/Data/rg_tba.xvg",
    ],
    "TBA_folded_0KCl": [
        "simulation_TBA_K_Thrombin/0KCl/run1/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/0KCl/run2/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/0KCl/run3/Data/rg_tba.xvg",
    ],
    "TBA_folded_5KCl": [
        "simulation_TBA_K_Thrombin/5KCl/run1/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/5KCl/run2/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/5KCl/run3/Data/rg_tba.xvg",
    ],
    "TBA_folded_100KCl": [
        "simulation_TBA_K_Thrombin/100KCl/run1/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/100KCl/run2/Data/rg_tba.xvg",
        "simulation_TBA_K_Thrombin/100KCl/run3/Data/rg_tba.xvg",
    ],
}

unfolded_keys = [
    "TBA_unfolded_0KCl",
    "TBA_unfolded_100KCl"
]

folded_keys = [
    "TBA_folded_0KCl",
    "TBA_folded_5KCl",
    "TBA_folded_100KCl"
]

rg_data = {}

for situation, files in situations.items():
    all_rg = []

    for file in files:
        file = Path(file)

        if not file.exists():
            print(f"Warning: file not found: {file}")
            continue

        data = read_xvg(file)

        if data.size == 0:
            print(f"Warning: no data in {file}")
            continue

        rg_values = data[:, 1] * 10  # nm --> Å
        all_rg.extend(rg_values)

    rg_data[situation] = np.array(all_rg)

unfolded_values = np.concatenate([
    rg_data[key] for key in unfolded_keys
    if len(rg_data[key]) > 0
])

folded_values = np.concatenate([
    rg_data[key] for key in folded_keys
    if len(rg_data[key]) > 0
])

n_bins = 50

unfolded_xmin = np.min(unfolded_values)
unfolded_xmax = np.max(unfolded_values)
unfolded_bins = np.linspace(unfolded_xmin, unfolded_xmax, n_bins + 1)

folded_xmin = np.min(folded_values)
folded_xmax = np.max(folded_values)
folded_bins = np.linspace(folded_xmin, folded_xmax, n_bins + 1)


unfolded_ymax = 0
for key in unfolded_keys:
    hist, _ = np.histogram(
        rg_data[key],
        bins=unfolded_bins,
        density=True
    )
    unfolded_ymax = max(unfolded_ymax, np.max(hist))

folded_ymax = 0
for key in folded_keys:
    hist, _ = np.histogram(
        rg_data[key],
        bins=folded_bins,
        density=True
    )
    folded_ymax = max(folded_ymax, np.max(hist))

# Add 10% space above highest bar
unfolded_ymax *= 1.10
folded_ymax *= 1.10

print(f"Unfolded x-axis: {unfolded_xmin:.2f} Å to {unfolded_xmax:.2f} Å")
print(f"Unfolded y-axis: 0 to {unfolded_ymax:.2f}")

print(f"Folded x-axis: {folded_xmin:.2f} Å to {folded_xmax:.2f} Å")
print(f"Folded y-axis: 0 to {folded_ymax:.2f}")

for situation, rg_values in rg_data.items():

    if len(rg_values) == 0:
        print(f"Skipping {situation}: no data")
        continue

    if situation in unfolded_keys:
        bins = unfolded_bins
        x_min = unfolded_xmin
        x_max = unfolded_xmax
        y_max = unfolded_ymax

    elif situation in folded_keys:
        bins = folded_bins
        x_min = folded_xmin
        x_max = folded_xmax
        y_max = folded_ymax

    else:
        raise ValueError(f"Unknown situation: {situation}")

    hist, bin_edges = np.histogram(
        rg_values,
        bins=bins,
        density=True
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    np.savetxt(
        output_dir / f"{situation}_histogram_Angstrom.txt",
        np.column_stack((bin_centers, hist)),
        header="Rg_bin_center_Angstrom Probability_density"
    )

    np.savetxt(
        output_dir / f"{situation}_Rg_values_Angstrom.txt",
        rg_values,
        header="Radius_of_Gyration_Angstrom"
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        rg_values,
        bins=bins,
        density=True,
        alpha=0.7
    )

    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)

    plt.xlabel("Radius of gyration (Å)")
    plt.ylabel("Probability density")
    plt.title(situation)

    plt.tight_layout()

    plt.savefig(output_dir / f"{situation}_histogram_Angstrom.png", dpi=300)
    plt.savefig(output_dir / f"{situation}_histogram_Angstrom.pdf")

    plt.close()

    print(f"Saved histogram for {situation}")

print(f"\nAll output saved in: {output_dir}")
