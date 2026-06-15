import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("End_to_end_histograms")
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
    # Unfolded
    "TBA_unfolded_0KCl": [
        "A_0KCl_run1_end_to_end.xvg",
        "A_0KCl_run2_end_to_end.xvg",
        "A_0KCl_run3_end_to_end.xvg",
    ],

    "TBA_unfolded_100KCl": [
        "B_100KCl_run1_end_to_end.xvg",
        "B_100KCl_run2_end_to_end.xvg",
        "B_100KCl_run3_end_to_end.xvg",
    ],

    # Folded
    "TBA_folded_0KCl": [
        "C_Thrombin_0KCl_run1_end_to_end.xvg",
        "C_Thrombin_0KCl_run2_end_to_end.xvg",
        "C_Thrombin_0KCl_run3_end_to_end.xvg",
    ],

    "TBA_folded_100KCl": [
        "D_Thrombin_100KCl_run1_end_to_end.xvg",
        "D_Thrombin_100KCl_run2_end_to_end.xvg",
        "D_Thrombin_100KCl_run3_end_to_end.xvg",
    ],

    "TBA_folded_5KCl": [
        "E_Thrombin_5KCl_run1_end_to_end.xvg",
        "E_Thrombin_5KCl_run2_end_to_end.xvg",
        "E_Thrombin_5KCl_run3_end_to_end.xvg",
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

distance_data = {}

for situation, files in situations.items():

    all_distances = []

    for file in files:

        file = Path(file)

        if not file.exists():
            print(f"Warning: file not found: {file}")
            continue

        data = read_xvg(file)

        if data.size == 0:
            print(f"Warning: empty file: {file}")
            continue

        # Column 1 = end-to-end distance in nm
        # Convert nm -> Å
        distance_A = data[:, 1] * 10

        all_distances.extend(distance_A)

    distance_data[situation] = np.array(all_distances)

unfolded_values = np.concatenate([
    distance_data[key]
    for key in unfolded_keys
    if len(distance_data[key]) > 0
])

folded_values = np.concatenate([
    distance_data[key]
    for key in folded_keys
    if len(distance_data[key]) > 0
])

n_bins = 50

# X-axis limits
unfolded_xmin = np.min(unfolded_values)
unfolded_xmax = np.max(unfolded_values)
unfolded_bins = np.linspace(unfolded_xmin, unfolded_xmax, n_bins + 1)

folded_xmin = np.min(folded_values)
folded_xmax = np.max(folded_values)
folded_bins = np.linspace(folded_xmin, folded_xmax, n_bins + 1)


def get_group_ymax(keys, bins):

    ymax = 0

    for key in keys:

        values = distance_data[key]

        if len(values) == 0:
            continue

        hist, _ = np.histogram(
            values,
            bins=bins,
            density=True
        )

        ymax = max(ymax, np.max(hist))

    return ymax * 1.10


unfolded_ymax = get_group_ymax(unfolded_keys, unfolded_bins)
folded_ymax = get_group_ymax(folded_keys, folded_bins)

print(f"Unfolded x-axis: {unfolded_xmin:.2f}–{unfolded_xmax:.2f} Å")
print(f"Folded x-axis: {folded_xmin:.2f}–{folded_xmax:.2f} Å")

for situation, distances in distance_data.items():

    if len(distances) == 0:
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
        distances,
        bins=bins,
        density=True
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    np.savetxt(
        output_dir / f"{situation}_end_to_end_histogram.txt",
        np.column_stack((bin_centers, hist)),
        header="End_to_end_distance_Angstrom Probability_density"
    )

    # Save raw distances
    np.savetxt(
        output_dir / f"{situation}_end_to_end_distances_Angstrom.txt",
        distances,
        header="End_to_end_distance_Angstrom"
    )


    plt.figure(figsize=(7, 5))

    plt.hist(
        distances,
        bins=bins,
        density=True,
        alpha=0.7
    )

    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)

    plt.xlabel("End-to-end distance (Å)")
    plt.ylabel("Probability density")
    plt.title(situation)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{situation}_end_to_end_histogram.png",
        dpi=300
    )

    plt.savefig(
        output_dir / f"{situation}_end_to_end_histogram.pdf"
    )

    plt.close()

    print(f"Saved histogram for {situation}")

print(f"\nAll histogram plots and data saved in: {output_dir}")
