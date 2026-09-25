import csv
from pathlib import Path

import matplotlib.pyplot as plt


# I find the main project folder from the location of this script.
project_folder = Path(__file__).resolve().parents[1]
results_file = project_folder / "results" / "sparse_scaling_results.csv"
plots_folder = project_folder / "plots"

plots_folder.mkdir(exist_ok=True)


# I use this helper so the script still works if I used a slightly
# different column name when I saved the benchmark CSV.
def find_column(row, possible_names):
    for name in possible_names:
        if name in row:
            return name

    raise KeyError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns: {list(row.keys())}"
    )


with open(results_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)


if not rows:
    raise ValueError("The sparse scaling results CSV is empty.")


matrix_size_column = find_column(
    rows[0],
    ["matrix_size", "n", "size"]
)

target_density_column = find_column(
    rows[0],
    ["target_density", "density_target"]
)

density_l_column = find_column(
    rows[0],
    ["density_L", "density_l", "factor_density"]
)


# I group the factor-density results by the target density used
# when I generated each sparse matrix.
grouped_results = {}

for row in rows:
    target_density = float(row[target_density_column])
    matrix_size = int(float(row[matrix_size_column]))
    density_l = float(row[density_l_column])

    if target_density not in grouped_results:
        grouped_results[target_density] = []

    grouped_results[target_density].append(
        (matrix_size, density_l)
    )


# I plot d(L) directly because it shows how sparse the Cholesky
# factor remains as matrix size and input density change.
plt.figure(figsize=(10, 6))

for target_density in sorted(grouped_results):
    values = sorted(grouped_results[target_density])

    matrix_sizes = [value[0] for value in values]
    factor_densities = [value[1] for value in values]

    plt.plot(
        matrix_sizes,
        factor_densities,
        marker="o",
        label=f"Target density {target_density:.3f}"
    )


plt.xlabel("Matrix size n")
plt.ylabel("Density of Cholesky factor L")
plt.title("Density of the Cholesky Factor L as Matrix Size Increases")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = plots_folder / "sparse_factor_density_scaling.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Factor-density plot generated successfully.")
print(output_file)