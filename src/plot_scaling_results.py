import csv
from pathlib import Path

import matplotlib.pyplot as plt


# I find the main project folder from the location of this script.
project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"
plots_folder = project_folder / "plots"

plots_folder.mkdir(exist_ok=True)


def read_csv(file_path):
    with file_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


# I load the two new scaling experiments.
factorisation_results = read_csv(
    results_folder
    / "factorisation_scaling_results.csv"
)

sparse_results = read_csv(
    results_folder
    / "sparse_scaling_results.csv"
)


# Plot 1: CPU-time scaling for LU, QR, Cholesky and SVD


methods = [
    "LU",
    "QR",
    "Cholesky (LLT)",
    "SVD"
]

plt.figure(figsize=(10, 6))

for method in methods:
    method_rows = [
        row
        for row in factorisation_results
        if row["method"] == method
    ]

    method_rows.sort(
        key=lambda row: int(row["matrix_size"])
    )

    sizes = [
        int(row["matrix_size"])
        for row in method_rows
    ]

    cpu_times = [
        float(row["median_cpu_time_s"])
        for row in method_rows
    ]

    plt.plot(
        sizes,
        cpu_times,
        marker="o",
        label=method
    )

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel("Median CPU time (seconds)")
plt.title(
    "CPU-Time Scaling of Dense Factorisation Methods"
)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder
    / "factorisation_cpu_scaling.png",
    dpi=300
)

plt.close()


# Plot 2: Sparse Cholesky CPU time against matrix size


target_densities = [
    0.001,
    0.002,
    0.005
]

plt.figure(figsize=(10, 6))

for target_density in target_densities:
    density_rows = [
        row
        for row in sparse_results
        if abs(
            float(row["target_density"])
            - target_density
        ) < 1e-12
    ]

    density_rows.sort(
        key=lambda row: int(row["matrix_size"])
    )

    sizes = [
        int(row["matrix_size"])
        for row in density_rows
    ]

    cpu_times = [
        float(
            row[
                "median_cpu_factorisation_time_s"
            ]
        )
        for row in density_rows
    ]

    plt.plot(
        sizes,
        cpu_times,
        marker="o",
        label=f"Target density {target_density:.3f}"
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel(
    "Median Cholesky CPU factorisation time (seconds)"
)
plt.title(
    "Sparse Cholesky CPU Time by Matrix Size and Input Density"
)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder
    / "sparse_cpu_scaling.png",
    dpi=300
)

plt.close()



# Plot 3: Fill ratio against matrix size


plt.figure(figsize=(10, 6))

for target_density in target_densities:
    density_rows = [
        row
        for row in sparse_results
        if abs(
            float(row["target_density"])
            - target_density
        ) < 1e-12
    ]

    density_rows.sort(
        key=lambda row: int(row["matrix_size"])
    )

    sizes = [
        int(row["matrix_size"])
        for row in density_rows
    ]

    fill_ratios = [
        float(row["fill_ratio"])
        for row in density_rows
    ]

    plt.plot(
        sizes,
        fill_ratios,
        marker="o",
        label=f"Target density {target_density:.3f}"
    )

plt.xlabel("Matrix size n")
plt.ylabel(
    "Fill ratio nnz(L) / nnz(tril(A))"
)
plt.title(
    "Growth of Cholesky Fill-In with Matrix Size"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder
    / "sparse_fill_ratio_scaling.png",
    dpi=300
)

plt.close()


# Plot 4: Number of non-zero entries in L against matrix size

plt.figure(figsize=(10, 6))

for target_density in target_densities:
    density_rows = [
        row
        for row in sparse_results
        if abs(
            float(row["target_density"])
            - target_density
        ) < 1e-12
    ]

    density_rows.sort(
        key=lambda row: int(row["matrix_size"])
    )

    sizes = [
        int(row["matrix_size"])
        for row in density_rows
    ]

    nnz_l = [
        int(row["nnz_L"])
        for row in density_rows
    ]

    plt.plot(
        sizes,
        nnz_l,
        marker="o",
        label=f"Target density {target_density:.3f}"
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Matrix size n")
plt.ylabel("Non-zero entries in L")
plt.title(
    "Growth of Non-Zero Entries in the Cholesky Factor L"
)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder
    / "sparse_nnz_L_scaling.png",
    dpi=300
)

plt.close()



# Plot 5: Fill ratio against the actual measured density of A

matrix_sizes = [
    500,
    1000,
    1500,
    2000
]

plt.figure(figsize=(10, 6))

for matrix_size in matrix_sizes:
    size_rows = [
        row
        for row in sparse_results
        if int(row["matrix_size"]) == matrix_size
    ]

    size_rows.sort(
        key=lambda row: float(
            row["actual_density_A"]
        )
    )

    actual_densities = [
        float(row["actual_density_A"])
        for row in size_rows
    ]

    fill_ratios = [
        float(row["fill_ratio"])
        for row in size_rows
    ]

    plt.plot(
        actual_densities,
        fill_ratios,
        marker="o",
        label=f"n = {matrix_size}"
    )

plt.xlabel("Actual density of A")
plt.ylabel(
    "Fill ratio nnz(L) / nnz(tril(A))"
)
plt.title(
    "Effect of Input Density on Cholesky Fill-In"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder
    / "sparse_fill_vs_actual_density.png",
    dpi=300
)

plt.close()


print("All scaling plots generated successfully.")
print()
print(
    plots_folder
    / "factorisation_cpu_scaling.png"
)
print(
    plots_folder
    / "sparse_cpu_scaling.png"
)
print(
    plots_folder
    / "sparse_fill_ratio_scaling.png"
)
print(
    plots_folder
    / "sparse_nnz_L_scaling.png"
)
print(
    plots_folder
    / "sparse_fill_vs_actual_density.png"
)