from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# I find the main project folder from the location of this script.
project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"
plots_folder = project_folder / "plots"

# I create the plots folder if it does not already exist.
plots_folder.mkdir(exist_ok=True)


# I load the results from the two benchmark experiments I ran earlier.
sparse_results = pd.read_csv(
    results_folder / "sparse_benchmark_results.csv"
)

factorisation_results = pd.read_csv(
    results_folder / "factorisation_benchmark_results.csv"
)


print("SIT292 HD Plot Generation")
print()


# Figure 1: Sparse solver library runtime comparison

# I first compare the median runtimes of the three sparse solver libraries.
plt.figure(figsize=(9, 6))

plt.bar(
    sparse_results["library"],
    sparse_results["median_runtime_s"]
)

plt.xlabel("Library")
plt.ylabel("Median solve time (seconds)")
plt.title("Sparse Linear System Solver Runtime Comparison")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# I save the figure at a high resolution so I can use it in my report.
plt.savefig(
    plots_folder / "sparse_library_runtime.png",
    dpi=300
)

plt.close()

print("Saved sparse_library_runtime.png")


# Figure 2: Factorisation method runtime comparison

# I compare LU, QR, Cholesky and SVD using their median runtimes.
plt.figure(figsize=(9, 6))

plt.bar(
    factorisation_results["method"],
    factorisation_results["median_runtime_s"]
)

plt.xlabel("Factorisation method")
plt.ylabel("Median solve time (seconds)")
plt.title("Runtime Comparison of Linear Algebra Factorisations")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder / "factorisation_runtime.png",
    dpi=300
)

plt.close()

print("Saved factorisation_runtime.png")


# Figure 3: Numerical accuracy comparison

# I compare both the residual and the solution error because runtime alone
# does not show how accurately each method solved the system.
plt.figure(figsize=(9, 6))

plt.plot(
    factorisation_results["method"],
    factorisation_results["relative_residual"],
    marker="o",
    label="Relative residual"
)

plt.plot(
    factorisation_results["method"],
    factorisation_results["relative_solution_error"],
    marker="o",
    label="Relative solution error"
)

# I use a logarithmic scale because the errors are very small
# and occur at different orders of magnitude.
plt.yscale("log")

plt.xlabel("Factorisation method")
plt.ylabel("Relative error")
plt.title("Numerical Accuracy of Linear Algebra Factorisations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plots_folder / "factorisation_accuracy.png",
    dpi=300
)

plt.close()

print("Saved factorisation_accuracy.png")


print()
print("All HD plots generated successfully.")
print(f"Plots saved to: {plots_folder}")