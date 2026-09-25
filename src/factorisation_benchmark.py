import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg


# I use the same seed each time so I can reproduce the experiment.
SEED = 292

# I keep this smaller than the sparse benchmark because QR and SVD
# require more work and use dense matrices.
N = 400
TRIALS = 5

rng = np.random.default_rng(SEED)


# I create an orthogonal matrix first.
random_matrix = rng.standard_normal((N, N))
Q, _ = np.linalg.qr(random_matrix)

# I choose positive eigenvalues so the matrix is symmetric
# positive definite and Cholesky can be used.
eigenvalues = np.geomspace(1.0, 1e-4, N)

A = Q @ np.diag(eigenvalues) @ Q.T

# I create a known solution so I can measure solution error later.
x_true = rng.standard_normal(N)
b = A @ x_true


print("SIT292 HD Factorisation Benchmark")
print()

print(f"Matrix size: {A.shape}")
print(f"Symmetric: {np.allclose(A, A.T)}")
print(f"Condition number: {np.linalg.cond(A):.3e}")
print(f"Trials per method: {TRIALS}")
print()


# LU factorisation
def solve_lu():
    lu, piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve((lu, piv), b)


# QR factorisation
def solve_qr():
    Q_qr, R = scipy.linalg.qr(A)

    # From Ax = b and A = QR:
    # QRx = b
    # Rx = Q^T b
    y = Q_qr.T @ b

    return scipy.linalg.solve_triangular(
        R,
        y
    )


# Cholesky factorisation
def solve_cholesky():
    # Because A is symmetric positive definite,
    # it can be written as A = LL^T.
    L = scipy.linalg.cholesky(
        A,
        lower=True
    )

    # First solve Ly = b.
    y = scipy.linalg.solve_triangular(
        L,
        b,
        lower=True
    )

    # Then solve L^T x = y.
    return scipy.linalg.solve_triangular(
        L.T,
        y,
        lower=False
    )


# SVD factorisation
def solve_svd():
    U, singular_values, Vh = scipy.linalg.svd(A)

    # For A = U Sigma V^T,
    # x = V Sigma^-1 U^T b.
    return (
        Vh.T
        @ ((U.T @ b) / singular_values)
    )


methods = {
    "LU": solve_lu,
    "QR": solve_qr,
    "Cholesky (LLT)": solve_cholesky,
    "SVD": solve_svd
}


# I run each method once before timing it so the first call
# does not have an unfair amount of setup overhead.
print("Running warm-up solves...")

for solve_function in methods.values():
    solve_function()

print("Warm-up complete.")
print()


results = []


print("Running benchmark...")
print()

for method_name, solve_function in methods.items():

    times = []

    for trial in range(1, TRIALS + 1):

        start = time.perf_counter()

        x_calculated = solve_function()

        end = time.perf_counter()

        runtime = end - start
        times.append(runtime)

        print(
            f"{method_name:16s} | "
            f"Trial {trial} | "
            f"{runtime:.6f} s"
        )

    median_runtime = np.median(times)

    # I measure how closely the calculated solution satisfies Ax = b.
    relative_residual = (
        np.linalg.norm(A @ x_calculated - b)
        / np.linalg.norm(b)
    )

    # I also compare the calculated solution with the known true solution.
    relative_solution_error = (
        np.linalg.norm(x_calculated - x_true)
        / np.linalg.norm(x_true)
    )

    results.append(
        {
            "method": method_name,
            "median_runtime_s": median_runtime,
            "relative_residual": relative_residual,
            "relative_solution_error": relative_solution_error
        }
    )

    print()


results_df = pd.DataFrame(results)

print("Factorisation benchmark results")
print()
print(results_df.to_string(index=False))


# Save the results for the report.
project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"

results_folder.mkdir(exist_ok=True)

output_file = (
    results_folder
    / "factorisation_benchmark_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print(f"Results saved to: {output_file}")
print()
print("Factorisation benchmark completed successfully.")