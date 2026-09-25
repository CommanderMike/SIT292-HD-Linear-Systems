import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh, spsolve

from sksparse.cholmod import cho_factor
from pypardiso import spsolve as pardiso_spsolve


# I use the same seed each time so I can reproduce the experiment.
SEED = 292

# A 50 x 50 grid produces a 2500 x 2500 sparse matrix.
GRID_SIZE = 50

# I repeat each solve so I am not relying on one timing.
TRIALS = 5


rng = np.random.default_rng(SEED)


# Build the sparse matrix


# I use the standard 2D finite-difference Laplacian.
# This gives me a sparse, symmetric and positive definite matrix.
main_diagonal = 2.0 * np.ones(GRID_SIZE)
off_diagonal = -1.0 * np.ones(GRID_SIZE - 1)

T = diags(
    diagonals=[
        off_diagonal,
        main_diagonal,
        off_diagonal
    ],
    offsets=[-1, 0, 1],
    format="csc"
)

I = eye(GRID_SIZE, format="csc")

A = (
    kron(I, T, format="csc")
    + kron(T, I, format="csc")
)

n = A.shape[0]


# I create a known true solution first.
# This lets me measure the actual solution error later.
x_true = rng.standard_normal(n)

# I calculate b from Ax = b.
b = A @ x_true


print("SIT292 HD Sparse Benchmark")
print()

print(f"Matrix size: {A.shape}")
print(f"Non-zero entries: {A.nnz}")
print(f"Density: {A.nnz / (n * n):.6f}")
print(f"Symmetric: {(A - A.T).nnz == 0}")
print(f"Length of b: {len(b)}")



# Check positive definiteness


# Since I want to use Cholesky, I check that the matrix is
# positive definite by finding its smallest eigenvalue.
smallest_eigenvalue = eigsh(
    A,
    k=1,
    which="SA",
    return_eigenvectors=False
)[0]

positive_definite = smallest_eigenvalue > 0

print()
print(f"Smallest eigenvalue: {smallest_eigenvalue:.6e}")
print(f"Positive definite: {positive_definite}")



# Solver functions


def solve_with_scipy():
    # SciPy uses a sparse direct solver for this system.
    return spsolve(A, b)


def solve_with_cholmod():
    # I use CHOLMOD's sparse Cholesky factorisation because
    # this matrix is symmetric and positive definite.
    cholmod_factor = cho_factor(A)
    return cholmod_factor.solve(b)


def solve_with_pardiso():
    # PyPardiso provides another sparse direct solver.
    return pardiso_spsolve(A, b)


# Accuracy calculations

def calculate_relative_residual(x_calculated):
    return (
        np.linalg.norm(A @ x_calculated - b)
        / np.linalg.norm(b)
    )


def calculate_solution_error(x_calculated):
    return (
        np.linalg.norm(x_calculated - x_true)
        / np.linalg.norm(x_true)
    )


# Timing helper

def benchmark_solver(name, solve_function):
    times = []
    final_solution = None

    for trial in range(1, TRIALS + 1):
        start = time.perf_counter()

        final_solution = solve_function()

        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        print(
            f"{name:12s} | Trial {trial} | "
            f"{elapsed:.6f} s"
        )

    # I use the median because one unusually slow run
    # should not dominate the result.
    median_time = float(np.median(times))

    return median_time, final_solution


# Warm-up


# I run each solver once before recording timings.
# This helps reduce first-run overhead in the comparison.
print()
print("Running warm-up solves...")

solve_with_scipy()
solve_with_cholmod()
solve_with_pardiso()

print("Warm-up complete.")



# Benchmark the three libraries


print()
print("Running benchmark...")
print()

scipy_time, x_scipy = benchmark_solver(
    "SciPy",
    solve_with_scipy
)

print()

cholmod_time, x_cholmod = benchmark_solver(
    "CHOLMOD",
    solve_with_cholmod
)

print()

pardiso_time, x_pardiso = benchmark_solver(
    "PyPardiso",
    solve_with_pardiso
)


# Calculate accuracy

scipy_residual = calculate_relative_residual(x_scipy)
scipy_error = calculate_solution_error(x_scipy)

cholmod_residual = calculate_relative_residual(x_cholmod)
cholmod_error = calculate_solution_error(x_cholmod)

pardiso_residual = calculate_relative_residual(x_pardiso)
pardiso_error = calculate_solution_error(x_pardiso)


# Store the results

results = pd.DataFrame(
    [
        {
            "library": "SciPy",
            "method": "Sparse direct / LU",
            "matrix_size": n,
            "non_zero_entries": A.nnz,
            "median_runtime_s": scipy_time,
            "relative_residual": scipy_residual,
            "relative_solution_error": scipy_error
        },
        {
            "library": "CHOLMOD",
            "method": "Sparse Cholesky (LLT)",
            "matrix_size": n,
            "non_zero_entries": A.nnz,
            "median_runtime_s": cholmod_time,
            "relative_residual": cholmod_residual,
            "relative_solution_error": cholmod_error
        },
        {
            "library": "PyPardiso",
            "method": "PARDISO sparse direct",
            "matrix_size": n,
            "non_zero_entries": A.nnz,
            "median_runtime_s": pardiso_time,
            "relative_residual": pardiso_residual,
            "relative_solution_error": pardiso_error
        }
    ]
)



# Display results

print()
print("Benchmark results")
print()

print(
    results[
        [
            "library",
            "method",
            "median_runtime_s",
            "relative_residual",
            "relative_solution_error"
        ]
    ].to_string(index=False)
)


# Save results


project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"

results_folder.mkdir(exist_ok=True)

output_file = results_folder / "sparse_benchmark_results.csv"

results.to_csv(
    output_file,
    index=False
)

print()
print(f"Results saved to: {output_file}")
print()
print("Sparse benchmark completed successfully.")