import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg


# I use the same base seed as my earlier benchmark so the experiment
# remains reproducible.
SEED = 292

# I use several matrix sizes so I can study how runtime increases
# instead of relying on only one 400 x 400 matrix.
SIZES = [100, 200, 400, 800]

# I keep five timing trials, which is consistent with my earlier benchmark.
TRIALS = 5

# I keep the condition number approximately the same at every matrix size.
TARGET_CONDITION_NUMBER = 1e4

# On Windows, very short CPU timings can be reported as zero.
# I therefore repeat fast solves inside each timing batch until the
# total CPU time is long enough to measure reliably.
MIN_BATCH_CPU_TIME = 0.20
MAX_BATCH_REPEATS = 2048


def build_problem(n):
    # I use a separate but reproducible random generator for each size.
    rng = np.random.default_rng(SEED + n)

    # I first create an orthogonal matrix using SciPy.
    random_matrix = rng.standard_normal((n, n))
    Q, _ = scipy.linalg.qr(
        random_matrix,
        mode="economic",
        check_finite=False
    )

    # I choose positive eigenvalues between 1 and 1e-4.
    # This gives me a symmetric positive definite matrix with
    # condition number approximately 1e4.
    eigenvalues = np.geomspace(
        1.0,
        1.0 / TARGET_CONDITION_NUMBER,
        n
    )

    A = Q @ np.diag(eigenvalues) @ Q.T

    # I create a known true solution so I can still check accuracy.
    x_true = rng.standard_normal(n)
    b = A @ x_true

    return A, b, x_true


def choose_batch_repeats(solve_function):
    # I increase the number of repeated solves until the measured
    # CPU time is long enough to avoid zero-valued timings.
    repeats = 1

    while True:
        cpu_start = time.process_time()

        for _ in range(repeats):
            solve_function()

        cpu_elapsed = time.process_time() - cpu_start

        if (
            cpu_elapsed >= MIN_BATCH_CPU_TIME
            or repeats >= MAX_BATCH_REPEATS
        ):
            return repeats

        repeats *= 2


def benchmark_method(method_name, solve_function, A, b, x_true):
    cpu_times = []
    wall_times = []
    final_solution = None

    # I run one warm-up solve before collecting timings.
    solve_function()

    # I calibrate the batch size separately for each method and matrix size.
    repeats = choose_batch_repeats(solve_function)

    print(
        f"{method_name:16s} | "
        f"Repeated solves per timing batch: {repeats}"
    )

    for trial in range(1, TRIALS + 1):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        for _ in range(repeats):
            final_solution = solve_function()

        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        # I divide by the number of repeated solves so the stored value
        # is still the estimated time for one factorisation and solve.
        cpu_elapsed = (cpu_end - cpu_start) / repeats
        wall_elapsed = (wall_end - wall_start) / repeats

        cpu_times.append(cpu_elapsed)
        wall_times.append(wall_elapsed)

        print(
            f"{method_name:16s} | "
            f"Trial {trial} | "
            f"CPU {cpu_elapsed:.6f} s | "
            f"Wall {wall_elapsed:.6f} s"
        )

    median_cpu_time = float(np.median(cpu_times))
    median_wall_time = float(np.median(wall_times))

    # I check that the calculated solution still satisfies Ax = b.
    relative_residual = (
        scipy.linalg.norm(A @ final_solution - b)
        / scipy.linalg.norm(b)
    )

    # I also compare it with the known true solution.
    relative_solution_error = (
        scipy.linalg.norm(final_solution - x_true)
        / scipy.linalg.norm(x_true)
    )

    return {
        "method": method_name,
        "batch_repeats": repeats,
        "median_cpu_time_s": median_cpu_time,
        "median_wall_time_s": median_wall_time,
        "relative_residual": relative_residual,
        "relative_solution_error": relative_solution_error
    }


print("SIT292 HD Factorisation Scaling Benchmark")
print()
print(f"Matrix sizes: {SIZES}")
print(f"Trials per method: {TRIALS}")
print(
    f"Target condition number: "
    f"{TARGET_CONDITION_NUMBER:.0e}"
)
print()


all_results = []


for n in SIZES:
    print("=" * 70)
    print(f"Matrix size: {n} x {n}")
    print("=" * 70)

    A, b, x_true = build_problem(n)

    # Since A is symmetric positive definite, I calculate its
    # 2-norm condition number from its largest and smallest eigenvalues.
    matrix_eigenvalues = scipy.linalg.eigvalsh(
        A,
        check_finite=False
    )

    actual_condition_number = float(
        matrix_eigenvalues[-1]
        / matrix_eigenvalues[0]
    )

    print(f"Symmetric: {np.allclose(A, A.T)}")
    print(
        f"Condition number: "
        f"{actual_condition_number:.3e}"
    )
    print()

    # LU factorisation
    def solve_lu():
        lu, piv = scipy.linalg.lu_factor(
            A,
            check_finite=False
        )

        return scipy.linalg.lu_solve(
            (lu, piv),
            b,
            check_finite=False
        )

    # QR factorisation
    def solve_qr():
        Q_qr, R = scipy.linalg.qr(
            A,
            mode="economic",
            check_finite=False
        )

        y = Q_qr.T @ b

        return scipy.linalg.solve_triangular(
            R,
            y,
            check_finite=False
        )

    # Cholesky factorisation
    def solve_cholesky():
        L = scipy.linalg.cholesky(
            A,
            lower=True,
            check_finite=False
        )

        y = scipy.linalg.solve_triangular(
            L,
            b,
            lower=True,
            check_finite=False
        )

        return scipy.linalg.solve_triangular(
            L.T,
            y,
            lower=False,
            check_finite=False
        )

    # Singular value decomposition
    def solve_svd():
        U, singular_values, Vh = scipy.linalg.svd(
            A,
            check_finite=False
        )

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

    for method_name, solve_function in methods.items():
        result = benchmark_method(
            method_name,
            solve_function,
            A,
            b,
            x_true
        )

        result["matrix_size"] = n
        result["condition_number"] = actual_condition_number

        all_results.append(result)

        print(
            f"Median CPU time: "
            f"{result['median_cpu_time_s']:.6f} s"
        )

        print(
            f"Median wall time: "
            f"{result['median_wall_time_s']:.6f} s"
        )

        print(
            f"Relative residual: "
            f"{result['relative_residual']:.6e}"
        )

        print(
            f"Relative solution error: "
            f"{result['relative_solution_error']:.6e}"
        )

        print()


results_df = pd.DataFrame(all_results)

results_df = results_df[
    [
        "matrix_size",
        "method",
        "condition_number",
        "batch_repeats",
        "median_cpu_time_s",
        "median_wall_time_s",
        "relative_residual",
        "relative_solution_error"
    ]
]


# I estimate the rate of runtime growth using
#
#     t(n) approximately C n^p
#
# Taking logarithms gives
#
#     log(t) approximately log(C) + p log(n)
#
# so the slope p gives me an empirical growth exponent.
growth_results = []

print()
print("=" * 70)
print("Estimated runtime growth")
print("=" * 70)
print()


def fit_growth_exponent(matrix_sizes, times):
    log_n = np.log(matrix_sizes)
    log_t = np.log(times)

    slope, intercept = np.polyfit(
        log_n,
        log_t,
        1
    )

    predicted = intercept + slope * log_n

    ss_res = np.sum(
        (log_t - predicted) ** 2
    )

    ss_tot = np.sum(
        (log_t - np.mean(log_t)) ** 2
    )

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)

    return float(slope), float(r_squared)


for method_name in results_df["method"].unique():
    method_results = (
        results_df[
            results_df["method"] == method_name
        ]
        .sort_values("matrix_size")
    )

    matrix_sizes = (
        method_results["matrix_size"]
        .to_numpy(dtype=float)
    )

    cpu_times = (
        method_results["median_cpu_time_s"]
        .to_numpy(dtype=float)
    )

    wall_times = (
        method_results["median_wall_time_s"]
        .to_numpy(dtype=float)
    )

    cpu_exponent, cpu_r_squared = fit_growth_exponent(
        matrix_sizes,
        cpu_times
    )

    wall_exponent, wall_r_squared = fit_growth_exponent(
        matrix_sizes,
        wall_times
    )

    growth_results.append(
        {
            "method": method_name,
            "estimated_cpu_exponent": cpu_exponent,
            "cpu_fit_r_squared": cpu_r_squared,
            "estimated_wall_exponent": wall_exponent,
            "wall_fit_r_squared": wall_r_squared
        }
    )

    print(
        f"{method_name:16s} | "
        f"CPU p = {cpu_exponent:.3f} "
        f"(R^2 = {cpu_r_squared:.3f}) | "
        f"Wall p = {wall_exponent:.3f} "
        f"(R^2 = {wall_r_squared:.3f})"
    )


growth_df = pd.DataFrame(growth_results)


# I save both the raw scaling data and the estimated growth rates.
project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"

results_folder.mkdir(exist_ok=True)

results_file = (
    results_folder
    / "factorisation_scaling_results.csv"
)

growth_file = (
    results_folder
    / "factorisation_scaling_growth.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

growth_df.to_csv(
    growth_file,
    index=False
)


print()
print("Factorisation scaling results")
print()
print(results_df.to_string(index=False))

print()
print("Growth estimates")
print()
print(growth_df.to_string(index=False))

print()
print(f"Scaling results saved to: {results_file}")
print(f"Growth estimates saved to: {growth_file}")

print()
print("Factorisation scaling benchmark completed successfully.")