import csv
import time
from pathlib import Path

import numpy as np

from scipy.sparse import diags, random as sparse_random, tril, triu
from sksparse.cholmod import cho_factor


# I use the same base seed as my other SIT292 benchmark scripts.
SEED = 292

# I vary both the matrix size and the sparsity of the input matrix.
# This lets me study the two effects separately instead of relying
# on one sparse matrix instance.
SIZES = [500, 1000, 1500, 2000]

# These are target off-diagonal densities used when generating the
# symmetric sparse matrix. I also record the actual final density.
TARGET_DENSITIES = [0.001, 0.002, 0.005]

# I keep five trials so the timing method is consistent with my
# earlier experiments.
TRIALS = 5

# I use repeated factorisations for very fast cases so Windows can
# measure CPU time reliably.
MIN_BATCH_CPU_TIME = 0.20
MAX_BATCH_REPEATS = 256


def build_sparse_spd_matrix(n, target_density, seed):
    # I generate a sparse random matrix first.
    rng = np.random.default_rng(seed)

    random_part = sparse_random(
        n,
        n,
        density=target_density,
        format="csc",
        random_state=seed,
        data_rvs=lambda k: rng.uniform(0.1, 1.0, size=k)
    )

    # I keep only the strict upper triangle and mirror it so the
    # off-diagonal part is symmetric.
    upper = triu(
        random_part,
        k=1,
        format="csc"
    )

    off_diagonal = (
        upper
        + upper.T
    ).tocsc()

    # I make the diagonal larger than the absolute row sum.
    # Because the matrix is symmetric with a positive strictly
    # dominant diagonal, the resulting matrix is positive definite.
    absolute_row_sum = np.asarray(
        abs(off_diagonal).sum(axis=1)
    ).ravel()

    diagonal_values = absolute_row_sum + 1.0

    A = (
        off_diagonal
        + diags(
            diagonal_values,
            offsets=0,
            format="csc"
        )
    ).tocsc()

    return A


def choose_batch_repeats(A):
    # I increase the number of repeated factorisations until the
    # total CPU time is long enough to measure reliably.
    repeats = 1

    while True:
        cpu_start = time.process_time()

        for _ in range(repeats):
            cho_factor(A)

        cpu_elapsed = time.process_time() - cpu_start

        if (
            cpu_elapsed >= MIN_BATCH_CPU_TIME
            or repeats >= MAX_BATCH_REPEATS
        ):
            return repeats

        repeats *= 2


def benchmark_factorisation(A):
    cpu_times = []
    wall_times = []
    final_factor = None

    # I run one warm-up factorisation before recording timings.
    cho_factor(A)

    repeats = choose_batch_repeats(A)

    for trial in range(1, TRIALS + 1):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        for _ in range(repeats):
            final_factor = cho_factor(A)

        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        cpu_per_factorisation = (
            cpu_end - cpu_start
        ) / repeats

        wall_per_factorisation = (
            wall_end - wall_start
        ) / repeats

        cpu_times.append(cpu_per_factorisation)
        wall_times.append(wall_per_factorisation)

        print(
            f"Trial {trial} | "
            f"CPU {cpu_per_factorisation:.6f} s | "
            f"Wall {wall_per_factorisation:.6f} s"
        )

    return (
        float(np.median(cpu_times)),
        float(np.median(wall_times)),
        repeats,
        final_factor
    )


def fit_growth_exponent(x_values, y_values):
    log_x = np.log(
        np.asarray(x_values, dtype=float)
    )

    log_y = np.log(
        np.asarray(y_values, dtype=float)
    )

    slope, intercept = np.polyfit(
        log_x,
        log_y,
        1
    )

    predicted = (
        intercept
        + slope * log_x
    )

    ss_res = np.sum(
        (log_y - predicted) ** 2
    )

    ss_tot = np.sum(
        (log_y - np.mean(log_y)) ** 2
    )

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    return float(slope), float(r_squared)


print("SIT292 HD Sparse Scaling Benchmark")
print()
print(f"Matrix sizes: {SIZES}")
print(f"Target densities: {TARGET_DENSITIES}")
print(f"Trials per case: {TRIALS}")
print()


all_results = []


for density_index, target_density in enumerate(TARGET_DENSITIES):

    for n in SIZES:
        print("=" * 76)
        print(
            f"Matrix size: {n} x {n} | "
            f"Target density: {target_density:.4f}"
        )
        print("=" * 76)

        case_seed = (
            SEED
            + density_index * 10000
            + n
        )

        A = build_sparse_spd_matrix(
            n,
            target_density,
            case_seed
        )

        actual_density = (
            A.nnz
            / (n * n)
        )

        # I compare L with the lower triangular part of A because
        # L itself stores only one triangular half of the factorisation.
        lower_A = tril(
            A,
            format="csc"
        )

        nnz_lower_A = lower_A.nnz

        # I verify the strict diagonal dominance used to guarantee
        # positive definiteness.
        diagonal = A.diagonal()

        off_diagonal_absolute_sum = np.asarray(
            abs(
                A
                - diags(
                    diagonal,
                    offsets=0,
                    format="csc"
                )
            ).sum(axis=1)
        ).ravel()

        minimum_diagonal_margin = float(
            np.min(
                diagonal
                - off_diagonal_absolute_sum
            )
        )

        print(f"Non-zero entries in A: {A.nnz}")
        print(f"Actual density of A: {actual_density:.6f}")
        print(
            f"Minimum diagonal dominance margin: "
            f"{minimum_diagonal_margin:.6f}"
        )

        median_cpu_time, median_wall_time, repeats, factor = (
            benchmark_factorisation(A)
        )

        # Different scikit-sparse versions expose L either as a method
        # or directly as a sparse matrix. I handle both forms here.
        L_object = factor.L

        if callable(L_object):
            L = L_object().tocsc()
        else:
            L = L_object.tocsc()

        density_L = (
            L.nnz
            / (n * n)
        )

        fill_ratio = (
            L.nnz
            / nnz_lower_A
        )

        additional_fill = (
            L.nnz
            - nnz_lower_A
        )

        # I also solve one known system so I can confirm that the
        # factorisation still produces an accurate numerical solution.
        rng = np.random.default_rng(
            case_seed + 1
        )

        x_true = rng.standard_normal(n)
        b = A @ x_true

        x_calculated = factor.solve(b)

        relative_residual = (
            np.linalg.norm(
                A @ x_calculated - b
            )
            / np.linalg.norm(b)
        )

        relative_solution_error = (
            np.linalg.norm(
                x_calculated - x_true
            )
            / np.linalg.norm(x_true)
        )

        print(
            f"Repeated factorisations per timing batch: "
            f"{repeats}"
        )
        print(
            f"Median CPU factorisation time: "
            f"{median_cpu_time:.6f} s"
        )
        print(
            f"Median wall factorisation time: "
            f"{median_wall_time:.6f} s"
        )
        print(f"Non-zero entries in L: {L.nnz}")
        print(f"Density of L: {density_L:.6f}")
        print(f"Fill ratio: {fill_ratio:.3f}")
        print(f"Additional fill entries: {additional_fill}")
        print(
            f"Relative residual: "
            f"{relative_residual:.6e}"
        )
        print(
            f"Relative solution error: "
            f"{relative_solution_error:.6e}"
        )
        print()

        all_results.append(
            {
                "matrix_size": n,
                "target_density": target_density,
                "actual_density_A": actual_density,
                "nnz_A": A.nnz,
                "nnz_lower_A": nnz_lower_A,
                "minimum_diagonal_margin": minimum_diagonal_margin,
                "batch_repeats": repeats,
                "median_cpu_factorisation_time_s": median_cpu_time,
                "median_wall_factorisation_time_s": median_wall_time,
                "nnz_L": L.nnz,
                "density_L": density_L,
                "fill_ratio": fill_ratio,
                "additional_fill_entries": additional_fill,
                "relative_residual": relative_residual,
                "relative_solution_error": relative_solution_error
            }
        )


# I estimate how CPU time, the number of non-zero entries in L,
# and the fill ratio change with matrix size at each input density.
growth_results = []

print()
print("=" * 76)
print("Estimated sparse scaling rates")
print("=" * 76)
print()


for target_density in TARGET_DENSITIES:
    density_results = [
        row
        for row in all_results
        if row["target_density"] == target_density
    ]

    density_results.sort(
        key=lambda row: row["matrix_size"]
    )

    sizes = [
        row["matrix_size"]
        for row in density_results
    ]

    cpu_times = [
        row["median_cpu_factorisation_time_s"]
        for row in density_results
    ]

    nnz_L_values = [
        row["nnz_L"]
        for row in density_results
    ]

    fill_ratios = [
        row["fill_ratio"]
        for row in density_results
    ]

    cpu_exponent, cpu_r_squared = fit_growth_exponent(
        sizes,
        cpu_times
    )

    nnz_L_exponent, nnz_L_r_squared = fit_growth_exponent(
        sizes,
        nnz_L_values
    )

    fill_exponent, fill_r_squared = fit_growth_exponent(
        sizes,
        fill_ratios
    )

    growth_results.append(
        {
            "target_density": target_density,
            "cpu_time_exponent": cpu_exponent,
            "cpu_time_r_squared": cpu_r_squared,
            "nnz_L_exponent": nnz_L_exponent,
            "nnz_L_r_squared": nnz_L_r_squared,
            "fill_ratio_exponent": fill_exponent,
            "fill_ratio_r_squared": fill_r_squared
        }
    )

    print(
        f"Target density {target_density:.4f} | "
        f"CPU p = {cpu_exponent:.3f} "
        f"(R^2 = {cpu_r_squared:.3f}) | "
        f"nnz(L) p = {nnz_L_exponent:.3f} "
        f"(R^2 = {nnz_L_r_squared:.3f}) | "
        f"fill p = {fill_exponent:.3f} "
        f"(R^2 = {fill_r_squared:.3f})"
    )


project_folder = Path(__file__).resolve().parents[1]
results_folder = project_folder / "results"

results_folder.mkdir(
    exist_ok=True
)

results_file = (
    results_folder
    / "sparse_scaling_results.csv"
)

growth_file = (
    results_folder
    / "sparse_scaling_growth.csv"
)


result_fields = [
    "matrix_size",
    "target_density",
    "actual_density_A",
    "nnz_A",
    "nnz_lower_A",
    "minimum_diagonal_margin",
    "batch_repeats",
    "median_cpu_factorisation_time_s",
    "median_wall_factorisation_time_s",
    "nnz_L",
    "density_L",
    "fill_ratio",
    "additional_fill_entries",
    "relative_residual",
    "relative_solution_error"
]

with results_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=result_fields
    )

    writer.writeheader()
    writer.writerows(all_results)


growth_fields = [
    "target_density",
    "cpu_time_exponent",
    "cpu_time_r_squared",
    "nnz_L_exponent",
    "nnz_L_r_squared",
    "fill_ratio_exponent",
    "fill_ratio_r_squared"
]

with growth_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=growth_fields
    )

    writer.writeheader()
    writer.writerows(growth_results)


print()
print("Sparse scaling results")
print()

for row in all_results:
    print(
        f"n={row['matrix_size']:4d} | "
        f"target density={row['target_density']:.4f} | "
        f"actual density={row['actual_density_A']:.6f} | "
        f"CPU={row['median_cpu_factorisation_time_s']:.6f} s | "
        f"nnz(A)={row['nnz_A']:8d} | "
        f"nnz(L)={row['nnz_L']:8d} | "
        f"fill={row['fill_ratio']:.3f}"
    )


print()
print("Sparse growth estimates")
print()

for row in growth_results:
    print(
        f"density={row['target_density']:.4f} | "
        f"CPU p={row['cpu_time_exponent']:.3f} | "
        f"nnz(L) p={row['nnz_L_exponent']:.3f} | "
        f"fill p={row['fill_ratio_exponent']:.3f}"
    )


print()
print(
    f"Scaling results saved to: "
    f"{results_file}"
)
print(
    f"Growth estimates saved to: "
    f"{growth_file}"
)
print()
print(
    "Sparse scaling benchmark completed successfully."
)