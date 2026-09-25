# SIT292 HD Linear Systems

This repository contains the Python code, benchmark results and plots used for my SIT292 Mathematics for Computing HD report on solving linear systems.

The project extends my earlier investigation of numerical linear algebra libraries by looking more closely at factorisation methods, sparse linear systems, runtime scaling and fill-in.

## What I investigated

I compared four dense factorisation methods:

- LU factorisation
- QR factorisation
- Cholesky factorisation (LLT)
- Singular Value Decomposition (SVD)

I also compared sparse linear system solvers using:

- SciPy
- CHOLMOD through scikit-sparse
- PyPardiso

The scaling experiments investigate how CPU time changes as matrix size increases. For sparse Cholesky factorisation, I also investigate how the sparsity of the factorised matrix \(L\) changes with matrix size and the sparsity of the original matrix \(A\).

## Project structure

```text
src/
    factorisation_benchmark.py
    factorisation_scaling.py
    sparse_benchmark.py
    sparse_scaling.py
    plot_hd_results.py
    plot_scaling_results.py
    plot_factor_density.py

results/
    Benchmark CSV files

plots/
    Figures generated from the benchmark results