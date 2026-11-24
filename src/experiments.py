"""goal: Run GA and MA on every benchmark and save the results to CSV."""
import csv
import os
import time
import tracemalloc

from . import ga, ma
from .benchmarks import FUNCTIONS

ALGORITHMS = {"GA": ga.run, "MA": ma.run}
COLUMNS = ["algorithm", "function", "dim", "run", "best_f", "time_s", "memory_mb", "evals", "success"]


def run_one(alg_name, func_name, dim, run_idx, tol=1e-4):
    """Run one algorithm once. Returns (csv_row, history)."""
    func, bounds = FUNCTIONS[func_name]
    seed = 1000 + run_idx            # GA and MA get the SAME seed so they have the same starting population
    params = {"seed": seed, "max_evals": 10000 * dim}
    res = ALGORITHMS[alg_name](func, bounds, dim, params)        # real run, no tracking

    tracemalloc.start()                                          # short run, only for memory
    ALGORITHMS[alg_name](func, bounds, dim, {**params, "max_evals": 20 * 100})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row = {
        "algorithm": alg_name,
        "function": func_name,
        "dim": dim,
        "run": run_idx,
        "best_f": res["best_f"],
        "time_s": res["time_s"],
        "memory_mb": peak / 1e6,
        "evals": res["evals"],
        "success": int(res["best_f"] < tol),
    }
    return row, res["history"]


def main(functions, dims, runs, out="results"):
    """Run every function x dim x run x algorithm and write results/results.csv."""
    conv_dir = os.path.join(out, "convergence")
    os.makedirs(conv_dir, exist_ok=True)               # create the folders if missing
    csv_path = os.path.join(out, "results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for func_name in functions:
            for dim in dims:
                for r in range(runs):
                    for alg in ALGORITHMS:
                        row, hist = run_one(alg, func_name, dim, r)
                        writer.writerow(row)
                        f.flush()                      # save right away, survives a crash

                        # convergence history -> its own small CSV
                        name = f"{alg}_{func_name}_d{dim}_r{r}.csv"
                        with open(os.path.join(conv_dir, name), "w") as h:
                            h.write("generation,best_f\n")
                            for g, v in enumerate(hist):
                                h.write(f"{g},{v}\n")

                        print(f"{alg} {func_name} d={dim} run={r}  best_f={row['best_f']:.2e}  "
                              f"{row['time_s']:.1f}s")
    print("Saved", csv_path)