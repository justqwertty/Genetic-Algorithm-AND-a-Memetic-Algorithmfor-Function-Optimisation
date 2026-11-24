import time
import numpy as np
from . import operators as op


def run(func, bounds, dim, params=None, callback=None):
    """Run the GA until max_evals fitness evaluations are used."""
    p = params or {}
    pop_size = p.get("pop_size", 100)
    max_evals = p.get("max_evals", 10000 * dim)
    rng = np.random.default_rng(p.get("seed"))
    start = time.perf_counter()

    pop = op.init_population(pop_size, dim, bounds, rng)                                  # step 2: random population
    fit = np.array([func(x) for x in pop])     # score everyone
    evals = pop_size                                # how many evaluations so far?
    history = [fit.min()]

    # step 3 goes here next

    return {"best_f": fit.min(), "evals": evals, "history": history,
            "time_s": time.perf_counter() - start}