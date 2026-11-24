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
        # read the settings (with defaults)
    k = p.get("tournament_k", 3)
    cx_method = p.get("crossover", "sbx")
    cx_rate = p.get("crossover_rate", 0.9)
    mut_method = p.get("mutation", "polynomial")
    mut_rate = p.get("mutation_rate", 1 / dim)
    n_elites = p.get("n_elites", 2)

    gen = 0
    if callback:
        callback(gen, fit.min())

    while evals + pop_size <= max_evals:
        parents = op.tournament_selection(pop, fit, pop_size, k, rng)
        children = op.crossover(parents, cx_rate, cx_method, bounds, rng)
        children = op.mutate(children, mut_rate, mut_method, bounds, rng)
        child_fit = np.array([func(x) for x in children])
        evals += pop_size
        pop, fit = op.apply_elitism(pop, fit, children, child_fit, n_elites)
        gen += 1
        history.append(fit.min())
        if callback:
            callback(gen, fit.min())

    return {"best_f": fit.min(), "evals": evals, "history": history,
            "time_s": time.perf_counter() - start}