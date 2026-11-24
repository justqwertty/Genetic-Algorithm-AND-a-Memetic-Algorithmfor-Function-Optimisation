import time
import numpy as np
from . import operators as op

def run(func, bounds, dim, params=None, callback=None):
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")

    # read the settings (with defaults)
    p = params or {}
    pop_size = p.get("pop_size", 100)
    max_evals = p.get("max_evals", 10000 * dim)
    k = p.get("tournament_k", 3)
    cx_method = p.get("crossover", "sbx")
    cx_rate = p.get("crossover_rate", 0.9)
    mut_method = p.get("mutation", "polynomial")
    mut_rate = p.get("mutation_rate", 1 / dim)
    n_elites = p.get("n_elites", 2)

    # check the inputs before doing any work
    low, high = bounds
    if low >= high:
        raise ValueError(f"bounds must have low < high, got {bounds}")
    if not (pop_size >= 2 and pop_size % 2 == 0):
        raise ValueError(f"pop_size must be an even number >= 2, got {pop_size}")
    if not (0 <= cx_rate <= 1):
        raise ValueError(f"crossover_rate must be between 0 and 1, got {cx_rate}")
    if not (0 <= mut_rate <= 1):
        raise ValueError(f"mutation_rate must be between 0 and 1, got {mut_rate}")

    # start
    rng = np.random.default_rng(p.get("seed"))
    start = time.perf_counter()
    pop = op.init_population(pop_size, dim, bounds, rng)
    fit = np.array([func(x) for x in pop])
    evals = pop_size
    history = [fit.min()]

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
    best = np.argmin(fit)            
    return {
        "best_x": pop[best],
        "best_f": fit[best],
        "history": history,
        "evals": evals,
        "time_s": time.perf_counter() - start,
        "generations": gen,
    }