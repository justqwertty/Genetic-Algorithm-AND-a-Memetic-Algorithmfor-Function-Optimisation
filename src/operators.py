import numpy as np

def init_population(pop_size, dim, bounds, rng):
    low, high = bounds
    pop= rng.uniform(low,high,size =(pop_size, dim))
    return pop
    
def tournament_selection(pop, fitness, n_select, k, rng):
    """Pick n_select parents. Each one is the best of k random individuals."""
    parents = []
    for _ in range(n_select):
        idx = rng.integers(0,len(pop),size =k)        # k random row numbers
        best = idx[np.argmin(fitness[idx])]      # which of those has the lowest fitness? 
        parents.append(pop[best])   # add that solution from pop
    return np.array(parents)

def blx_alpha_crossover(p1, p2, bounds, rng, alpha=0.5):
    """BLX-alpha: each child gene is random in the parents' range, stretched by alpha."""
    lo = np.minimum(p1,p2)        # smaller of p1 and p2, number by number (hint: np.minimum)
    hi = np.maximum(p1,p2)        # larger of p1 and p2 (hint: np.maximum)
    gap = hi - lo

    low_r,high_r = lo - alpha * gap, hi + alpha * gap

    c1 = rng.uniform(low_r,high_r)
    c2 = rng.uniform(low_r,high_r)        # same as c1 (a second random child)
    
    low, high = bounds
    c1 = np.clip(c1, low, high)    # keep children inside the bounds
    c2 = np.clip(c2, low,high)
    return c1, c2