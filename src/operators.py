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