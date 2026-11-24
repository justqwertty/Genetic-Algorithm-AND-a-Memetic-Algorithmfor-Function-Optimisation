import numpy as np

def init_population(pop_size, dim, bounds, rng):
    low, high = bounds
    pop= rng.uniform(low,high,size =(pop_size, dim))
    return pop
    
def tournament_selection(pop, fitness, n_select, k, rng):

    parents = []
    for _ in range(n_select):
        idx = rng.integers(0,len(pop),size =k)       
        best = idx[np.argmin(fitness[idx])]      
        parents.append(pop[best])   
    return np.array(parents)

def blx_alpha_crossover(p1, p2, bounds, rng, alpha=0.5):
    lo = np.minimum(p1,p2)        
    hi = np.maximum(p1,p2)       
    gap = hi - lo

    low_r,high_r = lo - alpha * gap, hi + alpha * gap

    c1 = rng.uniform(low_r,high_r)
    c2 = rng.uniform(low_r,high_r)       

    low, high = bounds
    c1 = np.clip(c1, low, high)    # keep children inside the bounds
    c2 = np.clip(c2, low,high)
    return c1, c2

def sbx_crossover(p1, p2, bounds, rng, eta=15.0):
    u = rng.random(p1.shape)          # one random number per gene

    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1))) 

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    low, high = bounds
    return np.clip(c1, low, high), np.clip(c2, low, high)