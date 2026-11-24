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



def crossover(parents, rate, method, bounds, rng, eta=15.0, alpha=0.5):
    if method not in ("sbx", "blx"):
        raise ValueError(f"Unknown crossover '{method}', use 'sbx' or 'blx'")
    children = parents.copy()
    for i in range(0, len(parents) - 1, 2):
        if rng.random() < rate:
            p1, p2 = parents[i], parents[i + 1]
            if method == "sbx":
                children[i], children[i + 1] = sbx_crossover(p1, p2, bounds, rng, eta)
            else:
                children[i], children[i + 1] = blx_alpha_crossover(p1, p2, bounds, rng, alpha)
    return children

def gaussian_mutation(pop, rate, bounds, rng, sigma=0.1):
    low, high = bounds
    mask = rng.random(pop.shape) < rate      
    noise =  rng.normal(0, sigma * (high - low), size=pop.shape)     
    new_pop = pop + mask * noise   
    return np.clip(new_pop, low , high)      