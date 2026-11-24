import numpy as np

def init_population(pop_size, dim, bounds, rng):
    """Create pop_size random solutions, each with dim numbers inside bounds."""
    low, high = bounds
    # your code: one line using rng.uniform(low, high, size=...)
    pop= rng.uniform(low,high,size =(pop_size, dim))
    return pop
    
