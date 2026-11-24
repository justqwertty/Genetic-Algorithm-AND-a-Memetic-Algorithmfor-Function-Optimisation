import random

def ga(random_solution, fitness, crossover, mutate,
       pop_size=30, generations=100, n_elites=2, seed=0):
    random.seed(seed)                                   # same seed -> same run (point 5)
    pop = [random_solution() for _ in range(pop_size)]
    for gen in range(generations):
        scored = sorted(pop, key=fitness, reverse=True) # bigger = better (point 2)
        new_pop = scored[:n_elites]                     # elitism (point 4)
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fitness)               # tournament selection (point 4)
            p2 = tournament(pop, fitness)
            new_pop.append(mutate(crossover(p1, p2)))
        pop = new_pop
    best = max(pop, key=fitness)
    return best, fitness(best)

def tournament(pop, fitness, k=3):
    return max(random.sample(pop, k), key=fitness)