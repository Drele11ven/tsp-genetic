# ga.py
import numpy as np
import random
from copy import deepcopy

def init_population(n_cities, pop_size):
    pop = []
    base = list(range(n_cities))
    for _ in range(pop_size):
        indiv = base.copy()
        random.shuffle(indiv)
        pop.append(indiv)
    return pop

def fitness_population(pop, dist_matrix):
    # fitness = 1 / length (maximize fitness)
    lengths = np.array([route_length(ind, dist_matrix) for ind in pop])
    fitness = 1.0 / (lengths + 1e-9)
    return fitness, lengths

def tournament_selection(pop, fitness, k=3):
    # return selected individual (copy)
    idxs = np.random.choice(len(pop), k, replace=False)
    best = idxs[np.argmax(fitness[idxs])]
    return deepcopy(pop[best])

def ordered_crossover(parent1, parent2):
    # OX (ordered crossover)
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n
    # copy slice from parent1
    child[a:b+1] = parent1[a:b+1]
    # fill remaining from parent2 in order
    p2 = [c for c in parent2 if c not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[idx]
            idx += 1
    return child

def swap_mutation(indiv, mutation_rate):
    n = len(indiv)
    for i in range(n):
        if random.random() < mutation_rate:
            j = random.randrange(n)
            indiv[i], indiv[j] = indiv[j], indiv[i]
    return indiv

def evolve(pop, dist_matrix, elite_size=2, mutation_rate=0.01, tournament_k=3):
    fitness, lengths = fitness_population(pop, dist_matrix)
    sorted_idx = np.argsort(-fitness)  # descending by fitness
    new_pop = []
    # Elitism: keep top elites
    for i in range(elite_size):
        new_pop.append(deepcopy(pop[sorted_idx[i]]))
    # generate rest
    while len(new_pop) < len(pop):
        p1 = tournament_selection(pop, fitness, k=tournament_k)
        p2 = tournament_selection(pop, fitness, k=tournament_k)
        child = ordered_crossover(p1, p2)
        child = swap_mutation(child, mutation_rate)
        new_pop.append(child)
    return new_pop, lengths[sorted_idx[0]], pop[sorted_idx[0]]

# helper route_length imported from utils to avoid circular import
from utils import route_length
