#!/usr/bin/env python
from argparse import ArgumentParser
from math import sqrt
from random import random, seed, shuffle
from typing import List, Optional

from evol import Evolution, Population
from evol.helpers.combiners.permutation import cycle_crossover
from evol.helpers.groups import group_stratified
from evol.helpers.mutators.permutation import swap_elements
from evol.helpers.pickers import pick_random

def run_travelling_salesman(population_size: int = 100,
                            n_iterations: int = 10,
                            random_seed: int = 42,
                            n_destinations: int = 50,
                            concurrent_workers: Optional[int] = None,
                            n_groups: int = 4,
                            silent: bool = False):
    seed(random_seed)
    destinations = [(random(), random()) for _ in range(n_destinations)]

    def evaluate(ordered_destinations: List[int]) -> float:
        return sum(sqrt((destinations[i][0] - destinations[j][0])**2 + (destinations[i][1] - destinations[j][1])**2)
                   for i, j in zip(ordered_destinations, ordered_destinations[1:] + [ordered_destinations[0]]))

    def generate_solution() -> List[int]:
        indexes = list(range(n_destinations))
        shuffle(indexes)
        return indexes

    def print_function(population: Population):
        if population.generation % 5000 == 0 and not silent:
            print(f'{population.generation}: {population.documented_best.fitness:.2f} / {population.current_best.fitness:.2f}')

    pop = Population(generate_solution, evaluate, maximize=False, size=population_size * n_groups, concurrent_workers=concurrent_workers)

    island_evo = Evolution().survive(fraction=0.5).breed(pick_random, cycle_crossover).mutate(swap_elements, elitist=True)

    evo = Evolution().evaluate(lazy=True).callback(print_function).repeat(island_evo, n=100, group_stratified, n_groups)

    result = pop.evolve(evo, n=n_iterations)

    if not silent:
        print(f'Shortest route: {result.documented_best.chromosome}')
        print(f'Route length: {result.documented_best.fitness}')

def parse_arguments():
    parser = ArgumentParser(description='Run the travelling salesman example.')
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--n-iterations', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--n-destinations', type=int, default=50)
    parser.add_argument('--n-groups', type=int, default=4)
    parser.add_argument('--concurrent-workers', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_travelling_salesman(**vars(args))
