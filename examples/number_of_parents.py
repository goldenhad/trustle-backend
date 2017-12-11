"""
There are a few worthwhile things to notice in this example: 

1. you can pass hyperparams into functions from the `.breed` and `.mutate` step 
2. the algorithm does not care how many parents it will use in the `breed` step
"""

import random
import math
import argparse

from evol import Population, Evolution

def run_evolutionary(opt_value=1, population_size=100, n_parents=2, num_iter=200, survival=0.5, noise=0.1, seed=42):

    random.seed(seed)

    def init_func():
        return (random.random() - 0.5) * 20 + 10

    def eval_func(x, opt_value=opt_value):
        return -((x - opt_value) ** 2) + math.cos(x - opt_value)

    def random_parent_picker(pop, n_parents):
        return [random.choice(pop) for i in range(n_parents)]

    def mean_parents(*parents):
        return sum(parents) / len(parents)

    def add_noise(chromosome, sigma):
        return chromosome + (random.random()-0.5) * sigma

    pop = Population(chromosomes=[init_func() for _ in range(population_size)],
                     eval_function=eval_func, maximize=True).evaluate()

    evo = (Evolution()
           .survive(fraction=survival)
           .breed(parent_picker=random_parent_picker, combiner=mean_parents, n_parents=n_parents)
           .mutate(func=add_noise, sigma=noise)
           .evaluate())

    print("will start the evolutionary program, will log progress every 10 steps")
    print(pop.maximize)
    for i in range(num_iter):
        pop = pop.evolve(evo)
        print(f"iteration:{i} best: {pop.max_individual.fitness} worst: {pop.min_individual.fitness}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an example evol algorithm against a simple continuous function.')
    parser.add_argument('--opt_value', type=int, default=0,
                        help='the true optimal value of the problem')
    parser.add_argument('--population_size', type=int, default=20,
                        help='the number of candidates to start the algorithm with')
    parser.add_argument('--n_parents', type=int, default=2,
                        help='the number of parents the algorithm with use to generate new indivuals')
    parser.add_argument('--num_iter', type=int, default=20,
                        help='the number of evolutionary cycles to run')
    parser.add_argument('--survival', type=float, default=0.7,
                        help='the fraction of individuals who will survive a generation')
    parser.add_argument('--noise', type=float, default=0.5,
                        help='the amount of noise the mutate step will add to each individual')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed for all this')

    args = parser.parse_args()
    print(f"i am aware of these arguments: {args}")
    run_evolutionary(opt_value=args.opt_value, population_size=args.population_size,
                     n_parents=args.n_parents, num_iter=args.num_iter,
                     noise=args.noise, seed=args.seed)