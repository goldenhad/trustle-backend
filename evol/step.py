from .population import Population
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evol.evolution import Evolution


class EvolutionStep:

    def __init__(self, name: Optional[str], **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.__class__.__name__} with name {self.name}"


class EvaluationStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.evaluate(**self.kwargs)


class ApplyStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.apply(**self.kwargs)


class CheckpointStep(EvolutionStep):

    def __init__(self, name, every=1, **kwargs):
        EvolutionStep.__init__(self, name, **kwargs)
        self.count = 0
        self.every = every

    def apply(self, population) -> Population:
        self.count += 1
        if self.count >= self.every:
            self.count = 0
            return population.checkpoint(**self.kwargs)
        return population


class MapStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.map(**self.kwargs)


class FilterStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.filter(**self.kwargs)


class UpdateStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.update(**self.kwargs)


class SurviveStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.survive(**self.kwargs)


class BreedStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.breed(**self.kwargs)


class MutateStep(EvolutionStep):

    def apply(self, population) -> Population:
        return population.mutate(**self.kwargs)


class LogStep(EvolutionStep):
    def __init__(self, name, every=1, **kwargs):
        EvolutionStep.__init__(self, name, **kwargs)
        self.count = 0
        self.every = every

    def apply(self, population) -> Population:
        self.count += 1
        if self.count >= self.every:
            self.count = 0
            return population.log(**self.kwargs)
        return population


class RepeatStep(EvolutionStep):
    def apply(self, population) -> Population:
        return population.evolve(**self.kwargs)


class GroupedStep(EvolutionStep):
    def __init__(self, name: str, evolution: 'Evolution', n: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self.evolution = evolution
        self.n = n

    def apply(self, population) -> Population:
        intended_size = population.intended_size
        groups = population.group(**self.kwargs)
        # TODO: parallellize
        return Population.combine(*[group.evolve(evolution=self.evolution, n=self.n) for group in groups],
                                  intended_size=intended_size)


class CallbackStep(EvolutionStep):
    def __init__(self, name, every=1, **kwargs):
        EvolutionStep.__init__(self, name, **kwargs)
        self.count = 0
        self.every = every

    def apply(self, population) -> Population:
        self.count += 1
        if self.count >= self.every:
            self.count = 0
            return population.callback(**self.kwargs)
        return population
