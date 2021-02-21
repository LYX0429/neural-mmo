from pdb import set_trace as TT
import abc
import math
import time
from timeit import default_timer as timer
from inspect import signature
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, \
    MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
from typing_extensions import runtime, Protocol
import warnings
import numpy as np
import copy
import traceback
import os

from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.algorithms.base import _severalEvalsWrapper, _evalWrapper
from qdpy.containers import *
from qdpy import tools
from qdpy.algorithms.evolution import CMAES
from evolution.individuals import EvoIndividual as EvoIndividual

import functools

partial = functools.partial  # type: ignore

class EvoCMAES(CMAES):
    def __init__(self, evo, *args, **kwargs):
        self.evo = evo
        super().__init__(*args, **kwargs)

    def _eval(ind):
        # print("starting: ", threading.get_ident(), ind[0])
        # print(type(ind[0]))
        #    x = float(ind[0])
        #    for i in range(1000000):
        #        x*x
        # tmp = np.zeros(100000)
        # for i in range(len(tmp)):
        #    tmp[i] = i * 42.
        # return illumination_rastrigin_normalised(ind, nb_features=nb_features)
        # res = illumination_rastrigin_normalised(ind, nb_features=nb_features)
        # res = [[np.random.random()], list(np.random.random(2))]
        #       res = illumination_rastrigin_normalised(ind, nb_features=2)
        #       fitness, features = res
        #       fitness[0] = 0.0 if fitness[0] < 0.90 else fitness[0]
        # print("finishing: ", threading.get_ident(), ind[0])
        # return res
        #       return fitness, features
        return ind.fitness.values, ind.features

    def optimise(self, evaluate: Callable, budget: Optional[int] = None,
                 batch_mode: bool = True, executor: Optional[ExecutorLike] = None,
                 send_several_suggestions_to_fn: bool = False) -> IndividualLike:
        """TODO"""
        optimisation_start_time: float = timer()
        # Init budget
        if budget is not None:
            self.budget = budget
        if self.budget is None:
            raise ValueError("`budget` must be provided.")

        _executor: ExecutorLike = executor if executor is not None else SequentialExecutor()

        # Call callback functions
        for fn in self._callbacks.get("started_optimisation"):
            fn(self)

        def optimisation_loop(budget_fn: Callable):
            budget = budget_fn()
            remaining_evals = budget
            batch_start_time: float = timer()
            futures: MutableMapping[int, FutureLike] = {}
            individuals: MutableMapping[int, Union[IndividualLike, Sequence[IndividualLike]]] = {}
            while remaining_evals > 0 or len(futures) > 0:
                # Update budget
                new_budget = budget_fn()
                if new_budget > budget:
                    remaining_evals += new_budget - budget
                elif new_budget < budget:
                    remaining_evals = max(0, remaining_evals - (budget - new_budget))
                budget = new_budget

                nb_suggestions = min(remaining_evals, self.batch_size - len(futures))

                if send_several_suggestions_to_fn:
                    # Launch evals on suggestions
                    eval_id: int = remaining_evals
                    inds: Sequence[IndividualLike] = [self.ask() for _ in range(nb_suggestions)]
                    individuals[eval_id] = inds
                    futures[eval_id] = _executor.submit(_severalEvalsWrapper, eval_id, evaluate, inds)
                    remaining_evals -= len(inds)

                    ### custom evoNMMO bit
                    self.evo.train_individuals(individuals)
                    ###

                    # Wait for next completed future
                    f = generic_as_completed(list(futures.values()))
                    ind_id, ind_elapsed, ind_res, ind_exc = f.result()
                    if ind_exc is None:
                        inds = individuals[ind_id] # type: ignore
                        for i in range(len(inds)):
                            self.tell(inds[i], fitness=ind_res[i][0], features=ind_res[i][1], elapsed=ind_elapsed)
                    else:
                        warnings.warn(f"Individual evaluation raised the following exception: {ind_exc} !")
                        self._nb_evaluations += len(ind_res)

                else:
                    # Launch evals on suggestions
                    for _ in range(nb_suggestions):
                        eval_id = remaining_evals
                        ind: IndividualLike = self.ask()
                        individuals[eval_id] = ind
                        futures[eval_id] = _executor.submit(_evalWrapper, eval_id, evaluate, ind)
                        remaining_evals -= 1

                    TT()
                    ### custom evoNMMO bit
                    self.evo.train_individuals(individuals)
                    ###

                    # Wait for next completed future
                    f = generic_as_completed(list(futures.values()))
                    #ind_id, ind_elapsed, ind_fitness, ind_features, ind_exc = f.result()
                    ind_id, ind_elapsed, ind_res, ind_exc = f.result()
                    if ind_exc is None:
                        if isinstance(ind_res, IndividualLike):
                            self.tell(ind_res, elapsed=ind_elapsed)
                        else:
                            ind_fitness, ind_features = ind_res
                            ind = individuals[ind_id] # type: ignore
                            self.tell(ind, fitness=ind_fitness, features=ind_features, elapsed=ind_elapsed)
                    else:
                        warnings.warn(f"Individual evaluation raised the following exception: {ind_exc} !")
                        self._nb_evaluations += 1

                # Clean up
                del futures[ind_id]
                del individuals[ind_id]
                # Verify if we finished an iteration
                if self._verify_if_finished_iteration(batch_start_time):
                    batch_start_time = timer()


        if batch_mode:
            budget = self.budget
            remaining_evals: int = budget
            new_budget: int
            while remaining_evals > 0:
                budget_iteration: int = min(remaining_evals, self.batch_size)
                optimisation_loop(lambda: budget_iteration)
                remaining_evals -= budget_iteration

                # Update budget
                new_budget = self.budget
                if new_budget > budget:
                    remaining_evals += new_budget - budget
                elif budget > new_budget:
                    remaining_evals = max(0, remaining_evals - (budget - new_budget))
                budget = new_budget

        else:
            def ret_budget():
                return self.budget
            optimisation_loop(ret_budget)

        # Call callback functions
        optimisation_elapsed: float = timer() - optimisation_start_time
        for fn in self._callbacks.get("finished_optimisation"):
            fn(self, optimisation_elapsed)

        return self.best()

    def run(self):
#       self.optimise(EvoCMAES._eval)
        pop_size = self.evo.config.N_EVO_MAPS
        for i in range(self.evo.n_epochs):
            models = [np.array(self.ask()) for _ in range(pop_size)]
            individuals = [EvoIndividual(iterable=models[j], rank=j, evolver=self.evo) for j in range(pop_size)]
            for j, ind in enumerate(individuals):
                if self.evo.CA:
                    ind.chromosome.nn.set_weights(models[j])
                    ind.chromosome.n_passes = models[j][-1]
                    ind.chromosome.gen_map()
                if self.evo.TILE_FLIP:
                    ind.chromosome.multi_hot = models[j]
                    pass
            self.evo.train_individuals(individuals)
            for ind in individuals:
            #   self.tell(np.hstack((ind.chromosome.nn.weights, [ind.chromosome.nn.n_passes])).tolist(), fitness=ind.fitness.values, features=ind.features)
                self.tell(ind)
            print('CMAES epoch {}'.format(i))
            if i % self.evo.config.EVO_SAVE_INTERVAL == 0:
#               self.evo.log_me()
                self.evo.log()
                self.evo.saveMaps(individuals)
                container = self.evo.container
                self.evo.container = None
#               self.evo.save()
                self.evo.container = container
#       for _ in range
