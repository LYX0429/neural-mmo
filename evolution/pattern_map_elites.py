import copy
import inspect
import operator
import os
import random
import warnings
from pdb import set_trace as TT
from timeit import default_timer as timer

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy

import deap
from deap import algorithms, base, creator, gp, tools
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums
from pcg import TILE_PROBS
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
# from qdpy.benchmarks import *
from qdpy.containers import Grid
from qdpy.plots import *

# from qdpy.plots import *

#!/usr/bin/env python3
#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.
"""A simple example of MAP-elites to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3. This code uses the library DEAP to implement the evolutionary part."""

mpl.use('Agg')

class Fitness():
   def __init__(self):
      self.values = None
      self.valid = False

   def dominates(self, fitness):
      assert len(self.values) == 1 and len(fitness.values) == 1
      my_fit = self.values[0]
      their_fit = fitness.values[0]

      return my_fit > their_fit

class Individual():
   def __init__(self, rank=None, evolver=None, mysterious_class=None):
      ind_idx, map_arr, atk_mults = evolver.genRandMap()
      self.fitness = Fitness()
      assert rank is not None
      self.data = {
              'ind_idx':   rank,
              'chromosome': evolver.chromosomes[ind_idx],
#             'map_arr':   map_arr,
#             'atk_muts':  atk_mults
              }




class NMMOGrid(Grid):
    def __init__(self, save_path, config, map_generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        render = config.TERRAIN_RENDER
        self.border = config.TERRAIN_BORDER
        self.save_path = save_path
        self.render = render
        self.map_generator = map_generator

    def add(self, individual):
        border = self.border
        index = super(NMMOGrid, self).add(individual)

        if index is not None:

            index_str = '('+ ', '.join([str(f) for f in self.index_grid(individual.features)]) + ')'
            map_path = os.path.join(self.save_path, 'maps', 'map' + index_str)
            try:
                os.makedirs(map_path)
            except FileExistsError:
                pass
            Save.np(individual.data['chromosome'][0].flat_map, map_path)
           #if render == True:
            Save.render(individual.data['chromosome'][0].flat_map[border:-border, border:-border], self.map_generator.textures, map_path + '.png')
            individual.data['ind_idx'] = self.index_grid((individual.features))
#           print('add ind with idx {}'.format(tuple(individual.features)))
            json_path = os.path.join(self.save_path, 'maps', 'atk_mults' + index_str + 'json')
            with open(json_path, 'w') as json_file:
               json.dump(individual.data['chromosome'][1], json_file)

        return index

class EvoDEAPQD(DEAPQDAlgorithm):
   def __init__(self, qd_fun, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ea_fn = qd_fun

class MapElites():
   def qdSimple(self, init_batch, toolbox, container, batch_size, niter, cxpb = 0.0, mutpb = 1.0, stats = None, halloffame = None, verbose = False, show_warnings = False, start_time = None, iteration_callback = None):
       """The simplest QD algorithm using DEAP.
       :param init_batch: Sequence of individuals used as initial batch.
       :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
       :param batch_size: The number of individuals in a batch.
       :param niter: The number of iterations.
       :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
       :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                          contain the best individuals, optional.
       :param verbose: Whether or not to log the statistics.
       :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
       :param start_time: Starting time of the illumination process, or None to take the current time.
       :param iteration_callback: Optional callback funtion called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
       :returns: The final batch
       :returns: A class:`~deap.tools.Logbook` with the statistics of the
                 evolution

       TODO
       """

       if start_time == None:
           start_time = timer()
       logbook = deap.tools.Logbook()
       logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + ["elapsed"]

       if len(init_batch) == 0:
           raise ValueError("``init_batch`` must not be empty.")

       # Evaluate the individuals with an invalid fitness
       invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
       fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

       for ind, fit in zip(invalid_ind, fitnesses):
           ind.fitness.values = fit[0]
           ind.features = fit[1]

       if len(invalid_ind) == 0:
           raise ValueError("No valid individual found !")

       # Update halloffame

       if halloffame is not None:
           halloffame.update(init_batch)

       # Store batch in container
       nb_updated = container.update(init_batch, issue_warning=show_warnings)

       # FIXME: we should warn about this when not reloading!
       if nb_updated == 0:
          print('Warning: empty container/grid')
          pass
   #      raise ValueError("No individual could be added to the container !")

       else:
          # Compile stats and update logs
          record = stats.compile(container) if stats else {}
          logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record)

          if verbose:
              print(logbook.stream)
       # Call callback function

       if iteration_callback != None:
           iteration_callback(0, init_batch, container, logbook)

       # Begin the generational process

       for i in range(1, niter + 1):
           start_time = timer()
           # Select the next batch individuals
           batch = toolbox.select(container, batch_size)

           ## Vary the pool of individuals
           offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)
           #for o in batch:
           #    newO = toolbox.clone(o)
           #    ind, = toolbox.mutate(newO)
           #    del ind.fitness.values
           #    offspring.append(ind)

           # Evaluate the individuals with an invalid fitness
           invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#          print('{} invalid individuals'.format(len(invalid_ind)))
           fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

           for ind, fit in zip(invalid_ind, fitnesses):
               ind.fitness.values = fit[0]
               ind.features = fit[1]

           # Replace the current population by the offspring
           nb_updated = container.update(offspring, issue_warning=show_warnings)

           # Update the hall of fame with the generated individuals

           if halloffame is not None:
               halloffame.update(container)

           # Append the current generation statistics to the logbook
           record = stats.compile(container) if stats else {}
           logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record)

           if verbose:
               print(logbook.stream)
           # Call callback function

           if iteration_callback != None:
               iteration_callback(i, batch, container, logbook)

       return batch, logbook


   def compile(self):

       pass

   def gen_individual(self):
       pass

   def mutate(self, individual):
      evo = self.evolver
      idx = individual.data['ind_idx']

#     print('mutate {}'.format(idx))
      idx = self.g_idxs.pop()
      self.mutated_idxs.add(idx)
      individual.data['ind_idx'] = idx
      # = self.container.index_grid(np.clip(inddividual.features, 0, 2000))
      #FIXME: big hack
      chrom, atk_mults = individual.data['chromosome']
      atk_mults = self.evolver.mutate_mults(atk_mults)
      chrom.mutate()
      self.evolver.validate_spawns(chrom.flat_map, chrom.multi_hot)

      if not hasattr(individual.fitness, 'values'):
         individual.fitness.values = None
      individual.fitness.valid = False
#     evo.genes[idxs] = map_arr, atk_mults

      return (individual, )


   def mate(self, parent_0, parent_1):
      evo = self.evolver
 #    idx_0 = parent_0.data['ind_idx']
 #    idx_1 = parent_1.data['ind_idx']
#     idx_0 = self.g_idxs.pop()
#     idx_1 = self.g_idxs.pop()
      chrom_0, atk_mults_0 = parent_0.data['chromosome']
      chrom_1, atk_mults_1 = parent_1.data['chromosome']
      prims_0 = chrom_0.patterns
      prims_1 = chrom_1.patterns
      new_atk_mults_0, new_atk_mults_1 = evo.mate_mults(atk_mults_0, atk_mults_1)
      len_0, len_1 = len(prims_0), len(prims_1)

      if len_0 < len_1:
         prims_0 = prims_0 + prims_1[-len_1 + len_0 - 1:]
      elif len_1 < len_0:
         prims_1 = prims_1 + prims_0[-len_0 + len_1 - 1:]
      new_prims_0 = [prims_0[i] if random.random() < 0.5 else prims_1[i] for i in range(len_0)]
      new_prims_1 = [prims_0[i] if random.random() < 0.5 else prims_1[i] for i in range(len_1)]
      chrom_0.patterns = new_prims_0
      chrom_1.patterns = new_prims_1
#     evo.chromosomes[idx_0] = chrom_0, new_atk_mults_0
#     evo.chromosomes[idx_1] = chrom_1, new_atk_mults_1
      chrom_0.update_features()
      chrom_1.update_features()

      if not hasattr(parent_0.fitness, 'values'):
         parent_0.fitness.values = None
      parent_0.fitness.valid = False

      if not hasattr(parent_1.fitness, 'values'):
         parent_1.fitness.values = None
      parent_1.fitness.valid = False
#     ind_0.data = {'ind_idx': idx_0}
#     ind_1.data = {'ind_idx': idx_1}

      return parent_0, parent_1

   def __init__(self, evolver, save_path='./'):
      self.n_gen = 0
      self.save_path = save_path
      self.evolver = evolver
      # Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
      self.init_toolbox()
      self.idxs = set()
      self.mutated_idxs = set()
      self.stats = None
      self.g_idxs = list(range(self.evolver.config.N_EVO_MAPS))
      feature_names = self.evolver.config.ME_DIMS
      self.feature_idxs = [self.evolver.config.SKILLS.index(n) for n in feature_names]

   def init_toolbox(self):
       fitness_weight = -1.0
       creator.create("FitnessMin", base.Fitness, weights=(fitness_weight, ))
       creator.create("Individual",
                      Individual,
                      evolver=self.evolver)
#                     fitness=creator.FitnessMin,
#                     features=list)

       # Create Toolbox
       self.max_size = max_size = self.evolver.config.ME_BIN_SIZE
       toolbox = base.Toolbox()
       #     toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
       toolbox.register("expr", self.expr)
       toolbox.register("individual", tools.initIterate, creator.Individual,
                        toolbox.expr)
       toolbox.register("population", self.init_pop)
       toolbox.register("compile", self.compile)  # gp.compile, pset=pset)
       # toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
       toolbox.register("evaluate", self.evaluate)  # , points=points)
       # toolbox.register("select", tools.selTournament, tournsize=3)
       toolbox.register("select", tools.selRandom)
       toolbox.register("mate", self.mate)
       # gp.genFull, min_=0, max_=2)
       toolbox.register("expr_mut", self.expr_mutate)
       # , expr=toolbox.expr_mut, pset=pset)
       toolbox.register("mutate", self.mutate)
#      toolbox.decorate(
#          "mate",
#          gp.staticLimit(key=operator.attrgetter("height"),
#                         max_value=max_size))
#      toolbox.decorate(
#          "mutate",
#          gp.staticLimit(key=operator.attrgetter("height"),
#                         max_value=max_size))
       # toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta, indpb=mutation_pb)
       # toolbox.register("select", tools.selRandom) # MAP-Elites = random selection on a grid container
       self.toolbox = toolbox
       self.max_skill = 2000

   def init_pop(self, n):
      return [Individual(rank=i, evolver=self.evolver) for i in range(n)]

   def expr_mutate(self):
       pass

   def protectedDiv(left, right):
       try:
           return left / right
       except ZeroDivisionError:
           return 1

   def iteration_callback(self, i, batch, container, logbook):
      print('pattern_map_elites iteration {}'.format(self.n_gen))
      # FIXME: doesn't work -- sync up these iteration/generation counts
      self.algo.current_iteration = self.n_gen
      evo = self.evolver
      evo.n_epoch = self.n_gen
      self.n_gen += 1
      self.idxs = set()
      evo.global_stats.reset.remote()
      self.g_idxs = set(range(self.evolver.config.N_EVO_MAPS))
      self.stats = None
      # update the elites to avoid stagnation (since simulation is stochastic)
      if len(container) > self.evolver.config.N_EVO_MAPS and np.random.random() < 0.1:
          invalid_elites = np.random.choice(container, min(max(1, len(container) - 6), self.evolver.config.N_EVO_MAPS), replace=False)
          elite_idxs = [container.index_grid(np.clip(ind.features, 0, self.max_skill)) for ind in invalid_elites]
          for el in invalid_elites:
             if el in container:
                 try:
                     container.discard(el, also_from_depot=True)
                 except ValueError as v:
                     # FIXME: why?
                     print(v)
             #FIXME: iterate through diff. features
          [ind.data.update({'ind_idx':idx}) for ind, idx in zip(invalid_elites, elite_idxs)]
          self.evolver.global_counter.set_idxs.remote(elite_idxs)
          self.evolver.trainer.train()
          self.stats = ray.get(evo.global_stats.get.remote())
          elite_fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_elites)
          for el, el_fit in zip(invalid_elites, elite_fitnesses):
             evo.genes.pop(el.data['ind_idx'])
             el.fitness.values = el_fit[0]
#            el.features = np.clip(el_fit[1], self.features_domain[0][0], self.features_domain[0][1])
             el.features = np.clip(el_fit[1], 0, self.max_skill)
          nb_updated = container.update(invalid_elites, issue_warning=True)
          print('Reinstated {} of {} disturbed elites.'.format(nb_updated, len(elite_idxs)))
      #nb_el_updated = container.update()


      self.idxs = set()
      self.stats = None
      evo.global_stats.reset.remote()
      self.evolver.global_counter.set_idxs.remote(list(range(self.evolver.config.N_EVO_MAPS)))
      self.g_idxs = set(range(self.evolver.config.N_EVO_MAPS))
      if not hasattr(self, 'mutated_idxs'):
          evo.saveMaps(evo.genes)
      else:
          evo.saveMaps(evo.genes, self.mutated_idxs)
      evo.log()
      self.mutated_idxs = set()
      if self.n_gen % 10 == 0:
         self.log_me(container)


   def evaluate(self, individual, elite=False):
      ind_idx = individual.data['ind_idx']
#     print('evaluating {}'.format(ind_idx))
      def calc_mean_agent(ind_stats):
         skill_mat = np.vstack(ind_stats['skills'])
         mean_agent = skill_mat.mean(axis=0)

         return mean_agent
      evo = self.evolver

      if ind_idx not in evo.genes:
          evo.genes[ind_idx] = individual.data['chromosome'][0].paint_map(), individual.data['chromosome'][1]
          evo.chromosomes[ind_idx] = individual.data['chromosome']

      if ind_idx in self.idxs:
         pass
      # if we have to run any sims, run the parallelized rllib trainer object

      if elite:
         raise Exception

      if self.stats is None or ind_idx not in self.stats:
#        print("Training batch 1")
         evo.trainer.train()
         self.stats = ray.get(evo.global_stats.get.remote())

      if ind_idx not in self.stats:
         print("Training batch 2")
         evo.trainer.train()
      assert ind_idx in self.stats
      ind_stats = self.stats[ind_idx]

      if 'skills' not in ind_stats:
         score = 0
      score = evo.calc_diversity(ind_stats)
      features = calc_mean_agent(ind_stats)

      if ind_idx in evo.population:
          (game, old_score, age) = evo.population[ind_idx]
          evo.population[ind_idx] = (game, score, age)
         #features = evo.chromosomes[ind_idx][0].features
      individual.fitness.values = [score]
      individual.fitness.valid = True
      features = [features[i] for i in self.feature_idxs]
      individual.features = features
      self.idxs.add(ind_idx)


      if self.n_gen % 10 == 0:
         algo = self.algo
         self.algo = None
         toolbox = self.toolbox
         self.toolbox = None
#        for k, v in inspect.getmembers(self.algo):
#           if k.startswith('_') and k != '__class__': #or inspect.ismethod(v):
#              setattr(self, k, lambda x: None)
         evo.save()
         algo.save(os.path.join(self.save_path, 'ME_archive.p'))
         self.algo = algo
         self.toolbox = toolbox

      return [[score], features]

   def load(self):
      import pickle
      self.init_toolbox()
      with open(os.path.join(self.save_path, 'ME_archive.p'), "rb") as f:
         archive = pickle.load(f)
         # NOTE: (Elite) individuals saved in the grid will have overlapping indexes.
         # TODO: Save all elite maps; do inference on one of them.
         algo = EvoDEAPQD(
                 self.qdSimple,
                 self.toolbox,
                               archive['container'],
                               init_batch_size=archive['init_batch_size'],
                               batch_size=archive['batch_size'],
                               niter=archive['nb_iterations'],
                               cxpb=self.cxpb,
                               mutpb=self.mutation_pb,
                               verbose=self.verbose,
                               show_warnings=self.show_warnings,
                               results_infos=self.results_infos,
                               log_base_path=self.log_base_path,
                               iteration_callback_fn=self.iteration_callback)
         self.algo = algo
         algo.current_iteration = archive['current_iteration']
#        algo.start_time = timer()
#        algo.run(archive['container'])
         if not self.evolver.config.RENDER:
            algo.run()

         return


   def expr(self):
      individual = Individual(Individual, evolver=self.evolver)


      return individual

   def evolve(self):
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed',
#                         type=int,
#                         default=None,
#                         help="Numpy random seed")
#     parser.add_argument(
#         '-p',
#         '--parallelismType',
#         type=str,
#         default='multiprocessing',
#         help=
#         "Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)"
#     )
#     parser.add_argument('-o',
#                         '--outputDir',
#                         type=str,
#                         default=None,
#                         help="Path of the output log files")
#     args = parser.parse_args()
      self.evolver.global_counter.set_idxs.remote(self.g_idxs)
      seed = 420
      seed = np.random.randint(1000000)
      output_dir = self.save_path


   #   # Algorithm parameters
   #   dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
      max_size = self.max_size
   # The number of features to take into account in the container
      nb_features = len(self.feature_idxs)
      assert nb_features == len(self.evolver.config.ME_DIMS)
      nb_bins = [max_size for _ in range(nb_features)]
      #    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
      # The domain (min/max values) of the features
#     features_domain = [(0, 2000), (0, 2000)]
      self.features_domain = features_domain = [(0, self.max_skill) for i in range(nb_features)]
      # The domain (min/max values) of the fitness
      fitness_domain = [(-np.inf, np.inf)]
      # The number of evaluations of the initial batch ('batch' = population)
      init_batch_size = self.evolver.config.N_EVO_MAPS
      # The number of evaluations in each subsequent batch
      batch_size = self.evolver.config.N_EVO_MAPS
      # The number of iterations (i.e. times where a new batch is evaluated)
      nb_iterations = self.evolver.n_epochs
      self.cxpb = cxpb = 0.5
      # The probability of mutating each value of a genome
      self.mutation_pb = mutation_pb = 1.0
      # The number of items in each bin of the grid
      max_items_per_bin = 1
      self.verbose = verbose = True
      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
      self.show_warnings = show_warnings = True
      self.log_base_path = log_base_path = output_dir if output_dir is not None else "."

      # Update and print seed
      np.random.seed(seed)
      random.seed(seed)
      print("Seed: %i" % seed)

      # Create a dict storing all relevant infos
      self.results_infos = results_infos = {}
      #    results_infos['dimension'] = dimension
      #    results_infos['ind_domain'] = ind_domain
      results_infos['features_domain'] = features_domain
      results_infos['fitness_domain'] = fitness_domain
      results_infos['nb_bins'] = nb_bins
      results_infos['init_batch_size'] = init_batch_size
      results_infos['nb_iterations'] = nb_iterations
      results_infos['batch_size'] = batch_size
      #    results_infos['mutation_pb'] = mutation_pb
      #    results_infos['eta'] = eta

      # Create container
      grid = NMMOGrid(
                  self.save_path,
                  self.evolver.config,
                  self.evolver.map_generator,
                  shape=nb_bins,
                  max_items_per_bin=max_items_per_bin,
                  fitness_domain=fitness_domain,
                  features_domain=features_domain,
                  storage_type=list)
      parallelism_type = 'sequential'
#     parallelism_type = 'multiprocessing'
      with ParallelismManager(parallelism_type,
                             toolbox=self.toolbox) as pMgr:
         # Create a QD algorithm
         algo = EvoDEAPQD(self.qdSimple,
                            pMgr.toolbox,
                            grid,
                                init_batch_size=init_batch_size,
                                batch_size=batch_size,
                                niter=nb_iterations,
                                cxpb=cxpb,
                                mutpb=mutation_pb,
                                verbose=verbose,
                                show_warnings=show_warnings,
                                results_infos=results_infos,
                                log_base_path=log_base_path,
                                iteration_callback_fn=self.iteration_callback)
         self.algo = algo
         # Run the illumination process !
         algo.run()
      self.log_me(container)

   def log_me(self, container):
      grid = container
      algo = self.algo
      log_base_path = self.log_base_path
      # Print results info
      print(f"Total elapsed: {algo.total_elapsed}\n")
      print(grid.summary())
      # print("Best ever fitness: ", container.best_fitness)
      # print("Best ever ind: ", container.best)
      # print("%s filled bins in the grid" % (grid.size_str()))
      # print("Solutions found for bins: ", grid.solutions)
      # print("Performances grid: ", grid.fitness)
      # print("Features grid: ", grid.features)

      # Search for the smallest best in the grid:
      smallest_best = grid.best
      smallest_best_fitness = grid.best_fitness
      smallest_best_length = grid.best_features[0]
      interval_match = 1e-10

      for ind in grid:
          if abs(ind.fitness.values[0] -
                 smallest_best_fitness.values[0]) < interval_match:

              if ind.features[0] < smallest_best_length:
                  smallest_best_length = ind.features[0]
                  smallest_best = ind
      print("Smallest best:", smallest_best)
      print("Smallest best fitness:", smallest_best.fitness)
      print("Smallest best features:", smallest_best.features)

      # It is possible to access the results (including the genomes of the solutions, their performance, etc) stored in the pickle file by using the following code:
      # ----8<----8<----8<----8<----8<----8<
      # from deap import base, creator, gp
      # import pickle
      # fitness_weight = -1.0
      # creator.create("FitnessMin", base.Fitness, weights=(fitness_weight,))
      # creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, features=list)
      # pset = gp.PrimitiveSet("MAIN", 1)
      # pset.addEphemeralConstant("rand101", lambda: random.randint(-4.,4.))
      # with open("final.p", "rb") as f:
      #    data = pickle.load(f)
      # print(data)
      # ----8<----8<----8<----8<----8<----8<
      # --> data is a dictionary containing the results.

      # Create plots
      plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
      plotGridSubplots(grid.quality_array[..., 0],
                       plot_path,
                       plt.get_cmap("nipy_spectral"),
                       grid.features_domain,
                       grid.fitness_extrema[0],
                       nbTicks=None)
      print("\nA plot of the performance grid was saved in '%s'." %
            os.path.abspath(plot_path))

      plot_path = os.path.join(log_base_path, "activityGrid.pdf")
      plotGridSubplots(grid.activity_per_bin,
                       plot_path,
                       plt.get_cmap("nipy_spectral"),
                       grid.features_domain,
                       [0, np.max(grid.activity_per_bin)],
                       nbTicks=None)
      print("\nA plot of the activity grid was saved in '%s'." %
            os.path.abspath(plot_path))

      print("All results are available in the '%s' pickle file." %
            algo.final_filename)

   # MODELINE "{{{1
   # vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
   # vim:foldmethod=marker
