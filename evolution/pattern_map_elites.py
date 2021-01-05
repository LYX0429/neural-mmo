import copy
import inspect
import operator
import os
import random
import warnings
from pdb import set_trace as TT

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy

from deap import algorithms, base, creator, gp, tools
from forge.blade.lib import enums
from pcg import TILE_PROBS
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
# from qdpy.benchmarks import *
from qdpy.containers import Grid
from qdpy.plots import *
from timeit import default_timer as timer

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

class MapElites():
   def compile(self):
       TT()

   def gen_individual(self):
       TT()

   def mutate(self, individual):
      evo = self.evolver
      idx = individual.data['ind_idx']
      chrom, atk_mults = evo.chromosomes[idx]
      atk_mults = evo.mutate_mults(atk_mults)
      chrom.mutate()

      if not hasattr(individual.fitness, 'values'):
         individual.fitness.values = None
      individual.fitness.valid = False

      return (individual, )


   def mate(self, parent_0, parent_1):
      evo = self.evolver
      idx_0 = parent_0.data['ind_idx']
      idx_1 = parent_1.data['ind_idx']
      chrom_0, atk_mults_0 = evo.chromosomes[idx_0]
      chrom_1, atk_mults_1 = evo.chromosomes[idx_1]
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
      evo.chromosomes[idx_0] = chrom_0, new_atk_mults_0
      evo.chromosomes[idx_1] = chrom_1, new_atk_mults_1
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
      self.stats = None

   def init_toolbox(self):
       fitness_weight = -1.0
       creator.create("FitnessMin", base.Fitness, weights=(fitness_weight, ))
       creator.create("Individual",
                      Individual,
                      evolver=self.evolver)
#                     fitness=creator.FitnessMin,
#                     features=list)

       # Create Toolbox
       self.max_size = max_size = 25
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

   def init_pop(self, n):
      return [Individual(rank=i, evolver=self.evolver) for i in range(n)]

   def expr_mutate(self):
       TT()

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
      self.stats = None
      evo.global_stats.reset.remote()
      evo.saveMaps(evo.genes)
      evo.log()

   def evaluate(self, individual):
      evo = self.evolver
      ind_idx = individual.data['ind_idx']

      if ind_idx in self.idxs:
         pass
      # if we have to run any sims, run the parallelized rllib trainer object

      if self.stats is None or ind_idx not in self.stats:
         evo.trainer.train()
         self.stats = ray.get(evo.global_stats.get.remote())
      assert ind_idx in self.stats
      ind_stats = self.stats[ind_idx]
      score = evo.calc_diversity(ind_stats)
      (game, old_score, age) = evo.population[ind_idx]
      evo.population[ind_idx] = (game, score, age)
      features = evo.chromosomes[ind_idx][0].features
      individual.fitness.values = [score]
      individual.fitness.valid = True
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
         algo = DEAPQDAlgorithm(self.toolbox,
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
         return algo.run()


   def expr(self):
      individual = Individual(Individual, evolver=self.evolver)


      return individual

   def evolve(self):
#      import argparse
#      parser = argparse.ArgumentParser()
#      parser.add_argument('--seed',
#                          type=int,
#                          default=None,
#                          help="Numpy random seed")
#      parser.add_argument(
#          '-p',
#          '--parallelismType',
#          type=str,
#          default='multiprocessing',
#          help=
#          "Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)"
#      )
#      parser.add_argument('-o',
#                          '--outputDir',
#                          type=str,
#                          default=None,
#                          help="Path of the output log files")
#      args = parser.parse_args()
       self.evolver.global_counter.set_idxs.remote(list(range(self.evolver.config.N_EVO_MAPS)))
       seed = 420
      # seed = np.random.randint(1000000)
       output_dir = self.save_path


   #    # Algorithm parameters
   #    dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
       max_size = self.max_size
   # The number of features to take into account in the container
       nb_features = 2
       nb_bins = [max_size, max_size]
       #    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
       # The domain (min/max values) of the features
       features_domain = [(0, self.evolver.max_primitives), (0, self.evolver.max_primitives)]
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
       grid = Grid(shape=nb_bins,
                   max_items_per_bin=max_items_per_bin,
                   fitness_domain=fitness_domain,
                   features_domain=features_domain,
                   storage_type=list)
       self.grid = grid
       parallelism_type = 'sequential'
#      parallelism_type = 'multiprocessing'
       with ParallelismManager(parallelism_type,
                               toolbox=self.toolbox) as pMgr:
           # Create a QD algorithm
           algo = DEAPQDAlgorithm(pMgr.toolbox,
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
