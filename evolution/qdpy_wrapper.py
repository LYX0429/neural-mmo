import copy
import inspect
import operator
import os
import random
import pickle
import warnings
from pdb import set_trace as TT
from timeit import default_timer as timer

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy
from PIL import Image

import deap
from deap import algorithms, base, creator, gp, tools
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.algorithms.evolution import CMAES
from qdpy.base import ParallelismManager
# from qdpy.benchmarks import *
from qdpy.containers import Grid
from qdpy.plots import *
from evolution.evo_map import EvolverNMMO, save_maps
from evolution.individuals import EvoIndividual, NeuralCA
from evolution.cmaes import EvoCMAES
import matplotlib.pyplot as plt

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

N_TRIALS = 1  # number of episodes run on a newly-generated map or a map being visited during re-evaluation

mpl.use('Agg')

def plot_qdpy_fitness(save_path, evolver=None, logbook=None):
   if logbook is None:
      logbook = evolver.algo.logbook
   gen = logbook.select("iteration")
   fit_mins = logbook.select("min")
   fit_avgs = logbook.select("avg")
   fit_stds = logbook.select("std")
   fit_maxs = logbook.select("max")

   fig, ax1 = plt.subplots()
   line0 = ax1.plot(gen, fit_mins, "b--")
   line1_err = ax1.errorbar(gen, fit_avgs, np.hstack(fit_stds), color='green', mfc='green', mec='green', linestyle="-",
                            label="Average Fitness",
                            ms=20, mew=4,
                            alpha=min(0.9, 100 / len(gen)),
                            # alpha=0.9,
                            )
   line1 = ax1.plot(gen, fit_avgs, 'b-', label='Average Fitness')
   line2 = ax1.plot(gen, fit_maxs, "b--")
   ax1.set_xlabel("Generation")
   ax1.set_ylabel("Fitness")
   # FIXME: figure out from logbook if we've got all-1 bin sizes so we don't plot size
   # if not np.all(self.config.ME_BIN_SIZES == 1):
   if True:
      # plot the size of the archive
      containerSize_avgs = logbook.select('containerSize')
      for tl in ax1.get_yticklabels():
         tl.set_color("b")
      ax2 = ax1.twinx()
      line2 = ax2.plot(gen, containerSize_avgs, "r-", label="Archive Size")
      ax2.set_ylabel("Size", color="r")
      # ax2_ticks = ax2.get_yticklabels()
      start, end = ax2.get_ylim()
      ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
      for tl in ax2.get_yticklabels():
         tl.set_color("r")
      lns = line1 + line2
      labs = [l.get_label() for l in lns]
      ax1.legend(lns, labs, loc="best")

   plt.tight_layout()
   # plt.show()
   plt.savefig(os.path.join(save_path, 'fitness.png'))


class NMMOGrid(Grid):
   def __init__(self, evolver, save_path, config, map_generator, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.evolver = evolver
       self.border = evolver.config.TERRAIN_BORDER
       self.save_path = save_path
       self.map_generator = map_generator
       self._nb_items_per_bin = self._nb_items_per_bin.astype(np.uint8)
       self.config = config

   def add(self, individual):
      border = self.border
      idx = self.index_grid(individual.features)
      if self.evolver.BASELINE_SIMPLEX and self.solutions[idx]:
         # If running baseline, never add to new cell
         return None
      index = super(NMMOGrid, self).add(individual)
      old_idx = individual.idx
#     if not (old_idx in self.evolver.population and old_idx in self.evolver.maps and old_idx in self.evolver.chromosomes):
      if index is not None:
         # if it has been added
         chromosome = individual.chromosome
         bin_idxs = set(range(self.config.N_EVO_MAPS))
         for s in self.solutions[idx]:
             if s is not individual:
                 bin_idxs.remove(s.bin_idx)
         individual.bin_idx = bin_idxs.pop()
         individual.idx = idx  + (individual.bin_idx,)
#        self.evolver.score_hists[idx] = individual.score_hists
#        self.evolver.score_hists[idx] = self.evolver.score_hists[old_idx]
#        self.evolver.chromosomes[idx] = chromosome
#        self.evolver.maps[idx] = (chromosome.map_arr, chromosome.multi_hot), chromosome.atk_mults
#        self.evolver.population[idx] = self.evolver.population[old_idx]
         if self.evolver.LEARNING_PROGRESS:
            self.evolver.ALPs[idx] = individual.ALPs

 #       if len(idx) == 1:
 #          index_str = '(' + str(idx[0]) + ',)'
 #       else:
 #          index_str = '('+ ', '.join([str(f) for f in idx]) + ')'
 #       map_path = os.path.join(self.save_path, 'maps', 'map' + index_str)
 #       try:
 #           os.makedirs(map_path)
 #       except FileExistsError:
 #           pass
 #       map_arr = chromosome.map_arr
 #       atk_mults = chromosome.atk_mults
#        if map_arr is None:
#           map_arr, _ = self.evolver.gen_cppn_map(chromosome)
#        Save.np(map_arr, map_path)
 #       if self.evolver.config.TERRAIN_RENDER == True:
 #          Save.render(map_arr[border:-border, border:-border], self.evolver.map_generator.textures, map_path + '.png')
#        print('add ind with idx {}'.format(tuple(individual.features)))
 #       json_path = os.path.join(self.save_path, 'maps', 'atk_mults' + index_str + 'json')
 #       with open(json_path, 'w') as json_file:
 #          json.dump(atk_mults, json_file)

      # individual is removed from population, whether or not it has been added to the container
      self.evolver.flush_individual(old_idx)

      return index

   def save(self):
      self.evolver = None
      return super().save()


class EvoDEAPQD(DEAPQDAlgorithm):
   def __init__(self, qd_fun, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ea_fn = qd_fun


class meNMMO(EvolverNMMO):
   def __init__(self, save_path, make_env, trainer, config, n_proc, n_pop, map_policy, n_epochs=10000):
      super().__init__(save_path, make_env, trainer, config, n_proc, n_pop, map_policy, n_epochs=n_epochs)
      # Track how many new elites have been added so that we can force population drift if necessary
      self.archive_update_hist = np.empty((self.config.ARCHIVE_UPDATE_WINDOW))
      self.archive_update_hist[:] = np.NaN
      self.n_gen = 0
      self.save_path = save_path
      # Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
      self.init_toolbox()
      self.idxs = set()
      self.mutated_idxs = set()
      self.reset_g_idxs()
      feature_names = self.config.ME_DIMS
      if self.config.FEATURE_CALC == 'skills':
          self.feature_idxs = [self.config.SKILLS.index(n) for n in feature_names]
      else:
          self.feature_idxs = [i for i in range(len(feature_names))]
#     try:
#        os.mkdir(os.path.join(self.save_path, 'temp_checkpoints'))
#     except FileExistsError:
#        pass
      # FIXME: we should handle this in the parent
      self.init_pop()
      self.logbook_path = os.path.join(self.save_path, 'logbook.pkl')

   def qdSimple(self, init_batch, toolbox, container, batch_size, niter, cxpb=0.0, mutpb=1.0, stats=None,
                halloffame=None, verbose=False, show_warnings=True, start_time=None, iteration_callback=None):
      """The simplest QD algorithm using DEAP, adapted for evolving maps in NMMO.
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
      def cull_invalid(offspring):
          if self.MAP_TEST:
              return offspring
          # Remove invalid mutants
          valid_ind = []
          [valid_ind.append(o) if o.valid_map else None for o in offspring]
          return valid_ind
      if start_time == None:
          start_time = timer()
      if self.n_gen == 0:
         logbook = deap.tools.Logbook()
         logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + [
            "elapsed"]
      else:
         with open(self.logbook_path, 'rb') as lb_file:
            logbook = pickle.load(lb_file)
      if len(init_batch) == 0:
          raise ValueError("``init_batch`` must not be empty.")
      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
      invalid_ind = cull_invalid(invalid_ind)
      while len(invalid_ind) == 0:
         print("No valid maps in initial batch. Re-generating initial batch.")
         invalid_ind = cull_invalid([EvoIndividual([], i, self) for i in range(len(init_batch))])
      for _ in range(N_TRIALS):
         self.train_individuals(invalid_ind)

      if self.LEARNING_PROGRESS:
         for _ in range(N_TRIALS):
            self.train_individuals(invalid_ind)
#     [self.evaluate(ind) for ind in invalid_ind]
      # No need for parallelism or toolbox here, we've already evaluated above
      if len(invalid_ind) == 0:
          raise ValueError("No valid individual found !")
      # Update halloffame
      if halloffame is not None:
          halloffame.update(init_batch)
      # Store batch in container
      nb_updated = container.update(invalid_ind, issue_warning=show_warnings)
      self.archive_update_hist = np.hstack((self.archive_update_hist[1:], [nb_updated]))
      if nb_updated == 0:
         #NOTE: For reloading.
         print('Warning: nothing added to container/grid. Not good if this is a fresh run.')
   #     raise ValueError("No individual could be added to the container !")
      else:
         # Compile stats and update logs
         record = stats.compile(container) if stats else {}
         logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                        elapsed=timer()-start_time, **record)
         if verbose:
             print(logbook.stream)
      # Call callback function
      if iteration_callback != None:
          iteration_callback(0, init_batch, container, logbook)

      # Begin the generational process
      for i in range(self.n_gen + 1, niter + 1):
         start_time = timer()
         # Select the next batch individuals
         batch = toolbox.select(container, batch_size)
         ## Vary the pool of individuals (actually doing this manually)
#        offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)
         offspring = []
         mutated = []
         maps = {}
         for (j, o) in enumerate(batch):
            rnd = np.random.random()
            # For every ~99 individuals we mutate, we inject a new random one
            if rnd < 0.01 or self.BASELINE_SIMPLEX:
                # If running a non-evolved baseline, never mutate, always generate anew
                newO = EvoIndividual([], j, self)
            else:
                newO = self.clone(o)
                newO.mutate()
            new_chrom = newO.chromosome
            newO.idx = j
            offspring.append(newO)
#           self.gen_cppn_map(newO.chromosome)
            self.maps[j] = ((new_chrom.map_arr, new_chrom.multi_hot), new_chrom.atk_mults)
            mutated.append(j)
         valid_ind = cull_invalid(offspring)

         # We should never get here?
         while len(valid_ind) == 0:
             raise Exception
             # FIXME: put this inside our own varAnd function
             self.reset_g_idxs()  # since cloned individuals need new indices
             # We should be doing this manually, as above, no?
             offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)
             valid_ind = cull_invalid(offspring)
         for _ in range(N_TRIALS):
            self.train_individuals(valid_ind)
#        fitness_stds = [ind.fitness_stds[0] for ind in valid_ind]
#        print('fitness stds: {}'.format(fitness_stds))
         if self.LEARNING_PROGRESS:
            for _ in range(N_TRIALS):
               self.train_individuals(valid_ind)
#        print('{} invalid individuals'.format(len(invalid_ind)))
         # Replace the current population with the offspring
         if self.MAP_TEST:
             show_warnings = True
         nb_updated = container.update(valid_ind, issue_warning=show_warnings)
         self.archive_update_hist = np.hstack((self.archive_update_hist[1:], [nb_updated]))
         # Update the hall of fame with the generated individuals
         if halloffame is not None:
             halloffame.update(container)
         # Append the current generation statistics to the logbook
         record = stats.compile(container) if stats else {}
         logbook.record(iteration=self.n_gen, containerSize=container.size_str(), evals=len(valid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record)
         if verbose:
            print(logbook.stream)
         # Call callback function
         if iteration_callback != None:
             iteration_callback(i, batch, container, logbook)

      save_maps(save_path=self.save_path, config=self.config, individuals=self.container,
                map_generator=self.map_generator)
      plot_qdpy_fitness(save_path=self.save_path, evolver=self)

      return batch, logbook


   def reset_g_idxs(self):
      self.g_idxs = set(range(self.config.N_EVO_MAPS))

   def iteration_callback(self, i, batch, container, logbook):
      # print('qdpy-based MAP-elites iteration {}'.format(self.n_gen))
#     if not len(self.population) == len(self.maps) == len(self.chromosomes):
#        raise Exception
      # FIXME: doesn't work -- sync up these iteration/generation counts
      self.algo.current_iteration = self.n_gen
      self.n_epoch = self.n_gen
      self.idxs = set()
#     stats = self.tats
      self.reset_g_idxs()
      # update the elites to avoid stagnation (since simulation is stochastic)
#     if self.n_gen > 0 and (len(container) > 0 and np.random.random() < 0.1):
      if self.config.ARCHIVE_UPDATE_WINDOW == 0:
         recent_mean_updates = 0
      else:
         recent_mean_updates = np.nanmean(self.archive_update_hist)
#     if self.n_epoch > 0 and len(container) > 0 and not self.MAP_TEST:
      if len(container) > 0 and recent_mean_updates < 0.01 and not self.MAP_TEST:
     #   try:
         disrupted_elites = [container[i] for i in np.random.choice(len(container), min(max(1, len(container)-1), self.config.N_EVO_MAPS), replace=False)]
         for _ in range(N_TRIALS):
            self.train_individuals(disrupted_elites)
         # NOTE: We're simply changing individuals in-place for now, since map features will not change
         # nb_updated = container.update(disrupted_elites, issue_warning=True)
         # print('Reinstated {} of {} disturbed elites.'.format(nb_updated, len(disrupted_elites)))
         print('Re-evaluated {} individuals in-place.'.format(len(disrupted_elites)))

      self.idxs = set()
      self.reset_g_idxs()
      self.log(verbose=False)
      self.mutated_idxs = set()
      self.n_gen += 1

      if self.n_gen == 1 or self.n_gen > 0 and self.n_gen % self.config.EVO_SAVE_INTERVAL == 0:
         self.save()
      # Remove mutants after each iteration, since they have either been added to the container/archive, or deleted.
      #FIXME: why wouldn't it be?

   def log(self, verbose=False):
      pass

   def reload_log(self):
      pass

   def save(self):
      evo_save_start = timer()
      self.log_me(self.container)
      # A bunch of workarounds to not save circular or reduntant stuff (elite archive)
      # TODO: These references could be a whole lot less circular, eh?
      algo = self.algo
      algo.container.evolver = None
      algo.save(os.path.join(self.save_path, 'ME_archive.p'))
      algo.container.evolver = self
      self.algo = None
      toolbox = self.toolbox
      self.toolbox = None
      self.container = None
      with open(self.logbook_path, 'wb') as lb_file:
         pickle.dump(algo.logbook, lb_file)
      super().save()
      self.algo = algo
      self.toolbox = toolbox
      self.container = algo.container
      evo_save_end = timer()
      evo_save_time = evo_save_end - evo_save_start
      print('qdpy save time elapsed: {}'.format(evo_save_time))

   def compile(self):

       pass

   def gen_individual(self):
       pass


   def clone(self, individual):
      child = individual.clone(self)
      idx = self.g_idxs.pop()
      child.idx = idx
#     assert child is not individual
#     assert child != individual
#     self.chromosomes[idx] = individual.chromosome

      return child


   def mutate(self, individual):
      individual.mutate()
      return (individual, )

  #def mutate(self, individual):
  #   idx = individual.idx
  #   self.mutated_idxs.add(idx)
# #   print('mutate {}'.format(idx))
  #   # = self.container.index_grid(np.clip(inddividual.features, 0, 2000))
  #   #FIXME: big hack
  #   chrom, atk_mults = individual.chromosome
  #   atk_mults = self.mutate_mults(atk_mults)
  #   chrom.mutate()
  #   self.validate_map(chrom.flat_map, chrom.multi_hot)

  #   individual.fitness.delValues()
# #   if not hasattr(individual.fitness, 'values'):
# #      individual.fitness.values = None
# #   individual.fitness.valid = False
# #   evo.maps[idxs] = map_arr, atk_mults

  #   return (individual, )

   def mate(self, p0, p1):
      return p0, p1

  #def mate(self, parent_0, parent_1):
  #   idx_0 = parent_0.idx
  #   idx_1 = parent_1.idx
  #   self.mutated_idxs.add(idx_0)
  #   self.mutated_idxs.add(idx_1)
  #   chrom_0, atk_mults_0 = parent_0.chromosome
  #   chrom_1, atk_mults_1 = parent_1.chromosome
  #   prims_0 = chrom_0.patterns
  #   prims_1 = chrom_1.patterns
# #   new_atk_mults_0, new_atk_mults_1 = self.mate_mults(atk_mults_0, atk_mults_1)
  #   len_0, len_1 = len(prims_0), len(prims_1)

  #   if len_0 < len_1:
  #      prims_0 = prims_0 + prims_1[-len_1 + len_0 - 1:]
  #   elif len_1 < len_0:
  #      prims_1 = prims_1 + prims_0[-len_0 + len_1 - 1:]
  #   new_prims_0 = [prims_0[i] if random.random() < 0.5 else prims_1[i] for i in range(len_0)]
  #   new_prims_1 = [prims_0[i] if random.random() < 0.5 else prims_1[i] for i in range(len_1)]
  #   chrom_0.patterns = new_prims_0
  #   chrom_1.patterns = new_prims_1
# #   self.chromosomes[idx_0] = chrom_0, new_atk_mults_0
# #   self.chromosomes[idx_1] = chrom_1, new_atk_mults_1
  #   chrom_0.update_features()
  #   chrom_1.update_features()

  #   parent_0.delValues()
# #   if not hasattr(parent_0.fitness, 'values'):
# #      parent_0.fitness.values = None
# #   parent_0.fitness.valid = False

  #   parent_1.delValues()
# #   if not hasattr(parent_1.fitness, 'values'):
# #      parent_1.fitness.values = None
# #   parent_1.fitness.valid = False

  #   return parent_0, parent_1

   def init_toolbox(self):
      fitness_weight = -1.0
      creator.create("FitnessMin", base.Fitness, weights=(fitness_weight, ))
      creator.create("Individual",
                     EvoIndividual,
                     iterable=[],
                     evolver=self)
#                    fitness=creator.FitnessMin,
#                    features=list)

      # Create Toolbox
      toolbox = base.Toolbox()
      toolbox.register("expr", self.expr)
      toolbox.register("individual", tools.initIterate, creator.Individual,
                       toolbox.expr)
      toolbox.register("clone", self.clone)
      toolbox.register("population", self.init_individuals)
      toolbox.register("compile", self.compile)  # gp.compile, pset=pset)
      toolbox.register("evaluate", self.evaluate)  # , points=points)
      # toolbox.register("select", tools.selTournament, tournsize=3)
      toolbox.register("select", tools.selRandom)
#     if self.CPPN:
#        toolbox.register("mate", self.mate_cppns)
#        toolbox.register("mutate", self.mutate_cppn)
#     elif self.PRIMITIVES:
#        toolbox.register("mate", self.mate)
#        toolbox.register("mutate", self.mutate)
#     else:
#        pass
#        raise Exception
      toolbox.register("mutate", self.mutate)
      toolbox.register("mate", self.mate)
      toolbox.register("expr_mut", self.expr_mutate)
#     toolbox.register("select", self.select_max_lifetime)
      self.toolbox = toolbox
#     self.max_skill = 2000

   def expr_mutate(self):
      raise NotImplementedError

   def init_individuals(self, n):
      individuals = [EvoIndividual([], rank=i, evolver=self) for i in range(n)]

      return individuals

   def protectedDiv(left, right):
       try:
           return left / right
       except ZeroDivisionError:
           return 1

   def select_max_lifetime(self, container, k):
      return sorted(container, key=lambda ind: ind.features[0], reverse=True)[:k]



   def evaluate(self, individual, elite=False):
      return [individual.fitness.getValues(), individual.features]

   def init_algo(self,
                 ea_fn,
                 toolbox,
                 container,
                 init_batch_size,
                 batch_size,
                 niter,
                ):
       if self.MAP_ELITES:
           algo = EvoDEAPQD(
               qd_fun=ea_fn,
               toolbox=toolbox,
               container=container,
               init_batch_size=init_batch_size,
               batch_size=batch_size,
               niter=niter,
               cxpb=self.cxpb,
               mutpb=self.mutation_pb,
               verbose=self.verbose,
               show_warnings=self.show_warnings,
               results_infos=self.results_infos,
               log_base_path=self.log_base_path,
               iteration_callback_fn=self.iteration_callback)
       elif self.CMAES:
           if self.TILE_FLIP:
               dimension = self.n_tiles * self.map_width * self.map_width
           elif self.CA:
               dimension = len(EvoIndividual([], rank=0, evolver=self).chromosome.nn.weights) + 1
           else:
               raise Exception
           budget = self.config.N_EVO_MAPS * self.n_epochs
           algo = EvoCMAES(self, container,
                        budget=budget,
                        dimension=dimension)
       self.algo = algo

   def reload_archive(self):
      import pickle
      with open(os.path.join(self.save_path, 'ME_archive.p'), "rb") as f:
         archive = pickle.load(f)
      self.container = archive['container']

   def resume(self):
      import pickle
      self.init_toolbox()
      with open(os.path.join(self.save_path, 'ME_archive.p'), "rb") as f:
         archive = pickle.load(f)
      # NOTE: (Elite) individuals saved in the grid will have overlapping indexes.
      self.init_algo(
              self.qdSimple,
              self.toolbox,
              container=archive['container'],
              init_batch_size=archive['init_batch_size'],
              batch_size=archive['batch_size'],
              # niter=archive['nb_iterations'],
              niter=self.config.N_GENERATIONS,
      )
      # ZOINKYS!
      self.algo.container.evolver = self

      self.algo.current_iteration = archive['current_iteration']
      self.algo.start_time = timer()
      if not self.config.RENDER:
         self.algo.run()

      return


   def expr(self):
      individual = EvoIndividual(iterable=[], rank=None, evolver=self)
      assert individual.chromosome.map_arr is not None


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
      seed = 420
      seed = np.random.randint(1000000)
      output_dir = self.save_path


   #   # Algorithm parameters
   #   dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
#     max_size = self.max_size
   # The number of features to take into account in the container
      nb_features = len(self.feature_idxs)
      assert nb_features == len(self.config.ME_DIMS)
#     nb_bins = [max_size for _ in range(nb_features)]
      nb_bins = self.config.ME_BIN_SIZES
      #    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
      # The domain (min/max values) of the features
#     features_domain = [(0, 2000), (0, 2000)]
#     self.features_domain = features_domain = [(0, self.max_skill) for i in range(nb_features)]
      self.features_domain = features_domain = self.config.ME_BOUNDS
      # The domain (min/max values) of the fitness
      fitness_domain = [(-np.inf, np.inf)]
      # The number of evaluations of the initial batch ('batch' = population)
      init_batch_size = self.config.N_EVO_MAPS
      # The number of evaluations in each subsequent batch
      batch_size = self.config.N_EVO_MAPS
      # The number of iterations (i.e. times where a new batch is evaluated)
      nb_iterations = self.n_epochs
      self.cxpb = cxpb = 0.5
      # The probability of mutating each value of a genome
      self.mutation_pb = mutation_pb = 1.0
      # The number of items in each bin of the grid
      max_items_per_bin = int(self.config.ITEMS_PER_BIN)
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
                  self,
                  self.save_path,
                  self.config,
                  self.map_generator,
                  shape=nb_bins,
                  max_items_per_bin=max_items_per_bin,
                  fitness_domain=fitness_domain,
                  features_domain=features_domain,
                  storage_type=list)
      self.container = grid
      # We don't need qdpy's parallelism since RLlib already ensures that evaluation of different maps happens in parallel
#     parallelism_type = 'sequential'
#     parallelism_type = 'multiprocessing'
#     with ParallelismManager(parallelism_type,
#                            toolbox=self.toolbox) as pMgr:
      # Create a QD algorithm
      self.init_algo(self.qdSimple, self.toolbox, grid, init_batch_size, batch_size, nb_iterations)
         # Run the illumination process !
      self.algo.run()
      self.log_me(grid)





   def log_me(self, container=None):
         if container is not None:
             grid = container
         else:
             grid = self.container
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
         # print(" grid: ", grid.features)

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
         # TODO: make this more efficient. Do qdpy archives already store this? extend it if not
         ages = np.zeros(shape=grid.shape, dtype=np.uint8)
         for ind in grid:
            ages[ind.idx[:-1]] = ind.age
         # Create age-over-archive heatmap(s)
         plot_path = os.path.join(log_base_path, "ages.pdf")
         plotGridSubplots(ages,
                          plot_path,
                          plt.get_cmap("nipy_spectral"),
                          grid.features_domain,
                          [0, np.max(ages)],
                          nbTicks=None)
         print("\nA plot of the age grid was saved in '%s'." %
               os.path.abspath(plot_path))

         print("All results are available in the '%s' pickle file." %
               algo.final_filename)

def render_map_grid(grid, save_path, d=5):
   map_generator = MapGenerator(grid.config)
   shape = grid.shape
   ind_keys = sorted(list(grid.solutions.keys()))
   # make a list of list of indices corresponding sufficiently populated rows
   ind_idxs = []
   x = -1
   y_idxs = []
   n_y = 0
   for k in ind_keys:
      if k[0] > x:
         if n_y >= 10:
            ind_idxs.append((x, y_idxs))
         x = k[0]
         n_y = 0
         y_idxs = []
      if grid.solutions[k]:
         n_y += 1
         y_idxs.append(k[1])

   # select indices to render with maximum spacing
   render_idxs = []
   x_rel_idxs = [int(i* (len(ind_idxs)-1) / (d-1)) for i in range(d)]
   for xi in x_rel_idxs:
      x_idx = ind_idxs[xi][0]
      y_rel_idxs = [int(i * (len(ind_idxs[xi][1])-1) / (d-1)) for i in range(d)]
      y_idxs = [(x_idx, ind_idxs[xi][1][yi]) for yi in y_rel_idxs]
      render_idxs.append(y_idxs)

   map_grid = []
   for (i, row) in enumerate(render_idxs):
      map_grid.append([])
      for (j, (x, y)) in enumerate(render_idxs[i]):
         ind = grid.solutions[(x, y)][0]
         map_arr = ind.chromosome.map_arr
         im = Save.render(map_arr[grid.config.TERRAIN_BORDER-1:-grid.config.TERRAIN_BORDER+1, grid.config.TERRAIN_BORDER-1:-grid.config.TERRAIN_BORDER+1],
         map_generator.textures, path=None)
         map_grid[i].append(im)
#        map_grid.append(grid.index_grid)
   map_grid = [np.hstack(row) for row in map_grid]
   map_grid = np.vstack(map_grid)
   img = Image.fromarray(map_grid, 'RGB')
   img.save(os.path.join(save_path, 'map_grid.png'))
