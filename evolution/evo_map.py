import copy

# import torch.cuda
from qdpy.phenotype import Features
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from pdb import set_trace as TT
from shutil import copyfile
from typing import Dict
import skimage
from skimage.morphology import disk

import numpy as np
import ray
import scipy
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import projekt
import neat
from evolution.diversity import diversity_calc
from evolution.diversity import calc_map_entropies, calc_global_map_entropy
from evolution.lambda_mu import LambdaMuEvolver
from evolution.individuals import EvoIndividual
#from evolution.individuals import CPPNGenome as DefaultGenome
from evolution.individuals import DefaultGenome
from evolution.individuals import CAGenome
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums
from forge.blade.lib import material
from forge.ethyr.torch import utils
from forge.trinity.overlay import OverlayRegistry

from pcg import get_tile_data
from plot_evo import plot_exp
#from projekt import rlutils
#from projekt.evaluator import Evaluator
from projekt.rllib_wrapper import (EvoPPOTrainer, RLlibLogCallbacks, RLlibEvaluator,
                                   actionSpace, frozen_execution_plan,
                                   observationSpace)
# from pureples.shared.visualize import draw_net

np.set_printoptions(threshold=sys.maxsize,
                    linewidth=120,
                    suppress=True,
                   #precision=10
                    )

# keep trainer here while saving evolver object
global TRAINER
TRAINER = None

# # FIXME: backward compatability, we point to correct function on restore()
# def calc_convex_hull():
#    pass
#
# def calc_differential_entropy():
#    pass
#
# def calc_discrete_entropy_2():
#    pass
#
# def calc_convex_hull():
#    pass
#
# def calc_fitness_l2():
#    pass
#
# def calc_map_diversity():
#    pass

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    k_lrg = np.column_stack(np.unravel_index(idx, a.shape))

    return k_lrg

def sigmoid_lifespan(x):
   res = 1 / (1 + np.exp(0.1*(-x+50)))
#  res = scipy.special.softmax(res)

   return res

def save_maps(save_path, config, individuals, map_generator=None, mutated=None):
   if map_generator is None:
      map_generator = MapGenerator(config)
   # can't have neat with non-cppn representation
   #     if self.NEAT:
   #        if mutated is None:
   #           mutated = list(maps.keys())
   #        #FIXME: hack
   # checkpoint_dir_path = os.path.join(self.save_path, 'maps', 'checkpoint_{}'.format(self.n_epoch))
   # checkpoint_dir_created = False
   # checkpointing = False
   for ind in individuals:
      i = ind.idx
      path = os.path.join(save_path, 'maps', 'map' + str(i), '')
      map_arr = ind.chromosome.map_arr
      Save.np(map_arr, path)
      png_path = os.path.join(save_path, 'maps', 'map' + str(i) + '.png')
      Save.render(map_arr[config.TERRAIN_BORDER:-config.TERRAIN_BORDER, config.TERRAIN_BORDER:-config.TERRAIN_BORDER], map_generator.textures, png_path)

class LogCallbacks(DefaultCallbacks):
   STEP_KEYS = 'env_step realm_step env_stim stim_process'.split()
   EPISODE_KEYS = ['env_reset']

   def init(self, episode):
      for key in LogCallbacks.STEP_KEYS + LogCallbacks.EPISODE_KEYS:
         episode.hist_data[key] = []

   def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy],
         episode: MultiAgentEpisode, **kwargs):
      self.init(episode)

   def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
         episode: MultiAgentEpisode, **kwargs):

      env = base_env.envs[0]

      for key in LogCallbacks.STEP_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

   def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
      env = base_env.envs[0]

      for key in LogCallbacks.EPISODE_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

   def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
      n_epis = result["episodes_this_iter"]
#     print("trainer.train() result: {} -> {} episodes".format(
#        trainer, n_epis))

      # you can mutate the result dict to add new fields to return
      # result['something'] = True

def calc_mean_agent(individual, config):
   skill_arrays = []
   ind_stats = individual.stats

   for skill_arr in ind_stats['skills']:
      if len(skill_arr) > 0:
         skill_arrays.append(skill_arr)
   skill_mat = np.vstack(skill_arr)
   mean_agent = skill_mat.mean(axis=0)

   return mean_agent

def calc_mean_lifetime(individual, config):
   ind_stats = individual.stats
   mean_lifetime = [np.hstack(ind_stats['lifetimes']).mean()]

   return mean_lifetime

def dummi_features(individual, config):
    return (50, 50)

def rand_features(individual, config):
    # Assuming the experiments we're comparing against measure map entropy, make feat_1 <= feat_0, since mean local map
    # entropy cannot be greater than global map entropy.
    feat_0 = np.random.randint(*config.ME_BOUNDS[0])
    feat_1 = np.random.randint(feat_0 + 1)
#   feats = [np.random.randint(*config.ME_BOUNDS[i]) for i in range(len(config.ME_BOUNDS))]
    return [feat_0, feat_1]

class EvolverNMMO(LambdaMuEvolver):
   def __init__(self, save_path, make_env, trainer, config, n_proc=12, n_pop=12, map_policy=None, n_epochs=10000):
      self.config = config
      if config.PAIRED:
         config.NPOLICIES = 2
         config.NPOP = 2
      if config.PRETRAIN:
          # assert self.BASELINE_SIMPLEX
          assert config.GENOME == 'Baseline'
          # This will allow the archive to be filled out quite rapidly, so that we cannot be accused of giving baseline models too few maps on which to train as compared to the jointly-optimized maps
          self.calc_features = rand_features
      elif config.FEATURE_CALC == "map_entropy":
         config.ME_DIMS = 'map_entropy'
         self.calc_features = calc_map_entropies
#        self.calc_features = calc_global_map_entropy
      elif config.FEATURE_CALC == 'skill':
         self.calc_features = calc_mean_agent
      elif config.FEATURE_CALC == 'agent_lifetime':
         self.calc_features = calc_mean_lifetime
      elif config.FEATURE_CALC is None:
          self.calc_features = dummi_features
      else:
         raise NotImplementedError('Provided feature calculation function: {} is invalid. In case of BCs (provide "None" if not applicable).'.format(config.FEATURE_CALC))
      self.policy_to_agent = {i: [] for i in range(config.NPOLICIES)}
#     self.gen_mults = gen_atk_mults
#     self.mutate_mults = mutate_atk_mults
#     self.mate_mults = mate_atk_mults
      config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', config.EVO_DIR, 'maps', 'map')
      super().__init__(save_path, n_proc=n_proc, n_pop=n_pop, n_epochs=n_epochs)
      self.lam = 1/4
      self.mu = 1/4
      self.make_env = make_env
      self.trainer = trainer
      self.map_width = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
      self.map_height = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
      TILE_TYPES, self.TILE_PROBS = get_tile_data(griddly=config.GRIDDLY)
      self.n_tiles = len(TILE_TYPES)
      self.mature_age = config.MATURE_AGE
      self.state = {}
      self.done = {}
      self.chromosomes = {}
      if self.config.GRIDDLY:
         from griddly_nmmo.map_gen import GdyMaterial, GriddlyMapGenerator
         self.mats = GdyMaterial
         self.SPAWN_IDX = GdyMaterial.SPAWN.value.index
         self.map_generator = GriddlyMapGenerator(self.config)
      else:
         self.mats = enums.MaterialEnum
         if self.config.GRIDDLY:
            self.SPAWN_IDX = self.mats.Spawn.index
         else:
            self.SPAWN_IDX = material.Spawn.index
         self.map_generator = MapGenerator(config)
      self.skill_idxs = {}
      self.idx_skills = {}
      self.reloading = False  # Have we just reloaded? For re-saving elite maps when reloading.
      self.epoch_reloaded = None
#     self.global_stats = ray.get_actor("global_stats")
      self.global_counter = ray.get_actor("global_counter")
      # FIXME: FUCK THIS, PROBABLY UNNECESSARY BY NOW???
      self.CPPN = config.GENOME == 'CPPN'
      self.CA = config.GENOME == 'CA'
      self.PRIMITIVES = config.GENOME == 'Pattern'
      self.TILE_FLIP = config.GENOME == 'Random'
      self.LSYSTEM = config.GENOME == 'LSystem'
      self.SIMPLEX_NOISE = config.GENOME == 'Simplex'
      self.BASELINE_SIMPLEX = config.GENOME == 'Baseline'
      self.ALL_GENOMES = config.GENOME == 'All'
      if not (self.CA or self.LSYSTEM or self.CPPN or self.PRIMITIVES or self.TILE_FLIP or self.SIMPLEX_NOISE or self.ALL_GENOMES or self.BASELINE_SIMPLEX):
         raise Exception('Invalid genome')
      if self.BASELINE_SIMPLEX:
          assert config.PRETRAIN
      self.NEAT = config.EVO_ALGO == 'NEAT'
      self.LEARNING_PROGRESS = config.FITNESS_METRIC == 'ALP'
      self.MAP_TEST = 'MapTest' in config.FITNESS_METRIC
      self.calc_fitness = diversity_calc(config)
      self.ALPs = {}
      self.LAMBDA_MU = config.EVO_ALGO == 'Simple'
      self.MAP_ELITES = config.EVO_ALGO == 'MAP-Elites'
      self.CMAES = config.EVO_ALGO == 'CMAES'
      self.CMAME = config.EVO_ALGO == 'CMAME'

      # because population will exceeded given limit

      if self.NEAT or self.MAP_ELITES:
         # g_idxs we might use
         self.g_idxs_reserve = set([i for i in range(self.n_pop*10)])

      if self.CPPN or self.ALL_GENOMES:
         self.neat_to_g = {}

         if self.config.GRIDDLY:
            neat_config_path = 'config_cppn_nmmo_griddly'
         else:
            neat_config_path = 'config_cppn_nmmo'
         self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                               neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                               neat_config_path)
         self.neat_config.fitness_threshold = np.float('inf')
         self.neat_config.pop_size = self.config.N_EVO_MAPS
         self.neat_config.elitism = int(self.lam * self.config.N_EVO_MAPS)
         self.neat_config.survival_threshold = self.mu
         self.neat_config.num_outputs = self.n_tiles
#        self.neat_pop = neat.population.Population(self.neat_config)
         #NOTE: NEAT indexing accomodation
#        self.chromosomes = dict([(i-1, genome) for (i, genome) in self.neat_pop.population.items()])


      if self.PRIMITIVES or self.ALL_GENOMES:
        #self.max_primitives = self.map_width**2 / 4
         self.max_primitives = 20

#        from evolution.individuals import PatternGenome
#        self.Chromosome = PatternGenome
#        self.chromosomes = {}
#        self.global_counter.set_idxs.remote(range(self.config.N_EVO_MAPS))
         # TODO: generalize this for tile-flipping, then NEAT

#     if self.MAP_ELITES:
#        self.me = MapElites(
#              evolver=self,
#              save_path=self.save_path
#              )
#        self.init_pop()
#           # Do MAP-Elites using qdpy

#     elif self.NEAT:
#        self.n_epoch = -1

#        stats = neat.statistics.StatisticsReporter()
#        self.neat_pop.add_reporter(stats)
#        self.neat_pop.add_reporter(neat.reporting.StdOutReporter(True))
#        winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)
      else:
         pass

   # Map agentID to policyID -- requires config global
   def mapPolicy(self, agentID):

      policy_id = agentID % self.config.NPOLICIES
      self.policy_to_agent[policy_id].append(agentID)
      return 'policy_{}'.format(policy_id)

   # Generate RLlib policies
   def createPolicies(self):
      config = self.config
      obs = observationSpace(config)
      atns = actionSpace(config)
      policies = {}

      for i in range(config.NPOLICIES):
         params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}

         key = self.mapPolicy(i)
         policies[key] = (None, obs, atns, params)

      return policies

   def update_fitness(self, individual, ALP):
      if self.config.PAIRED:
         protaganist_fitness, antagonist_fitness = [self.calc_fitness(individual.stats, pop=i) for i in range(2)]
         individual.score_hists.append(protaganist_fitness - antagonist_fitness)
      elif not self.MAP_TEST:
         individual.score_hists.append(self.calc_fitness(individual.stats))
      else:
         individual.score_hists.append(self.calc_fitness(individual=individual, config=self.config))

      if ALP:
         score = individual.score_hists[-2] - individual.score_hists[-1]

         if individual.ALPs is None:
            assert individual.age == 0
            individual.ALPs = []
         individual.ALPs.append(score)
#        score = abs(np.mean(individual.ALPs))
         score = np.mean(individual.ALPs)
         individual.fitness.setValues([-score])
      else:
         score = np.mean(individual.score_hists)

         if len(individual.score_hists) >= self.config.ROLLING_FITNESS:
            individual.score_hists = individual.score_hists[-self.config.ROLLING_FITNESS:]

         if ALP:
            if len(self.ALPs) >= self.config.ROLLING_FITNESS:
               individual.ALPs = individual.ALPs[-self.config.ROLLING_FITNESS:]

#        individual.fitness.values = [score]
      individual.fitness.setValues([-score])

   def update_features(self, individual):
      #     print('evaluating {}'.format(idx))
#     if not len(individual.features) == 0:
#        # assume features are fixed w.r.t. map: no need to recalculate except after mutation/mating
#        return

      features = self.calc_features(individual, self.config)
      individual.features = features
      #     features = [features[i] for i in self.me.feature_idxs]
      # FIXME: nip this in the bud
      #      features = [features[i] if i in features else 0 for i in self.feature_idxs]
#     individual.features = features
#     individual.features.setValues(features)
#     individual.features = Features(features)

     ##FIXME: would need to make sure this is resetting on clone
     #[individual.feature_hists[i].append(
     #   feat) if i in individual.feature_hists else individual.feature_hists.update(
     #   {i: [feat]}) for (i, feat) in enumerate(features)]
#    #individual.features = [np.mean(individual.feature_hists[i]) for i in range(len(features))]
     #individual.features.setValues([np.mean(individual.feature_hists[i]) for i in range(len(features))])

     #for (i, feat) in enumerate(features):
     #   if len(individual.feature_hists[i]) >= self.config.ROLLING_FITNESS:
     #      individual.feature_hists[i] = individual.feature_hists[i][-self.config.ROLLING_FITNESS:]

   def plot(self):
      plot_exp(self.config.EVO_DIR)

   def train_individuals(self, individuals):
      # if self.n_epoch % self.config.EVO_SAVE_INTERVAL == 0:
      #    self.saveMaps(self.container)
      #    if self.n_epoch > 0:
      #        self.plot()
      maps = dict([(ind.idx, ind.chromosome.map_arr) for ind in individuals])
#     if self.config.FROZEN and False:
#        stats = self.train_and_log_frozen(maps)
#     else:
      stats = self.train_and_log(maps)

      for ind in individuals:
         if not self.MAP_TEST and not ind.idx in stats:
            print('missing individual {} from training stats!'.format(ind.idx))
#           print(maps.keys())
#           print(stats.keys())
#           #FIXME: We'll try again for now.
            stats = self.train_and_log(maps)
#           stats = ray.get(self.global_stats.get.remote())

         if self.LEARNING_PROGRESS and len(ind.score_hists) < 2:
            stats = self.train_and_log(maps)
           #stats = ray.get(self.global_stats.get.remote())
            [setattr(i, 'stats', stats[i.idx]) for i in individuals]
            [self.update_fitness(i, ALP=False) for i in individuals]
            [self.update_features(i) for i in individuals]
         ind.stats = stats[ind.idx]
         self.update_fitness(ind, ALP=self.LEARNING_PROGRESS)
         self.update_features(ind)
#        ind.fitness.valid = True
         ind.age += 1
#        if not len(ind.score_hists) == ind.age:
#           T()
         #FIXME: what for?
#        self.idxs.add(idx)
#        if self.CPPN or self.CA or self.ALL_GENOMES:
            # Taken into account during CPPN reproduction in neat-python
            #NOTE: Assume single objective
         fitness = ind.fitness.getValues()
#        ind.chromosome.fitness = fitness

         if ind.idx in self.population:
             (game, old_score, age) = self.population[ind.idx]
             self.population[ind.idx] = (game, fitness, age)
         else:
             self.population[ind.idx] = (None, fitness, ind.age)

#     self.global_stats.reset.remote()

#  def train_and_log_frozen(self, maps):
#     from multiprocessing import Pipe, Process
#     processes = {}
#     n_proc = 0
#     if not hasattr(self, 'frozen_workers'):
#        self.frozen_workers = [self.make_env({'config':self.config}) for _ in range(self.config.N_EVO_MAPS)]
#        [self.frozen_workers[i].set_map(i, maps) for i in range(self.config.N_EVO_MAPS)]
#     for (g_hash, map_arr) in maps.items():
#        parent_conn, child_conn = Pipe()
#        game = self.frozen_workers[g_hash]
#        p = Process(target=self.simulate_game,
#                    args=(
#                       game,
#                       maps,
#                       self.n_sim_ticks,
#                       child_conn,
#                    ))
#        p.start()
#        processes[g_hash] = p, parent_conn, child_conn
#        #              # NB: specific to NMMO!!
#        #              # we simulate a bunch of envs simultaneously through the rllib trainer
#        #              self.simulate_game(game, map_arr, self.n_sim_ticks, g_hash=g_hash)
#        #              parent_conn, child_conn = None, None
#        #              processes[g_hash] = score, parent_conn, child_conn
#        #              for g_hash, (game, score, age) in population.items():
#        #                  try:
#        #                      with open(os.path.join('./evo_experiment', '{}'.format(self.config['config'].EVO_DIR), 'env_{}_skills.json'.format(g_hash))) as f:
#        #                          agent_skills = json.load(f)
#        #                          score = self.update_entropy_skills(agent_skills)
#        #                  except FileNotFoundError:
#        #                      raise Exception
#        #                      # hack
#        #                      score = None
#        #                      processes[g_hash] = score, parent_conn, child_conn
#        #                  self.population[g_hash] = (game, score, age)

#        #              n_proc += 1
#        #              break

#        if n_proc > 1 and n_proc % self.n_proc == 0:
#           self.join_procs(processes)

#     if len(processes) > 0:
#        self.join_procs(processes)

#     stats_list = [w.send_agent_stats() for w in self.frozen_workers]
#     stats = {}
#     for worker_stats in stats_list:
#        if not worker_stats: continue
#        for (envID, env_stats) in worker_stats:
#           if not env_stats: continue
#           if envID not in stats:
#              stats[envID] = env_stats
#           else:
#              for (k, v) in env_stats.items():
#                 if k not in stats[envID]:
#                    stats[envID][k] = v
#                 else:
#                    stats[envID][k] += v

#     return stats

   def train_and_log(self, maps):
      self.global_counter.set_idxs.remote([i for i in maps.keys()])
      if not self.MAP_TEST:
         stats = self.trainer.train(maps)
      else:
         stats = dict([(k, None) for k in maps.keys()])

      return stats

   def flush_elite(self, gi):
      # FIXME: we should know whether or not the genome is present in each of these structures

      if gi in self.population:
         self.population.pop(gi)

   def flush_individual(self, gi):
      self.g_idxs_reserve.add(gi)

      if gi in self.population:
         self.population.pop(gi)
#     self.maps.pop(gi)
#     self.score_hists.pop(gi)

#     if gi in self.chromosomes:
#        self.chromosomes.pop(gi)

#     if self.LEARNING_PROGRESS:
#        self.ALPs.pop(gi)


#  def gen_cppn_map(self, genome):
#     if genome.map_arr is not None and genome.multi_hot is not None:
#        return genome.map_arr, genome.multi_hot

#     cppn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
#       if self.config.NET_RENDER:
#          with open('nmmo_cppn.pkl', 'wb') a
#     multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_height), dtype=np.float)
#     map_arr = np.zeros((self.map_width, self.map_height), dtype=np.uint8)

#     for x in range(self.map_width):
#        for y in range(self.map_height):
#           # a decent scale for NMMO
#           x_i, y_i = x * 2 / self.map_width - 1, y * 2 / self.map_width - 1
#           x_i, y_i = x_i * 2, y_i * 2
#           v = cppn.activate((x_i, y_i))

#           if self.config.THRESHOLD:
#              # use NMMO's threshold logic
#              assert len(v) == 1
#              v = v[0]
#              v = self.map_generator.material_evo(self.config, v)
#           else:
#              # CPPN has output channel for each tile type; take argmax over channels
#              # also a spawn-point tile
#              assert len(v) == self.n_tiles
#              multi_hot[:, x, y] = v
#              # Shuffle before selecting argmax to prevent bias for certain tile types in case of ties
#              v = np.array(v)
#              v = np.random.choice(np.flatnonzero(v == v.max()))
#              v = np.argmax(v)
#              map_arr[x, y] = v
#     map_arr = self.validate_map(map_arr, multi_hot)
#     genome.map_arr = map_arr
#     genome.multi_hot = multi_hot

#     return map_arr, multi_hot



   def neat_eval_fitness(self, genomes, neat_config):
      if self.n_epoch == 0:
         self.last_map_idx = -1
         self.last_fitnesses = {}

      if self.reloading:
         # Turn off reloading after 1 epoch.

         if not self.epoch_reloaded:
            self.epoch_reloaded = self.n_epoch
         elif self.epoch_reloaded < self.n_epoch + 11:
            self.reloading = False

      # Remove old individuals from score history to make way for new ones
      maps = {}
#     global_counter = ray.get_actor("global_counter")
      neat_idxs = set([idx for idx, _ in genomes])
      # the g_idxs that end up being relevant
      g_idxs = self.g_idxs_reserve
      g_idxs_out = set()
      new_g_idxs = set()
      neat_to_g = {}
      skip_idxs = set()

      for idx, g in genomes:
#        if self.n_epoch == 0 or idx not in self.last_fitnesses:
#            self.last_fitnesses[idx] = []
         # neat-python indexes starting from 1

         if idx in self.neat_to_g:
             g_idx = self.neat_to_g[idx]
             neat_to_g[idx] = g_idx
            #g_idxs.remove(g_idx)
             (map_arr, multi_hot), atk_mults = self.maps[g_idx]
         else:
            g_idx = g_idxs.pop()
            new_g_idxs.add(g_idx)
            neat_to_g[idx] = g_idx
            self.maps[g_idx] = (None, g.atk_mults)

            map_arr, multi_hot = self.gen_cppn_map(self, g)

           #if idx <= self.last_map_idx and not self.reloading:
           #   continue
           #map_arr = self.add_border(map_arr)
            # Impossible maps are no good
            tile_counts = np.bincount(map_arr.reshape(-1))

            if False and (len(tile_counts) <= material.All.materials.FOREST.index or \
                  tile_counts[material.All.materials.FOREST.index] <= self.config.NENT * 3 or \
                  tile_counts[material.All.materials.WATER.index] <= self.config.NENT * 3):
               print('map {} rejected for lack of food and water'.format(g_idx))
               g.fitness = 0
               neat_idxs.remove(idx)
               skip_idxs.add(g_idx)
            #  self.maps.pop(g_idx)
#           self.maps[g_idx] = (map_arr, multi_hot), g.atk_mults
            self.maps[g_hash] = (map_arr, atk_mults)
         g_idxs_out.add(g_idx)
         maps[g_idx] = map_arr, g.atk_mults
      # remove dead guys

      for ni, gi in self.neat_to_g.items():
          if ni not in neat_to_g:
             self.flush_individual(gi)
      self.neat_to_g = neat_to_g
      self.maps = maps
      neat_idxs = list(neat_idxs)
      self.last_map_idx = neat_idxs[-1]
      g_idxs_envs = list(g_idxs_out)
      np.random.shuffle(g_idxs_envs)
#     global_counter.set_idxs.remote(g_idxs_envs)
      if self.n_epoch % self.config.EVO_SAVE_INTERVAL == 0:
         self.saveMaps(maps, new_g_idxs)
#     global_stats = self.global_stats
#     self.send_genes(global_stats)
      _ = self.train_and_log(maps)

      if self.LEARNING_PROGRESS: # and self.n_epoch == 0:
          _ = self.train_and_log()
#     stats = ray.get(global_stats.get.remote())
#     headers = ray.get(global_stats.get_headers.remote())
#    #n_epis = train_stats['episodes_this_iter']
#    #assert n_epis == self.n_pop
#     for g_idx in g_idxs_out:
#        g_idx = neat_to_g[idx]
#        #FIXME: hack

#        if g_idx in skip_idxs:
#           continue

#        if g_idx not in stats:
#           print('Missing stats for map {}, training again.'.format(g_idx))
#           print('Missing stats for map {}, using old stats.'.format(g_idx))
#           self.trainer.train()
#           stats = ray.get(global_stats.get.remote())
#           continue

#     last_fitnesses = self.last_fitnesses
#     new_fitness_hist = {}

      for idx, g in genomes:
         g_idx = neat_to_g[idx]
         #FIXME: hack

        #if g_idx not in stats:
        #   # do something clever?

        #   continue

         if g_idx not in g_idxs_out:
            score = 0
         else:
           if g_idx not in self.score_hists:
              _ = self.train_and_log()
#           if 'skills' not in stats[g_idx]:
#              score = 0
#           else:
           score = self.score_hists[g_idx][-1]
#              score = self.calc_fitness(stats[g_idx], skill_headers=self.config.SKILLS, verbose=self.config.EVO_VERBOSE)
        #self.population[g_hash] = (None, score, None)
#        print('Map {}, diversity score: {}\n'.format(idx, score))
#        last_fitness = last_fitnesses[idx]
#        last_fitness.append(score)
         if self.LEARNING_PROGRESS:
            if len(self.score_hists[g_idx]) <2:
                #FIXME: SHOULD NOT be happening FUCK
                _ = self.train_and_log()
         score = self.get_score(g_idx)

         g.fitness = score
         g.age += 1
#        new_fitness_hist[idx] = last_fitness
#        self.score_hists[g_idx] = new_fitness_hist[idx]
         self.population[g_idx] = (None, score, g.age)
#     self.last_fitnesses = new_fitness_hist
#     global_stats.reset.remote()

      if self.reloading:
         self.reloading = False

      if (self.n_epoch % 10 == 0 or self.n_epoch == 0) and not self.reloading:
         self.save()
      self.log()
#     self.neat_pop.reporters.reporters[0].save_genome_fitness(
#           delimiter=',',
#           filename=os.path.join(self.save_path, 'genome_fitness.csv'))
      self.n_epoch += 1

   def log(self, verbose=False):
      super().log([(ind.idx, None, ind.fitness.values[0], ind.age) for ind in self.container], verbose=verbose)


   def evolve(self):
     #if self.MAP_ELITES:
     #   self.evolve()

     #elif self.NEAT:
     #   if self.n_epoch == -1:
     #      self.init_pop()
     #   else:
     #      self.map_generator = MapGenerator(self.config)
     #   winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)
     #else:
      return super().evolve()



   def make_game(self, child_map):
      config = self.config
     #config['map_arr'] = child_map
      game = self.make_env(config)

      return game

   def restore(self, trash_trainer=False, inference=False):
      '''
      '''
      if self.MAP_TEST:
         return
      self.calc_fitness = diversity_calc(self.config)
      self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.EVO_DIR, 'maps', 'map')

#     self.global_counter.set_idxs.remote(range(self.config.N_EVO_MAPS))

      if inference:
         #TODO: avoid loading redundant envs, provide policy rather than trainer to Evaluator if possible
         pass
      global TRAINER
      self.trainer = TRAINER

      if self.config.GRIDDLY and False:
         model_config = {
               'conv_filters': [
                  [64, (7, 7), 1],
#      #          [3, (3, 3), 1],
#      #          [3, (3, 3), 1],
                  ],
               }
         griddly_config = {
               }
         multiagent_config = {}
      else:
         model_config = {
                'custom_model': 'test_model',
                'custom_model_config': {
                    'config':
                    self.config
                }}
         griddly_config = {}
         multiagent_config = {
            "policy_mapping_fn": self.mapPolicy,
         }

      if self.trainer is None or trash_trainer:
         self.trainer = None
         del(self.trainer)

         # Create policies
#        policies = createPolicies(self.config)
         policies = self.createPolicies()
         multiagent_config['policies'] = policies

         conf = self.config
         model_path = os.path.join(self.save_path, 'models')
         try:
            os.mkdir(model_path)
         except FileExistsError:
            print('Model directory already exists.',model_path)
         # Instantiate monolithic RLlib Trainer object.
         if self.config.RENDER:
            num_workers = 0
            self.config.N_EVO_MAPS = 0
            sgd_minibatch_size = 0
#           train_batch_size = 256 * num_workers
         elif self.config.N_EVO_MAPS == 1:
            num_workers = self.config.N_PROC
            sgd_minibatch_size = 100
#           train_batch_size = 256 * num_workers
         else:
            num_workers = self.config.N_PROC
            sgd_minibatch_size = 128
#           train_batch_size = 256 * num_workers
#        sgd_minibatch_size = min(512, train_batch_size)
      # EvoPPOTrainer = build_trainer(
       #     name="EvoPPO",
       #    #name="PPO",
       #    #default_config=DEFAULT_CONFIG,
       #    #validate_config=validate_config,
       #    #default_policy=PPOTFPolicy,
       #    #get_policy_class=get_policy_class,
       #     execution_plan=frozen_execution_plan,
       # )
         if self.config.RENDER or self.config.FROZEN:
            evaluation_interval = 1
            evaluation_num_workers = self.config.N_PROC
#           evaluation_num_episodes = self.config.N_EVO_MAPS
#           evaluation_num_episodes = None
            evaluation_config = {
               'evaluation_interval': evaluation_interval,
               #           'evaluation_num_episodes': evaluation_num_episodes,
               'evaluation_num_workers': evaluation_num_workers,
               'batch_mode': 'truncate_episodes',
#              'worker_config': {
#                 'batch_mode': 'thisgoesnowherehahafuck',
#                 }
            }
         else:
            evaluation_config = {
            }
         trainer = EvoPPOTrainer(
            execution_plan=frozen_execution_plan,
            env="custom",
            path=model_path,
            config={
            'num_workers': num_workers, # normally: 4
           #'num_gpus_per_worker': 0.083,  # hack fix
            'num_gpus_per_worker': 0,
            'num_gpus': 1,
           #'num_envs_per_worker': int(conf.N_EVO_MAPS / max(1, num_workers)),
            'num_envs_per_worker': 1,
            # batch size is n_env_steps * maps per generation
            # plus 1 to actually reset the last env
#           'train_batch_size': conf.MAX_STEPS * conf.N_EVO_MAPS, # normally: 4000
            'train_batch_size': 1200,
           #'train_batch_size': train_batch_size,
            'rollout_fragment_length': 100,
           #'lstm_bptt_horizon': 16,
            'sgd_minibatch_size': sgd_minibatch_size,  # normally: 128
            'num_sgd_iter': 1,
            'monitor': False,
            'framework': 'torch',
            'horizon': np.inf,
            'soft_horizon': False,
            # '_use_trajectory_view_api': False,
            'no_done_at_end': False,
            'callbacks': LogCallbacks,
            'env_config': {
                'config':
                self.config,
            },

            'model': model_config,
            'multiagent': multiagent_config,
            **evaluation_config,
            **griddly_config
            }

#           'optimizer': {
#              'frozen': True,
#              },
#           'multiagent': {
#              'policies_to_train': [],
#              },
          )

         # Print model size

#        utils.modelSize(trainer.defaultModel())
         trainer.restore(self.config.MODEL)

         TRAINER = trainer
         self.trainer = trainer

   def infer(self):
      ''' Do inference, just on the top individual for now.'''
#     global_counter = ray.get_actor("global_counter")
#     self.send_genes(self.global_stats)
     #best_g, best_score = None, -999

     #for g_hash, (_, score, age) in self.population.items():
     #    if score and score > best_score and age > self.mature_age:
     #        print('new best', score)
     #        best_g, best_score = g_hash, score

     #if not best_g:
     #    raise Exception('No population found for inference.')
     #print("Loading map {} for inference.".format(best_g))
      best_g = self.config.INFER_IDX
#     global_counter.set.remote(best_g)
      self.config.EVALUATE = True

      if not self.trainer:
         self.restore()
      evaluator = RLLibEvaluator(self.config, self.trainer, archive=self.container, createEnv=self.make_env)
      evaluator.render()


   def mate_cppns(self, ind_1, ind_2):
      g1 = ind_1.chromosome
      g2 = ind_2.chromosome
      gid_1, gid_2 = ind_1.idx, ind_2.idx
      #FIXME redundant since these are already clones, working around assetion in neat-python mutation
      g1_new = self.neat_config.genome_type(gid_1)
      g2_new = self.neat_config.genome_type(gid_2)
      g1_new.configure_crossover(g1, g2, self.neat_config.genome_config)
      g2_new.configure_crossover(g1, g2, self.neat_config.genome_config)
      ind_1.chromosome = g1_new
      ind_2.chromosome = g2_new

      if not hasattr(ind_1.fitness, 'values'):
         ind_1.fitness.values = None
      ind_1.fitness.valid = False

      if not hasattr(ind_2.fitness, 'values'):
         ind_2.fitness.values = None
      ind_2.fitness.valid = False

      return ind_1, ind_2

#  def mutate_cppn(self, ind_1):
#     g1 = ind_1.chromosome
#     g1.mutate(self.neat_config.genome_config)
#     ind_1.chromosome = g1
#     self.gen_cppn_map(g1)

#     if not hasattr(ind_1.fitness, 'values'):
#        ind_1.fitness.values = None
#     ind_1.fitness.valid = False

#     return (ind_1, )

   def genRandMap(self, g_hash):
      ind = EvoIndividual(iterable=[], rank=g_hash, evolver=self)
      chrom = ind.chromosome

      return chrom.map_arr, chrom.atk_mults

#     print('genRandMap {}'.format(g_hash))

#     if self.n_epoch > 0 or self.reloading:
#        print('generating new random map when I probably should not be...')

#     map_arr= np.random.randint(1, self.n_tiles,
#                                 (self.map_width, self.map_height))
#     if self.NEAT:
#        return None
#  #     raise Exception('CPPN maps should be generated inside neat_eval_fitness function if using NEAT')
#     if self.PATTERN_GEN:
#        if g_hash is None:
#           g_hash = ray.get(self.global_counter.get.remote())
#        chromosome = self.Chromosome(
#              self.map_width,
#              self.n_tiles,
#              self.max_primitives,
#              enums.Material.GRASS.value.index)
#        map_arr, multi_hot = chromosome.generate()
#        map_arr = self.validate_map(map_arr, multi_hot)
#        chromosome.flat_map = map_arr
#        atk_mults = self.gen_mults()
#        self.chromosomes[g_hash] = chromosome, atk_mults
#     elif self.CPPN:
#        if g_hash is None:
#           g_hash = ray.get(self.global_counter.get.remote())
#        chromosome = self.neat_pop.population[g_hash+1]
#        map_arr, multi_hot = self.gen_cppn_map(chromosome)
#        atk_mults = self.gen_mults()
#        chromosome.atk_mults = atk_mults
#        self.chromosomes[g_hash] = chromosome

#     elif self.CA:
#        pass

#     else:
#        # FIXME: hack: ignore lava
#        map_arr = np.random.choice(np.arange(0, self.n_tiles), (self.map_width, self.map_height),
#              p=self.TILE_PROBS[:])
#        map_arr = self.validate_map(map_arr)
#        atk_mults = self.gen_mults()

#        self.add_border(map_arr)

#        #FIXME: hack for map elites

#     if self.MAP_ELITES:
#        return g_hash, map_arr, atk_mults
#     else:
#        return map_arr, atk_mults


#  def mutate(self, g_hash, par_hash):
#     if self.CPPN:
#        raise Exception('CPPN-generated maps should be mutated inside NEAT code.')
#     elif self.PATTERN_GEN:
#        chromosome, atk_mults = copy.deepcopy(self.chromosomes[par_hash])
#        map_arr, multi_hot = chromosome.mutate()
#        self.validate_map(map_arr, multi_hot)
#        chromosome.flat_map = map_arr
#        self.chromosomes[g_hash] = chromosome, atk_mults
#     else:
#        map_arr, atk_mults = self.maps[par_hash]
#        map_arr = map_arr.copy()

#        for i in range(random.randint(0, self.n_mutate_actions)):
#           x= random.randint(0, self.map_width - 1)
#           y= random.randint(0, self.map_height - 1)
#           # FIXME: hack: ignore lava
#           t= np.random.randint(1, self.n_tiles)
#           map_arr[x, y]= t
#  #     map_arr = self.add_border(map_arr)
#        self.validate_map(map_arr)
#     atk_mults = self.mutate_mults(atk_mults)

#     return map_arr, atk_mults


   def add_spawn(self, g_hash, map_arr):
      self.add_border(map_arr, enums.Material.GRASS.value.index)
      idxs = map_arr == self.SPAWN_IDX
      spawn_points = np.vstack(np.where(idxs)).transpose()

      if len(spawn_points) == 0:
         self.add_border(map_arr, self.SPAWN_IDX)
#        spawn_points = [(b, j) for j in range(map_arr.shape[1])] + [(i, b) for i in range(map_arr.shape[0])] + [(i, -b) for i in range(map_arr.shape[0])] + [(-b, j) for j in range(map_arr.shape[1])]
     #else:
     #   map_arr[idxs] = enums.Material.GRASS.value.index
     #self.global_stats.add_spawn_points.remote(g_hash, spawn_points)
#     return spawn_points

   def update_max_skills(self, ent):
       skills = ent.skills.packet()

       for s, v in skills.items():
           if s == 'level':
               continue
           exp= v['exp']

           if s not in self.max_skills:
               self.max_skills[s]= exp
           else:
               if exp > self.max_skills[s]:
                   self.max_skills[s]= exp

   def update_specialized_skills(self, ent):
       skills = ent.skills.packet()

       n_skills = 0
       max_skill = None, -999
       total_xp = 0

       for s, v in skills.items():
           if s == 'level':
               continue
           n_skills += 1
           xp= v['exp']

           if xp > max_skill[1]:
               max_skill = s, xp
           total_xp += xp

       max_xp = max_skill[1]
       total_xp -= max_xp
       # subtract average in other skills from top skill score
       # max_xp is always nonnegative
       max_xp -= (total_xp / (n_skills - 1))
       skill = max_skill[0]

       if skill not in self.max_skills:
           self.max_skills[skill] = max_xp
       else:
           if xp > self.max_skills[skill]:
               self.max_skills[skill] = max_xp

   def simulate_game(self, game, maps, conn=None, g_hash=None):
      game.set_map(game.worldIdx, maps)
      ob = game.reset()
      for t in range(self.config.MAX_STEPS):

          actions, self.state, _ = self.trainer.compute_actions(
             ob)
          ob, _, _, _ = game.step(actions)



#      score = 0
#      score += self.tick_game(game, g_hash=g_hash)

#      if conn:
#          conn.send(score)

#      return score

   def run(self, game, map_arr):
       self.obs= game.reset(#map_arr=map_arr,
               idx=0)
       self.game= game
       from forge.trinity.twistedserver import Application
       Application(game, self.tick).run()

#  def send_genes(self, global_stats):
#     ''' Send (some) gene information to a global object to be retrieved by parallel environments.
#     '''

#     for g_hash, gene in self.maps.items():
#     for g_hash, gene in self.maps.items():
#        atk_mults = gene[-1]
#        global_stats.add_mults.remote(g_hash, atk_mults)

   def evolve_generation(self):
#     global_stats = self.global_stats
#     self.send_genes(global_stats)
      stats = self.trainer.train(maps=dict([(i, None) for i in range(self.config.N_EVO_MAPS)]))
#     stats = self.stats
     #headers = ray.get(global_stats.get_headers.remote())
#     n_epis = train_stats['episodes_this_iter']

#     if n_epis == 0:
#        print('Missing simulation stats. I assume this is the 0th generation? Re-running the training step.')

#        return self.evolve_generation()

#     global_stats.reset.remote()

      for g_hash in self.population.keys():
         if g_hash not in stats:
            print('Missing simulation stats for map {}. I assume this is the 0th generation, or re-load? Re-running the training step.'.format(g_hash))
            stats = self.trainer.train()
#           stats = self.stats

            break

      for g_hash, (game, score, age) in self.population.items():
        #print(self.config.SKILLS)
         score = self.calc_fitness(stats[g_hash], verbose=False)
         self.population[g_hash] = (game, score, age)

      for g_hash, (game, score_t, age) in self.population.items():
         # get score from latest simulation
         # cull score history

        #if len(self.score_hists[g_hash]) >= self.config.ROLLING_FITNESS:
        #   self.score_hists[g_hash] = self.score_hists[g_hash][-self.config.ROLLING_FITNESS:]
#           while len(self.score_hists[g_hash]) >= self.config.ROLLING_FITNESS:
#              self.score_hists[g_hash].pop(0)
          # hack

         #if score_t is None:
         #    score_t = 0
        #else:
         self.score_hists[g_hash].append(score_t)
         self.score_hists[g_hash] = self.score_hists[g_hash][-self.config.ROLLING_FITNESS:]

         if self.LEARNING_PROGRESS:
             score = self.score_hists[-1] - self.score_hists[0]
         else:
             score = np.mean(self.score_hists[g_hash])
         game, _, age = self.population[g_hash]
         self.population[g_hash] = (game, score, age + 1)
      super().mutate_gen()

   def tick_game(self, game, g_hash=None):
       return self.tick(game=game, g_hash=g_hash)

   def tick(self, game=None, g_hash=None):
       # check if we are doing inferenc
       FROZEN = True

       if FROZEN:
           if game is None:
               game = self.game
#              update_entropy_skills(game.desciples.values())

          #if self.n_tick > 15:
          #    self.obs = game.reset()
          #    self.done = {}
          #    self.n_tick = 0
           reward= 0
           # Remove dead agents

           for agentID in self.done:
               if self.done[agentID]:
                   assert agentID in self.obs
                   del self.obs[agentID]

                  #if agentID in game.desciples:
                  #    print('update max score during tick')
                  #    ent= game.desciples[agentID]
                  #    self.update_max_skills(ent)
                   reward -= 1
           # Compute batch of actions
               if self.config.GRIDDLY and False:
                  actions, self.state, _= self.trainer.compute_actions(
                      self.obs, state=self.state, policy_id='default_policy')
               else:
                  actions, self.state, _= self.trainer.compute_actions(
                      self.obs, state=self.state, policy_id='policy_0')

               # Compute overlay maps
               self.overlays.register(self.obs)

               # Step the environment
               self.obs, rewards, self.done, _= game.step(actions)
       else:
          #self.trainer.reset()
           print(dir(self.trainer))
          #self.trainer.train(self.maps[g_hash])
           stats = self.trainer.train()
           print('evo map trainer stats', stats)
       self.n_tick += 1
       reward = 0

       return reward

   def save(self):
      population = copy.deepcopy(self.population)
      self.population = None
#     for g_hash in self.population:
#         game, score, age= self.population[g_hash]
#         # FIXME: something weird is happening after reload. Not the maps though.
#         #   so for now, trash score history and re-calculate after reload
#         self.population[g_hash]= None, score, age
#        #self.population[g_hash]= None, score, age
      if not self.MAP_TEST:
         self.trainer.save()
      global TRAINER
      TRAINER = self.trainer
      self.trainer= None
      self.game= None
      global_counter = self.global_counter
      self.global_counter = None
      # map_arr = self.maps[g_hash]
      if os.path.exists(self.evolver_path):
         copyfile(self.evolver_path, self.evolver_path + '.bkp')
      with open(self.evolver_path, 'wb') as save_file:
         pickle.dump(self, save_file)
      self.global_counter = global_counter
      self.population = population
      self.restore(trash_trainer=False)
      raise Exception

   def init_pop(self):
       if not self.config.PRETRAINED and not self.reloading:
           # train from scratch
          self.config.MODEL = None
       elif self.reloading:
           # use pre-trained model
          self.config.MODEL = 'reload'
       elif self.config.FROZEN:
          pass
       else:
          self.config.MODEL = 'current'
       # load the model and initialize and save the population
       super().init_pop()
#      self.saveMaps(self.maps, list(self.maps.keys()))
       # reload latest model from evolution moving forward
       self.config.MODEL = 'reload'
