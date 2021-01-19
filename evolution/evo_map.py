import copy
import csv
import json
import os
import pickle
import random
import sys
import time
from pdb import set_trace as T
from shutil import copyfile
from typing import Dict

import numpy as np
import ray
import scipy
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import neat
import neat.nn
import projekt
from evolution.lambda_mu import LambdaMuEvolver
from evolution.pattern_map_elites import MapElites
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums
from forge.ethyr.torch import utils
from pcg import TILE_PROBS, TILE_TYPES
#from projekt import rlutils
#from projekt.evaluator import Evaluator
from projekt.rllib_wrapper import RLLibEvaluator, EvoPPOTrainer, LogCallbacks, observationSpace, actionSpace, frozen_execution_plan
from pureples.shared.visualize import draw_net
from plot_evo import plot_exp
from evolution.diversity import diversity_calc
from forge.trinity.overlay import OverlayRegistry
from ray.rllib.agents.trainer_template import build_trainer
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize,
                    linewidth=120,
                    suppress=True,
                   #precision=10
                    )

# keep trainer here while saving evolver object
global TRAINER
TRAINER = None

MELEE_MIN = 0.4
MELEE_MAX = 1.4
MAGE_MIN = 0.6
MAGE_MAX = 1.6
RANGE_MIN = 0.2
RANGE_MAX = 1

# FIXME: backward compatability, we point to correct function on restore()
def calc_convex_hull():
   pass

def calc_differential_entropy():
   pass

def calc_discrete_entropy_2():
   pass

def calc_convex_hull():
   pass

def calc_diversity_l2():
   pass

def calc_map_diversity():
   T()

class SpawnPoints():
   def __init__(self, map_width, n_players):
      self.max_spawns = n_players * 3
      n_spawns = np.random.randint(n_players, self.max_spawns)
      self.spawn_points = np.random.randint(0, map_width (2, n_spawns))

   def mutate(self):
      n_spawns = len(self.spawn_points)
      n_delete = np.random.randint(0, 5)
      n_add = np.random.randint(0, 5)


#NOTE: had to move this multiplier stuff outside of the evolver so the CPPN genome could incorporate them
# without pickling error.
#TODO: move these back into evolver class?
def gen_atk_mults():
   # generate melee, range, and mage attack multipliers for automatic game-balancing
  #atks = ['MELEE_MULT', 'RANGE_MULT', 'MAGE_MULT']
  #mults = [(atks[i], 0.2 + np.random.random() * 0.8) for i in range(3)]
  #atk_mults = dict(mults)
   # range is way too dominant, always

   atk_mults = {
         # b/w 0.2 and 1.0
         'MELEE_MULT': np.random.random() * (MELEE_MAX - MELEE_MIN) + MELEE_MIN,
         'MAGE_MULT': np.random.random() * (MAGE_MAX - MAGE_MIN) + MAGE_MIN,
         # b/w 0.0 and 0.8
         'RANGE_MULT': np.random.random() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN,
         }

   return atk_mults

def mutate_atk_mults(atk_mults):
   rand = np.random.random()

   if rand < 0.2:
      atk_mults = gen_atk_mults()
   else:
      atk_mults = {
         'MELEE_MULT': max(min(atk_mults['MELEE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                               MELEE_MAX), MELEE_MIN),
         'MAGE_MULT': max(min(atk_mults['MAGE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                              MAGE_MAX), MAGE_MIN),
         'RANGE_MULT': max(min(atk_mults['RANGE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                               RANGE_MAX), RANGE_MIN),
            }

   return atk_mults


def mate_atk_mults(atk_mults_0, atk_mults_1, single_offspring=False):
   new_atk_mults_0, new_atk_mults_1 = {}, {}

   for k, v in atk_mults_0.items():
      if np.random.random() < 0.5:
         new_atk_mults_0[k] = atk_mults_1[k]
      else:
         new_atk_mults_0[k] = atk_mults_0[k]

      if single_offspring:
         continue

      if np.random.random() < 0.5:
         new_atk_mults_1[k] = atk_mults_0[k]
      else:
         new_atk_mults_1[k] = atk_mults_1[k]

   return new_atk_mults_0, new_atk_mults_1



def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    k_lrg = np.column_stack(np.unravel_index(idx, a.shape))
    T()

    return k_lrg

def sigmoid_lifespan(x):
   res = 1 / (1 + np.exp(0.1*(-x+50)))
#  res = scipy.special.softmax(res)

   return res

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


# Map agentID to policyID -- requires config global
def mapPolicy(agentID,
        #config
        ):

    return 'policy_{}'.format(agentID % 1)


# Generate RLlib policies
def createPolicies(config):
    obs = observationSpace(config)
    atns = actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i
                #, config
                )
        policies[key] = (None, obs, atns, params)

    return policies


class DefaultGenome(neat.genome.DefaultGenome):
   evolver = None
   ''' A wrapper class for a NEAT genome, which smuggles in other evolvable params for NMMO,
   beyond the map.'''
   def __init__(self, key):
      super().__init__(key)
      self.atk_mults = gen_atk_mults()
      self.age = 0

   def configure_crossover(self, parent1, parent2, config):
      super().configure_crossover(parent1, parent2, config)
      mults_1, mults_2 = parent1.atk_mults, parent2.atk_mults
      self.atk_mults, _ = mate_atk_mults(mults_1, mults_2, single_offspring=True)

   def mutate(self, config):
      super().mutate(config)
      self.atk_mults = mutate_atk_mults(self.atk_mults)
      self.age += 1



class EvolverNMMO(LambdaMuEvolver):
   def __init__(self, save_path, make_env, trainer, config, n_proc=12, n_pop=12,):
      self.gen_mults = gen_atk_mults
      self.mutate_mults = mutate_atk_mults
      self.mate_mults = mate_atk_mults
      config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', config.EVO_DIR, 'maps', 'map')
      super().__init__(save_path, n_proc=n_proc, n_pop=n_pop)
      self.lam = 1/4
      self.mu = 1/4
      self.make_env = make_env
      self.trainer = trainer
      self.map_width = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
      self.map_height = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
      self.n_tiles = len(TILE_TYPES)
      self.mature_age = config.MATURE_AGE
      self.state = {}
      self.done = {}
      self.map_generator = MapGenerator(config)
      self.config = config
      self.skill_idxs = {}
      self.idx_skills = {}
      self.reloading = False  # Have we just reloaded? For re-saving elite maps when reloading.
      self.epoch_reloaded = None
      self.global_stats = ray.get_actor("global_stats")
      self.global_counter = ray.get_actor("global_counter")
      self.CPPN = config.GENOME == 'CPPN'
      self.PATTERN_GEN = config.GENOME == 'Pattern'
      self.RAND_GEN = config.GENOME == 'Random'
      self.LEARNING_PROGRESS = config.FITNESS_METRIC == 'ALP'
      self.calc_diversity = diversity_calc(config)

      if not (self.CPPN or self.PATTERN_GEN or self.RAND_GEN):
         raise Exception('Invalid genome')
      self.LAMBDA_MU = config.EVO_ALGO == 'Simple'
      self.MAP_ELITES = config.EVO_ALGO == 'MAP-Elites'

      if self.CPPN or self.MAP_ELITES:
         # g_idxs we might use
         self.g_idxs_reserve = set([i for i in range(self.n_pop*10)])

      if self.PATTERN_GEN:
         self.max_primitives = self.map_width**2 / 4

         from evolution.paint_terrain import Chromosome
         self.Chromosome = Chromosome
         self.chromosomes = {}
         self.global_counter.set_idxs.remote(range(self.config.N_EVO_MAPS))
         # TODO: generalize this for tile-flipping, then NEAT
         if self.MAP_ELITES:
            self.me = MapElites(
                  evolver=self,
                  save_path=self.save_path
                  )
            self.init_pop()
            # Do MAP-Elites using qdpy

      elif self.CPPN:
         self.neat_to_g = {}
         self.n_epoch = -1
         evolver = self
         self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_nmmo')
         self.neat_config.fitness_threshold = np.float('inf')
         self.neat_config.pop_size = self.config.N_EVO_MAPS
         self.neat_config.elitism = int(self.lam * self.config.N_EVO_MAPS)
         self.neat_config.survival_threshold = self.mu
         self.neat_pop = neat.population.Population(self.neat_config)
         stats = neat.statistics.StatisticsReporter()
         self.neat_pop.add_reporter(stats)
         self.neat_pop.add_reporter(neat.reporting.StdOutReporter(True))
#        winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)
      else:
         pass


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

      maps = {}
      global_counter = ray.get_actor("global_counter")
      neat_idxs = set([idx for idx, _ in genomes])
      # the g_idxs that end up being relevant
      g_idxs = self.g_idxs_reserve
      g_idxs_out = set()
      new_g_idxs = set()
      neat_to_g = {}
      skip_idxs = set()

      for idx, g in genomes:
         if self.n_epoch == 0 or idx not in self.last_fitnesses:
             self.last_fitnesses[idx] = []
         # neat-python indexes starting from 1
         if idx in self.neat_to_g:
             g_idx = self.neat_to_g[idx]
             neat_to_g[idx] = g_idx
            #g_idxs.remove(g_idx)
             (map_arr, multi_hot), atk_mults = self.genes[g_idx]
         else:
            g_idx = g_idxs.pop()
            new_g_idxs.add(g_idx)   
            neat_to_g[idx] = g_idx
            self.genes[g_idx] = (None, g.atk_mults)

           #if idx <= self.last_map_idx and not self.reloading:
           #   continue
            cppn = neat.nn.FeedForwardNetwork.create(g, self.neat_config)
    #       if self.config.NET_RENDER:
    #          with open('nmmo_cppn.pkl', 'wb') a
            multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_height), dtype=np.float)
            map_arr = np.zeros((self.map_width, self.map_height), dtype=np.uint8)

            for x in range(self.map_width):
               for y in range(self.map_height):
                  # a decent scale for NMMO
                  x_i, y_i = x * 2 / self.map_width - 1, y * 2 / self.map_width - 1
                  x_i, y_i = x_i * 2, y_i * 2
                  v = cppn.activate((x_i, y_i))

                  if self.config.THRESHOLD:
                     # use NMMO's threshold logic
                     assert len(v) == 1
                     v = v[0]
                     v = self.map_generator.material_evo(self.config, v)
                  else:
                     # CPPN has output channel for each tile type; take argmax over channels
                     # also a spawn-point tile
                     assert len(v) == self.n_tiles
                     multi_hot[:, x, y] = v
                     v = np.argmax(v)
                     map_arr[x, y] = v
            self.validate_spawns(map_arr, multi_hot)
           #map_arr = self.add_border(map_arr)
            # Impossible maps are no good
            tile_counts = np.bincount(map_arr.reshape(-1))

            if False and (len(tile_counts) <= enums.Material.FOREST.value.index or \
                  tile_counts[enums.Material.FOREST.value.index] <= self.config.NENT * 3 or \
                  tile_counts[enums.Material.WATER.value.index] <= self.config.NENT * 3):
               print('map {} rejected for lack of food and water'.format(g_idx))
               g.fitness = 0
               neat_idxs.remove(idx)
               skip_idxs.add(g_idx)
            #  self.genes.pop(g_idx)
            self.genes[g_idx] = (map_arr, multi_hot), g.atk_mults
         g_idxs_out.add(g_idx)
         maps[g_idx] = map_arr, g.atk_mults
      # remove dead guys
      for ni, gi in self.neat_to_g.items():
          if ni not in neat_to_g:
              self.g_idxs_reserve.add(gi)
              self.population.pop(gi)
              self.genes.pop(gi)
      self.neat_to_g = neat_to_g
      self.maps = maps
      neat_idxs = list(neat_idxs)
      self.last_map_idx = neat_idxs[-1]
      g_idxs_envs = list(g_idxs_out)
      np.random.shuffle(g_idxs_envs)
      global_counter.set_idxs.remote(g_idxs_envs)
      self.saveMaps(maps, new_g_idxs)
      global_stats = self.global_stats
      self.send_genes(global_stats)
      train_stats = self.trainer.train()
      stats = ray.get(global_stats.get.remote())
#     headers = ray.get(global_stats.get_headers.remote())
     #n_epis = train_stats['episodes_this_iter']
     #assert n_epis == self.n_pop
      for g_idx in g_idxs_out:
#        g_idx = neat_to_g[idx]
         #FIXME: hack

         if g_idx in skip_idxs:
            continue

         if g_idx not in stats:
#           T()
            print('Missing stats for map {}, training again.'.format(g_idx))
#           print('Missing stats for map {}, using old stats.'.format(g_idx))
            self.trainer.train()
            stats = ray.get(global_stats.get.remote())
            continue

      last_fitnesses = self.last_fitnesses
      new_fitness_hist = {}

      for idx, g in genomes:
         g_idx = neat_to_g[idx]
         #FIXME: hack

        #if g_idx not in stats:
        #   # do something clever?

        #   continue

         if g_idx not in g_idxs_out:
            score = 0
         else:
            if 'skills' not in stats[g_idx]:
               score = 0
#              T()
            else:
               score = self.calc_diversity(stats[g_idx], skill_headers=self.config.SKILLS, verbose=self.config.EVO_VERBOSE)
        #self.population[g_hash] = (None, score, None)
#        print('Map {}, diversity score: {}\n'.format(idx, score))
         last_fitness = last_fitnesses[idx]
         last_fitness.append(score)
         if self.LEARNING_PROGRESS:
             g.fitness = score = (last_fitness[-1] - last_fitness[0]) / len(last_fitness)
         else:
             g.fitness = np.mean(last_fitness)
         g.age += 1

         if len(last_fitness) >= self.config.ROLLING_FITNESS:
            last_fitness = last_fitness[-self.config.ROLLING_FITNESS:]
         new_fitness_hist[idx] = last_fitness
         self.score_hists[g_idx] = new_fitness_hist[idx]
         self.population[g_idx] = (None, score, g.age)
      self.last_fitnesses = new_fitness_hist
      global_stats.reset.remote()

      if self.reloading:
         self.reloading = False

      if (self.n_epoch % 10 == 0 or self.n_epoch == 0) and not self.reloading:
         self.save()
      self.log()
#     self.neat_pop.reporters.reporters[0].save_genome_fitness(
#           delimiter=',',
#           filename=os.path.join(self.save_path, 'genome_fitness.csv'))
      self.n_epoch += 1


   def evolve(self):
      if self.MAP_ELITES:
         self.me.evolve()

      elif self.CPPN:
         if self.n_epoch == -1:
            self.init_pop()
         else:
            self.map_generator = MapGenerator(self.config)
         winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)
      else:
         return super().evolve()


   def saveMaps(self, maps, mutated=None):
      if self.n_epoch % 100 == 0:
         checkpoint_maps = True
         mutated = self.genes.keys()
      else:
         checkpoint_maps = False
      if self.PATTERN_GEN:
         # map index, (map_arr, multi_hot), atk_mults
         if mutated is None:
            mutated = list(self.chromosomes.keys())
         maps = [(i, (self.chromosomes[i][0].paint_map(), self.chromosomes[i][1])) for i in mutated]
         [self.validate_spawns(m[1][0][0], m[1][0][1]) for m in maps]
         maps = dict(maps)
      elif self.CPPN:
         if mutated is None:
            mutated = list(maps.keys())
         #FIXME: hack

         if self.n_epoch == -1:
            self.n_epoch += 1

            return
#     if self.n_epoch % 100 == 0:
#        if self.PATTERN_GEN:
#            mutated = [i for i in self.chromosomes.keys()]
#        else:
#            mutated = list(range(self.n_pop))
     #for i, map_arr in maps.items():
      if mutated is None or self.reloading:
         if isinstance(maps, dict):
             mutated = maps.keys()
         else:
             mutated = [i for i, _ in enumerate(maps)]
         self.reloading = False

      checkpoint_dir_created = False

      for i in mutated:
         if self.CPPN:
            # TODO: find a way to mutate attack multipliers alongside the map-generating CPPNs?
            if not i in maps:
                T()
            map_arr, atk_mults = maps[i]
         elif self.RAND_GEN:
            map_arr, atk_mults = maps[i]
         else:
            map_arr, atk_mults = maps[i]

         if self.PATTERN_GEN:
            # ignore one-hot map
            map_arr = map_arr[0]
         path = os.path.join(self.save_path, 'maps', 'map' + str(i), '')
         try:
            os.mkdir(path)
         except FileExistsError:
            pass

         if map_arr is None:
            T()
         Save.np(map_arr, path)

         if self.config.TERRAIN_RENDER:
            png_path = os.path.join(self.save_path, 'maps', 'map' + str(i) + '.png')
            Save.render(map_arr[self.config.TERRAIN_BORDER:-self.config.TERRAIN_BORDER, self.config.TERRAIN_BORDER:-self.config.TERRAIN_BORDER], self.map_generator.textures, png_path)

         if self.n_epoch % 100 == 0:
            if self.n_epoch != 0:
                plot_exp(self.config.EVO_DIR)
            checkpoint_dir_path = os.path.join(self.save_path, 'maps', 'checkpoint_{}'.format(self.n_epoch))

            if not checkpoint_dir_created and not os.path.isdir(checkpoint_dir_path):
               os.mkdir(checkpoint_dir_path)
               checkpoint_dir_created = True
            png_path = os.path.join(checkpoint_dir_path, 'map' + str(i) + '.png')
            Save.render(map_arr, self.map_generator.textures, png_path)
            Save.np(map_arr, os.path.join(checkpoint_dir_path, 'map' + str(i) + '.np'))
            json_path = os.path.join(checkpoint_dir_path, 'atk_mults' + str(i) + 'json')
            with open(json_path, 'w') as json_file:
               json.dump(atk_mults, json_file)

        #if self.RAND_GEN or self.PATTERN_GEN:
         json_path = os.path.join(self.save_path, 'maps', 'atk_mults' + str(i) + 'json')
         with open(json_path, 'w') as json_file:
            json.dump(atk_mults, json_file)

   def make_game(self, child_map):
      config = self.config
     #config['map_arr'] = child_map
      game = self.make_env(config)

      return game

   def restore(self, trash_data=False, inference=False):
      '''
      trash_data: to avoid undetermined weirdness when reloading
      '''
      self.calc_diversity = diversity_calc(self.config)
      self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.EVO_DIR, 'maps', 'map')

      self.global_counter.set_idxs.remote(range(self.config.N_EVO_MAPS))

      if inference:
         #TODO: avoid loading redundant envs, provide policy rather than trainer to Evaluator if possible
         pass
      global TRAINER
      self.trainer = TRAINER

      if self.trainer is None:

         # Create policies
         policies = createPolicies(self.config)

         conf = self.config
         model_path = os.path.join(self.save_path, 'models')
         try:
            os.mkdir(model_path)
         except FileExistsError:
            print('Model directory already exists.',model_path)
         # Instantiate monolithic RLlib Trainer object.
         num_workers = self.config.N_PROC
       # EvoPPOTrainer = build_trainer(
       #     name="EvoPPO",
       #    #name="PPO",
       #    #default_config=DEFAULT_CONFIG,
       #    #validate_config=validate_config,
       #    #default_policy=PPOTFPolicy,
       #    #get_policy_class=get_policy_class,
       #     execution_plan=frozen_execution_plan,
       # )
         trainer = EvoPPOTrainer(
            execution_plan=frozen_execution_plan,
            env="custom",
            path=model_path,
            config={
            'num_workers': num_workers, # normally: 4
           #'num_gpus_per_worker': 0.083,  # hack fix
            'num_gpus_per_worker': 0,
            'num_gpus': 1,
            'num_envs_per_worker': int(conf.N_EVO_MAPS / num_workers),
           #'num_envs_per_worker': 1,
            # batch size is n_env_steps * maps per generation
            # plus 1 to actually reset the last env
            'train_batch_size': conf.MAX_STEPS * conf.N_EVO_MAPS, # normally: 4000
           #'train_batch_size': 5000,
            'rollout_fragment_length': 100,
            'sgd_minibatch_size': 128,  # normally: 128
            'num_sgd_iter': 1,
            'framework': 'torch',
            'horizon': np.inf,
            'soft_horizon': False,
            '_use_trajectory_view_api': False,
            'no_done_at_end': False,
            'callbacks': LogCallbacks,
            'env_config': {
                'config':
                self.config,
            },
            'multiagent': {
                "policies": policies,
                "policy_mapping_fn":
                mapPolicy
            },
            'model': {
                'custom_model': 'test_model',
                'custom_model_config': {
                    'config':
                    self.config
                },
            },
#           'optimizer': {
#              'frozen': True,
#              },
#           'multiagent': {
#              'policies_to_train': [],
#              },
          })

         # Print model size
         utils.modelSize(trainer.defaultModel())
         trainer.restore(self.config.MODEL)
         TRAINER = trainer
         self.trainer = trainer

   def infer(self):
      ''' Do inference, just on the top individual for now.'''
      global_counter = ray.get_actor("global_counter")
      self.send_genes(self.global_stats)
     #best_g, best_score = None, -999

     #for g_hash, (_, score, age) in self.population.items():
     #    if score and score > best_score and age > self.mature_age:
     #        print('new best', score)
     #        best_g, best_score = g_hash, score

     #if not best_g:
     #    raise Exception('No population found for inference.')
     #print("Loading map {} for inference.".format(best_g))
      best_g = self.config.INFER_IDX
      global_counter.set.remote(best_g)
      self.config.EVALUATE = True
      if not self.trainer:
         self.restore()
      evaluator = RLLibEvaluator(self.config, self.trainer)
      evaluator.render()

   def genRandMap(self, g_hash=None):
      print('genRandMap {}'.format(g_hash))
      if self.n_epoch > 0 or self.reloading:
         print('generating new random map when I probably should not be... \n\n')
#     map_arr= np.random.randint(1, self.n_tiles,
#                                 (self.map_width, self.map_height))
      if self.CPPN:
         return None
   #     raise Exception('CPPN maps should be generated inside neat_eval_fitness function')
      elif self.PATTERN_GEN:
         if g_hash is None:
            g_hash = ray.get(self.global_counter.get.remote())
         chromosome = self.Chromosome(
               self.map_width,
               self.n_tiles,
               self.max_primitives,
               enums.Material.GRASS.value.index)
         map_arr, multi_hot = chromosome.generate()
         self.validate_spawns(map_arr, multi_hot)
         chromosome.flat_map = map_arr
         atk_mults = self.gen_mults()
         self.chromosomes[g_hash] = chromosome, atk_mults

         #FIXME: hack for map elites

         if self.MAP_ELITES:
            return g_hash, map_arr, atk_mults
         else:
            return map_arr, atk_mults
      else:
         # FIXME: hack: ignore lava
         map_arr = np.random.choice(np.arange(0, self.n_tiles), (self.map_width, self.map_height),
               p=TILE_PROBS[:])
         self.validate_spawns(map_arr)
#        self.add_border(map_arr)
      atk_mults = self.gen_mults()

      return map_arr, atk_mults

   def validate_spawns(self, map_arr, multi_hot=None):
      self.add_border(map_arr, multi_hot)
      idxs = map_arr == enums.Material.SPAWN.value.index
      spawn_points = np.vstack(np.where(idxs)).transpose()
      n_spawns = len(spawn_points)

      if n_spawns >= self.config.NENT:
         return
      n_new_spawns = self.config.NENT - n_spawns

#     if multi_hot is not None:
#        spawn_idxs = k_largest_index_argsort(
#              multi_hot[enums.Material.SPAWN.value.index, :, :],
#              n_new_spawns)
#    #   map_arr[spawn_idxs[:, 0], spawn_idxs[:, 1]] = enums.Material.SPAWN.value.index
#     else:
      border = self.config.TERRAIN_BORDER
      spawn_idxs = np.random.randint(border, self.map_width - border, (2, n_new_spawns))
      map_arr[spawn_idxs[0], spawn_idxs[1]] = enums.Material.SPAWN.value.index


   def add_border(self, map_arr, multi_hot=None):
      b = self.config.TERRAIN_BORDER
      # the border must be lava
      map_arr[0:b, :]= enums.Material.LAVA.value.index
      map_arr[:, 0:b]= enums.Material.LAVA.value.index
      map_arr[-b:, :]= enums.Material.LAVA.value.index
      map_arr[:, -b:]= enums.Material.LAVA.value.index

      if multi_hot is not None:
         multi_hot[:, 0:b, :]= -1
         multi_hot[:, :, 0:b]= -1
         multi_hot[:, -b:, :]= -1
         multi_hot[:, :, -b:]= -1

   def mutate(self, g_hash, par_hash):
      if self.CPPN:
         raise Exception('CPPN-generated maps should be mutated inside NEAT code.')
      elif self.PATTERN_GEN:
         chromosome, atk_mults = copy.deepcopy(self.chromosomes[par_hash])
         map_arr, multi_hot = chromosome.mutate()
         self.validate_spawns(map_arr, multi_hot)
         chromosome.flat_map = map_arr
         self.chromosomes[g_hash] = chromosome, atk_mults
      else:
         map_arr, atk_mults = self.genes[par_hash]
         map_arr = map_arr.copy()

         for i in range(random.randint(0, self.n_mutate_actions)):
            x= random.randint(0, self.map_width - 1)
            y= random.randint(0, self.map_height - 1)
            # FIXME: hack: ignore lava
            t= np.random.randint(1, self.n_tiles)
            map_arr[x, y]= t
   #     map_arr = self.add_border(map_arr)
         self.validate_spawns(map_arr)
      atk_mults = self.mutate_mults(atk_mults)

      return map_arr, atk_mults


   def add_spawn(self, g_hash, map_arr):
      self.add_border(map_arr, enums.Material.GRASS.value.index)
      idxs = map_arr == enums.Material.SPAWN.value.index
      spawn_points = np.vstack(np.where(idxs)).transpose()

      if len(spawn_points) == 0:
         self.add_border(map_arr, enums.Material.SPAWN.value.index)
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

   def simulate_game(self, game, map_arr, n_ticks, conn=None, g_hash=None):
       score = 0
       score += self.tick_game(game, g_hash=g_hash)

       if conn:
           conn.send(score)

       return score

   def run(self, game, map_arr):
       self.obs= game.reset(#map_arr=map_arr,
               idx=0)
       self.game= game
       from forge.trinity.twistedserver import Application
       Application(game, self.tick).run()

   def send_genes(self, global_stats):
      ''' Send (some) gene information to a global object to be retrieved by parallel environments.
      '''

#     T()
#     for g_hash, gene in self.genes.items():
      for g_hash, gene in self.genes.items():
         atk_mults = gene[-1]
         global_stats.add_mults.remote(g_hash, atk_mults)

   def evolve_generation(self):
      global_stats = self.global_stats
      self.send_genes(global_stats)
      train_stats = self.trainer.train()
      stats = ray.get(global_stats.get.remote())
     #headers = ray.get(global_stats.get_headers.remote())
#     n_epis = train_stats['episodes_this_iter']

#     if n_epis == 0:
#        print('Missing simulation stats. I assume this is the 0th generation? Re-running the training step.')

#        return self.evolve_generation()

      global_stats.reset.remote()

      for g_hash in self.population.keys():
         if g_hash not in stats:
            print('Missing simulation stats for map {}. I assume this is the 0th generation, or re-load? Re-running the training step.'.format(g_hash))
            self.trainer.train()
            stats = ray.get(global_stats.get.remote())

            break

      for g_hash, (game, score, age) in self.population.items():
        #print(self.config.SKILLS)
         score = self.calc_diversity(stats[g_hash], verbose=False)
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
       # check if we are doing inference
       FROZEN = True

       if FROZEN:
           if game is None:
               game = self.game
               update_entropy_skills(game.desciples.values())

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
               actions, self.state, _= self.trainer.compute_actions(
                   self.obs, state=self.state, policy_id='policy_0')

               # Compute overlay maps
               self.overlays.register(self.obs)

               # Step the environment
               self.obs, rewards, self.done, _= game.step(actions)
       else:
          #self.trainer.reset()
           print(dir(self.trainer))
          #self.trainer.train(self.genes[g_hash])
           stats = self.trainer.train()
           print('evo map trainer stats', stats)
       self.n_tick += 1
       reward = 0

       return reward

   def save(self):
       save_file= open(self.evolver_path, 'wb')

       population = copy.deepcopy(self.population)

       for g_hash in self.population:
           game, score, age= self.population[g_hash]
           # FIXME: omething weird is happening after reload. Not the maps though.
           # so for now, trash score history and re-calculate after reload
           self.population[g_hash]= None, score, age
          #self.population[g_hash]= None, score, age
       global TRAINER
       TRAINER = self.trainer
       self.trainer= None
       self.game= None
       # map_arr = self.genes[g_hash]
       copyfile(self.evolver_path, self.evolver_path + '.bkp')
       pickle.dump(self, save_file)
       self.population = population
       self.restore()

   def init_pop(self):
       if not self.config.PRETRAINED and not self.reloading:
           # train from scratch
          self.config.MODEL = None
       elif self.reloading:
           # use pre-trained model
          self.config.MODEL = 'reload'
       else:
          self.config.MODEL = 'current'
       # load the model and initialize and save the population
       super().init_pop()
       self.saveMaps(self.genes, list(self.genes.keys()))
       # reload latest model from evolution moving forward
       self.config.MODEL = 'reload'

