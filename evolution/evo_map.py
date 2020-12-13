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
import neat
import neat.nn
import projekt
import ray
import scipy
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums
from forge.ethyr.torch import utils
from pcg import TILE_PROBS, TILE_TYPES
from projekt import rlutils
from projekt.evaluator import Evaluator
from projekt.overlay import OverlayRegistry
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from evolution.lambda_mu import LambdaMuEvolver

np.set_printoptions(threshold=sys.maxsize,
                    linewidth=120,
                    suppress=True,
                   #precision=10
                    )

# keep trainer here while saving evolver object
global TRAINER
TRAINER = None


def calc_diversity_l2(agent_skills71G):
   assert len(agent_skills) == 1
   a = agent_skills[0]
   n_agents = a.shape[0]
   b = a.reshape(n_agents, 1, a.shape[1])
   score = np.sum(np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b)))/2
#  print('agent skills:\n{}'.format(a))
#  print('score:\n{}\n'.format(
#      score))
   score = score / (n_agents**2)

   return score

def sigmoid_lifespan(x):
   return 3 / (1 + np.exp(0.1*(-x+20)))

def calc_differential_entropy(agent_stats, max_pop=8):
   # Penalize if under max pop agents living
  #for i, a_skill in enumerate(agent_skills):
  #   if a_skill.shape[0] < max_pop:
  #      a = np.mean(a_skill, axis=0)
  #      a_skill = np.vstack(np.array([a_skill] + [a for _ in range(max_pop - a_skill.shape[0])]))
  #      agent_skills[i] = a_skill
   # if there are stats from multiple simulations, we consider agents from all simulations together
   #FIXME: why
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']
   if not len(agent_skills) == 1:
      pass
   assert len(agent_skills) == len(lifespans)
   a_skills = np.vstack(agent_skills)
   a_lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(a_lifespans)
#  assert len(agent_skills) == 1
  #a_skills = agent_skills[0]
   mean = np.average(a_skills, axis=0, weights=weights)
   cov = np.cov(a_skills,rowvar=0, aweights=weights)
   gaussian = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
   score = gaussian.entropy()
#  print(np.array(a_skills))
#  print(score)
   # FIXME: Only applies to exploration-only experiment
   print(a_skills.transpose()[0])
   print('lifespans')
   print(a_lifespans)
   print(len(agent_skills), 'populations')

   return score

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
      print("trainer.train() result: {} -> {} episodes".format(
         trainer, n_epis))
      # you can mutate the result dict to add new fields to return
      # result['something'] = True


# Map agentID to policyID -- requires config global
def mapPolicy(agentID,
        #config
        ):

    return 'policy_{}'.format(agentID % 1)


# Generate RLlib policies
def createPolicies(config):
    obs = projekt.env.observationSpace(config)
    atns = projekt.env.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i
                #, config
                )
        policies[key] = (None, obs, atns, params)

    return policies

class EvolverNMMO(LambdaMuEvolver):
   def __init__(self, save_path, make_env, trainer, config, n_proc=12, n_pop=12,):
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
      self.global_stats = ray.get_actor("global_stats")
      if self.config.FITNESS_METRIC == 'L2':
         self.calc_diversity = calc_diversity_l2
      elif self.config.FITNESS_METRIC == 'Differential':
         self.calc_diversity = calc_differential_entropy
      elif self.config.FITNESS_METRIC == 'Discrete':
         self.calc_diversity = calc_discrete_entropy
      else:
         raise Exception('Unsupported fitness function: {}'.format(self.config.FITNESS_METRIC))
      self.CPPN = config.GENE == 'CPPN'
      if self.CPPN:
         self.n_epoch = -1
         self.global_counter = ray.get_actor("global_counter")
         self.neat_config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_nmmo')
         self.neat_config.pop_size = self.config.N_EVO_MAPS
         self.neat_pop = neat.population.Population(self.neat_config)
         stats = neat.statistics.StatisticsReporter()
         self.neat_pop.add_reporter(stats)
         self.neat_pop.add_reporter(neat.reporting.StdOutReporter(True))
#        winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)



   def neat_eval_fitness(self, genomes, neat_config):
      if self.n_epoch == 0:
         self.last_map_idx = 0
      maps = {}
      global_counter = ray.get_actor("global_counter")
      g_idxs = set([idx for idx, _ in genomes])
      print('current map IDs: {}'.format(sorted(list(g_idxs))))
      # FIXME NOPE
      self.save()
      skip_idxs = set()
      for idx, g in genomes:
         if idx <= self.last_map_idx:
            continue
         cppn = neat.nn.FeedForwardNetwork.create(g, self.neat_config)
         map_arr = np.zeros((self.map_width, self.map_height), dtype=np.uint8)
         for x in range(self.map_width):
            for y in range(self.map_height):
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
                  v = np.argmax(v)
                  map_arr[x, y] = v
         self.add_spawn(idx, map_arr)
        #map_arr = self.add_border(map_arr)
         # Impossible maps are no good
         tile_counts = np.bincount(map_arr.reshape(-1))
         if False and (len(tile_counts) <= enums.Material.FOREST.value.index or \
               tile_counts[enums.Material.FOREST.value.index] <= self.config.NENT * 3 or \
               tile_counts[enums.Material.WATER.value.index] <= self.config.NENT * 3):
            print('map {} rejected'.format(idx))
            g.fitness = 0
            g_idxs.remove(idx)
            skip_idxs.add(idx)
         else:
            maps[idx] = map_arr#, spawn_points
      g_idxs = list(g_idxs)
      self.last_map_idx = g_idxs[-1]
      global_counter.set_idxs.remote(g_idxs)
      self.saveMaps(maps)
      global_stats = self.global_stats
      self.send_genes(global_stats)
      train_stats = self.trainer.train()
      stats = ray.get(global_stats.get.remote())
      headers = ray.get(global_stats.get_headers.remote())
      n_epis = train_stats['episodes_this_iter']
     #assert n_epis == self.n_pop
      for idx, _ in genomes:
         #FIXME: hack
         if idx in skip_idxs:
            continue
         if idx not in stats:
            print('Missing stats for map {}, training again.'.format(idx))
            self.trainer.train()
            stats = ray.get(global_stats.get.remote())

      for idx, g in genomes:
         #FIXME: hack
         if idx in skip_idxs:
            continue
         print(headers)
         score = self.calc_diversity(stats[idx])
#        self.population[g_hash] = (game, score, age)
         print('Map {}, diversity score: {}\n'.format(idx, score))
         g.fitness = score
      global_stats.reset.remote()
      if self.n_epoch % 10 == 0:
         self.save()
      self.n_epoch += 1

   def evolve(self):
      if self.n_epoch == -1:
         self.init_pop()
      else:
         self.map_generator = MapGenerator(self.config)
      winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)

   def saveMaps(self, maps, mutated=None):
      if self.CPPN:
         mutated = list(maps.keys())
         #FIXME: hack
         if self.n_epoch == -1:
            self.n_epoch += 1
            return
     #for i, map_arr in maps.items():
      if mutated is None:
         mutated = list(self.population.keys())

      for i in mutated:
         if self.CPPN:
            # TODO: find a way to mutate attack multipliers alongside the map-generating CPPNs?
            map_arr = maps[i]
         else:
            map_arr, atk_mults = maps[i]
         print('Saving map ' + str(i))
         path = os.path.join(self.save_path, 'maps', 'map' + str(i), '')
         try:
            os.mkdir(path)
         except FileExistsError:
            pass
         Save.np(map_arr, path)

         if self.config.TERRAIN_RENDER:
            png_path = os.path.join(self.save_path, 'maps', 'map' + str(i) + '.png')
            Save.render(map_arr, self.map_generator.textures, png_path)

         if not self.CPPN:
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
            print('Model directory already exists.')
         # Instantiate monolithic RLlib Trainer object.
         trainer = rlutils.EvoPPOTrainer(
            env="custom",
            path=model_path,
            config={
            'num_workers': 6, # normally: 4
           #'num_gpus_per_worker': 0.083,  # hack fix
            'num_gpus_per_worker': 0,
            'num_gpus': 1,
           #'num_envs_per_worker': int(conf.N_EVO_MAPS / 6),
            'num_envs_per_worker': 1,
            # batch size is n_env_steps * maps per generation
            # plus 1 to actually reset the last env
            'train_batch_size': conf.MAX_STEPS * conf.N_EVO_MAPS, # normally: 4000
           #'train_batch_size': 4000,
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
                }
            },
          })

         # Print model size
         utils.modelSize(trainer.defaultModel())
         trainer.restore(self.config.MODEL)
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
      evaluator = Evaluator(self.config, self.trainer)
      evaluator.render()

   def genRandMap(self):
      if self.n_epoch > 0:
         print('generating new random map when I probably should not be... \n\n')
      # FIXME: hack: ignore lava
#     map_arr= np.random.randint(1, self.n_tiles,
#                                 (self.map_width, self.map_height))
      if self.CPPN:
         return None, None
      else:
         map_arr = np.random.choice(np.arange(1, self.n_tiles), (self.map_width, self.map_height),
               p=TILE_PROBS[1:])
         self.add_border(map_arr)
      atk_mults = self.gen_mults()

      return map_arr, atk_mults

   def gen_mults(self):
      # generate melee, range, and mage attack multipliers for automatic game-balancing
     #atks = ['MELEE_MULT', 'RANGE_MULT', 'MAGE_MULT']
     #mults = [(atks[i], 0.2 + np.random.random() * 0.8) for i in range(3)]
     #atk_mults = dict(mults)
      # range is way too dominant, always
      self.MELEE_MIN = 0.4
      self.MELEE_MAX = 1.4
      self.MAGE_MIN = 0.6
      self.MAGE_MAX = 1.6
      self.RANGE_MIN = 0.2
      self.RANGE_MAX = 1
      atk_mults = {
            # b/w 0.2 and 1.0
            'MELEE_MULT': np.random.random() * (self.MELEE_MAX - self.MELEE_MIN) + self.MELEE_MIN,
            'MAGE_MULT': np.random.random() * (self.MAGE_MAX - self.MAGE_MIN) + self.MAGE_MIN,
            # b/w 0.0 and 0.8
            'RANGE_MULT': np.random.random() * (self.RANGE_MAX - self.RANGE_MIN) + self.RANGE_MIN,
            }

      return atk_mults

   def mutate(self, gene):
      map_arr, atk_mults = gene
      map_arr= map_arr.copy()

      for i in range(random.randint(0, self.n_mutate_actions)):
         x= random.randint(0, self.map_width - 1)
         y= random.randint(0, self.map_height - 1)
         # FIXME: hack: ignore lava
         t= np.random.randint(1, self.n_tiles)
         map_arr[x, y]= t
      map_arr = self.add_border(map_arr)
      # kind of arbitrary, no?
     #atk_mults = dict([(atk, max(min(mult + (np.random.random() * 2 - 1) / 3, 1), 0.2)) for atk, mult in atk_mults.items()])
      rand = np.random.random()

      if rand < 0.2:
         atk_mults = self.gen_mults()
      else:
         atk_mults = {
            'MELEE_MULT': max(min(atk_mults['MELEE_MULT'] + (np.random.random() * 2 - 1) * 0.3, 
                                  self.MELEE_MAX), self.MELEE_MIN),
            'MAGE_MULT': max(min(atk_mults['MAGE_MULT'] + (np.random.random() * 2 - 1) * 0.3, 
                                 self.MAGE_MAX), self.MAGE_MIN),
            'RANGE_MULT': max(min(atk_mults['RANGE_MULT'] + (np.random.random() * 2 - 1) * 0.3, 
                                  self.RANGE_MAX), self.RANGE_MIN),
               }

      return map_arr, atk_mults

   def add_border(self, map_arr, border_mat_index):
      b = self.config.TERRAIN_BORDER
      # agents should not spawn and die immediately, as this may crash the env
      a = 1
      map_arr[b:b+a, :]= border_mat_index
      map_arr[:, b:b+a]=   border_mat_index
      map_arr[:, -b-a:-b]=  border_mat_index
      map_arr[-b-a:-b, :]=  border_mat_index
      # the border must be lava
      map_arr[0:b, :]= enums.Material.LAVA.value.index
      map_arr[:, 0:b]= enums.Material.LAVA.value.index
      map_arr[-b:, :]= enums.Material.LAVA.value.index
      map_arr[:, -b:]= enums.Material.LAVA.value.index

      return map_arr

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

      for g_hash, (_, atk_mults) in self.genes.items():
         global_stats.add_mults.remote(g_hash, atk_mults)

   def evolve_generation(self):
      global_stats = self.global_stats
      self.send_genes(global_stats)
      train_stats = self.trainer.train()
      stats = ray.get(global_stats.get.remote())
      headers = ray.get(global_stats.get_headers.remote())
      n_epis = train_stats['episodes_this_iter']

      if n_epis == 0:
         print('Missing simulation stats. I assume this is the 0th generation? Re-running the training step.')

         return self.evolve_generation()

      global_stats.reset.remote()

      for g_hash, (game, score, age) in self.population.items():
         score = self.calc_diversity(stats[g_hash])
         self.population[g_hash] = (game, score, age)
         print(headers)
         print(stats[g_hash][0])
         print('diversity score: ', score)

      for g_hash, (game, score_t, age) in self.population.items():
         # get score from latest simulation
         # cull score history

         if len(self.score_hists[g_hash]) >= 10:
            while len(self.score_hists[g_hash]) >= 10:
               self.score_hists[g_hash].pop(0)
          # hack

         #if score_t is None:
         #    score_t = 0
         else:
            self.score_hists[g_hash].append(score_t)
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
       self.restore()

   def init_pop(self):
       self.config.MODEL = None
       super().init_pop()
       self.config.MODEL = 'current'

def update_entropy_skills(skill_dict):
    agent_skills = [[] for _ in range(len(skill_dict))]
    i = 0

    for a_skills in skill_dict:
        j = 0

        for a_skill in a_skills:
            try:
                val = float(a_skill)
                agent_skills[i].append(val)
                j += 1
            except:
                pass
        i += 1

    return calc_diversity_l2(agent_skills)

def calc_discrete_entropy(agent_skills, alpha, skill_idxs=None):
    BASE_VAL = 0.0001
    # split between skill and agent entropy
    n_skills = len(agent_skills[0])
    n_pop = len(agent_skills)
    agent_sums = [sum(skills) for skills in agent_skills]
    i = 0

    for a in agent_sums:
        if a == 0:
            agent_sums[i] = BASE_VAL * n_skills
        i += 1
    skill_sums = [0 for i in range(n_skills)]

    for i in range(n_skills):

        for a_skills in agent_skills:
            skill_sums[i] += a_skills[i]

        if skill_sums[i] == 0:
            skill_sums[i] = BASE_VAL * n_pop

    skill_ents = []

    for i in range(n_skills):
        skill_ent = 0

        for j in range(n_pop):

            a_skill = agent_skills[j][i]

            if a_skill == 0:
                a_skill = BASE_VAL
            p = a_skill / skill_sums[i]

            if p == 0:
                skill_ent += 0
            else:
                skill_ent += p * np.log(p)
        skill_ent = skill_ent / (n_pop)
        skill_ents.append(skill_ent)

    agent_ents = []

    for j in range(n_pop):
        agent_ent = 0

        for i in range(n_skills):

            a_skill = agent_skills[j][i]

            if a_skill == 0:
                a_skill = BASE_VAL
            p = a_skill / agent_sums[j]

            if p == 0:
                agent_ent += 0
            else:
                agent_ent += p * np.log(p)
        agent_ent = agent_ent / (n_skills)
        agent_ents.append(agent_ent)
    agent_score = np.mean(agent_ents)
    skill_score = np.mean(skill_ents)
    score = (alpha * skill_score + (1 - alpha) * agent_score)
    score = score * 100
    print('agent skills:\n{}\n{}'.format(skill_idxs, np.array(agent_skills)))
    print('skill_ents:\n{}\nskill_mean:\n{}\nagent_ents:\n{}\nagent_mean:{}\nscore:\n{}\n'.format(
        np.array(skill_ents), skill_score, np.array(agent_ents), agent_score, score))

    return score
