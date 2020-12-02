import os
from pdb import set_trace as T
import numpy as np
import json
import csv

import time
from collections import defaultdict
import gym

import time

import ray
from ray import rllib

from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.flexdict import FlexDict

from forge.blade import core
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.systems import combat

from forge.trinity.dataframe import DataType

#Moved log to forge/blade/core/env
class RLLibEnv(core.Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      self.n_step = 0
      self.map_fitness = 0
      self.skill_headers = []
      self.n_epi = 0
      # names of per-agent stats used by evolution
      self.headers = None
      super().__init__(self.config)

   def init_skill_log(self):
      self.skill_log_path = './evo_experiment/{}/map_{}_skills.csv'.format(self.config.EVO_DIR, self.worldIdx)
      with open(self.skill_log_path, 'w', newline='') as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=self.skill_headers)
         writer.writeheader()
      assert csvfile.closed


   def reset(self, idx=None, step=True):
      '''Enable us to reset the Neural MMO environment.
      This is for training on limited-resource systems where
      simply using one env map per core is not feasible'''
      self.env_reset = time.time()
      self.n_epi += 1

      self.map_fitness = 0
      self.lifetimes = []

      ret = super().reset(idx=idx, step=step)

      self.n_step = 0
      self.env_reset = time.time() - self.env_reset
      return ret

   def step(self, decisions):
      '''Action postprocessing; small wrapper to fit RLlib'''
      #start = time.time()
      self.rllib_compat = time.time()

      actions = {}
      for entID in list(decisions.keys()):
         ent = self.realm.players[entID]
         r, c = ent.pos
         radius = self.config.STIM
         grid  = self.realm.dataframe.data['Entity'].grid
         index = self.realm.dataframe.data['Entity'].index
         cent = grid.data[r, c]
         rows = grid.window(
            r-radius, r+radius+1,
            c-radius, c+radius+1)
         rows.remove(cent)
         rows.insert(0, cent)
         rows = [index.teg(e) for e in rows]
         assert rows[0] == ent.entID

         actions[entID] = defaultdict(dict)
         if entID in self.dead:
            continue

         for atn, args in decisions[entID].items():
            for arg, val in args.items():
               val = int(val)
               if len(arg.edges) > 0:
                  actions[entID][atn][arg] = arg.edges[val]
               elif val < len(rows):
                  actions[entID][atn][arg] = self.realm.entity(rows[val])
               else:
                  actions[entID][atn][arg] = ent

      self.rllib_compat = time.time() - self.rllib_compat
      self.env_step     = time.time()

      obs, rewards, dones, infos = super().step(actions)

      self.env_step     = time.time() - self.env_step
      env_post          = time.time()

      #Cull dead agents
      for entID, ent in self.dead.items():
         lifetime = ent.history.timeAlive.val
         self.lifetimes.append(lifetime)
         if not self.config.EVO_MAP:
             if not self.config.EVALUATE and len(self.lifetimes) >= 1000:
                lifetime = np.mean(self.lifetimes)
                print('Lifetime: {}'.format(lifetime))
                dones['__all__'] = True

      self.env_step += time.time() - env_post
      skills = {}
      # are we doing evolution? 
      if self.config.EVO_MAP:
         # reset the env manually, to load from the new updated population of maps
         if self.n_step == self.config.MAX_STEPS:
            global_stats = ray.get_actor('global_stats')
           #print('preparing env {} for reset'.format(self.worldIdx))
            dones['__all__'] = True
            # Do not save skills data if rendering.
            if not self.config.RENDER:
                a_skills = None
                for d, v in self.realm.players.items():
                    a_skills = v.skills.packet()
                    a_skill_vals = {}
                    for k, v in a_skills.items():
                        if not isinstance(v, dict):
                            continue
                        a_skill_vals[k] = v['exp']
                    skills[d] = a_skill_vals
                if a_skills:
                    if not self.headers:
                        headers = list(a_skills.keys())
                        self.headers = ray.get(global_stats.get_headers.remote(headers))
                    stats = np.zeros((len(skills), len(self.headers)))
                    # over agents
                    for i, a_skills in enumerate(skills.values()):
                        # over skills
                        for j, k in enumerate(self.headers):
                            if k != 'level':
                                stats[i, j] = a_skills[k]
                    global_stats.add.remote(stats, self.worldIdx)
                   #if not self.skill_headers:
                   #    self.skill_headers = list(a_skills.keys())
                   #    self.init_skill_log()
                   #with open(self.skill_log_path, 'w') as outfile:
                   #   writer = csv.DictWriter(outfile, fieldnames=self.skill_headers)
                   #   writer.writeheader()
                   #   for skills in skills.values():
                   #       writer.writerow(skills)
      self.n_step += 1
      return obs, rewards, dones, infos

#Neural MMO observation space
def observationSpace(config):
   obs = FlexDict(defaultdict(FlexDict))
   for entity in sorted(Stimulus.values()):
      #attrDict = FlexDict({})
      #for attr in sorted(entity.values()):
      #   attrDict[attr] = attr(config, None).space
      nRows = entity.N(config)

      nContinuous = 0
      nDiscrete   = 0
      for _, attr in entity:
         if attr.DISCRETE:
            nDiscrete += 1
         if attr.CONTINUOUS:
            nContinuous += 1

      obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=0, high=2**16, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)

   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

