import os
from pdb import set_trace as T
import numpy as np
import json
import csv

import time
from collections import defaultdict
import gym

import time

from ray import rllib

from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.flexdict import FlexDict

from forge.blade import core
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.systems import combat

from forge.trinity.dataframe import DataType

class Env(core.Env):
   def log(self, quill, ent):
      print('logging')
      blob = quill.register('Lifetime', quill.HISTOGRAM, quill.LINE, quill.SCATTER, quill.GANTT)
      blob.log(ent.history.timeAlive.val)

      blob = quill.register('Skill Level', quill.HISTOGRAM, quill.STACKED_AREA, quill.STATS, quill.RADAR)
      blob.log(ent.skills.range.level,        'Range')
      blob.log(ent.skills.mage.level,         'Mage')
      blob.log(ent.skills.melee.level,        'Melee')
      blob.log(ent.skills.constitution.level, 'Constitution')
      blob.log(ent.skills.defense.level,      'Defense')
      blob.log(ent.skills.fishing.level,      'Fishing')
      blob.log(ent.skills.hunting.level,      'Hunting')

      #TODO: swap these entries when equipment is reenabled
      blob = quill.register('Wilderness', quill.HISTOGRAM, quill.SCATTER)
      blob.log(combat.wilderness(self.config, ent.pos))

      blob = quill.register('Equipment', quill.HISTOGRAM, quill.STACKED_AREA)
      blob.log(ent.loadout.chestplate.level, 'Chestplate')
      blob.log(ent.loadout.platelegs.level,  'Platelegs')

      quill.stat('Lifetime',  ent.history.timeAlive.val)
      quill.stat('Skilling',  (ent.skills.fishing.level + ent.skills.hunting.level)/2.0)
      quill.stat('Combat',    combat.level(ent.skills))
      quill.stat('Equipment', ent.loadout.defense)

class RLLibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      self.n_step = 0
      self.map_fitness = 0
      self.skill_headers = []
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
         self.map_fitness -= 1
         if not self.config.RENDER and len(self.lifetimes) >= 1000:
        ##FIXME: very sketchy hack -smearle
        #if self.n_step >= self.config.MAX_STEPS:
            lifetime = np.mean(self.lifetimes)
            print('Lifetime: {}'.format(lifetime))
            dones['__all__'] = True

      self.n_step += 1
      self.env_step += time.time() - env_post
      skills = {}
      if self.n_step > 0 and self.n_step % (self.config.MAX_STEPS - 1) == 0:
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
              if not self.skill_headers:
                  self.skill_headers = list(a_skills.keys())
                  self.init_skill_log()
              with open(self.skill_log_path, 'w') as outfile:
                 writer = csv.DictWriter(outfile, fieldnames=self.skill_headers)
                 writer.writeheader()
                 for skills in skills.values():
                     writer.writerow(skills)
      if self.n_step % (self.config.MAX_STEPS * self.config.MATURE_AGE) == 0:
          dones['__all__'] = True
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

