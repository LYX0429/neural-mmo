#<<<<<<< HEAD
import os
from pathlib import Path
from collections import defaultdict
from pdb import set_trace as T
from typing import Dict
import json
#=======
from pdb import set_trace as T

from collections import defaultdict
from itertools import chain
import shutil
import contextlib
import os
import re

from tqdm import tqdm
import numpy as np
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

import gym
from matplotlib import pyplot as plt
import matplotlib
from ray.rllib.agents import Trainer
from ray.rllib.env.normalize_actions import NormalizeActionWrapper
from ray.rllib.utils import override
from ray.rllib.utils.from_config import from_config
from ray.tune import Trainable
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.utils import merge_dicts

matplotlib.use('Agg')
import numpy as np
import ray
import ray.rllib.agents.ppo.ppo as ppo
import torch
#<<<<<<< HEAD
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy import Policy
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.spaces.flexdict import FlexDict
from forge.blade.lib.material import Water, Lava, Stone
from torch import nn
from tqdm import tqdm
from plot_diversity import heatmap
import projekt
from forge.blade.io.action.static import Action
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.lib.log import InkWell
from forge.blade.core. terrain import Save, MapGenerator
from forge.ethyr.torch import io, policy
from forge.ethyr.torch.policy import baseline
from forge.trinity import Env, evaluator
from forge.trinity.dataframe import DataType
from forge.trinity.overlay import OverlayRegistry
from forge.blade.io import action

from evolution.diversity import DIV_CALCS, diversity_calc

from ray.rllib.execution.metric_ops import StandardMetricsReporting

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep

from ray.rllib.utils.typing import EnvConfigDict, EnvType, ResultDict, TrainerConfigDict, PartialTrainerConfigDict


#Moved log to forge/trinity/env
#class RLLibEnv(Env, rllib.MultiAgentEnv):
#=======
from torch import nn

from ray import rllib
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action.static import Action
from forge.blade.lib import overlay

from forge.ethyr.torch.policy import baseline

from forge.trinity import Env, evaluator, formatting
from forge.trinity.dataframe import DataType
from forge.trinity.overlay import Overlay, OverlayRegistry

###############################################################################
### RLlib Env Wrapper
class RLlibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      if self.config.GRIDDLY:
         from griddly_nmmo.env import NMMO
         from griddly import GymWrapperFactory, gd
      self.headers = self.config.SKILLS
      super().__init__(self.config)
      self.evo_dones = None
      if config['config'].FITNESS_METRIC == 'Actions':
         self.ACTION_MATCHING = True
         self.realm.target_action_sequence = [action.static.South] * config['config'].TRAIN_HORIZON
         # A list of net actions matched by all dead agents
         self.actions_matched = []
      else:
         self.ACTION_MATCHING = False


   def init_skill_log(self):
      self.skill_log_path = './evo_experiment/{}/map_{}_skills.csv'.format(self.config.EVO_DIR, self.worldIdx)
      with open(self.skill_log_path, 'w', newline='') as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=self.skill_headers)
         writer.writeheader()
      assert csvfile.closed


   def step(self, decisions, omitDead=False, preprocessActions=True):
#     print('decisions keys', decisions.keys())
#     print('ent keys', self.ents.keys())
      obs, rewards, dones, infos = super().step(decisions,
            omitDead=omitDead, preprocessActions=preprocessActions)

      t, mmean = len(self.lifetimes), np.mean(list(self.lifetimes.values()))

      # We don't need this, set_map does it for us?

#     if not self.config.EVALUATE and self.realm.tick > self.config.TRAIN_HORIZON:
#        dones['__all__'] = True

      # are we doing evolution?

 #    if self.config.EVO_MAP and not self.config.FIXED_MAPS:# and not self.config.RENDER:
 #       if self.realm.tick >= self.config.MAX_STEPS or self.config.RENDER:
 #          # reset the env manually, to load from the new updated population of maps
##          print('resetting env {} after {} steps'.format(self.worldIdx, self.n_step))
 #          dones['__all__'] = True

      if self.config.EVO_MAP and hasattr(self, 'evo_dones') and self.evo_dones is not None:
         dones = self.evo_dones
         self.evo_dones = None
#     print('obs keys', obs.keys())


      return obs, rewards, dones, infos


   def send_agent_stats(self):
      global_stats = ray.get_actor('global_stats')
      stats = self.get_all_agent_stats()
      global_stats.add.remote(stats, self.worldIdx)
      self.evo_dones = {}
      self.evo_dones['__all__'] = True

      return stats

   def get_agent_stats(self, player):
      player_packet = player.packet()
      a_skills = player_packet['skills']
      a_skill_vals = {}

      for k, v in a_skills.items():
         if not isinstance(v, dict):
            continue

         if k in ['exploration']:
            continue

         if k in ['cooking', 'smithing', 'level']:
            continue
         a_skill_vals[k] = v['exp']

         if k in ['fishing', 'hunting', 'constitution']:
            # FIXME: hack -- just easier on the eyes, mostly. Don't change config.RESOURCE !
            a_skill_vals[k] -= 1154
      # a_skill_vals['wilderness'] = player_packet['status']['wilderness'] * 10
#     a_skill_vals['exploration'] = player.exploration_grid.sum() * 20
      a_skill_vals['exploration'] = len(player.explored) * 20
      # timeAlive will only add expressivity if we fit more than one gaussian.
      a_skill_vals['time_alive'] = player_packet['history']['timeAlive']
      if self.ACTION_MATCHING:
         a_skill_vals['actions_matched'] = player.actions_matched

      return a_skill_vals

   def get_all_agent_stats(self, verbose=False):
      skills = {}
      a_skills = None

      # Get stats of dead (note the order here)
      l = 0
      for player_id, skill_vals in self.agent_skills.items():
         skills[player_id] = skill_vals
         l += 1

      # Get stats of living
      d = 0
      for player_id, player in self.realm.players.items():
         a_skill_vals = self.get_agent_stats(player)
         skills[player_id] = a_skill_vals
         d += 1


#     if a_skills:
      stats = np.zeros((len(skills), len(self.headers)))
     #stats = np.zeros((len(skills), 1))
      lifespans = np.zeros((len(skills)))
      if self.ACTION_MATCHING:
         actions_matched = np.zeros((len(skills)))

      for player_id, a_skills in skills.items():
         # over agents

         for j, k in enumerate(self.headers):
            # over skills
            if k not in ['level', 'cooking', 'smithing']:
#             if k in ['exploration']:
               stats[player_id - 1, j] = a_skills[k]
               j += 1
         lifespans[player_id - 1] = a_skills['time_alive']
         if self.ACTION_MATCHING:
            actions_matched[player_id - 1] = a_skills['actions_matched']

      # Add lifespans of the living to those of the dead
      stats = {
            'skills': [stats],
            'lifespans': [lifespans],
           #'lifetimes': lifetimes,
            }
      if self.ACTION_MATCHING:
         actions_matched = np.hstack((self.actions_matched, actions_matched))
         stats['actions_matched'] = [actions_matched],

      return stats

     #return {'skills': [0] * len(self.config.SKILLS),
     #      'lifespans': [],
     #      'lifetimes': []}


#Neural MMO observation space
#      obs, rewards, dones, infos = super().step(
#            decisions, omitDead, preprocessActions)
#
#      config = self.config
#      dones['__all__'] = False
#      test = config.EVALUATE or config.RENDER
#      if not test and self.realm.tick  >= config.TRAIN_HORIZON:
#         dones['__all__'] = True
#
#      return obs, rewards, dones, infos
#
def observationSpace(config):
   if hasattr(config, 'GRIDDLY') and config.GRIDDLY:
      #TODO: this, not manually!
      obs = gym.spaces.Box(0, 1, (7, 7, 10))
      return obs
   obs = FlexDict(defaultdict(FlexDict))

   for entity in sorted(Stimulus.values()):
      nRows       = entity.N(config)
      nContinuous = 0
      nDiscrete   = 0

      for _, attr in entity:
         if attr.DISCRETE:
            nDiscrete += 1

         if attr.CONTINUOUS:
            nContinuous += 1

      obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=-2**20, high=2**20, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)

   return obs

#Neural MMO action space
def actionSpace(config, n_act_i=3, n_act_j=5):
   if config.GRIDDLY:
      print('WARNING: Are you sure the griddly env action space is {} {}?'.format(n_act_i, n_act_j))
      atns = gym.spaces.MultiDiscrete((n_act_i, n_act_j))
      return atns
   atns = FlexDict(defaultdict(FlexDict))

   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)

   return atns

def plot_diversity(x, y, div_names, exp_name, config, render=False):
   colors = ['darkgreen', 'm', 'g', 'y', 'salmon', 'darkmagenta', 'orchid', 'darkolivegreen', 'mediumaquamarine',
            'mediumturquoise', 'cadetblue', 'slategrey', 'darkblue', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet',
            'fuchsia', 'deeppink', 'olive', 'orange', 'maroon', 'lightcoral', 'firebrick', 'black', 'dimgrey', 'tomato',
            'saddlebrown', 'greenyellow', 'limegreen', 'turquoise', 'midnightblue', 'darkkhaki', 'darkseagreen', 'teal',
            'cyan', 'lightsalmon', 'springgreen', 'mediumblue', 'dodgerblue', 'mediumpurple', 'darkslategray', 'goldenrod',
            'indigo', 'steelblue', 'coral', 'mistyrose', 'indianred']
#   fig, ax = plt.subplots(figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
   fig, axs = plt.subplots(len(div_names) + 1) 
   fig.suptitle(exp_name)
   plt.subplots_adjust(right=0.78)
   for i, div_name in enumerate(div_names):
      ax = axs[i]
      markers, caps, bars = ax.errorbar(x, y[:,i,:].mean(axis=0), yerr=y[:,i,:].std(axis=0), label=div_name, alpha=1)
      [bar.set_alpha(0.2) for bar in bars]
      plt.text(0.8, 0.8-i*0.162, '{:.2}'.format(y[:,i,:].mean()), fontsize=12, transform=plt.gcf().transFigure)
     #ax.text(0.8, 0.2, '{:.2}'.format(y[:,i,:].mean()))
      ax.legend(loc='upper left')
#     if div_name == 'mean pairwise L2':
#        ax.set_ylim(0, 50000)
      if div_name == 'differential entropy':
         ax.set_ylim(20, 57)
      if div_name == 'discrete entropy':
         ax.set_ylim(-13, -7)
   ax.set_ylabel('diversity')
   #markers, caps, bars = ax.errorbar(x, avg_scores, yerr=std,
   #                                   ecolor='purple')
   #[bar.set_alpha(0.03) for bar in bars]
  #plt.ylabel('diversity')
  #plt.subplots_adjust(top=0.9)
  #plt.legend()
   ax = axs[i+1]
   ax.errorbar(x, y[:,i+1,:].mean(axis=0), yerr=y[:,i+1,:].std(axis=0), label='lifespans')
  #ax.text(10, 0, '{:.2}'.format(y[:,i+1,:].mean()))
  #plt.ylabel('lifespans')
   ax.set_ylabel('lifespans')
   ax.set_ylim(0, config.EVALUATION_HORIZON)
   plt.text(0.8, 0.8-(i+1)*0.162, '{:.2}'.format(y[:,i+1,:].mean()), fontsize=12, transform=plt.gcf().transFigure)
   plt.xlabel('tick')
   plt.tight_layout()
   ax.legend(loc='upper left')

   if render:
      plt.show()


import copy
def unregister():
   for env in copy.deepcopy(gym.envs.registry.env_specs):
      if 'GDY' in env:
         print("Remove {} from registry".format(env))
         del gym.envs.registry.env_specs[env]


class RLlibEvaluator(evaluator.Base):
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config, trainer, archive=None, createEnv=None):
      super().__init__(config)
      self.i = 0
      self.trainer  = trainer

      if config.GRIDDLY:
#        self.policy_id = 'default_policy'
         self.policy_id = 'policy_0'
      else:
         self.policy_id = 'policy_0'
      self.model    = self.trainer.get_policy(self.policy_id).model
      if self.config.MAP != 'PCG':
#        self.config.ROOT = self.config.MAP
         self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.MAP, 'maps', 'map')
      if self.config.GRIDDLY:

         self.env = createEnv({'config': config})
      else:
         self.env      = projekt.rllib_wrapper.RLlibEnv({'config': config})

      if archive is not None:
         self.maps = maps = dict([(ind.idx, ind.chromosome.map_arr) for ind in archive])
         idx = list(maps.keys())[np.random.choice(len(maps))]
         self.env.set_map(idx=idx, maps=maps)
      self.env.reset(idx=config.INFER_IDX, step=False)
#     self.env.reset(idx=0, step=False)
      if not config.GRIDDLY:
         self.registry = OverlayRegistry(self.env, self.model)#, trainer, config)
      self.obs      = self.env.step({})[0]

      self.state    = {}

      if config.EVALUATE:
         self.eval_path_map = os.path.join('eval_experiment', self.config.MAP.split('/')[-1])

         try:
            os.mkdir(self.eval_path_map)
         except FileExistsError:
            print('Eval result directory exists for this map, will overwrite any existing files: {}'.format(self.eval_path_map))

         self.eval_path_map = os.path.join(self.eval_path_map, str(self.config.INFER_IDX))

         try:
            os.mkdir(self.eval_path_map)
         except FileExistsError:
            print('Eval result directory exists for this map, will overwrite any existing files: {}'.format(self.eval_path_map))

         self.eval_path_model = os.path.join(self.eval_path_map, self.config.MODEL.split('/')[-1])

         try:
            os.mkdir(self.eval_path_model)
         except FileExistsError:
            print('Eval result directory exists for this model, will overwrite any existing files: {}'.format(self.eval_path_model))

         self.calc_diversity = diversity_calc(config)

   def render(self):
      self.obs = self.env.reset(idx=self.config.INFER_IDX)
      self.registry = RLlibOverlayRegistry(
            self.config, self.env).init(self.trainer, self.model)
      super().render()

#   def tick(self, pos, cmd):
#      '''Simulate a single timestep
#
#      Args:
#          pos: Camera position (r, c) from the server)
#          cmd: Console command from the server
#      '''
#      actions, self.state, _ = self.trainer.compute_actions(
#            self.obs, state=self.state, policy_id='policy_0')
#      super().tick(self.obs, actions, pos, cmd)


   def evaluate(self, generalize=False):

      model_name = self.config.MODEL.split('/')[-1]
      map_name = self.config.MAP.split('/')[-1] 
      map_idx = self.config.INFER_IDX
      exp_name = 'MODEL_{}_MAP_{}_ID{}_{}steps'.format(model_name, map_name, map_idx, self.config.EVALUATION_HORIZON)
      # Render the map in case we hadn't already
      map_arr = self.env.realm.map.inds()
      map_generator = MapGenerator(self.config)
      t_start = self.config.TERRAIN_BORDER
      t_end = self.config.TERRAIN_SIZE - self.config.TERRAIN_BORDER
      Save.render(map_arr[t_start:t_end, t_start:t_end],
            map_generator.textures, os.path.join(self.eval_path_map, '{} map {}.png'.format(self.config.MAP.split('/')[-1], self.config.INFER_IDX)))
      ts = np.arange(self.config.EVALUATION_HORIZON)
      n_evals = self.config.N_EVAL
      n_metrics = len(DIV_CALCS) + 1 
      n_skills = len(self.config.SKILLS)
      div_mat = np.zeros((n_evals, n_metrics, self.config.EVALUATION_HORIZON))
#     heatmaps = np.zeros((n_evals, self.config.EVALUATION_HORIZON, n_skills + 1, self.config.TERRAIN_SIZE, self.config.TERRAIN_SIZE))
      heatmaps = np.zeros((n_evals, n_skills + 1, self.config.TERRAIN_SIZE, self.config.TERRAIN_SIZE))
      final_stats = []

      data_path = os.path.join(self.eval_path_model, '{} eval.npy'.format(exp_name))
      if self.config.NEW_EVAL:
         for i in range(n_evals):
            self.env.reset(idx=self.config.INFER_IDX)
            self.obs = self.env.step({})[0]
            self.state = {}
            self.registry = OverlayRegistry(self.config, self.env)
            # array of data: diversity scores, lifespans...
            divs = np.zeros((len(DIV_CALCS) + 1, self.config.EVALUATION_HORIZON))
            for t in tqdm(range(self.config.EVALUATION_HORIZON)):
               self.tick(None, None)
   #           print(len(self.env.realm.players.entities))
               div_stats = self.env.get_all_agent_stats()
               for j, (calc_diversity, div_name) in enumerate(DIV_CALCS):
                  diversity = calc_diversity(div_stats, verbose=False)
                  divs[j, t] = diversity
               lifespans = div_stats['lifespans']
               divs[j + 1, t] = np.mean(lifespans)
               div_mat[i] = divs
               # This is a crazy bit where we construct heatmaps but should we not just use get_agent_stats()?
               for _, ent in self.env.realm.players.entities.items():
                  r, c = ent.pos
                  for si, skill in enumerate(self.config.SKILLS):
                     if skill == 'exploration':
                        xp = len(ent.explored) * 20
                     else:
                        xp = getattr(ent.skills, skill).exp
#                    heatmaps[i, t, si, r, c] = xp
                     heatmaps[i, si, r, c] += xp
                  # What is this doing?
                  heatmaps[i, si+1, r, c] += 1
            final_stats.append(div_stats)
         with open(data_path, 'wb') as f:
            np.save(f, np.array(final_stats))
            np.save(f, div_mat)
            np.save(f, heatmaps)
      else:
         with open(data_path, 'rb') as f:
            final_stats = np.load(f, allow_pickle=True)
            div_mat = np.load(f)
            heatmaps = np.load(f)

      plot_name = 'diversity {}'.format(exp_name)
      plot_diversity(ts, div_mat, [d[1] for d in DIV_CALCS], exp_name, self.config)
      plt.savefig(os.path.join(self.eval_path_model, exp_name), dpi=96)
      plt.close()
#     heat_out = heatmaps.mean(axis=0).mean(axis=0)
      # mean over evals
      heat_out = heatmaps.mean(axis=0)
      for s_heat, s_name in zip(heat_out, self.config.SKILLS + ['visited']):
         fig, ax = plt.subplots()
         ax.title.set_text('{} heatmap'.format(s_name))
         map_arr = self.env.realm.map.inds()
         mask = (map_arr == Water.index ) + (map_arr == Lava.index) + (map_arr == Stone.index)
         s_heat = np.ma.masked_where((mask==True), s_heat)
         s_heat = np.flip(s_heat, 0)
#        s_heat = np.log(s_heat + 1)
         im = ax.imshow(s_heat, cmap='cool')
         ax.set_xlim(self.config.TERRAIN_BORDER, self.config.TERRAIN_SIZE-self.config.TERRAIN_BORDER)
         ax.set_ylim(self.config.TERRAIN_BORDER, self.config.TERRAIN_SIZE-self.config.TERRAIN_BORDER)
         cbar = ax.figure.colorbar(im, ax=ax)
         cbar.ax.set_ylabel('{} (log(xp)/tick)'.format(s_name))
         plt.savefig(os.path.join(self.eval_path_model, '{} heatmap {}.png'.format(s_name, exp_name)))

      mean_divs = {}
      means_np = div_mat.mean(axis=-1).mean(axis=0)
      stds_np = div_mat.mean(axis=-1).std(axis=0)
      for j, (_, div_name) in enumerate(DIV_CALCS):
         mean_divs[div_name] = {}
         mean_divs[div_name]['mean'] = means_np[j]
         mean_divs[div_name]['std'] = stds_np[j]
      mean_divs['lifespans'] = means_np[j+1]
      with open(os.path.join(self.eval_path_model, 'stats.json'), 'w') as outfile:
         json.dump(mean_divs, outfile, indent=2)

      from sklearn.manifold import TSNE
      tsne = TSNE(n_components=2, random_state=0)
      final_agent_skills = np.vstack([stats['skills'][0] for stats in final_stats])
      X_2d = tsne.fit_transform(final_agent_skills)
      plt.close()
      plt.figure()
      plt.title('TSNE plot of agents')
      colors = np.hstack([stats['lifespans'] for stats in final_stats])
     #colors = lifespans
      sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors)
      cbar = plt.colorbar(sc)
      cbar.ax.set_ylabel('lifespans')
      plt.savefig(os.path.join(self.eval_path_model, 'TSNE {}.png'.format(exp_name)))
      plt.close()
      plt.figure()
      p1 = plt.bar(np.arange(final_agent_skills.shape[0]), final_agent_skills.mean(axis=1), 5, yerr=final_agent_skills.std(axis=1))
      plt.title('agent bars {}'.format(exp_name))
      plt.close()
      plt.figure()
      p1 = plt.bar(np.arange(final_agent_skills.shape[1]), final_agent_skills.mean(axis=0), 1, yerr=final_agent_skills.std(axis=0))
      plt.xticks(np.arange(final_agent_skills.shape[1]), self.config.SKILLS)
      plt.ylabel('experience points')
      plt.title('skill bars {}'.format(exp_name))
      plt.savefig(os.path.join(self.eval_path_model, 'skill bars {}.png'.format(exp_name)))
      plt.close()
      plt.figure()
      plt.title('agent-skill matrix {}'.format(exp_name))
      im, cbar = heatmap(final_agent_skills, {}, self.config.SKILLS)
      plt.savefig('agent-skill matrix {}'.format(exp_name))
      if final_agent_skills.shape[1] == 2:
         plot_div_2d(final_stats)
#        plt.figure()
#        plt.title('Agents')
#        sc = plt.scatter(final_agent_skills[:, 0], final_agent_skills[:, 1], c=colors)
#        cbar = plt.colorbar(sc)
#        cbar.ax.set_ylabel('lifespans')
#        plt.ylabel('woodcutting')
#        plt.xlabel('mining')
#        plt.savefig(os.path.join(self.eval_path_model, 'agents scatter.png'.format(exp_name)))

#     print('Diversity: {}'.format(diversity))

      log = InkWell()
      log.update(self.env.terminal())

      fpath = os.path.join(self.config.LOG_DIR, self.config.LOG_FILE)
      np.save(fpath, log.packet)


   def tick(self, pos, cmd):
      '''Compute actions and overlays for a single timestep
      Args:
          pos: Camera position (r, c) from the server)
          cmd: Consol command from the server
      '''

      #Compute batch of actions
      actions, self.state, _ = self.trainer.compute_actions(
            self.obs, state=self.state, policy_id=self.policy_id)
#     actions = dict([(i, (2, np.random.randint(5))) for (i, val) in self.env.env.action_space.sample().items()])
#     actions = dict([(i, val) for (i, val) in self.env.env.action_space.sample().items()])
      if not self.config.GRIDDLY:
         self.registry.step(self.obs, pos, cmd,
          # update='counts values attention wilderness'.split())
                            )

      #Step environment
      if hasattr(self.env, 'evo_dones') and self.env.evo_dones is not None:
         self.env.evo_dones['__all__'] = False
      ret = super().tick(self.obs, actions, pos, cmd)

      if self.config.GRIDDLY:
         if self.env.dones['__all__'] == True:
               self.reset_env()

      stats = self.env.get_all_agent_stats()
      score = self.calc_diversity(stats, verbose=True)

      self.i += 1

   def reset_env(self):
      stats = self.env.send_agent_stats()
      score = self.calc_diversity(stats, verbose=True)
      #     score = DIV_CALCS[1][0](stats, verbose=True)
      print(score)
      self. i = 0
      maps = self.maps
      idx = list(maps.keys())[np.random.choice(len(maps))]
      self.env.set_map(idx=idx, maps=maps)
      self.env.reset()

#class RLlibPolicy(RecurrentNetwork, nn.Module):
###############################################################################
### RLlib Policy, Evaluator, and Trainer wrappers
class RLlibPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      action_space = actionSpace(self.config)
      if hasattr(action_space, 'spaces'):
         self.space  = actionSpace(self.config).spaces
      else:
         self.space = action_space

      #Select appropriate baseline model

      if self.config.MODEL == 'attentional':
         self.model  = baseline.Attentional(self.config)
      elif self.config.MODEL == 'convolutional':
         self.model  = baseline.Simple(self.config)
      else:
         self.model  = baseline.Recurrent(self.config)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
      #Flatten structured logits for RLlib

      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn

#class RLlibEvaluator(evaluator.Base):
#   '''Test-time evaluation with communication to
#   the Unity3D client. Makes use of batched GPU inference'''
#   def __init__(self, config, trainer):
#      super().__init__(config)
#      self.trainer  = trainer
#
#      self.model    = self.trainer.get_policy('policy_0').model
#      self.env      = RLlibEnv({'config': config})
#      self.state    = {}
#
###      env = base_env.envs[0]
##
##      for key in RLlibLogCallbacks.STEP_KEYS:
##         if not hasattr(env, key):
##            continue
##         episode.hist_data[key].append(getattr(env, key))
##
##   def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
##         policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
##      env = base_env.envs[0]
##
##      for key in RLlibLogCallbacks.EPISODE_KEYS:
##         if not hasattr(env, key):
##            continue
##         episode.hist_data[key].append(getattr(env, key))
#   def render(self):
#      self.obs = self.env.reset(idx=1)
#      self.registry = RLlibOverlayRegistry(
#            self.config, self.env).init(self.trainer, self.model)
#      super().render()
#
#   def tick(self, pos, cmd):
#      '''Simulate a single timestep
#
#      Args:
#          pos: Camera position (r, c) from the server)
#          cmd: Console command from the server
#      '''
#      actions, self.state, _ = self.trainer.compute_actions(
#            self.obs, state=self.state, policy_id='policy_0')
#      super().tick(self.obs, actions, pos, cmd)
#
#global GOT_DUMMI
#GOT_DUMMI = False
#global EXEC_RETURN


def frozen_execution_plan(workers: WorkerSet, config: TrainerConfigDict):
    # Collects experiences in parallel from multiple RolloutWorker actors.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    global EXEC_RETURN
    if GOT_DUMMI:
       train_op = rollouts.combine(ConcatBatches(min_batch_size=config["train_batch_size"])).for_each(lambda x: None)
       return None
    else:

       # Combine experiences batches until we hit `train_batch_size` in size.
       # Then, train the policy on those experiences and update the workers.
       train_op = rollouts \
           .combine(ConcatBatches(
               min_batch_size=config["train_batch_size"])) \
           .for_each(TrainOneStep(workers))

       # Add on the standard episode reward, etc. metrics reporting. This returns
       # a LocalIterator[metrics_dict] representing metrics for each train step.
       config['timesteps_per_iteration'] = -1
       config['min_iter_time_s'] = -1
       config['metrics_smoothing_episodes'] = -1
       EXEC_RETURN = StandardMetricsReporting(train_op, workers, config)
    return EXEC_RETURN


import logging
logger = logging.getLogger(__name__)

class EvoPPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer. Evolution edition.'''
   def __init__(self, env, path, config, execution_plan):
      self.nmmo_config = config['env_config']['config']
      super().__init__(env=env, config=config)
#     self.execution_plan = execution_plan
#     self.train_exec_impl = execution_plan(self.workers, config)
      self.saveDir = path
      self.pathDir = '/'.join(path.split(os.sep)[:-1])
      self.init_epoch = True


   # FIXME: AWFUL hack, purely to override overriding of batch mode when
   # initializing eval workers.
   @override(Trainable)
   def setup(self, config: PartialTrainerConfigDict):
      env = self._env_id
      if env:
         config["env"] = env
         # An already registered env.
         if _global_registry.contains(ENV_CREATOR, env):
            self.env_creator = _global_registry.get(ENV_CREATOR, env)
         # A class specifier.
         elif "." in env:
            self.env_creator = \
               lambda env_context: from_config(env, env_context)
         # Try gym/PyBullet.
         else:

            def _creator(env_context):
               import gym
               # Allow for PyBullet envs to be used as well (via string).
               # This allows for doing things like
               # `env=CartPoleContinuousBulletEnv-v0`.
               try:
                  import pybullet_envs
                  pybullet_envs.getList()
               except (ModuleNotFoundError, ImportError):
                  pass
               return gym.make(env, **env_context)

            self.env_creator = _creator
      else:
         self.env_creator = lambda env_config: None

      # Merge the supplied config with the class default, but store the
      # user-provided one.
      self.raw_user_config = config
      self.config = Trainer.merge_trainer_configs(self._default_config,
                                                  config)
# NMMO won't use tf1 :D

#     # Check and resolve DL framework settings.
#     # Enable eager/tracing support.
#     if tf1 and self.config["framework"] in ["tf2", "tfe"]:
#        if self.config["framework"] == "tf2" and tfv < 2:
#           raise ValueError("`framework`=tf2, but tf-version is < 2.0!")
#        if not tf1.executing_eagerly():
#           tf1.enable_eager_execution()
#        logger.info("Executing eagerly, with eager_tracing={}".format(
#           self.config["eager_tracing"]))
#     if tf1 and not tf1.executing_eagerly() and \
#             self.config["framework"] != "torch":
#        logger.info("Tip: set framework=tfe or the --eager flag to enable "
#                    "TensorFlow eager execution")

      if self.config["normalize_actions"]:
         inner = self.env_creator

         def normalize(env):
            import gym  # soft dependency
            if not isinstance(env, gym.Env):
               raise ValueError(
                  "Cannot apply NormalizeActionActionWrapper to env of "
                  "type {}, which does not subclass gym.Env.", type(env))
            return NormalizeActionWrapper(env)

         self.env_creator = lambda env_config: normalize(inner(env_config))

      Trainer._validate_config(self.config)
      if not callable(self.config["callbacks"]):
         raise ValueError(
            "`callbacks` must be a callable method that "
            "returns a subclass of DefaultCallbacks, got {}".format(
               self.config["callbacks"]))
      self.callbacks = self.config["callbacks"]()
      log_level = self.config.get("log_level")
      if log_level in ["WARN", "ERROR"]:
         logger.info("Current log_level is {}. For more information, "
                     "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
                     "-vv flags.".format(log_level))
      if self.config.get("log_level"):
         logging.getLogger("ray.rllib").setLevel(self.config["log_level"])

      def get_scope():
#        if tf1 and not tf1.executing_eagerly():
#           return tf1.Graph().as_default()
#        else:
         return open(os.devnull)  # fake a no-op scope

      with get_scope():
         self._init(self.config, self.env_creator)

         # Evaluation setup.
         if self.config.get("evaluation_interval"):
            # Update env_config with evaluation settings:
            extra_config = copy.deepcopy(self.config["evaluation_config"])
            # Assert that user has not unset "in_evaluation".
            assert "in_evaluation" not in extra_config or \
                   extra_config["in_evaluation"] is True
            extra_config.update({
#              "batch_mode": "complete_episodes",
               # FIXME: what is this shit hahah
               "rollout_fragment_length": 10,
               "in_evaluation": True,
            })
            logger.debug(
               "using evaluation_config: {}".format(extra_config))

            self.evaluation_workers = self._make_workers(
               env_creator=self.env_creator,
               validate_env=None,
               policy_class=self._policy_class,
               config=merge_dicts(self.config, extra_config),
               num_workers=self.config["evaluation_num_workers"])
            self.evaluation_metrics = {}


   def log_result(self, stuff):
      return
#     if self.init_epoch:
#        self.init_epoch = False
#        return
#     else:
#        super().log_result(stuff)

   def reset(self):
      #TODO: is this doing anythiing??
     #print('sane reset evoTrainer \n')
     #print(self.workers.local_worker, self.workers.remote_workers)
      super().reset(self.config)
#     raise Exception

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      savedir = super().save(self.saveDir)
     #with open('evo_experiment/path.txt', 'w') as f:
      with open(os.path.join(self.pathDir, 'path.txt'), 'w') as f:
         f.write(savedir)
      print('Saved to: {}'.format(savedir))

      return savedir

   def restore(self, model):
      '''Restore model from path'''

      if model is None:
         print('Training from scratch...')

         return

      if model == 'current':
          with open('experiment/path.txt') as f:
             path = f.read().splitlines()[0]

      elif model == 'pretrained':
          with open(os.path.join(Path(self.pathDir).parent.parent, 'experiment', 'path.txt')) as f:
             path = f.read().splitlines()[0]
#         with open(os.path.join(self.pathDir, 'path.txt')) as f:
      elif model == 'reload':
#        path = '/'.join(model.split('/')[1:])
         path = os.path.join(self.pathDir, 'path.txt')
         with open(path) as f:
            path = f.read().splitlines()[0]
         path = os.path.abspath(path)
      elif self.nmmo_config.FROZEN:
         path = os.path.join('evo_experiment', model, 'path.txt')
         with open(path) as f:
            path = f.read().splitlines()[0]
         path = os.path.abspath(path)
      else:
         path = model
#        pass
     #else:
     #   raise Exception("Invalid model. {}".format(path))
     #   path = 'experiment/{}/checkpoint'.format(model)

      print('Loading from: {}'.format(path))
      super().restore(path)

#     if self.config['env_config']['config'].FROZEN:
#        workers = self.evaluation_workers
#        for worker in [workers.local_worker()] + workers.remote_workers():
#            worker.batch_mode = 'truncate_episodes'

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      model = self.get_policy(policyID).model
     #print('sending evo trainer model to gpu\n')
    #     model.cuda()
      return model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self, maps):
      # TODO: pass only the relevant map?
#     idxs = iter(maps.keys())
      idxs = list(maps.keys())
#     if self.config['env_config']['config'].GRIDDLY:
#        self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.set_map(idx=None, maps=maps)))
      #NOTE: you can't iteratively send indexes to environment with 'foreach_worker', multiprocessing will thwart you
      i = 0
      if self.nmmo_config.FROZEN:
         workers = self.evaluation_workers
      else:
         workers = self.workers

      if self.nmmo_config.N_PROC == self.nmmo_config.N_EVO_MAPS:
         for worker in [workers.local_worker()] + workers.remote_workers():
            if len(idxs) > 0:
               fuck_id = idxs[i % len(idxs)]
            else:
               fuck_id = idxs[i]
            i += 1
            # FIXME: must have N_PROC = N_EVO_MAPS?
         #  worker.foreach_env.remote(lambda env: env.set_map(idx=next(idxs), maps=maps))
            if isinstance(worker, RolloutWorker):
               worker.foreach_env(lambda env: env.set_map(idx=fuck_id, maps=maps))
            else:
               worker.foreach_env.remote(lambda env: env.set_map(idx=fuck_id, maps=maps))
      else:
         # Ha ha what the fuck
         if 'maps' in maps:
            maps = maps['maps']
         workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.set_map(idx=None, maps=maps)))

      if self.config['env_config']['config'].FROZEN:
         stats = self.simulate_frozen()
      else:
         stats = self.simulate_unfrozen()
      if self.config['env_config']['config'].FROZEN and False:
         global_stats = ray.get_actor('global_stats')
         stats = ray.get(global_stats.get.remote())
         global_stats.reset.remote()
         print('stats keys', stats.keys())
      else:
         stats_list = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: (env.worldIdx, env.send_agent_stats())))
         stats = {}
         for worker_stats in stats_list:
            if not worker_stats: continue
            for (envID, env_stats) in worker_stats:
               if not env_stats: continue
               if envID not in stats:
                  stats[envID] = env_stats
               else:
                  for (k, v) in env_stats.items():
                     if k not in stats[envID]:
                        stats[envID][k] = v
                     else:
                        stats[envID][k] += v

      return stats

   def simulate_frozen(self):
      stats = super()._evaluate()

      # FIXME: switch this off when already saving for other reasons; control from config
      if self.training_iteration < 100:
         save_interval = 10
      else:
         save_interval = 100

      if self.training_iteration % save_interval == 0:
         self.save()

   def reset_envs(self):
      obs = self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.reset({}, step=True)))
#     obs = [ob for worker_obs in obs for ob in worker_obs]

   def simulate_unfrozen(self):
      stats = super().train()

      # FIXME: switch this off when already saving for other reasons; control from config
      if self.training_iteration < 100:
         save_interval = 10
      else:
         save_interval = 100

      if self.training_iteration % save_interval == 0:
         self.save()

      nSteps = stats['info']['num_steps_trained']
      VERBOSE = False

      if VERBOSE:
         print('Epoch: {}, Samples: {}'.format(self.training_iteration, nSteps))
      hist = stats['hist_stats']

      for key, stat in hist.items():
         if len(stat) == 0 or key == 'map_fitness':
            continue

         if VERBOSE:
            print('{}:: Total: {:.4f}, N: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}'.format(
                  key, np.sum(stat), len(stat), np.mean(stat), np.std(stat), np.min(stat), np.max(stat)))
        #if key == 'map_fitness':
        #    print('DEBUG MAP FITNESS PRINTOUT')
        #    print(hist[key])
         hist[key] = []

      return stats



class SanePPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer'''
   def __init__(self, config):
      self.envConfig = config['env_config']['config']
      super().__init__(env=self.envConfig.ENV_NAME, config=config)

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
#     savedir = super().save(self.saveDir)
#     with open('experiment/path.txt', 'w') as f:
#        f.write(savedir)

      config   = self.envConfig
      saveFile = super().save(config.PATH_CHECKPOINTS)
      saveDir  = os.path.dirname(saveFile)

      #Clear current save dir
      shutil.rmtree(config.PATH_CURRENT, ignore_errors=True)
      os.mkdir(config.PATH_CURRENT)

      #Copy checkpoints
      for f in os.listdir(saveDir):
         stripped = re.sub('-\d+', '', f)
         src      = os.path.join(saveDir, f)
         dst      = os.path.join(config.PATH_CURRENT, stripped)
         shutil.copy(src, dst)

      print('Saved to: {}'.format(saveDir))

#     return savedir

   def restore(self, model):
      '''Restore model from path'''

      config   = self.envConfig
      if model is None:
         print('Initializing new model...')
         trainPath = config.PATH_TRAINING_DATA.format('current')
         np.save(trainPath, {})
         return


      if model == 'current':
         modelPath = config.PATH_MODEL.format(config.MODEL)
         print('Loading from: {}'.format(modelPath))
         path     = os.path.join(modelPath, 'checkpoint')
         super().restore(path)
#        with open('experiment/path.txt') as f:
#           path = f.read().splitlines()[0]
      elif model.startswith('evo_experiment'):
#        path = '/'.join(model.split('/')[1:])
         path = os.path.join(model, 'path.txt')
         with open(path) as f:
            path = f.read().splitlines()[0]
         #FIXME dumb hack
         path = '{}/{}/{}'.format(path.split('/')[0],
               'greene',
               '/'.join(path.split('/')[1:]),
               )
         path = os.path.abspath(path)
      else:
         path = os.path.join(model, 'path.txt')
         path = os.path.join('evo_experiment', path)
         with open(path) as f:
            path = f.read().splitlines()[0]
         path = os.path.abspath(path)
         #FIXME dumb hack

#     else:
#        path = 'experiment/{}/checkpoint'.format(model)
#     saveFile = super().save(config.PATH_CHECKPOINTS)
#     saveDir  = os.path.dirname(saveFile)
#
#     #Clear current save dir
#     shutil.rmtree(config.PATH_CURRENT, ignore_errors=True)
#     os.mkdir(config.PATH_CURRENT)

#     #Copy checkpoints
#     for f in os.listdir(saveDir):
#        stripped = re.sub('-\d+', '', f)
#        src      = os.path.join(saveDir, f)
#        dst      = os.path.join(config.PATH_CURRENT, stripped)
#        shutil.copy(src, dst)

#     print('Saved to: {}'.format(saveDir))

#  def restore(self, model):
#     '''Restore model from path'''
      config    = self.envConfig





   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      '''Train forever, printing per epoch'''
      config = self.envConfig

      logo   = open(config.PATH_LOGO).read().splitlines()

      model         = config.MODEL if config.MODEL is not None else 'current'
      trainPath     = config.PATH_TRAINING_DATA.format(model)
      training_logs = np.load(trainPath, allow_pickle=True).item()

      total_sample_time = 0
      total_learn_time = 0

      epoch   = 0
      blocks  = []

      while True:
          #Train model
          stats = super().train()
          self.save()
          epoch += 1

          if epoch == 1:
            continue

          #Time Stats
          timers             = stats['timers']
          sample_time        = timers['sample_time_ms'] / 1000
          learn_time         = timers['learn_time_ms'] / 1000
          sample_throughput  = timers['sample_throughput']
          learn_throughput   = timers['learn_throughput']

          #Summary
          nSteps             = stats['info']['num_steps_trained']
          total_sample_time += sample_time
          total_learn_time  += learn_time
          summary = formatting.box([formatting.line(
                title  = 'Neural MMO v1.5',
                keys   = ['Epochs', 'kSamples', 'Sample Time', 'Learn Time'],
                vals   = [epoch, nSteps/1000, total_sample_time, total_learn_time],
                valFmt = '{:.1f}')])

          #Block Title
          sample_stat = '{:.1f}/s ({:.1f}s)'.format(sample_throughput, sample_time)
          learn_stat  = '{:.1f}/s ({:.1f}s)'.format(learn_throughput, learn_time)
          header = formatting.box([formatting.line(
                keys   = 'Epoch Sample Train'.split(),
                vals   = [epoch, sample_stat, learn_stat],
                valFmt = '{}')])

          #Format stats (RLlib callback format limitation)
          for k, vals in stats['hist_stats'].items():
             if not k.startswith('_'):
                continue
             k                 = k.lstrip('_')
             track, stat       = re.split('_', k)

             if track not in training_logs:
                training_logs[track] = {}

#          epoch += 1
#
#          block = []
#
#          for key, stat in stats['hist_stats'].items():
#             if key.startswith('_') and len(stat) > 0:
#                stat       = stat[-self.envConfig.TRAIN_BATCH_SIZE:]
#                mmin, mmax = np.min(stat),  np.max(stat)
#                mean, std  = np.mean(stat), np.std(stat)
#
#                block.append(('   ' + left + '{:<12}{}Min: {:8.1f}{}Max: {:8.1f}{}Mean: {:8.1f}{}Std: {:8.1f}').format(
#                      key.lstrip('_'), sep, mmin, sep, mmax, sep, mean, sep, std))
#
#             if not self.envConfig.v:
#                continue
             if stat not in training_logs[track]:
                training_logs[track][stat] = []

             training_logs[track][stat] += vals

          np.save(trainPath, training_logs)

          #Representation for CLI
          cli = {}
          for track, stats in training_logs.items():
             cli[track] = {}
             for stat, vals in stats.items():
                mmean = np.mean(vals[-config.TRAIN_SUMMARY_ENVS:])
                cli[track][stat] = mmean

#          if len(block) > 0:
#             mmax = max(len(l) for l in block) + 1
#
#             for idx, l in enumerate(block):
#                block[idx] = ('{:<'+str(mmax)+'}').format(l + right)
#
#             blocks.append([top*len(line), line, bot*len(line), '   ' +
#                   top*(mmax-3)] + block + ['   ' + bot*(mmax-3)])
#
#
#          if len(blocks) > 3:
#             blocks = blocks[1:]
#
#          for block in blocks:
#             for line in block:
#                lines.append(' ' + line)
#
#          line = summary.format(sep, epoch, sep, nSteps, sep, total_sample_time, sep, total_learn_time)
#          lines.append(' ' + top*len(line))
#          lines.append(' ' + line)
#          lines.append(' ' + bot*len(line))
          lines = formatting.precomputed_stats(cli)
          if config.v:
             lines += formatting.timings(timings)

          #Extend blocks
          if len(lines) > 0:
             blocks.append(header + formatting.box(lines, indent=4))
             
          if len(blocks) > 3:
             blocks = blocks[1:]
          
          #Assemble Summary Bar Title
          lines = logo.copy() + list(chain.from_iterable(blocks)) + summary

          #Cross-platform clear screen
          os.system('cls' if os.name == 'nt' else 'clear')

          for idx, line in enumerate(lines):
             print(line)

###############################################################################
### RLlib Overlays
class RLlibOverlayRegistry(OverlayRegistry):
   '''Host class for RLlib Map overlays'''
   def __init__(self, config, realm):
      super().__init__(config, realm)

      self.overlays['values']       = Values
      self.overlays['attention']    = Attention
      self.overlays['tileValues']   = TileValues
      self.overlays['entityValues'] = EntityValues

class RLlibOverlay(Overlay):
   '''RLlib Map overlay wrapper'''
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      tiles      = self.realm.realm.map.tiles
      players    = self.realm.realm.players

      attentions = defaultdict(list)
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         player = players[playerID]
         r, c   = player.pos

         rad     = self.config.STIM
         obTiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()

         for tile, a in zip(obTiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      sz    = self.config.TERRAIN_SIZE
      data  = np.zeros((sz, sz))
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.register(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      players = self.realm.realm.players
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         r, c = players[playerID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = overlay.twoTone(self.values[:, :])
      self.realm.register(colorized)

def zeroOb(ob, key):
   for k in ob[key]:
      ob[key][k] *= 0

class GlobalValues(RLlibOverlay):
   '''Abstract base for global value functions'''
   def init(self, zeroKey):
      if self.trainer is None:
         return

      print('Computing value map...')
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.dense()
      values    = 0 * self.values

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         ob             = obs[agentID]
         batch[agentID] = ob
         zeroOb(ob, zeroKey)
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c         = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = overlay.twoTone(values)

   def register(self, obs):
      print('Computing Global Values. This requires one NN pass per tile')
      self.init()

      self.realm.register(self.colorized)

class TileValues(GlobalValues):
   def init(self, zeroKey='Entity'):
      '''Compute a global value function map excluding other agents. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

class EntityValues(GlobalValues):
   def init(self, zeroKey='Tile'):
      '''Compute a global value function map excluding tiles. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)


###############################################################################
### Logging
class RLlibLogCallbacks(DefaultCallbacks):
   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      assert len(base_env.envs) == 1, 'One env per worker'
      env    = base_env.envs[0]
      config = env.config

      for key, vals in env.terminal()['Stats'].items():
         logs = episode.hist_data
         key  = '_' + key

         logs[key + '_Min']  = [np.min(vals)]
         logs[key + '_Max']  = [np.max(vals)]
         logs[key + '_Mean'] = [np.mean(vals)]
         logs[key + '_Std']  = [np.std(vals)]
