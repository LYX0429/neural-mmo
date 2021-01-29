import os
from pathlib import Path
from collections import defaultdict
from pdb import set_trace as T
from typing import Dict
import json

import gym
from matplotlib import pyplot as plt
import numpy as np
import ray
import ray.rllib.agents.ppo.ppo as ppo
import torch
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy import Policy
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.spaces.flexdict import FlexDict
from forge.blade.lib.enums import Water, Lava, Stone
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

from evolution.diversity import DIV_CALCS

from ray.rllib.execution.metric_ops import StandardMetricsReporting

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep

from ray.rllib.utils.typing import EnvConfigDict, EnvType, ResultDict, TrainerConfigDict

#Moved log to forge/trinity/env
class RLLibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      self.headers = self.config.SKILLS
      super().__init__(self.config)
      self.evo_dones = None


   def init_skill_log(self):
      self.skill_log_path = './evo_experiment/{}/map_{}_skills.csv'.format(self.config.EVO_DIR, self.worldIdx)
      with open(self.skill_log_path, 'w', newline='') as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=self.skill_headers)
         writer.writeheader()
      assert csvfile.closed


   def step(self, decisions, omitDead=False, preprocessActions=True):
      obs, rewards, dones, infos = super().step(decisions,
            omitDead=omitDead, preprocessActions=preprocessActions)

      t, mmean = len(self.lifetimes), np.mean(self.lifetimes)

      if not self.config.EVALUATE and t >= self.config.TRAIN_HORIZON:
         dones['__all__'] = True

      # are we doing evolution?

 #    if self.config.EVO_MAP and not self.config.FIXED_MAPS:# and not self.config.RENDER:
 #       if self.realm.tick >= self.config.MAX_STEPS or self.config.RENDER:
 #          # reset the env manually, to load from the new updated population of maps
##          print('resetting env {} after {} steps'.format(self.worldIdx, self.n_step))
 #          dones['__all__'] = True

      if self.config.EVO_MAP and hasattr(self, 'evo_dones') and self.evo_dones is not None:
         dones = self.evo_dones
         self.evo_dones = None

      return obs, rewards, dones, infos

   def send_agent_stats(self):
      global_stats = ray.get_actor('global_stats')
      stats = self.get_agent_stats()
      global_stats.add.remote(stats, self.worldIdx)
      self.evo_dones = {}
      self.evo_dones['__all__'] = True

   def get_agent_stats(self):
      skills = {}
      a_skills = None

      for d, player in self.realm.players.items():
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
         a_skill_vals['exploration'] = player.exploration_grid.sum() * 20
         # timeAlive will only add expressivity if we fit more than one gaussian.
         a_skill_vals['time_alive'] = player_packet['history']['timeAlive']
         skills[d] = a_skill_vals

      if a_skills:
         stats = np.zeros((len(skills), len(self.headers)))
        #stats = np.zeros((len(skills), 1))
         lifespans = np.zeros((len(skills)))
         # over agents

         for i, a_skills in enumerate(skills.values()):
            # over skills

            for j, k in enumerate(self.headers):
               if k not in ['level', 'cooking', 'smithing']:
 #             if k in ['exploration']:
                  stats[i, j] = a_skills[k]
                  j += 1
            lifespans[i] = a_skills['time_alive']
         stats = {
               'skills': stats,
               'lifespans': lifespans,
               'lifetimes': self.lifetimes,
               }

         return stats

      return {}


#Neural MMO observation space
def observationSpace(config):
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
            low=-2**16, high=2**16, shape=(nRows, nContinuous),
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

   if config.GRIDDLY:
      pass
   return atns

def plot_diversity(x, y, div_names, exp_name, render=False):
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
   ax.set_ylim(0, 1000)
   plt.text(0.8, 0.8-(i+1)*0.162, '{:.2}'.format(y[:,i+1,:].mean()), fontsize=12, transform=plt.gcf().transFigure)
   plt.xlabel('tick')
   plt.tight_layout()
   ax.legend(loc='upper left')

   if render:
      plt.show()




class RLLibEvaluator(evaluator.Base):
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config, trainer):
      super().__init__(config)
      self.trainer  = trainer

      self.model    = self.trainer.get_policy('policy_0').model
      if self.config.MAP != 'PCG':
#        self.config.ROOT = self.config.MAP
         self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.MAP, 'maps', 'map')
      self.env      = projekt.rllib_wrapper.RLLibEnv({'config': config})

      self.env.reset(idx=self.config.INFER_IDX, step=False)
#     self.env.reset(idx=0, step=False)
      self.registry = OverlayRegistry(self.env, self.model, trainer, config)
      self.obs      = self.env.step({})[0]

      self.state    = {}
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

   def test(self):

      model_name = self.config.MODEL.split('/')[-1]
      map_name = self.config.MAP.split('/')[-1] 
      map_idx = self.config.INFER_IDX
      exp_name = 'MODEL_{}_MAP_{}_ID{}_{}steps'.format(model_name, map_name, map_idx, self.config.EVALUATION_HORIZON)
      # Render the map in case we hadn't already
      map_arr = self.env.realm.map.np()
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
            self.registry = OverlayRegistry(self.env, self.model, self.trainer, self.config)
            # array of data: diversity scores, lifespans...
            divs = np.zeros((len(DIV_CALCS) + 1, self.config.EVALUATION_HORIZON))
            for t in tqdm(range(self.config.EVALUATION_HORIZON)):
               self.tick(None, None)
   #           print(len(self.env.realm.players.entities))
               div_stats = self.env.get_agent_stats()
               for j, (calc_diversity, div_name) in enumerate(DIV_CALCS):
                  diversity = calc_diversity(div_stats, verbose=False)
                  divs[j, t] = diversity
               lifespans = div_stats['lifespans']
               divs[j + 1, t] = np.mean(lifespans)
               div_mat[i] = divs
               for _, ent in self.env.realm.players.entities.items():
                  r, c = ent.pos
                  for si, skill in enumerate(self.config.SKILLS):
                     if skill == 'exploration':
                        xp = ent.exploration_grid.sum()
                     else:
                        xp = getattr(ent.skills, skill).exp
#                    heatmaps[i, t, si, r, c] = xp
                     heatmaps[i, si, r, c] += xp
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
      plot_diversity(ts, div_mat, [d[1] for d in DIV_CALCS], exp_name)
      plt.savefig(os.path.join(self.eval_path_model, exp_name), dpi=96)
      plt.close()
#     heat_out = heatmaps.mean(axis=0).mean(axis=0)
      # mean over evals
      heat_out = heatmaps.mean(axis=0)
      for s_heat, s_name in zip(heat_out, self.config.SKILLS + ['visited']):
         fig, ax = plt.subplots()
         ax.title.set_text('{} heatmap'.format(s_name))
         mask = (self.env.realm.map.np() == Water.index ) + (self.env.realm.map.np() == Lava.index) + (self.env.realm.map.np() == Stone.index)
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
      final_agent_skills = np.vstack([stats['skills'] for stats in final_stats])
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
      stats = self.env.get_agent_stats()
      score = DIV_CALCS[1][0](stats, verbose=True)
      print(score)

      #Compute batch of actions
      actions, self.state, _ = self.trainer.compute_actions(
            self.obs, state=self.state, policy_id='policy_0')
      self.registry.step(self.obs, pos, cmd,
            update='counts values attention wilderness'.split())

      #Step environment
      super().tick(actions)

class Policy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.space  = actionSpace(self.config).spaces

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

class LogCallbacks(DefaultCallbacks):
   STEP_KEYS    = 'env_step preprocess_actions realm_step env_stim'.split()
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

      for key, val in env.terminal()['Stats'].items():
         episode.hist_data['_'+key] = val

global GOT_DUMMI
GOT_DUMMI = False
global EXEC_RETURN


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
       print(config.keys())
       EXEC_RETURN = StandardMetricsReporting(train_op, workers, config)
    return EXEC_RETURN



class EvoPPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer. Evolution edition.'''
   def __init__(self, env, path, config, execution_plan):
      super().__init__(env=env, config=config)
#     self.execution_plan = execution_plan
#     self.train_exec_impl = execution_plan(self.workers, config)
      self.saveDir = path
      self.pathDir = '/'.join(path.split(os.sep)[:-1])

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
      else:
         path = model
#        pass
     #else:
     #   raise Exception("Invalid model. {}".format(path))
     #   path = 'experiment/{}/checkpoint'.format(model)

      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      model = self.get_policy(policyID).model
     #print('sending evo trainer model to gpu\n')
    #     model.cuda()
      return model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
#     self.reset()
#     self.reset_envs()
      stats = self.simulate_unfrozen()
      self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.send_agent_stats()))
#     self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.reset()))
#     return self.simulate_frozen()
#     if self.training_iteration % 2 == 0:
#        return self.simulate_unfrozen()

   def simulate_frozen(self):
#           update='counts values attention wilderness'.split())
      obs = self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.reset({})))
#     actions = self.workers.foreach_policy(lambda pol, str: pol.compute_actions(obs))

      for _ in range(100):
         actions = [self.compute_actions(ob, policy_id='policy_0', state={}) for w_ob in obs for ob in w_ob]
        #actions, self.state, _ = self.compute_actions(
        #      self.obs, state=self.state, policy_id='policy_0')
#     self.registry.step(self.obs, pos, cmd,

         self.workers.foreach_env(lambda env: env.step(actions, env.id))

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
   def __init__(self, env, path, config):
      super().__init__(env=env, config=config)
      self.envConfig = config['env_config']['config']
      self.saveDir   = path

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      savedir = super().save(self.saveDir)
      with open('experiment/path.txt', 'w') as f:
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

      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      '''Train forever, printing per epoch'''
      logo   = open(self.envConfig.LOGO_DIR).read().splitlines()
      epoch  = 0

      total_sample_time = 0
      total_learn_time = 0

      sep     = u'\u2595\u258f'
      block   = u'\u2591'
      top     = u'\u2581'
      bot     = u'\u2594'
      left    = u'\u258f'
      right   = u'\u2595'

      summary = left + 'Neural MMO v1.5{}Epochs: {}{}Samples: {}{}Sample Time: {:.1f}s{}Learn Time: {:.1f}s' + right
      blocks  = []

      while True:
          stats = super().train()
          self.save()

          lines = logo.copy()

          nSteps = stats['info']['num_steps_trained']

          timers             = stats['timers']
          sample_time        = timers['sample_time_ms'] / 1000
          learn_time         = timers['learn_time_ms'] / 1000
          sample_throughput  = timers['sample_throughput']
          learn_throughput   = timers['learn_throughput']

          total_sample_time += sample_time
          total_learn_time  += learn_time

          line = (left + 'Epoch: {}{}Sample: {:.1f}/s ({:.1f}s){}Train: {:.1f}/s ({:.1f}s)' + right).format(
               epoch, sep, sample_throughput, sample_time, sep, learn_throughput, learn_time)

          epoch += 1

          block = []

          for key, stat in stats['hist_stats'].items():
             if key.startswith('_') and len(stat) > 0:
                stat       = stat[-self.envConfig.TRAIN_BATCH_SIZE:]
                mmin, mmax = np.min(stat),  np.max(stat)
                mean, std  = np.mean(stat), np.std(stat)

                block.append(('   ' + left + '{:<12}{}Min: {:8.1f}{}Max: {:8.1f}{}Mean: {:8.1f}{}Std: {:8.1f}').format(
                      key.lstrip('_'), sep, mmin, sep, mmax, sep, mean, sep, std))

             if not self.envConfig.v:
                continue

             if len(stat) == 0:
                continue

             lines.append('{}:: Total: {:.4f}, N: {:.4f}, Mean: {:.4f}, Std: {:.4f}'.format(
                   key, np.sum(stat), len(stat), np.mean(stat), np.std(stat)))

          if len(block) > 0:
             mmax = max(len(l) for l in block) + 1

             for idx, l in enumerate(block):
                block[idx] = ('{:<'+str(mmax)+'}').format(l + right)

             blocks.append([top*len(line), line, bot*len(line), '   ' +
                   top*(mmax-3)] + block + ['   ' + bot*(mmax-3)])


          if len(blocks) > 3:
             blocks = blocks[1:]

          for block in blocks:
             for line in block:
                lines.append(' ' + line)

          line = summary.format(sep, epoch, sep, nSteps, sep, total_sample_time, sep, total_learn_time)
          lines.append(' ' + top*len(line))
          lines.append(' ' + line)
          lines.append(' ' + bot*len(line))

          #Cross-platform clear screen
          os.system('cls' if os.name == 'nt' else 'clear')

          for idx, line in enumerate(lines):
             print(line)
