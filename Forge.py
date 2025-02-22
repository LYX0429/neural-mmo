'''Main file for the neural-mmo/projekt demo

/projeckt contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

Associated docs and tutorials are hosted on jsuarez5341.github.io.'''
from pdb import set_trace as TT

import numpy as np
import torch

from fire import Fire
import json
import copy

import ray
from ray import rllib

from forge.ethyr.torch import utils
from forge.blade.systems import ai

from forge.trinity.visualize import BokehServer
from forge.trinity.evaluator import Evaluator

import projekt
from projekt import rllib_wrapper as wrapper
from forge.blade.core import terrain
from evolution.utils import get_exp_shorthand


def createPolicies(config, mapPolicy):
   '''Generate RLlib policies'''
   obs      = wrapper.observationSpace(config)
   atns     = wrapper.actionSpace(config)
   policies = {}

   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
      key           = mapPolicy(i)
      policies[key] = (None, obs, atns, params)
   return policies

def loadTrainer(config):
   '''Create monolithic RLlib trainer object'''
   if config.load_arguments != -1:
      load_args = json.load(
         open('configs/settings_{}.json'.format(config.load_arguments), 'r'))
      [config.set(k, v) for (k, v) in load_args.items()]
   if config.PAIRED and not config.EVALUATE:
      config.set('NPOP', 2)
      config.set('NPOLICIES', 2)

   torch.set_num_threads(1)
   ray.init(local_mode=config.LOCAL_MODE, ignore_reinit_error=True)

   #Register custom env
   ray.tune.registry.register_env("Neural_MMO",
         lambda config: wrapper.RLlibEnv(config))

   #Create policies
   rllib.models.ModelCatalog.register_custom_model('godsword', wrapper.RLlibPolicy)
   mapPolicy = lambda agentID: 'policy_{}'.format(agentID % config.NPOLICIES)
   policies  = createPolicies(config, mapPolicy)

   #Instantiate monolithic RLlib Trainer object.
   return wrapper.SanePPOTrainer(config={
      'num_workers': config.NUM_WORKERS,
      'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
      'num_gpus': config.NUM_GPUS,
      'num_envs_per_worker': 1,
      'train_batch_size': config.TRAIN_BATCH_SIZE,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
      'num_sgd_iter': config.NUM_SGD_ITER,
      'framework': 'torch',
      'horizon': np.inf,
      'soft_horizon': False, 
      # '_use_trajectory_view_api': False,
      'no_done_at_end': False,
      'callbacks': wrapper.RLlibLogCallbacks,
      'env_config': {
         'config': config,
      },
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': mapPolicy,
         'count_steps_by': 'agent_steps'
      },
      'model': {
         'custom_model': 'godsword',
         'custom_model_config': {'config': config}
      },
   })

def loadEvaluator(config):
   '''Create test/render evaluator'''
   if config.PAIRED:
      pass
   elif config.NPOLICIES > 1 or config.MODEL.startswith('['): # the latter is just in case we're doing multi-policy eval with only 1 model for some reason
      models = config.MODEL.strip('[').strip(']').split(',')
      # Randomize order of models to randomize spawn order for fair evaluation over multiple trials
      np.random.shuffle(models)
      config.set('MULTI_MODEL_EXPERIMENTS', models)
      model_names = [get_exp_shorthand(m) for m in models]
      config.set('MULTI_MODEL_NAMES', model_names)
      return wrapper.RLlibMultiEvaluator(config, loadModels(config))
   else:
      config.set('MULTI_MODEL_NAMES', [get_exp_shorthand(config.MODEL)])
   if config.MODEL not in ('scripted-forage', 'scripted-combat'):
      return wrapper.RLlibEvaluator(config, loadModel(config))

   #Scripted policy backend
   if config.MODEL == 'scripted-forage':
      policy = ai.policy.forage 
   else:
      policy = ai.policy.combat

   #Search backend
   err = 'SCRIPTED_BACKEND may be either dijkstra or dynamic_programming'
   assert config.SCRIPTED_BACKEND in ('dijkstra', 'dynamic_programming'), err
   if config.SCRIPTED_BACKEND == 'dijkstra':
      backend = ai.behavior.forageDijkstra
   elif config.SCRIPTED_BACKEND == 'dynamic_programming':
      backend = ai.behavior.forageDP

   return Evaluator(config, policy, config.SCRIPTED_EXPLORE, backend)

def loadModels(config):
   models = config.MULTI_MODEL_EXPERIMENTS
   trainers = []
   for m in models:
      # Initialize a separate trainer for each model
      m_config = copy.deepcopy(config)
      m_config.NPOLICIES = 1
      m_config.NPOP = 1
      m_config.MODEL = m
      trainer = loadTrainer(m_config)
      utils.modelSize(trainer.defaultModel())
      trainer.restore(m)
      trainers.append(trainer)
   return trainers

def loadModel(config):
   '''Load NN weights and optimizer state'''
   trainer = loadTrainer(config)
   utils.modelSize(trainer.defaultModel())
   trainer.restore(config.MODEL)
   return trainer

class Anvil():
   '''Neural MMO CLI powered by Google Fire

   Main file for the RLlib demo included with Neural MMO.

   Usage:
      python Forge.py <COMMAND> --config=<CONFIG> --ARG1=<ARG1> ...

   The User API documents core env flags. Additional config options specific
   to this demo are available in projekt/config.py. 

   The --config flag may be used to load an entire group of options at once.
   The Debug, SmallMaps, and LargeMaps options are included in this demo with
   the latter being the default -- or write your own in projekt/config.py
   '''
   def __init__(self, **kwargs):
      if 'help' in kwargs:
         kwargs.pop('help')

      if 'config' in kwargs:
         config = kwargs.pop('config')
         config = getattr(projekt.config, config)()
      else:
         config = projekt.config.LargeMaps()
      config.override(**kwargs)
      self.config = config

   def train(self, **kwargs):
      '''Train a model starting with the current value of --MODEL'''
      loadModel(self.config).train()

   def evaluate(self, **kwargs):
      '''Evaluate a model on --EVAL_MAPS maps'''
      self.config.EVALUATE = True
      loadEvaluator(self.config).evaluate(self.config.GENERALIZE)

   def render(self, **kwargs):
      '''Start a WebSocket server that autoconnects to the 3D Unity client'''
      self.config.RENDER = True
      loadEvaluator(self.config).render()

   def generate(self, **kwargs):
      '''Generate game maps for the current --config setting'''
      terrain.MapGenerator(self.config).generate()

   def visualize(self, **kwargs):
      '''Training/Evaluation results Web dashboard'''
      BokehServer(self.config)
      
if __name__ == '__main__':
   def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

   from fire import core
   core.Display = Display
   Fire(Anvil)
