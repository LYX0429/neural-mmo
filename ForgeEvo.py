import os
import json
import pickle
import sys
# My favorite debugging macro
from pdb import set_trace as TT

import re
import numpy as np
import ray
import torch
from fire import Fire
from ray import rllib

import projekt
from evolution.evolver import init_evolver
from projekt import rllib_wrapper
from evolution.global_actors import Counter, Stats
from projekt.config import get_experiment_name
from pcg import get_tile_data
TILE_TYPES, TILE_PROBS = get_tile_data(griddly=False)

'''Main file for the neural-mmo/projekt demo

/projeckt will give you a basic sense of the training
loop, infrastructure, and IO modules for handling input
and output spaces. From there, you can either use the
prebuilt IO networks in PyTorch to start training your
own models immediately or hack on the environment'''

# Instantiate a new environment

def createEnv(config):
#   map_arr = config['map_arr']

    return rllib_wrapper.RLlibEnv(#map_arr,
            config)

# Map agentID to policyID -- requires config global

def mapPolicy(agentID):
    return 'policy_{}'.format(agentID % config.NPOLICIES)


# Generate RLlib policies

def createPolicies(config):
    obs = rllib_wrapper.observationSpace(config)
    atns = rllib_wrapper.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i)
        policies[key] = (None, obs, atns, params)

    return policies

def process_config(config):

   config.set('EVO_DIR', experiment_name)

   # TODO: put this in a dictionary that provides alternative skills for the griddly environment, or maybe just use strings that map to different skillsets for each environment?
   if config.SKILLS == 'ALL':
      SKILLS = ['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']
   elif config.SKILLS == 'HARVEST':
      SKILLS = ['woodcutting','mining']
   elif config.SKILLS == 'EXPLORATION':
      SKILLS = ['exploration']
   elif config.SKILLS == 'COMBAT':
      SKILLS = ['mage','range','melee']
   elif config.SKILLS == 'NONE':
      SKILLS = []
   else:
      raise Exception

   config.set('SKILLS', SKILLS)

   return config

if __name__ == '__main__':
   # Setup ray
#  torch.set_num_threads(1)
   torch.set_num_threads(torch.get_num_threads())
   ray.init()


   global config

   config = projekt.config.EvoNMMO()
#  config = projekt.config.Griddly()

   # Built config with CLI overrides
   if len(sys.argv) > 1:
       sys.argv.insert(1, 'override')
       Fire(config)

   # Load config from json
   if config.load_arguments != -1:
      load_args = json.load(
         open('configs/settings_{}.json'.format(config.load_arguments), 'r'))
      [config.set(k, v) for (k, v) in load_args.items()]

   # on the driver
   counter = Counter.options(name="global_counter").remote(config)
   stats = Stats.options(name="global_stats").remote(config)

   # RLlib registry
   rllib.models.ModelCatalog.register_custom_model('test_model',
                                                   rllib_wrapper.RLlibPolicy)
   ray.tune.registry.register_env("custom", createEnv)

 # save_path = 'evo_experiment/skill_entropy_life'
 # save_path = 'evo_experiment/scratch'
 # save_path = 'evo_experiment/skill_ent_0'

   experiment_name = get_experiment_name(config)

   config.set('ROOT', re.sub('evo_experiment/.*/', 'evo_experiment/{}/'.format(experiment_name), config.ROOT))
   #) config.ROOT.replace('evo_experiment/{}'.format(config.EVO_DIR), 'evo_experiment/{}'.format(experiment_name)))
   save_path = os.path.join('evo_experiment', '{}'.format(experiment_name))
   if not os.path.isdir(save_path):
       os.mkdir(save_path)

   with open(os.path.join(save_path, 'settings.json'), 'w') as f:
      json.dump(config.data, f, indent=4)

   config = process_config(config)


   try:
      evolver_path = os.path.join(save_path, 'evolver')
      with open(evolver_path, 'rb') as save_file:
         evolver = pickle.load(save_file)

         print('loading evolver from save file')
      # change params on reload here
      evolver.config.RENDER = config.RENDER
      evolver.config.TERRAIN_RENDER = config.TERRAIN_RENDER
      evolver.config.NENT = config.NENT
      evolver.config.MODEL = 'reload'
      evolver.config.ROOT = config.ROOT
      evolver.config.N_EVO_MAPS = config.N_EVO_MAPS
      evolver.config.N_PROC = config.N_PROC
      evolver.config.EVO_SAVE_INTERVAL = config.EVO_SAVE_INTERVAL
      evolver.reloading = True
      evolver.epoch_reloaded = evolver.n_epoch
      # Running out of RAM depending on size/number of map genomes... try trashing the archive while ray does its
      # multiprocessing nonsense garbage?
      # Why does this work !!!
      # I DO NOT UNDERSTAND MY CODE
      global container
      container = evolver.container
      evolver.container = None
      evolver.restore(trash_trainer=True)
      # evolver.container = container
      archive = pickle.load(open(os.path.join('evo_experiment', experiment_name, 'ME_archive.p'), 'rb'))
      evolver.container = archive['container']
      if config.RENDER:
         evolver.config.INFER_IDX = config.INFER_IDX
      if evolver.MAP_ELITES:
         evolver.load()

   except FileNotFoundError as e:
      print(e)
      print('Cannot load; missing evolver and/or model checkpoint. Evolving from scratch.')

      evolver = init_evolver(save_path,
                            createEnv,
                            None,  # init the trainer in evolution script
                            config,
                            n_proc=   config.N_PROC,
                            n_pop=    config.N_EVO_MAPS,
                            n_epochs= config.N_GENERATIONS,
                            )
#  print(torch.__version__)

#  print(torch.cuda.current_device())
#  print(torch.cuda.device(0))
#  print(torch.cuda.device_count())
#  print(torch.cuda.get_device_name(0))
#  print(torch.cuda.is_available())
#  print(torch.cuda.current_device())

   if config.RENDER:
      evolver.infer()
   else:
      evolver.evolve()
