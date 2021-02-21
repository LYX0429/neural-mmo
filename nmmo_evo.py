import copy
from evolution.evolver import init_evolver
import os
import pickle
import sys
# My favorite debugging macro
from pdb import set_trace as T

import gym
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import torch
import griddly_nmmo
from fire import Fire
from ray import rllib, tune
from ray.rllib.env import MultiAgentEnv

import evolution
import griddly
import projekt
import python_griddly
from evolution.evo_map import EvolverNMMO
from forge.ethyr.torch import utils
from griddly import GymWrapperFactory, gd
from griddly_nmmo.map_gen import MapGen
from griddly_nmmo.wrappers import NMMOWrapper
from projekt import rllib_wrapper
sep = os.pathsep
os.environ['PYTHONPATH'] = sep.join(sys.path)
from griddly_nmmo.env import NMMO


def unregister():
    for env in copy.deepcopy(gym.envs.registry.env_specs):
        if 'GDY' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registry.env_specs[env]


unregister()
'''Main file for the neural-mmo/projekt demo

/projeckt will give you a basic sense of the training
loop, infrastructure, and IO modules for handling input
and output spaces. From there, you can either use the
prebuilt IO networks in PyTorch to start training your
own models immediately or hack on the environment'''

# Instantiate a new environment

def process_obs(obs):
   obs = dict([(i, val.reshape(*val.shape)) for (i, val) in enumerate(obs)])

   return obs


def createEnv(config):
    import python_griddly
    #   map_arr = config['map_arr']

    #   return projekt.RLLibEnv(#map_arr,
    #           config)
    #  if not config.REGISTERED:
    unregister()
    wrapper = GymWrapperFactory()
    yaml_path = 'nmmo.yaml'
    wrapper.build_gym_from_yaml(
        'nmmo',
        yaml_path,
        level=0,
        player_observer_type=gd.ObserverType.VECTOR,
        global_observer_type=gd.ObserverType.ISOMETRIC,
    )

    return NMMO(config)


# Map agentID to policyID -- requires config global


def mapPolicy(agentID):
#  return 'default_policy'
   return 'policy_{}'.format(agentID % config.NPOLICIES)


# Generate RLlib policies


def createPolicies(config):
    obs = env.observationSpace(config)
    atns = env.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i)
        policies[key] = (None, obs, atns, params)

    return policies


# @ray.remote
# class Counter:
#   ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
#   responsible for providing unique indexes to parallel environments.'''
#   def __init__(self, config):
#      self.count = 0
#   def get(self):
#      self.count += 1
#
#      if self.count == config.N_EVO_MAPS:
#          self.count = 0
#
#      return self.count
#   def set(self, i):
#      self.count = i - 1


@ray.remote
class Counter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indexes to parallel environments.'''
    def __init__(self, config):
        self.count = 0
        self.idxs = None

    def get(self):

        if not self.idxs:
            # Then we are doing inference and have set the idx directly

            return self.count
        idx = self.idxs[self.count % len(self.idxs)]
        self.count += 1

        return idx

    def set(self, i):
        # For inference
        self.count = i

    def set_idxs(self, idxs):
        self.count = 0
        self.idxs = idxs


@ray.remote
class Stats:
    def __init__(self, config):
        self.stats = {}
        self.mults = {}
        self.spawn_points = {}
        self.config = config

    def add(self, stats, mapIdx):
        if config.RENDER:
            #        print(self.headers)
            #        print(stats)
           #calc_differential_entropy(stats)

            return

        if mapIdx not in self.stats:
            self.stats[mapIdx] = {}
            self.stats[mapIdx]['skills'] = [stats['skills']]
            self.stats[mapIdx]['lifespans'] = [stats['lifespans']]
        else:
            self.stats[mapIdx]['skills'].append(stats['skills'])
            self.stats[mapIdx]['lifespans'].append(stats['lifespans'])

    def get(self):
        return self.stats

    def reset(self):
        self.stats = {}

    def add_mults(self, g_hash, mults):
        self.mults[g_hash] = mults

    def get_mults(self, g_hash):
        if g_hash not in self.mults:
            return None

        return self.mults[g_hash]

    def add_spawn_points(self, g_hash, spawn_points):
        self.spawn_points[g_hash] = spawn_points

    def get_spawn_points(self, g_hash):
        return self.spawn_points[g_hash]



if __name__ == '__main__':
    unregister()
    # Setup ray
    #  torch.set_num_threads(1)
    torch.set_num_threads(torch.get_num_threads())
    ray.init()

    global config

    config = projekt.config.Griddly()

    # Built config with CLI overrides

    if len(sys.argv) > 1:
        sys.argv.insert(1, 'override')
        Fire(config)

    # on the driver
    counter = Counter.options(name="global_counter").remote(config)
    stats = Stats.options(name="global_stats").remote(config)

    # RLlib registry
    rllib.models.ModelCatalog.register_custom_model('test_model',
                                                    rllib_wrapper.Policy)
    ray.tune.registry.register_env("custom", createEnv)

    # save_path = 'evo_experiment/skill_entropy_life'
    # save_path = 'evo_experiment/scratch'
    # save_path = 'evo_experiment/skill_ent_0'

    save_path = os.path.join('evo_experiment', '{}'.format(config.EVO_DIR))

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    wrapper = GymWrapperFactory()
    yaml_path = 'nmmo.yaml'
    map_gen = MapGen(config)
    init_tiles, probs, skill_names = map_gen.get_init_tiles(yaml_path, write_game_file=True)

    try:
        evolver_path = os.path.join(save_path, 'evolver')
        with open(evolver_path, 'rb') as save_file:
            evolver = pickle.load(save_file)

            print('loading evolver from save file')
        # change params on reload here
#     evolver.config.ROOT = config.ROOT
        evolver.config.TERRAIN_RENDER = config.TERRAIN_RENDER
        evolver.config.TRAIN_RENDER = config.TRAIN_RENDER
        evolver.config.INFER_IDX = config.INFER_IDX
        evolver.config.RENDER = config.RENDER
        #     evolver.config.SKILLS = config.SKILLS
        #     evolver.config.MODEL = config.MODEL
        #     evolver.config['config'].MAX_STEPS = 200
        #     evolver.n_epochs = 15000
        evolver.reloading = True
        evolver.epoch_reloaded = evolver.n_epoch
        evolver.restore()
        evolver.trainer.reset()
        evolver.load()

    except FileNotFoundError as e:
        print(e)
        print(
            'Cannot load; missing evolver and/or model checkpoint. Evolving from scratch.'
        )

#       evolver = EvolverNMMO(
#           save_path,
#           createEnv,
#           None,  # init the trainer in evolution script
#           config,
#           n_proc=config.N_PROC,
#           n_pop=config.N_EVO_MAPS,
#           map_policy=mapPolicy,
#       )

        evolver = init_evolver(save_path,
                       createEnv,
                       None,  # init the trainer in evolution script
                       config,
                       n_proc=config.N_PROC,
                       n_pop=config.N_EVO_MAPS,
                       )
#  print(torch.__version__)

#  print(torch.cuda.current_device())
#  print(torch.cuda.device(0))
#  print(torch.cuda.device_count())
#  print(torch.cuda.get_device_name(0))
#  print(torch.cuda.is_available())
#  print(torch.cuda.current_device())

    unregister()
    wrapper.build_gym_from_yaml(
        'nmmo',
        yaml_path,
        level=0,
        player_observer_type=gd.ObserverType.VECTOR,
        global_observer_type=gd.ObserverType.ISOMETRIC,
    #   global_observer_type=gd.ObserverType.ISOMETRIC,
    )

    rllib_config = {
        "env": NMMO,
        "framework": "torch",
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "train_batch_size": 128,
        "sgd_minibatch_size": 128,
        "model": {
            "conv_filters": [[32, (7, 7), 3]],
           },
      # "no_done_at_end": True,
        "env_config": {
           "config": config,
           },

       }



    if config.TEST:
       env = NMMO(rllib_config['env_config'])

       for i in range(20):
          env.reset()

          for j in range(1000):
             obs, rew, done, infos = env.step(dict([(i, val) for (i, val) in env.action_space.sample().items()]))
#            env.render()

             if done['__all__']:
                break

    else:
   #   trainer = ppo.PPOTrainer(config=rllib_config, env=NMMO)
   #   while True:
   #      res = trainer.train()
   #      print(res)

      #results = tune.run("PG", config=config, verbose=1)

       if config.RENDER:
           evolver.infer()
       else:
           unregister()
           evolver.evolve()
