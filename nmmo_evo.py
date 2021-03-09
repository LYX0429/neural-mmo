import copy

from ray.rllib.models import ModelCatalog
from evolution.evolver import init_evolver
from griddly.util.rllib.torch import GAPAgent
import os
import pickle
import sys
# My favorite debugging macro
from pdb import set_trace as T

import gym
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from gym.utils.play import play
import torch
import griddly_nmmo
from fire import Fire
from ray import rllib, tune
from ray.rllib.env import MultiAgentEnv

import evolution
import griddly
import projekt
#import python_griddly
from evolution.evo_map import EvolverNMMO
from forge.ethyr.torch import utils
from griddly import GymWrapperFactory, gd
from griddly_nmmo.map_gen import MapGen
from griddly_nmmo.wrappers import NMMOWrapper
from projekt import rllib_wrapper
sep = os.pathsep
os.environ['PYTHONPATH'] = sep.join(sys.path)
from griddly_nmmo.env import NMMO

import matplotlib
matplotlib.use('Agg')

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
#    test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'griddly_nmmo/nmmo.yaml')
    test_path = config['config'].NMMO_YAML_PATH
    import python_griddly
    print("test path is: ", test_path)
    #   map_arr = config['map_arr']

    #   return projekt.RLLibEnv(#map_arr,
    #           config)
    #  if not config.REGISTERED:
#   unregister()
#   wrapper = GymWrapperFactory()
#   yaml_path = 'nmmo.yaml'
#   wrapper.build_gym_from_yaml(
#       'nmmo',
#       test_path,
#       level=None,
#       player_observer_type=gd.ObserverType.VECTOR,
#       global_observer_type=gd.ObserverType.ISOMETRIC,
#   )


    env_name = 'nmmo.yaml'
    config.update({
        'env': "custom",
        'num_gpus': 1,
        'env_config': {
            # in the griddly environment we set a variable to let the training environment
            # know if that player is no longer active
#           'player_done_variable': 'player_done',

#           'record_video_config': {
#               'frequency': 10000  # number of rollouts
#           },

            'yaml_file': test_path,
            'global_observer_type': gd.ObserverType.ISOMETRIC,
            'level': None,
            'max_steps': config['config'].MAX_STEPS,
        },
    })
#   return RLlibMultiAgentWrapper(env_config)
    env = NMMO(config)
    return env


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
#   def get(self)      self.count += 1
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
    config.NMMO_YAML_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'griddly_nmmo/nmmo.yaml')

    # on the driver
    counter = Counter.options(name="global_counter").remote(config)
    stats = Stats.options(name="global_stats").remote(config)

    # RLlib registry
#   rllib.models.ModelCatalog.register_custom_model('test_model',
#                                                   rllib_wrapper.Policy)

    from griddly.util.rllib.torch import GAPAgent
    rllib.models.ModelCatalog.register_custom_model('test_model', GAPAgent)

    ray.tune.registry.register_env("custom", createEnv)

    # save_path = 'evo_experiment/skill_entropy_life'
    # save_path = 'evo_experiment/scratch'
    # save_path = 'evo_experiment/skill_ent_0'

    save_path = os.path.join('evo_experiment', '{}'.format(config.EVO_DIR))

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    wrapper = GymWrapperFactory()
    yaml_path = 'griddly_nmmo/nmmo.yaml'
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
        evolver.config.ARCHIVE_UPDATE_WINDOW = config.ARCHIVE_UPDATE_WINDOW
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
#       )

        evolver = init_evolver(
                       save_path=save_path,
                       make_env=createEnv,
                       trainer=None,  # init the trainer in evolution script
                       config=config,
                       n_proc=config.N_PROC,
                       n_pop=config.N_EVO_MAPS,
                       map_policy=mapPolicy,
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
#       image_path='~/Griddly/resources/images',
        level=0,
        player_observer_type=gd.ObserverType.VECTOR,
        global_observer_type=gd.ObserverType.ISOMETRIC,
    #   global_observer_type=gd.ObserverType.ISOMETRIC,
    )

    ray.tune.registry.register_env('nmmo', NMMO)
    rllib_config = {
        "env": "custom",
        "framework": "torch",
        "num_workers": 6,
        "num_gpus": 1,
        "num_envs_per_worker": 1,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        'rollout_fragment_length': 100,
        "model": {
            "conv_filters": [[32, (7, 7), 3]],
           },
      # "no_done_at_end": True,
        "env_config": {
           "config": config,
        },

       }



    if config.TEST:
       def train_ppo(config, reporter):
           agent = ray.rllib.agents.ppo.PPOTrainer(config)
#          agent.restore("/home/sme/ray_results/PPO_custom_2021-03-07_23-31-41cnv2ax4i/checkpoint_32/checkpoint-32")  # continue training
           # training curriculum, start with phase 0
#          phase = 0
#          agent.workers.foreach_worker(
#              lambda ev: ev.foreach_env(
#                  lambda env: env.set_phase(phase)))
           episodes = 0
           i = 0
           while True:
               result = agent.train()
               if reporter is None:
                   continue
               else:
                   reporter(**result)
               if i % 10 == 0:  # save every 10th training iteration
                   checkpoint_path = agent.save()
                   print(checkpoint_path)
               i += 1
                # you can also change the curriculum here
       result = ray.tune.run(train_ppo, config = rllib_config,
               #             resources_per_trial={
               #                 "cpu": 6,
               #                 "gpu": 1,
               #         #       "extra_cpu": 0,
               #             },
                             )
#      result = ray.tune.run(ray.rllib.agents.ppo.PPOTrainer, config = rllib_config)

 #     env = NMMO(rllib_config)
##     play(env)

 #     for i in range(20):
 #        env.reset()

 #        for j in range(1000):
 #           obs, rew, done, infos = env.step(dict([(i, val) for (i, val) in env.action_space.sample().items()]))
##           env.render()

 #           if done['__all__']:
 #              break

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
