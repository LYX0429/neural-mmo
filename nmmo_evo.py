import copy
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

reshp = (1, 2, 0)

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
            calc_differential_entropy(stats)

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


class NMMO(MultiAgentEnv):
   def __init__(self, config):
      self.config = config
      self.map_gen = MapGen(config['config'])
      yaml_path = 'nmmo.yaml'
      self.init_tiles, self.probs, self.skill_names = self.map_gen.get_init_tiles(yaml_path, write_game_file=False)
      self.chars_to_terrain = dict([(v, k) for (k, v) in self.map_gen.chars.items()])
      level_string = self.map_gen.gen_map(self.init_tiles, self.probs)
      self.env = None
      env = gym.make('GDY-nmmo-v0',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.ISOMETRIC)
      env = NMMOWrapper(env)
      self.env = env
      self.env.reset(level_id=None, level_string=level_string)
     #self.env.reset(level_id=0)
      if isinstance(self.env.observation_space, list):
         self.observation_space = self.env.observation_space[0]
      else:
         self.observation_space = self.env.observation_space
      self.observation_space = gym.spaces.Box(self.observation_space.low.transpose(reshp),
            self.observation_space.high.transpose(reshp))
      #     self.observation_space.shape = self.env.observation_space[0].shape
      self.action_space = self.env.action_space
      self.action_space = gym.spaces.MultiDiscrete(self.env.action_space.nvec)
      print('NMMO env action space is {}'.format(self.action_space))
      del(self.env)
      self.env = None
      self.lifetimes = {}
      self.maps = None
     #del (self.env)
     #self.env = None

   def reset(self, config=None, step=None, maps=None):
      #FIXME: looks like the trainer is resetting of its own accord? Bad.
      if maps is None:
         maps = self.maps
      assert maps is not None      
      self.maps = maps
      if self.env is None:
         env = gym.make('GDY-nmmo-v0',
                        player_observer_type=gd.ObserverType.VECTOR,
                        global_observer_type=gd.ObserverType.ISOMETRIC)
         env = NMMOWrapper(env)
         self.env = env
      self.lifetimes = dict([(i, 0) for i in range(self.env.player_count)])
      self.skills = dict([(skill_name, dict([(i, 0) for i in range(self.env.player_count)])) for skill_name in self.skill_names])
      global_counter = ray.get_actor("global_counter")

      if self.config['config'].TEST:
         map_str_out = self.map_gen.gen_map(self.init_tiles, self.probs)
      else:
         self.mapIdx = ray.get(global_counter.get.remote())
#        fPath = os.path.join(self.config['config'].ROOT + str(self.mapIdx), 'map.npy')
#        map_arr = np.load(fPath)
         map_arr = maps[self.mapIdx]
         map_str = map_arr.astype('U'+str(1 + len(str(self.config['config'].NENT))))

         for i, char in enumerate(self.init_tiles):
            if char == self.map_gen.player_char:
               continue
            j = self.chars_to_terrain[char]
            j = self.map_gen.tile_types.index(j)
            map_str[map_arr==j] = char

         idxs = np.vstack(np.where(map_arr==griddly_nmmo.map_gen.GdyMaterial.SPAWN.value.index)).transpose()
         np.random.shuffle(idxs)
         for i, (x, y) in enumerate(idxs):
            if i < self.config['config'].NENT:
               map_str[x, y] = self.map_gen.player_char + str(i+1)
            else:
               map_str[x, y] = '.'

         map_str_out = np.append(map_str, np.zeros((map_str.shape[1], 1), dtype=map_str.dtype),axis=1)
         map_str_out[:, -1] = '\n'
         map_str_out = ' '.join(s for s in map_str_out.reshape(-1))
      obs = self.env.reset(level_id=None, level_string=map_str_out)
      [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]

      return obs


   def step(self, a):
    # if isinstance(a[0], np.ndarray):
      obs, rew, dones, info = self.env.step(a)
      [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]
      [self.lifetimes.update({k: v+1}) if k not in self.env.deads else None for (k, v) in self.lifetimes.items()]

      if self.config['config'].TRAIN_RENDER:
         self.render()

      return obs, rew, dones, info

   def render(self, observer='global'):
       return self.env.render(observer=observer)

   def send_agent_stats(self):
      global_vars = self.env.get_state()['GlobalVariables']
      [self.skills[skill_name].update({k: global_vars[skill_name][k]}) if k in global_vars[skill_name] else self.skills[skill_name].update({k: 0}) for k in range(self.env.player_count) for skill_name in self.config['config'].SKILLS]
      global_stats = ray.get_actor('global_stats')
      global_stats.add.remote(
             {
                'skills': [[self.skills[i][j] for i in self.skills] for j in range(self.env.player_count)],
                'lifespans': [self.lifetimes[i] for i in range(len(self.lifetimes))],
                'lifetimes': [self.lifetimes[i] for i in range(len(self.lifetimes))],
                },
             self.mapIdx)

   def __reduce__(self):
      pass


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
        #     evolver.config.SKILLS = config.SKILLS
        #     evolver.config.MODEL = config.MODEL
        #     evolver.config['config'].MAX_STEPS = 200
        #     evolver.n_epochs = 15000
        evolver.reloading = True
        evolver.epoch_reloaded = evolver.n_epoch
        evolver.restore()
        evolver.trainer.reset()

        if evolver.MAP_ELITES:
            evolver.me.load()

    except FileNotFoundError as e:
        print(e)
        print(
            'Cannot load; missing evolver and/or model checkpoint. Evolving from scratch.'
        )

        evolver = EvolverNMMO(
            save_path,
            createEnv,
            None,  # init the trainer in evolution script
            config,
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