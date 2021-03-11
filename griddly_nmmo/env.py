from pdb import set_trace as T
import gym
#from griddly import gd
import ray
from ray.rllib import MultiAgentEnv
import numpy as np

import griddly_nmmo
from griddly_nmmo.map_gen import MapGen
from griddly_nmmo.wrappers import NMMOWrapper

reshp = (1, 2, 0)
#reshp = (0, 1, 2)
import os

class NMMO(NMMOWrapper):
   def __init__(self, config):
      self.config = config
      self.map_gen = MapGen(config['config'])
      yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nmmo.yaml')
#     yaml_path = 'griddly_nmmo/nmmo.yaml'
      self.init_tiles, self.probs, self.skill_names = self.map_gen.get_init_tiles(yaml_path, write_game_file=False)
      self.chars_to_terrain = dict([(v, k) for (k, v) in self.map_gen.chars.items()])
      level_string = self.map_gen.gen_map(self.init_tiles, self.probs)
      self.env = None
#     env = gym.make('GDY-nmmo-v0',
#                    player_observer_type=gd.ObserverType.VECTOR,
#                    global_observer_type=gd.ObserverType.ISOMETRIC)
     #if 'env_config' in config:
      self.env = NMMOWrapper(config['env_config'])
   #else:
     #   self.env = NMMOWrapper(config)
#     self.env_config = env_config
#     env = RLlibMultiAgentWrapper(config['env_config'])
      self.env.reset(level_id=None, level_string=level_string)
     #self.env.reset(level_id=0)
      if isinstance(self.env.observation_space, list):
         self.observation_space = self.env.observation_space[0]
      else:
         self.observation_space = self.env.observation_space
#     self.observation_space = gym.spaces.Box(self.observation_space.low.transpose(reshp),
#           self.observation_space.high.transpose(reshp))
      #     self.observation_space.shape = self.env.observation_space[0].shape
      self.action_space = self.env.action_space
      self.action_space = gym.spaces.MultiDiscrete(self.env.action_space.nvec)
      print('NMMO env action space is {}'.format(self.action_space))
      del(self.env)
      self.env = None
      self.lifetimes = {}
      self.dones = {'__all__': False}
      self.maps = None
      self.map_arr = None
      self.past_deads = set()
      self.must_send_stats = False

   #del (self.env)
     #self.env = None

   def set_map(self, idx=None, maps=None):
#     print('setting map', self.worldIdx)
      if idx is None:
         global_counter = ray.get_actor("global_counter")
         self.mapIdx = ray.get(global_counter.get.remote())
     #   fPath = os.path.join(self.config['config'].ROOT + str(self.mapIdx), 'map.npy')
     #   map_arr = np.load(fPath)
      else:
         self.mapIdx = idx
      #FIXME: agree w/ NMMO
      self.worldIdx = self.mapIdx
      map_arr = maps[self.mapIdx]
      self.map_arr = map_arr
#     print(maps.keys())
#     assert map_arr is not Nonem

      # If we have se the map, we need to load it immediately
      self.dones['__all__'] = True


   def reset(self, config=None, step=None, maps=None):
#     print('resetting {}'.format(self.worldIdx))
#     if self.must_send_stats:
#        self.send_agent_stats()
      # If we are about to simulate, we will need to send stats afterward
      self.must_send_stats = True
      self.past_deads = set()
      self.dones = {'__all__': False}
#     if maps is None:
#        maps = self.maps
#     assert maps is not None
#     self.maps = maps
      if self.env is None:
#        env = gym.make('GDY-nmmo-v0',
#                       player_observer_type=gd.ObserverType.VECTOR,
#                       global_observer_type=gd.ObserverType.ISOMETRIC)
         env = NMMOWrapper(self.config['env_config'])
#        env = RLlibMultiAgentWrapper(self.config['env_config'])
         self.env = env
      self.lifetimes = dict([(i, 0) for i in range(self.env.player_count)])
      self.skills = dict([(skill_name, dict([(i, 0) for i in range(self.env.player_count)])) for skill_name in self.skill_names])
#     self.init_pos = {}

      if self.config['config'].TEST or self.config['config'].PRETRAIN:
         map_str_out = self.map_gen.gen_map(self.init_tiles, self.probs)
      else:
         map_arr = self.map_arr
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
#              self.init_pos[i] = (x, y)
            else:
               map_str[x, y] = '.'

         map_str_out = np.append(map_str, np.zeros((map_str.shape[1], 1), dtype=map_str.dtype),axis=1)
         map_str_out[:, -1] = '\n'
         map_str_out = ' '.join(s for s in map_str_out.reshape(-1))
      obs = self.env.reset(level_id=None, level_string=map_str_out)
#     [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]
      self.get_init_stats()

      return obs


   def step(self, a, omitDead=False, preprocessActions=False):
#     print('step env ',self.worldIdx)
    # if isinstance(a[0], np.ndarray):
      obs, rew, dones, info = self.env.step(a)

   #     [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]
      [self.lifetimes.update({k: v+1}) if k not in self.env.past_deads else None for (k, v) in self.lifetimes.items()]

      if self.config['config'].TRAIN_RENDER or self.config['config'].RENDER:
         self.render()
#     dones['__all__'] = self.dones['__all__']


      if self.config['config'].RENDER or (self.config['config'].GRIDDLY and self.config['config'].TEST):
         if len(self.env.past_deads) == self.config['config'].NENT:
            print('Evaluated map {}'.format(self.worldIdx))
            lifetimes = list(self.lifetimes.values())
            print('Lifetimes: {}'.format(lifetimes))
            print('Mean lifetime: {}'.format(np.mean(lifetimes)))
            self.dones['__all__'] = True

      # In case we need to load a map assigned to this env at the previous step
      # More generally: we can mark the environment as done from outside or as a result of processes other than usual
      # simulation.
      if self.dones['__all__'] == True:
         dones['__all__'] = True
      self.dones = dones
      self.last_obs = obs
      return obs, rew, dones, info

   def render(self, observer='global', mode='rgb'):
       if self.config['config'].RENDER or self.worldIdx == 0:
          return self.env.render(observer=observer)

   def unwrapped(self):
      return self

   def get_init_stats(self):
      self.init_stats = {}
      self.init_stats['init_pos'] = np.empty((self.env.player_count, 2))
      env_state = self.env.get_state()
      objects = env_state['Objects']
      for o in objects:
         if o['Name'] == 'gnome':
            self.init_stats['init_pos'][o['PlayerId'] - 1] = o['Location']


   def send_agent_stats(self):
#     print('sending stats', self.worldIdx)
      env_state = self.env.get_state()
      global_vars = env_state['GlobalVariables']
      [self.skills[skill_name].update({k: global_vars[skill_name][k]}) if k in global_vars[skill_name] else self.skills[skill_name].update({k: 0}) for k in range(self.env.player_count) for skill_name in self.config['config'].SKILLS]
      scores = [global_vars['score'][i] for i in range(self.env.player_count)]
      # FIXME: only necessary when frozen due to evaluator resets
      global_stats = ray.get_actor('global_stats')

      objects = env_state['Objects']
      end_pos = np.empty((self.env.player_count, 2))
      for o in objects:
         if o['Name'] == 'gnome':
            end_pos[o['PlayerId'] - 1] = o['Location']
      # assign map fitness when agent ends up in ``opposite corner'' of map
      y_deltas = [(end_pos[i][1] - self.init_stats['init_pos'][i][1]) + (end_pos[i][0] - self.init_stats['init_pos'][i][0]) for i in range(self.env.player_count)]


      stats = {
         'skills': [[self.skills[i][j] for i in self.skills] for j in range(self.env.player_count)],
         'lifespans': [self.lifetimes[i] for i in range(len(self.lifetimes))],
         #        'lifetimes': [self.lifetimes[i] for i in range(len(self.lifetimes))],
         'y_deltas': y_deltas,
         'scores': scores,
      }
      # FIXME: only necessary when frozen due to evaluator resets
      global_stats.add.remote(stats, self.mapIdx)

      # If we are sending stats, we must reset immediately, without again sending stats
      self.dones['__all__'] = True
      self.must_send_stats = False

      return stats

   def __reduce__(self):
      pass