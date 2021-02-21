import gym
from griddly import gd
import ray
from ray.rllib import MultiAgentEnv
import numpy as np

import griddly_nmmo
from griddly_nmmo.map_gen import MapGen
from griddly_nmmo.wrappers import NMMOWrapper

reshp = (1, 2, 0)

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
      self.dones = {'__all__': False}
      self.maps = None
     #del (self.env)
     #self.env = None

   def set_map(self, idx=None, maps=None):
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
      assert map_arr is not None

   def reset(self, config=None, step=None, maps=None):
      self.dones = {'__all__': False}
#     if maps is None:
#        maps = self.maps
#     assert maps is not None
#     self.maps = maps
      if self.env is None:
         env = gym.make('GDY-nmmo-v0',
                        player_observer_type=gd.ObserverType.VECTOR,
                        global_observer_type=gd.ObserverType.ISOMETRIC)
         env = NMMOWrapper(env)
         self.env = env
      self.lifetimes = dict([(i, 0) for i in range(self.env.player_count)])
      self.skills = dict([(skill_name, dict([(i, 0) for i in range(self.env.player_count)])) for skill_name in self.skill_names])

      if self.config['config'].TEST:
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
            else:
               map_str[x, y] = '.'

         map_str_out = np.append(map_str, np.zeros((map_str.shape[1], 1), dtype=map_str.dtype),axis=1)
         map_str_out[:, -1] = '\n'
         map_str_out = ' '.join(s for s in map_str_out.reshape(-1))
      obs = self.env.reset(level_id=None, level_string=map_str_out)
      [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]

      return obs


   def step(self, a, omitDead=False, preprocessActions=False):
    # if isinstance(a[0], np.ndarray):
      obs, rew, dones, info = self.env.step(a)
      [obs.update({k: v.transpose(reshp)}) for (k, v) in obs.items()]
      [self.lifetimes.update({k: v+1}) if k not in self.env.past_deads else None for (k, v) in self.lifetimes.items()]

      if self.config['config'].TRAIN_RENDER or self.config['config'].RENDER:
         self.render()
      dones['__all__'] = self.dones['__all__']


      if self.config['config'].RENDER:
         if len(self.env.past_deads) == self.config['config'].NENT:
            print('Evaluated map {}'.format(self.worldIdx))
            lifetimes = list(self.lifetimes.values())
            print('Lifetimes: {}'.format(lifetimes))
            print('Mean lifetime: {}'.format(np.mean(lifetimes)))
            self.dones['__all__'] = True

      return obs, rew, dones, info

   def render(self, observer='global'):
       return self.env.render(observer=observer)

   def send_agent_stats(self):
      global_vars = self.env.get_state()['GlobalVariables']
      [self.skills[skill_name].update({k: global_vars[skill_name][k]}) if k in global_vars[skill_name] else self.skills[skill_name].update({k: 0}) for k in range(self.env.player_count) for skill_name in self.config['config'].SKILLS]
      global_stats = ray.get_actor('global_stats')
      stats ={
         'skills': [[self.skills[i][j] for i in self.skills] for j in range(self.env.player_count)],
         'lifespans': [self.lifetimes[i] for i in range(len(self.lifetimes))],
         'lifetimes': [self.lifetimes[i] for i in range(len(self.lifetimes))],
         }
#     global_stats.add.remote(stats, self.mapIdx)
      self.dones['__all__'] = True

      return stats

   def __reduce__(self):
      pass