import os
import time
from pdb import set_trace as T

import gym
import numpy as np


import copy

for env in copy.deepcopy(gym.envs.registry.env_specs):
    if 'GDY' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registry.env_specs[env]

#from gym.utils.play import play
import griddly
from griddly import GymWrapperFactory, gd
from griddly_nmmo.map_gen import MapGen
from griddly_nmmo.wrappers import NMMOWrapper
from griddly.util.rts_tools import InvalidMaskingRTSWrapper

if __name__ == '__main__':


    for env in copy.deepcopy(gym.envs.registry.env_specs):
        if 'GDY' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registry.env_specs[env]


    # NB: The nmmo environment is designed to to render with an ISOMETRIC global observer only.
    wrapper = GymWrapperFactory()

    yaml_path = 'nmmo.yaml'
    map_gen = MapGen()
    init_tiles, probs, tile_names = map_gen.get_init_tiles(yaml_path)

    wrapper.build_gym_from_yaml('nmmo', yaml_path, level=0,
             player_observer_type=gd.ObserverType.VECTOR,
             global_observer_type=gd.ObserverType.ISOMETRIC,
            )
    # Create the Environment
    env = gym.make(f'GDY-nmmo-v0')
#   env = InvalidMaskingRTSWrapper(env)
    env = NMMOWrapper(env)

    def reset():
        level_string = map_gen.gen_map(init_tiles, probs)
        env.reset(level_id=None, level_string=level_string)
        env.render(observer='global')

    i = 0
    while i < 10:
        i += 1
        reset()
        j = 0
        while True:
            act = env.action_space.sample()
            obs, reward, done, infos = env.step(act)
            env.render(observer='global')

           #if j != 1 and j % 10 == 0:
           #    T()

            if done['__all__'] or j == 1000:
                break
            j += 1 
