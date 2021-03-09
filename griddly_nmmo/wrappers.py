from pdb import set_trace as T
from griddly.util.rllib.wrappers.core import RLlibMultiAgentWrapper, RLlibEnv

import gym
import numpy as np
from griddly import gd


class ValidatedMultiDiscreteNMMO(gym.spaces.MultiDiscrete):
    """
    The same action space as MultiDiscrete, however sampling this action space only results in valid actions
    """

    def __init__(self, nvec, masking_wrapper):
        self._masking_wrapper = masking_wrapper
        super().__init__(nvec)

    def sample(self):
        actions = {}

        for player_id in range(self._masking_wrapper.player_count):
            actions[player_id] = self.sample_player(player_id)

        return actions

    def sample_player(self, player_id):
        # Sample a location with valid actions
        available_actions = [a for a in self._masking_wrapper.env.game.get_available_actions(player_id + 1).items()]
        n_avail_actions = len(available_actions)
        if n_avail_actions == 0:
#           return None
            return 0, 0
        available_actions_choice = np.random.choice(len(available_actions))
        location, actions = available_actions[available_actions_choice]

        available_action_ids = [aid for aid in self._masking_wrapper.env.game.get_available_action_ids(location, list(
            actions)).items() if len(aid[1])>0]

        num_action_ids = len(available_action_ids)

        # If there are no available actions at all, we do a NOP (which is any action_name with action_id 0)

        if num_action_ids == 0:
            action_name_idx = 0
            action_id = 0
        else:
            available_action_ids_choice = np.random.choice(num_action_ids)
            action_name, action_ids = available_action_ids[available_action_ids_choice]
            action_name_idx = self._masking_wrapper.action_names.index(action_name)
            action_id = np.random.choice(action_ids)

        return [action_name_idx, action_id]

import os

class NMMOWrapper(RLlibMultiAgentWrapper):
    def __init__(self, config):
#       max_steps = 100
        env = RLlibEnv(config)
        super().__init__(env, config)
        self.n_step = 0
#       self.max_steps = max_steps

    def step(self, action):
        rew = {}
        obs = {}
        done = {}
        info = {}
        all_done = True
      # [None if (i in self.past_deads) else action.update({i: list(val)}) for (i, val) in action.items()]
#       action = dict([(i, action[i]) if i in action else (i, [0,0]) for i in range(self.player_count)])
#       action.reverse()
#       action = [action[i] for i in range(self.player_count)]
        obs, rew, done, info = super().step(action)
#       obs, rew, done, info = super().step(self.action_space.sample())
       #print(obs.shape)
#       obs = dict([(i, val) for (i, val) in enumerate(obs)])
       #rew = dict([(i, rew[i]) for i in range(self.player_count)])
        def cull_gnomes():
            env_state = self.get_state()
            env_deads = env_state['GlobalVariables']['player_done']
       #    done = dict([(i, env_deads[i+1] > 0 and i not in self.past_deads) for i in range(self.player_count)])
       #    info = dict([(i, info) for i in range(self.player_count)])
       #    # Pop anyone who is already dead
       #    [(obs.pop(i), rew.pop(i), done.pop(i), info.pop(i)) for i in self.past_deads]
            [self.past_deads.add(i) if isinstance(i, int) and env_deads[i + 1] > 0 else None for i in done]
       #    #FIXME: Trippy griddly bug fucks negative reward due to player starvation/thirst death
       #    [rew.update({i: 1}) if i not in self.past_deads else None for i in rew]
            # Reward gnomes for potentiall undying after death
           #[rew.update({i: 1}) if env_deads[i + 1] else None for i in rew]
        #[self.deads.add(i) if rew[i] < 0 else None for i in rew]
     #  done = dict([(i, False) for i in range(self.player_count)])
#       if not len(self.past_deads) == self.player_count:
#          cull_gnomes()

    #   for player_id, player_action in action.items():
    #       p_obs, p_rew, p_done, p_info = self.step_player(player_id, player_action)
    #       obs[player_id] = p_obs
    #       rew[player_id] = p_rew
    #       done[player_id] = p_done
#   #       done = done and p_done
    #       info[player_id] = p_info
    #       # This will remain true if all agents are done
    #       all_done = all_done and p_done

       #done['__all__'] = len(self.deads) == self.player_count
#       done['__all__'] = self.n_step >= self.max_steps
#       done['__all__'] = False
        self.n_step += 1

#       print('past deads:', self.past_deads)
#       print('done:', done)
#       print('rew:', rew)
        return obs, rew, done, info

#   def step_player(self, player_id, action):
#       if action is None:
#           reward = 0
#           done = True
#           info = {}
#           # Just a no-op.
#           _ = self._players[player_id].step('move', [0, 0], True)
#       else:
#          #x = action[0]
#          #y = action[1]
#           action_name = self.action_names[action[0]]
#           action_id = action[1]
#          #action_data = [x, y, action_id]
#          #reward, done, info = self.env._players[player_id].step(action_name, action_data)
#           print(player_id, action_name, action_id)

#           reward, done, info = self.env._players[player_id].step(action_name, [action_id], True)
#           self.env._player_last_observation[player_id] = np.array(self.env._players[player_id].observe(), copy=False)

#       return self.env._player_last_observation[player_id], reward, done, info

    def reset(self, level_id=None, level_string=None):
        self.past_deads = set()
        obs = super().reset(level_id=level_id, level_string=level_string)

        # Overwrite the action space
#       self.env.action_space = self._create_action_space()
#       self.action_space = self.env.action_space
#       self.observation_space = self.env.observation_space

#       obs = dict([(i, val) for i, val in enumerate(obs)])

        self.n_step = 0

        return obs

   #def _create_action_space(self):

   #    # Convert action to GriddlyActionASpace
#  #    self.player_count = self.player_count
#  #    self.action_input_mappings = self.action_input_mappings

   #    self._grid_width = self.game.get_width()
   #    self._grid_height = self.game.get_height()

   #    self.avatar_object = self.gdy.get_avatar_object()

   #    has_avatar = self.avatar_object is not None and len(self.avatar_object) > 0

   #   #if has_avatar:
   #   #    raise RuntimeError("Cannot use MultiDiscreteRTSWrapper with environments that control single avatars")

   #    self.valid_action_mappings = {}
   #    self.action_names = []
   #    self.max_action_ids = 0

   #    for action_name, mapping in sorted(self.action_input_mappings.items()):
   #        if not mapping['Internal']:
   #            self.action_names.append(action_name)
   #            num_action_ids = len(mapping['InputMappings']) + 1

   #            if self.max_action_ids < num_action_ids:
   #                self.max_action_ids = num_action_ids
   #            self.valid_action_mappings[action_name] = num_action_ids

   #   #multi_discrete_space = [1, self._grid_width, self._grid_height, len(self.valid_action_mappings),
   #   #                        self.max_action_ids]
   #    multi_discrete_space = [len(self.valid_action_mappings), self.max_action_ids]

   #    return ValidatedMultiDiscreteNMMO(multi_discrete_space, self)
