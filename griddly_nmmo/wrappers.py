from pdb import set_trace as T

import gym
import numpy as np


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


class NMMOWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        self.deads = set()
        self.n_step = 0
        self.max_steps = max_steps

    def step(self, action):
        rew = {}
        obs = {}
        done = {}
        info = {}
        all_done = True
        [action.update({i: [0,0]}) if (i in self.deads) else action.update({i: list(val)}) for (i, val) in action.items()]
        action = [action[i] if i in action else np.zeros(len(list(action.values())[0])) for i in range(self.player_count)]
#       action.reverse()
        obs, rew, done, info = self.env.step(action)
        obs = dict([(i, val) for (i, val) in enumerate(obs)])
        rew = dict([(i, rew[i]) for i in range(self.player_count)])
        done = dict([(i, self.env.get_state()['GlobalVariables']['player_dead'][i+1] > 0 and i not in self.deads and False) for i in range(self.player_count)])
        [self.deads.add(i) if self.env.get_state()['GlobalVariables']['player_dead'][i+1] > 0 else None for i in done] 
     #  done = dict([(i, False) for i in range(self.player_count)])
        info = dict([(i, info) for i in range(self.player_count)])

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
        done['__all__'] = self.n_step >= self.max_steps
        self.n_step += 1

        return obs, rew, done, info

    def step_player(self, player_id, action):
        if action is None:
            reward = 0
            done = True
            info = {}
            # Just a no-op.
            _ = self.env._players[player_id].step('move', [0, 0], True)
        else:
           #x = action[0]
           #y = action[1]
            action_name = self.action_names[action[0]]
            action_id = action[1]
           #action_data = [x, y, action_id]
           #reward, done, info = self.env._players[player_id].step(action_name, action_data)
            print(player_id, action_name, action_id)
            reward, done, info = self.env._players[player_id].step(action_name, [action_id], True)
            self.env._player_last_observation[player_id] = np.array(self.env._players[player_id].observe(), copy=False)

        return self.env._player_last_observation[player_id], reward, done, info

    def reset(self, level_id=None, level_string=None):
        self.deads = set()
        reset_result = super().reset(level_id=level_id, level_string=level_string)

        # Overwrite the action space
        self.env.action_space = self._create_action_space()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        obs = dict([(i, val) for i, val in enumerate(reset_result)])

        self.n_step = 0

        return obs

    def _create_action_space(self):

        # Convert action to GriddlyActionASpace
        self.player_count = self.env.player_count
        self.action_input_mappings = self.env.action_input_mappings

        self._grid_width = self.env.game.get_width()
        self._grid_height = self.env.game.get_height()

        self.avatar_object = self.env.gdy.get_avatar_object()

        has_avatar = self.avatar_object is not None and len(self.avatar_object) > 0

       #if has_avatar:
       #    raise RuntimeError("Cannot use MultiDiscreteRTSWrapper with environments that control single avatars")

        self.valid_action_mappings = {}
        self.action_names = []
        self.max_action_ids = 0

        for action_name, mapping in sorted(self.action_input_mappings.items()):
            if not mapping['Internal']:
                self.action_names.append(action_name)
                num_action_ids = len(mapping['InputMappings']) + 1

                if self.max_action_ids < num_action_ids:
                    self.max_action_ids = num_action_ids
                self.valid_action_mappings[action_name] = num_action_ids

       #multi_discrete_space = [1, self._grid_width, self._grid_height, len(self.valid_action_mappings),
       #                        self.max_action_ids]
        multi_discrete_space = [len(self.valid_action_mappings), self.max_action_ids]

        return ValidatedMultiDiscreteNMMO(multi_discrete_space, self)
