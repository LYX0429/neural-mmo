from pdb import set_trace as TT
import json
import os
import numpy as np

from collections import defaultdict
from itertools import chain
from copy import deepcopy

from forge.blade import entity, core
from forge.blade.io import stimulus
from forge.blade.io.stimulus import Static
from forge.blade.systems import combat

from forge.blade.lib.enums import MaterialEnum
from forge.blade.lib.material import Material
from forge.blade.lib import log
import ray

class Env:
   '''Environment wrapper for Neural MMO

   Note that the contents of (ob, reward, done, info) returned by the standard
   OpenAI Gym API reset() and step(actions) methods have been generalized to
   support variable agent populations and more expressive observation/action
   spaces. This means you cannot use preexisting optimizer implementations that
   strictly expect the OpenAI Gym API. We recommend PyTorch+RLlib to take
   advantage of our prebuilt baseline implementations, but any framework that
   supports RLlib's fairly popular environment API and extended OpenAI
   gym.spaces observation/action definitions should work as well.'''
   def __init__(self, config):
      '''
      Args:
         config : A forge.blade.core.Config object or subclass object
      '''
      super().__init__()
      self.realm     = core.Realm(config, self.spawn)

      self.config    = config
      self.overlay   = None
      self.worldIdx = None


   def set_map(self, idx, maps):
      if idx is None:
         counter = ray.get_actor("global_counter")
         idx = ray.get(counter.get.remote())
      map_arr = maps[idx]
      self.realm.set_map(idx, map_arr)
      self.worldIdx = idx

   ############################################################################
   ### Core API
   def reset(self, idx=None, step=True):
      '''OpenAI Gym API reset function

      Loads a new game map and returns initial observations

      Args:
         idx: Map index to load. Selects a random map by default

         step: Whether to step the environment and return initial obs

      Returns:
         obs: Initial obs if step=True, None otherwise 

      Notes:
         Neural MMO simulates a persistent world. Ideally, you should reset
         the environment only once, upon creation. In practice, this approach
         limits the number of parallel environment simulations to the number
         of CPU cores available. At small and medium hardware scale, we
         therefore recommend the standard approach of resetting after a long
         but finite horizon: ~1000 timesteps for small maps and
         5000+ timesteps for large maps

      Returns:
         observations, as documented by step()
      '''

      # ad hoc reset of player indexing
      self.realm.players.reset_pop_counts()
#     self.quill = log.Quill(self.realm.identify)
#
#     if idx is None:
#        idx = np.random.randint(self.config.N_TRAIN_MAPS) + 1

#     if self.worldIdx is None:
#        self.worldIdx = idx
#     self.realm.reset(self.worldIdx)

#     obs = None
#     if step:
#        obs, _, _, _ = self.step({})

#     return obs

      self.quill = log.Quill(self.realm.identify)

#     self.env_reset = time.time()

      self.agent_skills = {i: {} for i in range(self.config.NPOP)}
      self.lifetimes = {}
      self.actions_matched = {}

      # if idx is None:
      if self.config.EVO_MAP and not self.config.FIXED_MAPS:
         idx = self.worldIdx
      #        counter = ray.get_actor("global_counter")
      #        idx = ray.get(counter.get.remote())
      #        self.worldIdx = idx
      #        idx = self.worldIdx
      #        global_stats = ray.get_actor("global_stats")
      #        atk_mults = ray.get(global_stats.get_mults.remote(idx))
      #        if atk_mults:
      #           for k, v in atk_mults.items():
      #              setattr(self.config, k, v)
      elif self.config.EVALUATE and idx is not None and not self.config.MAP == 'PCG':
         json_path = os.path.join(os.curdir, 'evo_experiment', self.config.MAP, 'maps', 'atk_mults{}json'.format(idx))
         #       with open(json_path, 'r') as f:
         #          atk_mults = json.load(f)
         pass
      elif idx is None:
         idx = np.random.randint(self.config.NMAPS)

      #     self.worldIdx = idx
      #     print('trinity env idx: {}'.format(idx))
      self.dead = {}

      self.realm.reset(idx)

      if self.config.EVO_MAP:  # and not self.config.MAP == 'PCG':
         #   self.realm.spawn_points = ray.get(global_stats.get_spawn_points.remote(idx))
         self.realm.spawn_points = np.vstack(np.where(self.realm.map.inds() == MaterialEnum.SPAWN.value.index)).transpose()

      obs = None
      if step:
         obs, _, _, _ = self.step({})

      return obs

   def step(self, actions, omitDead=False, preprocessActions=True):
      '''OpenAI Gym API step function simulating one game tick or timestep

      Args:
         actions: A dictionary of agent decisions of format::

               {
                  agent_1: {
                     action_1: [arg_1, arg_2],
                     action_2: [...],
                     ...
                  },
                  agent_2: {
                     ...
                  },
                  ...
               }

            Where agent_i is the integer index of the i\'th agent 

            The environment only evaluates provided actions for provided
            agents. Unprovided action types are interpreted as no-ops and
            illegal actions are ignored

            It is also possible to specify invalid combinations of valid
            actions, such as two movements or two attacks. In this case,
            one will be selected arbitrarily from each incompatible sets.

            A well-formed algorithm should do none of the above. We only
            Perform this conditional processing to make batched action
            computation easier.

         omitDead: Whether to omit dead agents observations from the returned
            obs. Provided for conformity with some optimizer APIs

         preprocessActions: Whether to treat actions as raw indices or
            as game objects. Typically, this value should be True for
            neural models and false for scripted baselines.

      Returns:
         (dict, dict, dict, None):

         observations:
            A dictionary of agent observations of format::

               {
                  agent_1: obs_1,
                  agent_2: obs_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            obs_i is the observation of the i\'th' agent. Note that obs_i
            is a structured datatype -- not a flat tensor. It is automatically
            interpretable under an extended OpenAI gym.spaces API. Our demo
            code shows how do to this in RLlib. Other frameworks must
            implement the same extended gym.spaces API to do the same.
            
         rewards:
            A dictionary of agent rewards of format::

               {
                  agent_1: reward_1,
                  agent_2: reward_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            reward_i is the reward of the i\'th' agent.

            By default, agents receive -1 reward for dying and 0 reward for
            all other circumstances. Override Env.reward to specify
            custom reward functions
 
         dones:
            A dictionary of agent done booleans of format::

               {
                  agent_1: done_1,
                  agent_2: done_2,
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent and
            done_i is a boolean denoting whether the i\'th agent has died.

            Note that obs_i will be a garbage placeholder if done_i is true.
            This is provided only for conformity with OpenAI Gym. Your
            algorithm should not attempt to leverage observations outside of
            trajectory bounds. You can omit garbage obs_i values by setting
            omitDead=True.

         infos:
            An empty dictionary provided only for conformity with OpenAI Gym.
      '''
      #Preprocess actions
      if preprocessActions:
         for entID in list(actions.keys()):
            ent = self.realm.players[entID]
            if not ent.alive:
               continue

            for atn, args in actions[entID].items():
               for arg, val in args.items():
                  if len(arg.edges) > 0:
                     actions[entID][atn][arg] = arg.edges[val]
                  elif val < len(ent.targets):
                     targ                     = ent.targets[val]
                     actions[entID][atn][arg] = self.realm.entity(targ)
                  else: #Need to fix -inf in classifier before removing this
                     actions[entID][atn][arg] = ent

      #Step: Realm, Observations, Logs
      dead = self.realm.step(actions)
      obs, rewards, dones, self.raw = {}, {}, {}, {}
      for entID, ent in self.realm.players.items():
         ob             = self.realm.dataframe.get(ent)
         obs[entID]     = ob

         rewards[entID] = self.reward(entID)
         dones[entID]   = False

      for entID, ent in dead.items():
         self.log(ent)

         for entID, ent in dead.items():
            lifetime = ent.history.timeAlive.val
            self.lifetimes[entID] = lifetime
            self.agent_skills[ent.base.population.val][entID] = self.get_agent_stats(ent)
            if hasattr(self, 'actions_matched'):
               actions_matched = ent.actions_matched
               self.actions_matched[entID] = actions_matched

      #Postprocess dead agents
      if omitDead:
         return obs, rewards, dones, {}

      for entID, ent in dead.items():
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = ob
      dones['__all__'] = False

      return obs, rewards, dones, {}

   ############################################################################
   ### Logging
   def log(self, ent) -> None:
      '''Logs agent data upon death

      This function is called automatically when an agent dies. Logs are used
      to compute summary stats and populate the dashboard. You should not
      call it manually. Instead, override this method to customize logging.

      Args:
         ent: An agent
      '''

      quill = self.quill

      blob = quill.register('Population', self.realm.tick,
            quill.HISTOGRAM, quill.LINE, quill.SCATTER)
      blob.log(self.realm.population)

      blob = quill.register('Lifetime', self.realm.tick,
            quill.HISTOGRAM, quill.LINE, quill.SCATTER, quill.GANTT)
      blob.log(ent.history.timeAlive.val)

      blob = quill.register('Skill Level', self.realm.tick,
            quill.HISTOGRAM, quill.STACKED_AREA, quill.STATS, quill.RADAR)
      blob.log(ent.skills.range.level,        'Range')
      blob.log(ent.skills.mage.level,         'Mage')
      blob.log(ent.skills.melee.level,        'Melee')
      blob.log(ent.skills.constitution.level, 'Constitution')
      blob.log(ent.skills.defense.level,      'Defense')
      blob.log(ent.skills.fishing.level,      'Fishing')
      blob.log(ent.skills.hunting.level,      'Hunting')
      blob.log(ent.skills.woodcutting.level,  'Woodcutting')
      blob.log(ent.skills.mining.level,       'Mining')
      blob.log(len(ent.explored),    'Exploration')

      blob = quill.register('Equipment', self.realm.tick,
            quill.HISTOGRAM, quill.SCATTER)
      blob.log(ent.loadout.chestplate.level, 'Chestplate')
      blob.log(ent.loadout.platelegs.level,  'Platelegs')

      blob = quill.register('Exploration', self.realm.tick,
            quill.HISTOGRAM, quill.SCATTER)
      blob.log(ent.history.exploration)

      quill.stat('Population', self.realm.population)
      quill.stat('Lifetime',  ent.history.timeAlive.val)
      quill.stat('Skilling',  (ent.skills.fishing.level + ent.skills.hunting.level)/2.0)
      quill.stat('Combat',    combat.level(ent.skills))
      quill.stat('Equipment', ent.loadout.defense)
#<<<<<<< HEAD
#     quill.stat('Exploration', ent.exploration_grid.sum())
#=======
      quill.stat('Exploration', ent.history.exploration)
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

   def terminal(self):
      '''Logs currently alive agents and returns all collected logs

#<<<<<<< HEAD
#      return self.quill.packet
#
#   def set_map(self, idx, maps):
#      if idx is None:
#         counter = ray.get_actor("global_counter")
#         idx = ray.get(counter.get.remote())
#      map_arr = maps[idx]
#      self.realm.set_map(idx, map_arr)
#      self.worldIdx = idx
#
#   def reset(self, idx=None, step=True):
##     print('reset', idx)
#      Instantiates the environment and returns initial observations
#
#      Neural MMO simulates a persistent world. It is best-practice to call
#      reset() once per environment upon initialization and never again.
#      Treating agent lifetimes as episodes enables training with all on-policy
#      and off-policy reinforcement learning algorithms.
#
#      We provide this function for conformity with OpenAI Gym and
#      compatibility with various existing off-the-shelf reinforcement
#      learning algorithms that expect a hard environment reset. If you
#      absolutely must call this method after the first initialization,
#      we suggest using very long (1000+) timestep environment simulations.
#=======
      Automatic log calls occur only when agents die. To evaluate agent
      performance over a fixed horizon, you will need to include logs for
      agents that are still alive at the end of that horizon. This function
      performs that logging and returns the associated a data structure
      containing logs for the entire evaluation
>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

      Returns:
         Log datastructure. Use them to update an InkWell logger.
         
      Args:
         ent: An agent
      '''
#<<<<<<< HEAD

#=======
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

      for entID, ent in self.realm.players.entities.items():
         self.log(ent)

      return self.quill.packet



   ############################################################################
   ### Override hooks
   def reward(self, entID):
      '''Computes the reward for the specified agent

      Override this method to create custom reward functions. You have full
      access to the environment state via self.realm. Our baselines do not
      modify this method; specify any changes when comparing to baselines

      Returns:
         float:

         reward:
            The reward for the actions on the previous timestep of the
            entity identified by entID.
      '''
      if entID not in self.realm.players:
         return -1
      return 0


   def spawn(self):
      '''Called when an agent is added to the environment

      Override this method to specify name/population upon spawning.

      Returns:
         (int, str):

         popID:
            An integer used to identity membership within a population

         prefix:
            The agent will be named prefix + entID

      Notes:
         Mainly intended for population-based research. In particular, it
         allows you to define behavior that selectively spawns agents into
         particular populations based on the current game state -- for example,
         current population sizes or performance.'''

      pop = np.random.randint(self.config.NPOP)
      return pop, 'Neural_'

   ############################################################################
   ### Client data
   @property
   def packet(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      packet = {
            'config': self.config,
            'pos': self.overlayPos,
            'wilderness': combat.wilderness(self.config, self.overlayPos)
            }

      packet = {**self.realm.packet(), **packet}

      if self.overlay is not None:
         print('Overlay data: ', len(self.overlay))
         packet['overlay'] = self.overlay
         self.overlay      = None

      return packet

   def register(self, overlay):
      '''Register an overlay to be sent to the client

<<<<<<< HEAD
      Args:
         packets: A dictionary of Packet objects

      Returns:
         The packet dictionary populated with agent data
      '''
      self.raw = {}
      obs, rewards, dones = {}, {}, {'__all__': False}
      for entID, ent in self.realm.players.items():
         start = time.time()
         ob             = self.realm.dataframe.get(ent)
         obs[entID]     = ob

         rewards[entID] = self.reward(entID)
         dones[entID]   = False
         self.dummi_ob = ob

      if omitDead:
         return obs, rewards, dones

      #RLlib quirk: requires obs for dead agents
      for entID, ent in dead.items():
         #Currently just copying one over
         dones[ent.entID]   = True
         rewards[ent.entID] = self.reward(ent)
         obs[ent.entID]     = self.dummi_ob



      return obs, rewards, dones

   @property
   def size(self):
      '''Returns the size of the game map

      You can override this method to create custom reward functions.
      This method has access to the full environment state via self.
      The baselines do not modify this method. You should specify any
      changes you may have made to this method when comparing to the baselines

      Returns:
         tuple(int, int):

         size:
            The size of the map as (rows, columns)
      '''
      return self.realm.map.tiles.shape

   def registerOverlay(self, overlay):
      '''Registers an overlay to be sent to the client

      This variable is included in client data passed to the renderer and is
      typically used to send value maps computed using getValStim to the
      client in order to render as an overlay.
=======
      The intended use of this function is: User types overlay ->
      client sends cmd to server -> server computes overlay update -> 
      register(overlay) -> overlay is sent to client -> overlay rendered
>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

      Args:
         values: A map-sized (self.size) array of floating point values
      '''
      err = 'overlay must be a numpy array of dimension (*(env.size), 3)'
      assert type(overlay) == np.ndarray, err
      self.overlay = overlay.tolist()

   def dense(self):
      '''Simulates an agent on every tile and returns observations

      This method is used to compute per-tile visualizations across the
      entire map simultaneously. To do so, we spawn agents on each tile
      one at a time. We compute the observation for each agent, delete that
      agent, and go on to the next one. In this fashion, each agent receives
      an observation where it is the only agent alive. This allows us to
      isolate potential influences from observations of nearby agents

      This function is slow, and anything you do with it is probably slower.
      As a concrete example, consider that we would like to visualize a
      learned agent value function for the entire map. This would require
      computing a forward pass for one agent per tile. To cut down on
      computation costs, we omit lava tiles from this method

      Returns:
         (dict, dict):

         observations:
            A dictionary of agent observations as specified by step()

         ents:
            A corresponding dictionary of agents keyed by their entID
      '''
      config  = self.config
      R, C    = self.realm.map.tiles.shape

      entID   = 100000
      pop     = 0
      name    = "Value"
      color   = (255, 255, 255)


      observations, ents = {}, {}
      for r in range(R):
         for c in range(C):
            tile    = self.realm.map.tiles[r, c]
            if not tile.habitable:
               continue

            current = tile.ents
            n       = len(current)
            if n == 0:
               ent = entity.Player(self.realm, (r, c), entID, pop, name, color)
            else:
               ent = list(current.values())[0]

            obs = self.realm.dataframe.get(ent)
            if n == 0:
               self.realm.dataframe.remove(Static.Entity, entID, ent.pos)

            observations[entID] = obs
            ents[entID] = ent
            entID += 1

      return observations, ents
