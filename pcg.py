from collections import defaultdict

import numpy as np
import gym
import ray
from ray import rllib
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated

import projekt
from forge.blade.io import stimulus
from forge.blade.io.action.node import Node, NodeType
from forge.blade.io.action.static import Direction, Fixed
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.lib.enums import Palette
from forge.blade.lib.utils import staticproperty
from projekt import rlutils
from forge.blade.lib import enums
from forge.trinity.ascend import runtime, Timed

#from forge.blade.io.action.static import Action

class PCGRealm(Timed):
   '''Neural MMO environment class implementing the OpenAI Gym API function
   signatures. The actual (ob, reward, done, info) data contents returned by
   the canonical reset() and step(action) methods conform to RLlib's Gym
   extensions in order to support multiple and variably sized agent
   populations. This means you cannot use preexisting optimizer
   implementations that expect the OpenAI Gym API. We recommend
   PyTorch+RLlib to take advantage of our prebuilt baseline implementations,
   but any framework that supports RLlib's fairly popular environment API and
   extended OpenAI gym.spaces observation/action definitions works as well.'''
   def __init__(self, config, idx=0):
      '''
      Args:
         config : A forge.blade.core.Config (or subclass) specification object
         idx    : Index of the map file to load (0 to number of maps)
      '''
      super().__init__()
      self.spawner   = Spawner(config)
      self.world     = core.Env(config, idx)
      self.env       = self.world.env

      self.globalValues = None
      self.config    = config
      self.worldIdx  = idx
      self.desciples = {}
      self.overlay   = {}

      self.entID     = 1
      self.tick      = 0

   @runtime
   def step(self, decisions):
      '''OpenAI Gym API step function simulating one game tick or timestep

      Args:
         decisions: A dictionary of agent action choices of format::

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

            You do not need to provide actions for each agent
            You do not need to provide an action of each type for each agent
            Only provided actions for provided agents will be evaluated
            Unprovided action types are interpreted as no-ops
            Invalid actions are ignored

            It is also possible to specify invalid combinations of valid
            actions, such as two movements or two attacks. In this case,
            one will be selected arbitrarily from each incompatible sets.

            A well-formed algorithm should do none of the above. We only
            Perform this conditional processing to make batched action
            computation easier.

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
            all other circumstances. Realm.hook provides an interface for
            creating custom reward functions using full game state.
 
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
            trajectory bounds.

         infos:
            An empty dictionary provided only for conformity with OpenAI Gym.
      '''
      print('PCGRealm decisions {}'.format(decisions))
      self.tick += 1

      #Spawn an ent
      iden, pop, name = self.spawn()
      assert iden is not None

      self.spawner.spawn(self, iden, pop, name)
      self.stepEnts(decisions)

      self.stepEnv()
      obs, rewards, dones = self.getStims()

      return obs, rewards, dones, {}

   def reset(self):
      '''Instantiates the environment and returns initial observations

      Neural MMO simulates a persistent world. It is best-practice to call
      reset() once per environment upon initialization and never again.
      Treating agent lifetimes as episodes enables training with all on-policy
      and off-policy reinforcement learning algorithms.

      We provide this function for conformity with OpenAI Gym and
      compatibility with various existing off-the-shelf reinforcement
      learning algorithms that expect a hard environment reset. If you
      absolutely must call this method after the first initialization,
      we suggest using very long (1000+) timestep environment simulations.

      Returns:
         observations, as documented by step()
      '''
      err = 'Neural MMO is persistent and may only be reset once upon initialization'
      assert self.tick == 0, err
      return self.step({})

   def reward(self, entID):
      '''Computes the reward for the specified agent

      You can override this method to create custom reward functions.
      This method has access to the full environment state via self.
      The baselines do not modify this method. You should specify any
      changes you may have made to this method when comparing to the baselines

      Returns:
         float:

         reward:
            The reward for the actions on the previous timestep of the
            entity identified by entID.
      '''
      if entID in self.dead:
         return -1
      return 0

   def spawn(self):
      '''Called when an agent is added to the environment

      You can override this method to specify custom spawning behavior
      with full access to the environment state via self.

      Returns:
         (int, int, str):

         entID:
            An integer used to uniquely identify the entity

         popID:
            An integer used to identity membership within a population

         prefix:
            The agent will be named prefix + entID

      Notes:
         This API hook is mainly intended for population-based research. In
         particular, it allows you to define behavior that selectively
         spawns agents into particular populations based on the current game
         state -- for example, current population sizes or performance.'''
      pop        =  hash(str(self.entID)) % self.config.NPOP
      self.entID += 1

      return self.entID, pop, 'Neural_'

   def clientData(self):
      '''Data packet used by the renderer

      Returns:
         packet: A packet of data for the client
      '''
      packet = {
            'environment': self.world.env,
            'entities': dict((k, v.packet())
               for k, v in self.desciples.items()),
            'overlay': self.overlay
            }
      return packet

   def act(self, actions):
      '''Execute agent actions

      Args:
         actions: A dictionary of agent actions
      '''
      for priority, tups in actions.items():
         for entID, atnArgs in tups.items():
            ent = self.desciples[entID]
            ent.act(self.world, atnArgs)

   def prioritize(self, decisions):
      '''Reorders actions according to their priorities

      Args:
         decisions: A dictionary of agent actions

      Returns:
         Repriotized actions

      Notes:
         Only saves the first action of each priority
      '''
      actions = defaultdict(dict)
      for entID, atns in decisions.items():
         for atn, args in atns.items():
            actions[atn.priority][entID] = [atn, args]
      return actions

   def stepEnv(self):
      '''Advances the environment'''
      ents = list(chain(self.desciples.values()))

      #Stats
      self.world.step(ents, [])
      self.world.env.step()

      self.env = self.world.env.np()

   def stepEnts(self, decisions):
      '''Advance agents

      Args:
         decisions: A dictionary of agent actions

      Returns:
         packets : State-reward-done packets
         dones   : A list of dead agent IDs
      '''
      #Step all ents first
      for entID, actions in decisions.items():
         ent = self.desciples[entID]
         ent.step(self.world, actions)

      actions = self.prioritize(decisions)
      self.act(actions)

      #Finally cull dead. This will enable MAD melee
      dead = set()
      for entID, ent in self.desciples.items():
         self.postmortem(ent, dead)

      self.dead = dead
      self.cullDead(dead)
      return dead

   def postmortem(self, ent, dead):
      '''Add agent to the graveyard if it is dead

      Args:
         ent  : An agent object
         dead : A list of dead agents

      Returns:
         bool: Whether the agent is dead
      '''
      if not ent.base.alive:
         dead.add(ent)
         return True
      return False

   def cullDead(self, dead):
      '''Deletes the specified list of agents

      Args:
         dead: A list of dead agent IDs to remove
      '''
      for ent in dead:
         r, c  = ent.base.pos
         entID = ent.entID

         self.world.env.tiles[r, c].delEnt(entID)
         self.spawner.cull(ent.annID)

         del self.desciples[entID]

   def getStims(self):
      '''Gets agent stimuli from the environment

      Args:
         packets: A dictionary of Packet objects

      Returns:
         The packet dictionary populated with agent data
      '''
      self.raw = {}
      obs, rewards, dones = {}, {}, {'__all__': False}
      for entID, ent in self.desciples.items():
         r, c = ent.base.pos
         tile = self.world.env.tiles[r, c].tex
         stim = self.world.env.stim(
                ent.base.pos, self.config.STIM)

         obs[entID], self.raw[entID] = stimulus.Dynamic.process(
               self.config, stim, ent)
         ob = obs[entID]

         rewards[entID] = self.reward(entID)
         dones[entID]   = False

      for ent in self.dead:
         #Why do we have to provide an ob for the last timestep?
         #Currently just copying one over
         rewards[ent.entID] = self.reward(ent)
         dones[ent.entID]   = True
         obs[ent.entID]     = ob

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
      return self.world.env.tiles.shape

   def registerOverlay(self, overlay, name):
      '''Registers an overlay to be sent to the client

      This variable is included in client data passed to the renderer and is
      typically used to send value maps computed using getValStim to the
      client in order to render as an overlay.

      Args:
         values: A map-sized (self.size) array of floating point values
      '''
      err = 'overlay must be a numpy array of dimension (*(env.size), 3)'
      assert type(overlay) == np.ndarray, err
      self.overlay[name] = overlay.tolist()

   def getValStim(self):
      '''Simulates an agent on every tile and returns observations

      This method is used to compute per-tile visualizations across the
      entire map simultaneously. To do so, we spawn agents on each tile
      one at a time. We compute the observation for each agent, delete that
      agent, and go on to the next one. In this fashion, each agent receives
      an observation where it is the only agent alive. This allows us to
      isolate potential influences from observations of nearby agents

      This function is slow, and anything you do with it is porbably slower.
      As a concrete example, consider that we would like to visualize a
      learned agent value function for the entire map. This would require
      computing a forward pass for one agent per tile. To cut down on
      computation costs, we omit lava tiles from this method

      Returns:
         (dict, dict):

         observations:
            A dictionary of agent observations as specified by step()

         stimuli:
            A dictionary of raw game object observations as follows::

               {
                  agent_1: (tiles, agent),
                  agent_2: (tiles, agent),
                  ...
               ]

            Where agent_i is the integer index of the i\'th agent,
            tiles is an array of observed game tiles, and agent is the
            game object corresponding to agent_i
      '''
      config  = self.config
      R, C    = self.world.env.tiles.shape
      B       = config.BORDER

      entID   = 0
      pop     = 0
      name    = "Value"
      color   = (255, 255, 255)

      observations, stims = {}, {}
      for r in range(B-1, R-B):
         for c in range(B-1, C-B):
            pos   = tuple([r, c])
            ent   = entity.Player(config, entID, pop, name, color)

            ent.base.r.update(r)
            ent.base.c.update(c)

            self.world.env.tiles[r, c].addEnt(entID, ent)
            stim   = self.world.env.stim(pos, self.config.STIM)
            obs, _ = stimulus.Dynamic.process(self.config, stim, ent)
            self.world.env.tiles[r, c].delEnt(entID)

            observations[entID] = obs
            stims[entID] = (stim, ent)
            entID += 1

      return observations, stims

class Action(Node):
    nodeType = NodeType.SELECTION

    @staticproperty
    def edges():
        print('Action edges pcg')
        #return [Move, Attack, Exchange, Skill]

        return [Move, Terraform]

    @staticproperty
    def n():
        return len(Action.arguments)

    def args(stim, entity, config):
        return Static.edges
    #Called upon module import (see bottom of file)
    #Sets up serialization domain
    def hook():
        idx = 0
        arguments = []

        for action in Action.edges:
            for args in action.edges:
                if not 'edges' in args.__dict__:
                    continue

                for arg in args.edges:
                    arguments.append(arg)
                    arg.serial = tuple([idx])
                    arg.idx = idx
                    idx += 1
        Action.arguments = arguments

class Terraform(Node):
    nodeType = NodeType.SELECTION
    @staticproperty
    def n():
        print('len terraform args', Terraform.arguments)
        return len(Terraform.arguments)

    @staticproperty
    def edges():
        return [Terrain]

    @staticproperty
    def leaf():
        return True

    def call(world, entity, terr):
        print('world', world)

        return 1

class Terrain(Node):
    argType = Fixed

    @staticmethod
    def n():
        return len(enums.Material)

    def args(stim, entity, config):
        print('enums.Material', enums.Material)
        raise Exception
        return enums.Material

class Move(Node):
    priority = 0
    nodeType = NodeType.SELECTION
    def call(world, entity, direction):
        r, c = entity.base.pos
       #entity.history.lastPos = (r, c)
        rDelta, cDelta = direction.delta
        rNew, cNew = r+rDelta, c+cDelta
       #if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
       #    return
        if not utils.inBounds(rNew, cNew, world.shape):
            return
       #if entity.status.freeze > 0:
       #   return

        entity.base.pos = (rNew, cNew)
       #entity.base.r.update(rNew)
       #entity.base.c.update(cNew)
       #entID = entity.entID

       #r, c = entity.history.lastPos
       #world.env.tiles[r, c].delEnt(entID)

       #r, c = entity.base.pos
       #world.env.tiles[r, c].addEnt(entID, entity)

    @staticproperty
    def edges():
       return [Direction]

    @staticproperty
    def leaf():
       return True


def createSimPolicies(config):
   obs      = projekt.realm.observationSpace(config)
   atns     = projekt.realm.actionSpace(config)
   policies = {}

   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
      key           = mapPolicy(i, config)
      policies[key] = (None, obs, atns, params)

   return policies


def createPCGPolicies(config):
    # Generate RLlib policies
    obs = projekt.realm.observationSpace(config)
    atns = actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i, config)
        policies[key] = (None, obs, atns, params)

    return policies


# Instantiate a new environment
def createEnv(config):
    return projekt.Realm(config)


# Map agentID to policyID -- requires config global


def mapPolicy(agentID, config):
    return 'policy_{}'.format(agentID % config.NPOLICIES)


class Spawner():
    def __init__(self):
        self.palette = Palette(1)


sim_config = projekt.Config()

class PCG(PCGRealm, rllib.MultiAgentEnv):
    def __init__(self, config):
        self.done = False
        self.config = config['config']
        self.sim = projekt.Realm(config)
        self.sim_obs = self.sim.reset()
        self.n_step = 0
        self.max_steps = 200
        self.sim_steps = 200
        self.sim_done = {}
        self.sim_state = {}
        self.stimSize = 9
        self.desciples={0: PCGAgent()}
        self.spawner = Spawner()

        # RLlib registry
        rllib.models.ModelCatalog.register_custom_model(
            'test_model_agent', projekt.Policy)
        ray.tune.registry.register_env("custom_sim", createEnv)

        pcg_policies = createPCGPolicies(self.config)

        # Create policies
        sim_policies = createSimPolicies(sim_config)

        # Instantiate monolithic RLlib Trainer object.
        self.sim_trainer = rlutils.SanePPOTrainer(
            env="custom_sim",
            path='experiment',
            config={
                'num_workers': 1,
                'num_gpus': 1,
                'num_envs_per_worker': 1,
                'train_batch_size': 4000,
                'rollout_fragment_length': 100,
                'sgd_minibatch_size': 128,
                'num_sgd_iter': 1,
                'use_pytorch': True,
                'horizon': np.inf,
                'soft_horizon': False,
                'no_done_at_end': False,
                'env_config': {
                    'config': sim_config
                },
                'multiagent': {
                    "policies": sim_policies,
                    "policy_mapping_fn": mapPolicy
                },
                'model': {
                    'custom_model': 'test_model',
                    'custom_options': {
                        'config': sim_config
                    }
                },
            })
        self.world = self.sim.world
       #self.size = self.sim.world.env.size = self.sim.world.env.tiles.shape

    def registerOverlay(self, colorized, counts):
        pass

    def reset(self, idx=0):
        self.sim_obs = self.sim.reset()
        self.pcg_reward = {0: 0}
        self.done = False

        return self.sim_obs

   #def stim(self, pos):
   #    return self.sim.world.env.getPadded(self.sim.world.env.tiles,
   #                                        pos,
   #                                        self.stimSize,
   #                                        key=lambda e: 0).astype(
   #                                            np.int8)
    def reward(self, entID):
        return self.pcg_reward[entID]

# WHO IS CALLING THIS FUNCTION AND SENDING IT THE WRONG ACTION???

    def step(self, action):
        print('PCG.step action: {}'.format(action))

        if self.n_step == self.max_steps:
            self.done = True


        # Remove dead agents

        for agentID in self.sim_done:
            if self.sim_done[agentID]:
                del self.sim_obs[agentID]
                self.pcg_reward[0] += 1

       #print('sim obs\n{}'.format(self.sim_obs))
       #print('sim state\n{}'.format(self.sim_state))
        # Compute batch of actions
        actions, self.sim_state, _ = self.sim_trainer.compute_actions(
            self.sim_obs, state=self.sim_state, policy_id='policy_0')
        print('sim actions {}'.format(actions))
        # Compute overlay maps
        # self.overlays.register(self.sim_obs)

        # Step the environment
        self.sim_obs, sim_rewards, self.sim_done, _ = self.sim.step(
            actions)


# obs = self.sim_obs.items()[0][1]
       #obs = self.stim((0, 0))
        # obs = self.sim.world.tiles:w
        obs, reward, dones = self.getStims()
       #obs = {0: {'map': obs}}
       #rewards = {0: self.reward}
       #dones = {0: self.done, '__all__': False}
        self.n_step += 1
       #print('obs', obs)

        return obs, reward, dones, {}

    def getStims(self):
        '''Gets agent stimuli from the environment

        Args:
           packets: A dictionary of Packet objects

        Returns:
           The packet dictionary populated with agent data
        '''
       #print('getting pcg stim?')
        self.raw = {}
        obs, rewards, dones = {}, {}, {'__all__': False}

        for entID, ent in self.desciples.items():
            r, c = ent.base.pos
            tile = self.sim.world.env.tiles[r, c].tex
            stim = self.sim.world.env.stim(
                   ent.base.pos, 3)
#           print('stim', stim)

            obs[entID], self.raw[entID] = stimulus.Dynamic.process(
                  self.config, stim, ent)
            ob = obs[entID]

            rewards[entID] = self.reward(entID)
            dones[entID]   = self.done

       #for ent in self.dead:
       #    #Why do we have to provide an ob for the last timestep?
       #    #Currently just copying one over
       #    rewards[ent.entID] = self.reward(ent)
       #    dones[ent.entID]   = True
       #    obs[ent.entID]     = ob

        return obs, rewards, dones


    def clientData(self):
        '''Data packet used by the renderer

        Returns:
           packet: A packet of data for the client
        '''
        packet = {
              'environment': self.sim.world.env,
              'entities': dict((k, v.packet())
                 for k, v in self.sim.desciples.items()),
              'overlay': self.sim.overlay
              }

        return packet

class Population():
    def __init__(self):
        self.val = 0

class AgentBase():
    def __init__(self):
        self.pos = (10, 10)
        self.population = Population()

class PCGAgent():
    def __init__(self):
        self.base = AgentBase()


#Neural MMO observation space
def observationSpace(config):
   obs = FlexDict({})

   for entity in sorted(Stimulus.values()):
      attrDict = FlexDict({})

      for attr in sorted(entity.values()):
         attrDict[attr] = attr(config).space
      n           = entity.N(config)
      obs[entity] = Repeated(attrDict, max_len=n)

   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))

   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)

   return atns


