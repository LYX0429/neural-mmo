import numpy as np
import ray
from ray import rllib

import projekt
from projekt import rlutils
from forge.blade.io import stimulus

# Generate RLlib policies


def createPolicies(config):
    obs = projekt.realm.observationSpace(config)
    atns = projekt.realm.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i)
        policies[key] = (None, obs, atns, params)

    return policies


# Instantiate a new environment
def createEnv(config):
    return projekt.Realm(config)


# Map agentID to policyID -- requires config global


def mapPolicy(agentID):
    return 'policy_{}'.format(agentID % sim_config.NPOLICIES)


sim_config = projekt.Config()


class PCG(rllib.MultiAgentEnv):
    def __init__(self, config):
        self.done = False
        self.config = config['config']
        self.sim = projekt.Realm(config)
        self.sim_obs = self.sim.reset()
        self.n_step = 0
        self.max_steps = 200
        self.sim_steps = 200
        self.sim_done = {}
        self.stimSize = 100
        self.desciples={0: PCGAgent()}

        # RLlib registry
        rllib.models.ModelCatalog.register_custom_model(
            'test_model_agent', projekt.Policy)
        ray.tune.registry.register_env("custom_sim", createEnv)

        # Create policies
        policies = createPolicies(sim_config)

        # Instantiate monolithic RLlib Trainer object.
        self.sim_trainer = rlutils.SanePPOTrainer(
            env="custom_sim",
            path='experiment',
            config={
                'num_workers': 4,
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
                    "policies": policies,
                    "policy_mapping_fn": mapPolicy
                },
                'model': {
                    'custom_model': 'test_model',
                    'custom_options': {
                        'config': sim_config
                    }
                },
            })
        self.size = self.sim.world.env.size = self.sim.world.env.tiles.shape

    def reset(self, idx):
        self.sim_obs = self.sim.reset()
        self.pcg_reward = {0: 0}
        self.done = False

        return self.sim_obs

    def stim(self, pos):
        return self.sim.world.env.getPadded(self.sim.world.env.tiles,
                                            pos,
                                            self.stimSize,
                                            key=lambda e: 0).astype(
                                                np.int8)
    def reward(self, entID):
        return self.pcg_reward[entID]

    def step(self, action):
        if self.n_step == self.max_steps:
            self.done = True

            for i in range(self.sim_steps):

                # Remove dead agents

                for agentID in self.sim_done:
                    if self.sim_done[agentID]:
                        del self.sim_obs[agentID]
                        self.pcg_reward[0] += 1

                # Compute batch of actions
                actions, self.sim_state, _ = self.sim_trainer.compute_actions(
                    self.sim_obs, state=self.sim_state, policy_id='policy_0')

                # Compute overlay maps
                # self.overlays.register(self.sim_obs)

                # Step the environment
                self.sim_obs, sim_rewards, self.sim_done, _ = self.sim.step(
                    actions)


# obs = self.sim_obs.items()[0][1]
        obs = self.stim((0, 0))
        # obs = self.sim.world.tiles:w
        obs, reward, dones = self.getStims()
       #obs = {0: {'map': obs}}
       #rewards = {0: self.reward}
       #dones = {0: self.done, '__all__': False}

        return obs, reward, dones, {}

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
            tile = self.sim.world.env.tiles[r, c].tex
            stim = self.sim.world.env.stim(
                   ent.base.pos, self.config.STIM)
 
            obs[entID], self.raw[entID] = stimulus.Dynamic.process(
                  self.config, stim, ent)
            ob = obs[entID]
 
            rewards[entID] = self.reward(entID)
            dones[entID]   = False
 
       #for ent in self.dead:
       #    #Why do we have to provide an ob for the last timestep?
       #    #Currently just copying one over
       #    rewards[ent.entID] = self.reward(ent)
       #    dones[ent.entID]   = True
       #    obs[ent.entID]     = ob
 
        return obs, rewards, dones

class Population():
    def __init__(self):
        self.val = 0

class AgentBase():
    def __init__(self):
        self.pos = (0, 0)
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
