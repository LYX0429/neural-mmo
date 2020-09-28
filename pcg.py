from collections import defaultdict

import numpy as np
import ray
from ray import rllib

import projekt
from pcgrl.game.io.stimulus import Stimulus, Dynamic
#from forge.blade.io import stimulus
from forge.blade.lib.enums import Palette
import pcgrl
from pcgrl import rlutils
from pcgrl.realm import PCGRealm, Spawner

# from forge.blade.io.action.static import Action


def SPAWN(self):
  '''Generates spawn positions for new agents

  Default behavior randomly selects a tile position
  along the borders of the square game map

  Returns:
     tuple(int, int):

     position:
        The position (row, col) to spawn the given agent
  '''
  R, C = Config.R, Config.C
  spawn, border, sz = [], Config.BORDER, Config.SZ
  spawn += [(border, border+i) for i in range(sz)]
  spawn += [(border+i, border) for i in range(sz)]
  spawn += [(R-1, border+i) for i in range(sz)]
  spawn += [(border+i, C-1) for i in range(sz)]
  idx = np.random.randint(0, len(spawn))
  return spawn[idx]

def createSimPolicies(config):
    obs = projekt.realm.observationSpace(config)
    atns = projekt.realm.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapSimPolicy(i, config)
        policies[key] = (None, obs, atns, params)

    return policies


#def createPCGPolicies(config):
#    # Generate RLlib policies
#    obs = static.observationSpace(config)
#    atns = static.actionSpace(config)
#    policies = {}
#
#    for i in range(config.NPOLICIES):
#        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
#        key = mapPCGPolicy(i, config)
#        policies[key] = (None, obs, atns, params)
#
#    return policies


# Instantiate a new environment
def createEnv(config):
    return projekt.Realm(config)


# Map agentID to policyID -- requires config global


def mapPCGPolicy(agentID, config):
    return 'policy_pcg_{}'.format(agentID % config.NPOLICIES)


def mapSimPolicy(agentID, config):
    return 'policy_sim_{}'.format(agentID % config.NPOLICIES)


sim_config = projekt.Config()


class PCG(PCGRealm, rllib.MultiAgentEnv):
    def __init__(self, pcg_config):
        self.done = False
       #sim_config = projekt.Config()
        sim_realm_config = {
                'config': sim_config
                }
        self.sim = projekt.Realm(sim_realm_config)
        self.sim_obs = self.sim.reset()
#       print(self.sim.world.env)
#       print(dir(self.sim.world.env))
       #self.world_map = self.sim.world.env
        self.n_step = 0
        self.max_steps = 200
        self.sim_steps = 200
        self.sim_done = {}
        self.sim_state = {}
        self.stimSize = 9
       #self.desciples = {0: PCGAgent()}

        #FIXME: how could you
        pcg_config = pcg_config['config']
       #print(pcg_config, type(pcg_config))
      # pcg_config.NENT = 12
      # pcg_config.NPOP = 1
      # pcg_config.STIM = 4
      # pcg_config.SPAWN = SPAWN
        self.pcg_config = pcg_config

        self.spawner = Spawner(pcg_config)

        # RLlib registry
        rllib.models.ModelCatalog.register_custom_model(
            'test_sim_model', projekt.Policy)
        ray.tune.registry.register_env("custom_sim", createEnv)

        #       pcg_policies = createPCGPolicies(self.config)

        # Create policies
        sim_policies = createSimPolicies(sim_config)

        # Instantiate monolithic RLlib Trainer object.
        self.sim_trainer = rlutils.SanePPOSimTrainer(
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
                    "policy_mapping_fn": mapSimPolicy
                },
                'model': {
                    'custom_model': 'test_sim_model',
                    'custom_options': {
                        'config': sim_config
                    }
                },
            })
#       self.sim_trainer.defaultModel().cuda()
        self.world = self.sim.world

    #self.size = self.sim.world.env.size = self.sim.world.env.tiles.shape

    def registerOverlay(self, colorized, counts):
        pass

    def reset(self, idx=0):
        self.n_step = 0
        self.pcg_reward = 0
        self.sim_obs = self.sim.reset()
        self.done = False
        super().__init__(self.pcg_config, idx, self.sim)

        return self.step({})[0]

    # def stim(self, pos):
    #    return self.sim.world.env.getPadded(self.sim.world.env.tiles,
    #                                        pos,
    #                                        self.stimSize,
    #                                        key=lambda e: 0).astype(
    #                                            np.int8)
    def reward(self, entID):
#       print('PCG reward ', self.pcg_reward)
        #FIXME: wtf? go back to joseph's way
        return self.pcg_reward


    def step(self, decisions):
#       print('PCG.step action: {}'.format(decisions))

        actions = {}

        for entID in list(decisions.keys()):
           #entID = 0
            actions[entID] = defaultdict(dict)
            if entID in self.dead:
               continue

           #print('self.raw[entID]', self.raw[entID])
           #print('entID', entID)
           #print('pcg raw: {}'.format(self.raw))
            ents = self.raw[entID][Stimulus.Entity]

            for atn, args in decisions[entID].items():
                for arg, val in args.items():
                    val = int(val)

                    if len(arg.edges) > 0:
                        actions[entID][atn][arg] = arg.edges[val]
                    elif val < len(ents):
                        actions[entID][atn][arg] = ents[val]
                    else:
                        actions[entID][atn][arg] = ents[0]

        obs, rewards, dones, infos = super().step(actions)

       #print(self.n_step, dones)
        if self.n_step == self.max_steps:
            self.done = True
            for k in dones:
                dones[k] = True


        # Remove dead agents

        self.pcg_reward = 0


    #print('sim obs\n{}'.format(self.sim_obs))
    #print('sim state\n{}'.format(self.sim_state))
    # Compute batch of actions
        actions, self.sim_state, _ = self.sim_trainer.compute_actions(
            self.sim_obs, state=self.sim_state, policy_id='policy_sim_0')
       #print('sim actions {}'.format(actions))
        # Compute overlay maps
        # self.overlays.register(self.sim_obs)

        # Step the environment
        self.sim_obs, sim_rewards, self.sim_done, _ = self.sim.step(actions)

        for agentID in self.sim_done:
            if self.sim_done[agentID]:
                del self.sim_obs[agentID]
                self.pcg_reward += 1

        # obs = self.sim_obs.items()[0][1]
        #obs = self.stim((0, 0))
        # obs = self.sim.world.tiles:w
       #obs, reward, dones = self.getStims()


        #obs = {0: {'map': obs}}
        #rewards = {0: self.reward}
        #dones = {0: self.done, '__all__': False}
        self.n_step += 1
        #print('obs', obs)
       #print('pcg rewards', rewards)

        return obs, rewards, dones, {}

#   def getStims(self):
#       '''Gets agent stimuli from the environment

#       Args:
#          packets: A dictionary of Packet objects

#       Returns:
#          The packet dictionary populated with agent data
#       '''
#       #print('getting pcg stim?')
#       self.raw = {}
#       obs, rewards, dones = {}, {}, {'__all__': False}

#       for entID, ent in self.desciples.items():
#           r, c = ent.base.pos
#           tile = self.sim.world.env.tiles[r, c].tex
#           stim = self.sim.world.env.stim(ent.base.pos, 3)
#           #           print('stim', stim)

#           obs[entID], self.raw[entID] = Dynamic.process(
#               self.config, stim, ent)
#           ob = obs[entID]

#           rewards[entID] = self.reward(entID)
#           dones[entID] = self.done

#       for ent in self.dead:
#          #Why do we have to provide an ob for the last timestep?
#          #Currently just copying one over
#          rewards[ent.entID] = self.reward(ent)
#          dones[ent.entID]   = True
#          obs[ent.entID]     = ob

#       return obs, rewards, dones

#   def clientData(self):
#       '''Data packet used by the renderer

#       Returns:
#          packet: A packet of data for the client
#       '''
#       packet = {
#           'environment':
#           self.sim.world.env,
#           'entities':
#           dict((k, v.packet()) for k, v in self.sim.desciples.items()),
#           'overlay':
#           self.sim.overlay
#       }

#       return packet


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
