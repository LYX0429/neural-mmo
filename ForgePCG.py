'''Main file for the neural-mmo/projekt demo

/projeckt will give you a basic sense of the training
loop, infrastructure, and IO modules for handling input 
and output spaces. From there, you can either use the 
prebuilt IO networks in PyTorch to start training your 
own models immediately or hack on the environment'''

#My favorite debugging macro
from pdb import set_trace as T

from fire import Fire
import sys

import numpy as np
import torch

import ray
from ray import rllib

from forge.ethyr.torch import utils

import projekt
from projekt import realm
from pcgrl import rlutils

import pcg
from pcg import PCG

import pcgrl
from pcgrl.game.io.action import static



#Instantiate a new environment
def createPCGEnv(pcg_config):
  #return projekt.Realm(config)
   return PCG(pcg_config)

def createEnv(config):
    return projekt.Realm(config)

#Map agentID to policyID -- requires config global
def mapPCGPolicy(agentID):
   return 'policy_pcg_{}'.format(agentID % pcg_config.NPOLICIES)

#Generate RLlib policies
def createPolicies(config):
   obs      = projekt.realm.observationSpace(config)
   atns     = projekt.realm.actionSpace(config)
   policies = {}

   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
      key           = mapPolicy(i)
      policies[key] = (None, obs, atns, params)

   return policies

def createPCGPolicies(pcg_config):
   obs      = static.observationSpace(pcg_config)
   atns     = static.actionSpace(pcg_config)
   policies = {}

   for i in range(pcg_config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
      key           = mapPCGPolicy(i)
      policies[key] = (None, obs, atns, params)

   return policies

if __name__ == '__main__':
   #Setup ray
   torch.set_num_threads(1)
   ray.init()
   
   #Built config with CLI overrides
   pcg_config = pcgrl.Config()
   pcg_config.STIM = 4
   if len(sys.argv) > 1:
      sys.argv.insert(1, 'override')
      Fire(pcg_config)

   #RLlib registry
   rllib.models.ModelCatalog.register_custom_model(
         'test_pcg_model', pcgrl.PCGPolicy)
   ray.tune.registry.register_env("custom_pcg", createPCGEnv)

   #Create policies
   pcg_policies  = createPCGPolicies(pcg_config)

   print('top level config', vars(pcg_config))

   #Instantiate monolithic RLlib Trainer object.
   trainer = rlutils.SanePPOPCGTrainer(
         env="custom_pcg", path='experiment_pcg', config={
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
         'config': pcg_config
      },
      'multiagent': {
         "policies": pcg_policies,
         "policy_mapping_fn": mapPCGPolicy
      },
      'model': {
         'custom_model': 'test_pcg_model',
         'custom_options': {'config': pcg_config}
      },
   })
#  trainer.defaultModel().cuda()

   #Print model size
   utils.modelSize(trainer.model('policy_pcg_0'))
   trainer.restore(pcg_config.MODEL)

   if pcg_config.RENDER:
      env = createPCGEnv({'config': pcg_config})
      pcgrl.Evaluator(trainer, env, pcg_config).run()
   else:
      trainer.train()
