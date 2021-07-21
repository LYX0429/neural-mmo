import os
from pdb import set_trace as T
import numpy as np

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


class EvoPPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer. Evolution edition.'''
   def __init__(self, env, path, config):
      super().__init__(env=env, config=config)
      self.saveDir = path
      self.pathDir = '/'.join(path.split(os.sep)[:-1])

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      savedir = super().save(self.saveDir)
     #with open('evo_experiment/path.txt', 'w') as f:
      with open(os.path.join(self.pathDir, 'path.txt'), 'w') as f:
         f.write(savedir)
      print('Saved to: {}'.format(savedir))
      return savedir

   def restore(self, model):
      '''Restore model from path'''
      if model is None:
         print('Training from scratch...')
         return
      if model == 'current':
        #with open('experiment/path.txt') as f:
         with open(os.path.join(self.pathDir, 'path.txt')) as f:
            path = f.read().splitlines()[0]
      else:
         raise Exception("Invalid model.")
         path = 'experiment/{}/checkpoint'.format(model)

      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      model = self.get_policy(policyID).model
     #print('sending evo trainer model to gpu\n')
#     model.cuda()
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      stats = super().train()
      if self.training_iteration < 100:
         save_interval = 10
      else:
         save_interval = 100
      if self.training_iteration % save_interval == 0:
         self.save()

      nSteps = stats['info']['num_steps_trained']
      VERBOSE = False
      if VERBOSE:
         print('Epoch: {}, Samples: {}'.format(self.training_iteration, nSteps))
      hist = stats['hist_stats']
      for key, stat in hist.items():
         if len(stat) == 0 or key == 'map_fitness':
            continue

         if VERBOSE:   
            print('{}:: Total: {:.4f}, N: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}'.format(
                  key, np.sum(stat), len(stat), np.mean(stat), np.std(stat), np.min(stat), np.max(stat)))
        #if key == 'map_fitness':
        #    print('DEBUG MAP FITNESS PRINTOUT')
        #    print(hist[key])
         hist[key] = []
   
      return stats

   def reset(self):
     #print('sane reset evoTrainer \n')
     #print(self.workers.local_worker, self.workers.remote_workers)
      super().reset(self.config)
#     raise Exception


class SanePPOTrainer(ppo.PPOTrainer):
   '''Small utility class on top of RLlib's base trainer'''
   def __init__(self, env, path, config):
      super().__init__(env=env, config=config)
      self.saveDir = path

   def save(self):
      '''Save model to file. Note: RLlib does not let us chose save paths'''
      savedir = super().save(self.saveDir)
      with open('{}/path.txt'.format(self.saveDir), 'w') as f:
         f.write(savedir)
      print('Saved to: {}'.format(savedir))
      return savedir

   def restore(self, model):
      '''Restore model from path'''
      if model is None:
         print('Training from scratch...')
         return
      if model == 'current':
         with open('{}/path.txt'.format(self.saveDir)) as f:
            path = f.read().splitlines()[0]
      elif model.startswith('evo_experiment'):
#        path = '/'.join(model.split('/')[1:])
         path = os.path.join(model, 'path.txt')
         with open(path) as f:
            path = f.read().splitlines()[0]
         #FIXME dumb hack
         path = '{}/{}/{}'.format(path.split('/')[0],
               'greene',
               '/'.join(path.split('/')[1:]),
               )
         path = os.path.abspath(path)
      else:
         path = '{}/{}/checkpoint'.format(self.saveDir, model)

      print('Loading from: {}'.format(path))
      super().restore(path)

   def policyID(self, idx):
      return 'policy_{}'.format(idx)

   def model(self, policyID):
      print('rlutils get policy ', policyID)
      return self.get_policy(policyID).model

   def defaultModel(self):
      return self.model(self.policyID(0))

   def train(self):
      '''Train forever, printing per epoch'''
      epoch = 0
      while True:
          stats = super().train()
          self.save()

          nSteps = stats['info']['num_steps_trained']
          print('Epoch: {}, Samples: {}'.format(epoch, nSteps))
          hist = stats['hist_stats']
          for key, stat in hist.items():
             if len(stat) == 0:
                continue

             print('{}:: Total: {:.4f}, N: {:.4f}, Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}'.format(
                   key, np.sum(stat), len(stat), np.mean(stat), np.std(stat), np.min(stat), np.max(stat)))
             hist[key] = []
       
          epoch += 1
