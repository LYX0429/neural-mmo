from pdb import set_trace as T
import numpy as np

import os

from collections import defaultdict
from tqdm import tqdm

import projekt

from forge.trinity import Env
from forge.trinity.overlay import OverlayRegistry

from forge.blade.io.action import static as Action
from forge.blade.lib.log import InkWell
from evolution.diversity import diversity_calc
import plot_diversity

from matplotlib import pyplot as plt



class Base:
   '''Base class for test-time evaluators'''
   def __init__(self, config):
      self.config  = config
      self.done    = {}

   def render(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      if self.config.GRIDDLY:
         while True:
            self.tick(None, None)
      from forge.trinity.twistedserver import Application
      Application(self.env, self.tick)

#<<<<<<< HEAD
#  def test(self):
   def evaluate(self, generalize=True):
      '''Evaluate the model on maps according to config params'''
      if self.config.MAP != 'PCG':
         self.config.ROOT = self.config.MAP
         self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.MAP, 'maps', 'map')

      ts = np.arange(self.config.EVALUATION_HORIZON)
      n_evals = 2
      div_mat = np.zeros((n_evals, self.config.EVALUATION_HORIZON))
      calc_diversity = diversity_calc(self.config)

      for i in range(n_evals):
         self.env.reset(idx=self.config.INFER_IDX)
         self.obs = self.env.step({})[0]
         self.state = {}
         self.registry = OverlayRegistry(self.config, self.env)
         divs = np.zeros((self.config.EVALUATION_HORIZON))
         for t in tqdm(range(self.config.EVALUATION_HORIZON)):
            self.tick(None, None)
#           print(len(self.env.realm.players.entities))
            div_stats = self.env.get_all_agent_stats()
            diversity = calc_diversity(div_stats, verbose=False)
            divs[t] = diversity
         div_mat[i] = divs

      plot_div_2d(ts, div_mat, self.config.MODEL.split('/')[-1], self.config.MAP.split('/')[-1], self.config.INFER_IDX)

      print('Diversity: {}'.format(diversity))
#=======
#   def evaluate(self, generalize=True):
#      '''Evaluate the model on maps according to config params'''
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

      config = self.config
      log    = InkWell()

      if generalize:
         maps = range(-1, -config.EVAL_MAPS-1, -1)
      else:
         maps = range(1, config.EVAL_MAPS+1)

      print('Number of evaluation maps: {}'.format(len(maps)))
      for idx in maps:
         self.obs = self.env.reset(idx)
         for t in tqdm(range(config.EVALUATION_HORIZON)):
            self.tick(None, None)

         log.update(self.env.terminal())

      #Save data
      path = config.PATH_EVALUATION.format(config.NAME, config.MODEL)
      np.save(path, log.packet)

   def tick(self, obs, actions, pos, cmd, preprocessActions=True):
      '''Simulate a single timestep

      Args:
          obs: dict of agent observations
          actions: dict of policy actions
          pos: Camera position (r, c) from the server
          cmd: Console command from the server
          preprocessActions: Required for actions provided as indices
      '''
      self.obs, rewards, self.done, _ = self.env.step(
            actions, omitDead=True, preprocessActions=preprocessActions)
      if self.config.RENDER:
         self.registry.step(obs, pos, cmd)


class Evaluator(Base):
   '''Evaluator for scripted models'''
   def __init__(self, config, policy, *args):
      super().__init__(config)
      self.policy   = policy
      self.args     = args

      self.env      = Env(config)

   def render(self):
      '''Render override for scripted models'''
      self.obs      = self.env.reset()
      self.registry = OverlayRegistry(self.config, self.env).init()
      super().render()

   def tick(self, pos, cmd):
      '''Simulate a single timestep

      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      realm, actions    = self.env.realm, {}
      for agentID in self.obs:
         agent              = realm.players[agentID]
         agent.skills.style = Action.Range
         actions[agentID]   = self.policy(realm, agent, *self.args)

      super().tick(self.obs, actions, pos, cmd, preprocessActions=False)
