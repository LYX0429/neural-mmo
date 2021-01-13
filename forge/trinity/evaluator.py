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

from matplotlib import pyplot as plt



class Base:
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config):
      self.config  = config
      self.done    = {}

   def render(self):
      '''Rendering launches a Twisted WebSocket server with a fixed
      tick rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.'''
      from forge.trinity.twistedserver import Application
      Application(self.env, self.tick)

   def test(self):
      if self.config.MAP != 'PCG':
         self.config.ROOT = self.config.MAP
         self.config.ROOT = os.path.join(os.getcwd(), 'evo_experiment', self.config.MAP, 'maps', 'map')

      ts = np.arange(self.config.EVALUATION_HORIZON)
      n_evals = 20
      div_mat = np.zeros((n_evals, self.config.EVALUATION_HORIZON))

      for i in range(n_evals):
         self.env.reset(idx=self.config.INFER_IDX)
         self.obs = self.env.step({})[0]
         self.state = {}
         self.registry = OverlayRegistry(self.env, self.model, self.trainer, self.config)
         divs = np.zeros((self.config.EVALUATION_HORIZON))
         for t in tqdm(range(self.config.EVALUATION_HORIZON)):
            self.tick(None, None)
#           print(len(self.env.realm.players.entities))
            div_stats = self.env.get_agent_stats()
            calc_diversity = diversity_calc(self.config)
            diversity = calc_diversity(div_stats, verbose=False)
            divs[t] = diversity
         div_mat[i] = divs
      plot_diversity(ts, div_mat, self.config.MODEL.split('/')[-1], self.config.MAP.split('/')[-1], self.config.INFER_IDX)
      print('Diversity: {}'.format(diversity))

      log = InkWell()
      log.update(self.env.terminal())

      fpath = os.path.join(self.config.LOG_DIR, self.config.LOG_FILE)
      np.save(fpath, log.packet)

   def tick(self, actions, preprocessActions=True):
      '''Compute actions and overlays for a single timestep
      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      #Step the environment
      self.obs, rewards, self.done, _ = self.env.step(
            actions, omitDead=True, preprocessActions=preprocessActions)

class Evaluator(Base):
   def __init__(self, config, policy):
      super().__init__(config)
      self.policy   = policy

      self.env      = Env(config)
      self.obs      = self.env.reset()
      self.registry = OverlayRegistry(self.env, None, None, config)

   def tick(self, pos, cmd):
      realm, actions    = self.env.realm, {}
      for agentID in self.obs:
         agent              = realm.players[agentID]
         agent.skills.style = Action.Range
         actions[agentID]   = self.policy(realm, agent)

         self.registry.step(self.obs, pos, cmd, update=
               'counts wilderness'.split())

      super().tick(actions, preprocessActions=False)
 

