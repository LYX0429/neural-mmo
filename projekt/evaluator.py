import os
import time
from collections import defaultdict
from pdb import set_trace as T

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import projekt
from evolution.diversity import diversity_calc
from forge.blade.io.action import static as Action
from forge.blade.lib.log import InkWell
from forge.blade.systems import ai
from projekt.overlay import OverlayRegistry


def plot_diversity(x, y, model_name, map_name, render=True):
    my_dpi = 96
    colors = ['darkgreen', 'm', 'g', 'y', 'salmon', 'darkmagenta', 'orchid', 'darkolivegreen', 'mediumaquamarine',
            'mediumturquoise', 'cadetblue', 'slategrey', 'darkblue', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet',
            'fuchsia', 'deeppink', 'olive', 'orange', 'maroon', 'lightcoral', 'firebrick', 'black', 'dimgrey', 'tomato',
            'saddlebrown', 'greenyellow', 'limegreen', 'turquoise', 'midnightblue', 'darkkhaki', 'darkseagreen', 'teal',
            'cyan', 'lightsalmon', 'springgreen', 'mediumblue', 'dodgerblue', 'mediumpurple', 'darkslategray', 'goldenrod',
            'indigo', 'steelblue', 'coral', 'mistyrose', 'indianred']
    fig, ax = plt.subplots(figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
    ax.plot(y, c='indigo')
    plt.tight_layout()
    #markers, caps, bars = ax.errorbar(x, avg_scores, yerr=std,
    #                                   ecolor='purple')
    #[bar.set_alpha(0.03) for bar in bars]
    plt.ylabel('diversity')
    plt.xlabel('tick')
    plt.savefig('experiment/diversity_MODEL_{}_MAP_{}.png'.format(model_name, map_name),
          dpi=my_dpi)

    if render:
       plt.show()
    plt.close()

class Log:
   def __init__(self):
      self.data = []

   def update(self, infos):
      #Log performance

      for entID, e in infos.items():
         self.data.append(e)

class Evaluator:
   '''Test-time evaluation with communication to
   the Unity3D client. Makes use of batched GPU inference'''
   def __init__(self, config, trainer=None, policy=None):
      assert (trainer is None) ^ (policy is None)
      self.trainer = trainer
      self.policy  = policy
      self.config  = config
      self.done    = {}
      self.infos   = {}

      self.log = InkWell()

      if trainer:
         self.model    = self.trainer.get_policy('policy_0').model
         self.env      = projekt.RLLibEnv({'config': config})

         # So evo_map can pick map to load
#        self.env.reset(idx=0, step=False)
         self.env.reset(idx=None, step=False)
         self.registry = OverlayRegistry(self.env, self.model, trainer, config)
         self.obs      = self.env.step({})[0]

         self.state    = {}
      else:
         self.env      = projekt.Env(config)
         self.obs      = self.env.reset()
         self.registry = OverlayRegistry(self.env, None, None, config)

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
      divs = np.zeros((self.config.EVALUATION_HORIZON))

      for t in tqdm(range(self.config.EVALUATION_HORIZON)):
         self.tick(None, None)
         div_stats = self.env.get_agent_stats()
         calc_diversity = diversity_calc(self.config)
         diversity = calc_diversity(div_stats, verbose=False)
         divs[t] = diversity
      plot_diversity(ts, divs, self.config.MODEL.split('/')[-1], self.config.MAP.split('/')[-1])
      print('Diversity: {}'.format(diversity))
      self.log.update([self.env.terminal()])
      data = self.log.packet

      fpath = os.path.join(self.config.LOG_DIR, self.config.LOG_FILE)
      np.save(fpath, data)

   def tick(self, pos, cmd):
      '''Compute actions and overlays for a single timestep
      Args:
          pos: Camera position (r, c) from the server)
          cmd: Console command from the server
      '''
      #Remove dead agents

      for agentID in self.done:
         if self.done[agentID] and agentID != '__all__':
            del self.obs[agentID]

      #Compute batch of actions

      if self.trainer:
         actions, self.state, _ = self.trainer.compute_actions(
               self.obs, state=self.state, policy_id='policy_0')
         self.registry.step(self.obs, pos, cmd,
               update='counts values attention wilderness'.split())
      else:
         realm, actions = self.env.realm, {}

         for agentID in self.obs:
            agent              = realm.players[agentID]
            agent.skills.style = Action.Range
            actions[agentID]   = self.policy(realm, agent)

         self.registry.step(self.obs, pos, cmd, update=
               'counts wilderness'.split())

      #Step the environment
      self.obs, rewards, self.done, self.infos = self.env.step(actions)
      #FIXME: this breaks evolution inference
     #self.log.update([self.infos])
