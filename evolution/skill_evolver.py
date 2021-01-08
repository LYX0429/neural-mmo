'''
This is a file in which we mutate the hypothetical [agent X skill] matrix directly, allowing us to
test and compare different diversity metrics.
'''
import argparse
import copy
import os
import random
import pickle
from pdb import set_trace as T

import numpy as np

# NOTE: to prevent "relative import" error, run `python -m evolution.skill_evolver
from .evo_map import (calc_differential_entropy, calc_discrete_entropy, calc_discrete_entropy_2,
                      calc_diversity_l2, calc_convex_hull)
from .lambda_mu import LambdaMuEvolver


class SkillEvolver(LambdaMuEvolver):
   def __init__(self, *args, **kwargs):
      alpha = kwargs.pop('alpha')
      self.alpha = alpha
      super().__init__(*args, **kwargs)
      self.n_skills = 10
      self.n_agents = 16
      self.max_xp = 20000
      self.max_lifespan = 100
      self.MATURE_AGE = 1

   def infer(self):
      for g_hash, (_, score, age) in self.population.items():
          agent_skills = self.genes[g_hash]
          print(np.array(agent_skills))
          print('score: {}, age: {}'.format(score, age))

   def genRandMap(self, g_hash):
     # agent_skills = [
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
     #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.], ]
       agent_skills = np.zeros((self.n_agents, self.n_skills))
       agent_lifespans = np.random.randint(0, self.max_lifespan, (self.n_agents))
#      agent_skills = [[0 for i in range(n_skills)] for j in range(n_agents)]
#      agent_lifespans = [0 for j in range(n_agents)]
       agent_stats = {
             'skills': agent_skills,
             'lifespans': agent_lifespans,
             }

       return agent_stats

   def simulate_game(self,
                       game,
                       agent_stats,
                       n_sim_ticks,
                       child_conn,):
#     score = calc_diversity_l2(agent_stats, self.alpha)
#     score = calc_differential_entropy(agent_stats)
      score = calc_discrete_entropy_2(agent_stats)
#     score = calc_discrete_entropy(agent_stats)
#     score = calc_convex_hull(agent_stats)

      if child_conn:
          child_conn.send(score)

      return score

   def mutate(self, g_hash):
      gene = self.genes[g_hash]
      agent_skills = gene['skills']
      agent_lifespans = gene['lifespans']
      agent_skills = agent_skills.copy()
      agent_lifespans = agent_lifespans.copy()
      n_agents = agent_skills.shape[0]
      n_skills = agent_skills.shape[1]

      for i in range(random.randint(1, 5)):
         for j in range(i):
            a_i = random.randint(0, n_agents - 1)
            s_i = random.randint(0, n_skills - 1)
           #if s_i in [0, 5, 6]:
           #    min_xp = 1000
           #else:
            min_xp = 0
            max_xp = self.max_xp
            agent_skills[a_i][s_i] = \
                  min(max(min_xp, agent_skills[a_i][s_i] + random.randint(-100, 100)), max_xp)
            l_i = random.randint(0, n_agents - 1)
            max_lifespan = 100
            agent_lifespans[l_i] = \
                  min(max(0, agent_lifespans[l_i] + random.randint(-10, 10)), max_lifespan)

      if n_agents > 1 and random.random() < 0.05:
         # remove agent
         i = random.randint(0, n_agents - 1)
         agent_skills = np.concatenate((agent_skills[0:i, :], agent_skills[i+1:, :]), axis=0)
         agent_lifespans = np.concatenate((agent_lifespans[0:i], agent_lifespans[i+1:]), axis=0)
        #agent_skills = np.delete(agent_skills, i, 0)
        #agent_lifespans = np.delete(agent_skills, i, 0)
         n_agents -= 1

      if self.n_agents > n_agents > 0 and random.random() < 0.05:
         # add agent
         i = random.randint(0, n_agents - 1)
         new_agent = np.zeros((1, self.n_skills))
         print('new_agent', new_agent)
         new_lifespan = np.random.randint(0, self.max_xp, (1))
         agent_skills = np.concatenate((agent_skills, new_agent))
         agent_lifespans = np.concatenate((agent_lifespans, new_lifespan))
#        agent_skills = np.append(agent_skills, np.randint(0, self.max_xp, (self.n_skills)))
#        agent_lifespans = np.append(agent_lifespans, np.randint(0, self.max_xp, 1))
      if not agent_skills.shape[0] == agent_lifespans.shape[0]:
         T()

      stats = {
            'skills': agent_skills,
            'lifespans': agent_lifespans,
            }

      return stats

   def make_game(self, agent_skills):
      return None

   def restore(self, **kwargs):
      pass

   def saveMaps(self, genes, mutated):
      pass


   def join_procs(self, processes):

     #for g_hash, (game, score_t, age) in self.population.items():

      for g_hash, (p, parent_conn, _) in processes.items():
         score_t = parent_conn.recv()
         p.join()
         p.close()
         # get score from latest simulation
         # cull score history

         if len(self.score_hists[g_hash]) >= 100:
            while len(self.score_hists[g_hash]) >= 100:
                 self.score_hists[g_hash].pop(0)
         # hack

         if score_t is None:
            score_t = 0
         else:
            self.score_hists[g_hash].append(score_t)
            score = np.mean(self.score_hists[g_hash])
            game, _, age = self.population[g_hash]
            self.population[g_hash] = (game, score, age + 1)
      processes = {}   

      return processes


def some_examples():
   #print('bad situation')
    agent_skills = [
            [0, 0, 0],
            ]
    ent = calc_diversity(agent_skills, 0.5)

   #print('less bad situation')
    agent_skills = [
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)

    agent_skills = [
            [0, 0, 0],
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)

    agent_skills = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)


    print('NMMO: pacifism')
    agent_skills = [
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish and one murder')
    agent_skills = [
    [1200.,  0.,  0. ,  0. ,  0. , 1500.,1500. ,  0. ,  0.],
    [1200. ,  0.,1080.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0., 240.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0., 240.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
   ]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish')
    agent_skills = [
    [1200.,   0., 120. ,  0.,  0., 1500.,1500. ,  0.,   0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,120.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish, smaller population')
    agent_skills = [
    [1200.,   0., 1200. ,  0.,  0.,1500.,1500. ,  0.,   0.],
    [1200. ,  0. ,  0.  , 240., 0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,120.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
#   [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],]
]
    ent = calc_diversity(agent_skills)

if __name__ == '__main__':
#   some_examples()
   parser= argparse.ArgumentParser()
   parser.add_argument('--experiment-name',
                       default='skill_evolver_scratch',
                       help='name of the experiment')
   parser.add_argument('--load',
                       default=False,
                       action='store_true',
                       help='whether or not to load a previous experiment')
   parser.add_argument('--n-pop',
                       type=int,
                       default=12,
                       help='population size')
   parser.add_argument('--lam',
                       type=float,
                       default=1 / 3,
                       help='number of reproducers each epoch')
   parser.add_argument(
                       '--mu',
                       type=float,
                       default=1 / 3,
                       help='number of individuals to cull and replace each epoch')
   parser.add_argument('--inference',
                       default=False,
                       action='store_true',
                       help='watch simulations on evolved maps')
   parser.add_argument('--n-epochs',
                       default=10000,
                       type=int,
                       help='how many generations to evolve')
   parser.add_argument('--alpha',
                       default=0.66,
                       type=float,
                       help='balance between skill and agent entropies')
   args = parser.parse_args()
   n_epochs = args.n_epochs
   experiment_name = args.experiment_name
   load = args.load


   save_path = 'evo_experiment/{}'.format(args.experiment_name)

   if not os.path.isdir(save_path):
      os.mkdir(save_path)
   try:
      evolver_path = os.path.join(save_path, 'evolver')
      with open(evolver_path, 'rb') as save_file:
        evolver = pickle.load(save_file)
        print('loading evolver from save file')
      evolver.restore(trash_data=True)
   except FileNotFoundError as e:
      print(e)
      print('no save file to load')

      evolver = SkillEvolver(save_path,
          n_pop=12,
          lam=1 / 3,
          mu=1 / 3,
          n_proc=12,
          n_epochs=n_epochs,
          alpha=args.alpha,
      )

   if args.inference:
       evolver.infer()
   else:
       evolver.evolve(n_epochs=n_epochs)
