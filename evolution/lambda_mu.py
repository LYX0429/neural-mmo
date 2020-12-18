from pdb import set_trace as T
import os
from shutil import copyfile
import pickle
import numpy as np
from multiprocessing import Pipe, Process
import csv
import time

class LambdaMuEvolver():
   def __init__(
           self,
           save_path,
          #make_env,  # a function that creates the environment
           n_pop=3,
           lam=1 / 3,
           mu=1 / 3,
           n_proc=12,
           n_epochs= 10000
   ):
       self.genes = {}
       self.save_path = save_path
       self.evolver_path = os.path.join(self.save_path, 'evolver')
       self.log_path = os.path.join(self.save_path, 'log.csv')
       self.n_proc = n_proc
      #self.make_env = make_env
       self.lam = lam
       self.mu = mu
       self.n_pop = n_pop
       self.n_epochs = n_epochs
       self.n_sim_ticks = 100
       self.population = {}  # hash: (game, score, age)
       self.max_skills = {}
       self.n_tick = 0
       self.n_gen = 0
       self.score_hists = {}
       self.n_epoch = 0
       self.n_pop = n_pop
       self.g_hashes = list(range(self.n_pop))
       self.n_init_builds = 30
       self.n_mutate_actions = self.n_init_builds
       self.mature_age = 1

   def join_procs(self, processes):

      for g_hash, (game, score_t, age) in self.population.items():
         # get score from latest simulation
         # cull score history

         if len(self.score_hists[g_hash]) >= 5:
            while len(self.score_hists[g_hash]) >= 5:
                 self.score_hists[g_hash].pop(0)
         # hack
         if score_t is None:
            score_t = 0
         else:
            self.score_hists[g_hash].append(score_t)
            score = np.mean(self.score_hists[g_hash])
            game, _, age = self.population[g_hash]
            self.population[g_hash] = (game, score, age + 1)

   def evolve_generation(self):
      print('epoch {}'.format(self.n_epoch))
      population = self.population
      processes = {}
      n_proc = 0

      for g_hash, (game, score, age) in population.items():
         map_arr = self.genes[g_hash]
         parent_conn, child_conn = Pipe()
         p = Process(target=self.simulate_game,
                 args=(
                     game,
                     map_arr,
                     self.n_sim_ticks,
                     child_conn,
                     ))
         p.start()
         processes[g_hash] = p, parent_conn, child_conn
#              # NB: specific to NMMO!!
#              # we simulate a bunch of envs simultaneously through the rllib trainer
#              self.simulate_game(game, map_arr, self.n_sim_ticks, g_hash=g_hash)
#              parent_conn, child_conn = None, None
#              processes[g_hash] = score, parent_conn, child_conn
#              for g_hash, (game, score, age) in population.items():
#                  try:
#                      with open(os.path.join('./evo_experiment', '{}'.format(self.config['config'].EVO_DIR), 'env_{}_skills.json'.format(g_hash))) as f:
#                          agent_skills = json.load(f)
#                          score = self.update_entropy_skills(agent_skills)
#                  except FileNotFoundError:
#                      raise Exception
#                      # hack
#                      score = None
#                      processes[g_hash] = score, parent_conn, child_conn
#                  self.population[g_hash] = (game, score, age)

#              n_proc += 1
#              break

         if n_proc > 1 and n_proc % self.n_proc == 0:
            self.join_procs(processes)

      if len(processes) > 0:
         self.join_procs(processes)

      return self.mutate_gen()

   def log(self):
      pop_list = [(g_hash, game, score, age) for g_hash, (game, score, age) in self.population.items() if score is not None] 
      ranked_pop = sorted(pop_list, key=lambda tpl: tpl[2])
      print('Ranked population: (id, running_mean_score, last_score, age)')

      with open(self.log_path, mode='a') as log_file:
          log_writer = csv.writer(
                  log_file, delimiter=',',
                  quotechar='"',
                  quoting=csv.QUOTE_MINIMAL)
          log_writer.writerow(['epoch {}'.format(self.n_epoch)])
          log_writer.writerow(
                  ['id', 'running_score', 'last_score', 'age'])

          for g_hash, game, score, age in ranked_pop:

              if g_hash in self.score_hists:
                  score_hist = self.score_hists[g_hash]
                  if len(score_hist) > 0:
                      last_score = self.score_hists[g_hash][-1]
                  else:
                      last_score = 0
              else:
                  last_score = -1
              print('{}, {:2f}, {:2f}, {}'.format(
                  g_hash, score, last_score, age))
              log_writer.writerow([g_hash, score, last_score, age])

   def mutate_gen(self):
      population = self.population
      n_cull = int(self.n_pop * self.mu)
      n_parents = int(self.n_pop * self.lam)
      dead_hashes = []
      pop_list = [(g_hash, game, score, age) for g_hash, (game, score, age) in self.population.items() if score is not None] 
      ranked_pop = sorted(pop_list, key=lambda tpl: tpl[2])
      print('Ranked population: (id, running_mean_score, last_score, age)')

      with open(self.log_path, mode='a') as log_file:
          log_writer = csv.writer(
                  log_file, delimiter=',',
                  quotechar='"',
                  quoting=csv.QUOTE_MINIMAL)
          log_writer.writerow(['epoch {}'.format(self.n_epoch)])
          log_writer.writerow(
                  ['id', 'running_score', 'last_score', 'age'])

          for g_hash, game, score, age in ranked_pop:

              if g_hash in self.score_hists:
                  score_hist = self.score_hists[g_hash]
                  if len(score_hist) > 0:
                      last_score = self.score_hists[g_hash][-1]
                  else:
                      last_score = 0
              else:
                  last_score = -1
              print('{}, {:2f}, {:2f}, {}'.format(
                  g_hash, score, last_score, age))
              log_writer.writerow([g_hash, score, last_score, age])

      for j in range(n_cull):
          dead_hash = ranked_pop[j][0]

          if self.population[dead_hash][2] > self.mature_age:
              dead_hashes.append(dead_hash)

      par_hashes = []

      for i in range(n_parents):
          par_hash, _, _, age = ranked_pop[-(i + 1)]

          if age > self.mature_age:
              par_hashes.append(par_hash)

      # for all we have culled, add new mutated individual
      j = 0

      mutated = []
      if par_hashes:
          while dead_hashes:
              n_parent = j % len(par_hashes)
              par_hash = par_hashes[n_parent]
              # parent = population[par_hash]
              par_map = self.genes[par_hash]
              # par_game = parent[0]  # get game from (game, score, age) tuple
              g_hash = dead_hashes.pop()
              mutated.append(g_hash)
              population.pop(g_hash)
             #self.score_var.pop(g_hash)
              child_map = self.mutate(par_hash)
             #child_game = self.make_game(child_map)
              child_game = None
              population[g_hash] = (child_game, None, 0)
              self.genes[g_hash] = child_map
              self.score_hists[g_hash] = []
              j += 1

      self.saveMaps(self.genes, mutated)

      if self.n_epoch % self.mature_age == 0 or self.n_epoch == 2:
          self.save()
      self.n_epoch += 1


   def save(self):
       save_file = open(self.evolver_path, 'wb')
       copyfile(self.evolver_path, self.evolver_path + '.bkp')
       pickle.dump(self, save_file)



   def init_pop(self):

       for i in range(self.n_pop):
           rank= self.g_hashes.pop()

          #if rank in self.genes:
          #    map_arr = self.genes[rank]
          #else:
           map_arr = self.genRandMap(rank)
           game = None
           self.population[rank]= (game, None, 0)
           self.genes[rank]= map_arr
           self.score_hists[rank] = []

       try:
           os.mkdir(os.path.join(self.save_path, 'maps'))
       except FileExistsError:
           print('Overwriting evolved maps at {}.'.format(self.save_path))
       self.saveMaps(self.genes, list(self.genes.keys()))
       print('restoring')
       self.restore()
       print('restoring')

   def evolve(self, n_epochs=None):
   #   self.save()
   #   raise Exception
       if n_epochs:
           self.n_epochs = n_epochs
       if self.n_epoch == 0:
           self.init_pop()


       while self.n_epoch < self.n_epochs:
           start_time= time.time()
           self.evolve_generation()
           time_elapsed= time.time() - start_time
           hours, rem= divmod(time_elapsed, 3600)
           minutes, seconds= divmod(rem, 60)
           print("time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(
               int(hours), int(minutes), seconds))

   def genRandMap(self, g_hash):
       raise NotImplementedError()

   def mutate(self, map_arr):
       raise NotImplementedError()

   def restore():
       ''' Use saved maps to instantiate games. '''
       raise NotImplementedError()

   def infer(self):
       raise NotImplementedError()


