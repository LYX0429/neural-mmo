import argparse
from pdb import set_trace as T
import copy
import csv
import os
import pickle
import random
import sys
import time
import json
from multiprocessing import Pipe, Process
from shutil import copyfile

import numpy as np

import projekt
from forge.ethyr.torch import utils
from pcg import TILE_TYPES
from projekt import rlutils
#from projekt.overlay import Overlays
from scripts.new_terrain import grid, material, index, textures

from typing import Dict
from ray import rllib
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

np.set_printoptions(threshold=sys.maxsize,
                    linewidth=120,
                    suppress=True,
                   #precision=10
                    )
import torch

MULTIPROCESSING = False


def calc_diversity_l2(agent_skills):
    score = 0
    agent_skills = np.array(agent_skills)
    for a in agent_skills:
        for b in agent_skills:
            score += np.linalg.norm(a-b)
    print('agent skills:\n{}'.format(np.array(agent_skills)))
    print('score:\n{}\n'.format(
        score))

    return score


class LogCallbacks(DefaultCallbacks):
   STEP_KEYS    = 'rllib_compat env_step realm_step env_stim stim_process'.split()
   EPISODE_KEYS = ['env_reset']
   
   def init(self, episode):
      for key in LogCallbacks.STEP_KEYS + LogCallbacks.EPISODE_KEYS: 
         episode.hist_data[key] = []

   def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy],
         episode: MultiAgentEpisode, **kwargs):
      self.init(episode)

   def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
         episode: MultiAgentEpisode, **kwargs):

      env = base_env.envs[0]
      for key in LogCallbacks.STEP_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

   def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
      env = base_env.envs[0]
      for key in LogCallbacks.EPISODE_KEYS:
         if not hasattr(env, key):
            continue
         episode.hist_data[key].append(getattr(env, key))

sz = 16 + 16
scale = int(sz / 5)
root = 'resource/maps/'
tex = textures()
def saveMaps(maps, mutated):

   #for i, map_arr in maps.items():
    for i in mutated:
       map_arr = maps[i]
       print('Generating map ' + str(i))
       path = root + 'evolved/map' + str(i) + '/'
       try:
          os.mkdir(path)
       except:
          pass
       seed = 0
       terrain = grid(map_arr.shape[0], map_arr.shape[0], scale=scale, seed=seed)
   #   terrain = copy.deepcopy(terrain).astype(object)
   #   for y in range(map_arr.shape[1]):
   #       for x in range(map_arr.shape[0]):
   #           tile_type = TILE_TYPES[map_arr[x, y]]
   #          #print(tile_type)
   #           texture = tex[tile_type]
   #          #print(texture)
   #           terrain[y, x] = texture
   #   #fractal(terrain, path+'fractal.png')
   #   #render(tiles, path+'map.png')
       tiles = material(terrain, tex, sz, sz)
       index(terrain, path)

# Map agentID to policyID -- requires config global
def mapPolicy(agentID, 
        #config
        ):
    return 'policy_{}'.format(agentID % 1)


# Generate RLlib policies


def createPolicies(config):
    obs = projekt.env.observationSpace(config)
    atns = projekt.env.actionSpace(config)
    policies = {}

    for i in range(config.NPOLICIES):
        params = {"agent_id": i, "obs_space_dict": obs, "act_space_dict": atns}
        key = mapPolicy(i
                #, config
                )
        policies[key] = (None, obs, atns, params)

    return policies


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
        self.n_init_builds = 100
        self.n_mutate_actions = self.n_init_builds
        self.mature_age = 1

    def join_procs(self, processes):

        for g_hash, (game, score_t, age) in self.population.items():
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

    def evolve_generation(self):
        print('epoch {}'.format(self.n_epoch))
        population = self.population
        n_cull = int(self.n_pop * self.mu)
        n_parents = int(self.n_pop * self.lam)
        dead_hashes = []
        processes = {}
        n_proc = 0

        for g_hash, (game, score, age) in population.items():
            map_arr = self.genes[g_hash]
            if MULTIPROCESSING:
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
            else:
                # NB: specific to NMMO!!
                # we simulate a bunch of envs simultaneously through the rllib trainer
                self.simulate_game(game, map_arr, self.n_sim_ticks, g_hash=g_hash)
                parent_conn, child_conn = None, None
                processes[g_hash] = score, parent_conn, child_conn
                for g_hash, (game, score, age) in population.items():
                    try:
                        with open(os.path.join('./evo_experiment', '{}_alpha{}'.format(self.config['config'].EVO_DIR,
                        self.config['config'].DIVERSITY_ALPHA), 'env_{}_skills.json'.format(g_hash))) as f:
                            agent_skills = json.load(f)
                            score = self.update_entropy_skills(agent_skills)
                    except FileNotFoundError:
                        # hack
                        score = None
                        processes[g_hash] = score, parent_conn, child_conn
                    self.population[g_hash] = (game, score, age)

                n_proc += 1
                break

        self.join_procs(processes)
        if n_proc % self.n_proc == 0:
            self.join_procs(processes)

        if len(processes) > 0:
            self.join_procs(processes)

        print(self.population)
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
                child_map = self.mutate(par_map)
               #child_game = self.make_game(child_map)
                child_game = None
                population[g_hash] = (child_game, None, 0)
                self.genes[g_hash] = child_map
                self.score_hists[g_hash] = []
                j += 1

        saveMaps(self.genes, mutated)

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
            map_arr = self.genRandMap()
            game = None
            self.population[rank]= (game, None, 0)
            self.genes[rank]= map_arr
            self.score_hists[rank] = []
        saveMaps(self.genes, list(self.genes.keys()))
        self.restore()

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

    def genRandMap(self):
        raise NotImplementedError()

    def mutate(self, map_arr):
        raise NotImplementedError()

    def restore():
        ''' Use saved maps to instantiate games. '''
        raise NotImplementedError()

    def infer(self):
        raise NotImplementedError()


class SkillEvolver(LambdaMuEvolver):
    def __init__(self, *args, **kwargs):
        alpha = kwargs.pop('alpha')
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def infer(self):
        for g_hash, (_, score, age) in self.population.items():
            agent_skills = self.genes[g_hash]
            print(np.array(agent_skills))
            print('score: {}, age: {}'.format(score, age))

    def genRandMap(self):
      # agent_skills = [
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.], ]
        agent_skills = [[0 for i in range(3)] for j in range(8)]

        return agent_skills

    def simulate_game(self,
                        game,
                        agent_skills,
                        n_sim_ticks,
                        child_conn,):
        score = calc_diversity_l2(agent_skills, self.alpha)

        if child_conn:
            child_conn.send(score)

        return score

    def mutate(self, gene):
        agent_skills = copy.deepcopy(gene)
        n_agents = len(agent_skills)
        for i in range(random.randint(1, 5)):
            for j in range(i):
                a_i = random.randint(0, n_agents - 1)
                s_i = random.randint(0, len(agent_skills[0]) - 1)
               #if s_i in [0, 5, 6]:
               #    min_xp = 1000
               #else:
                min_xp = 0
                agent_skills[a_i][s_i] = \
                        min(max(min_xp, agent_skills[a_i][s_i] + random.randint(-100, 100)), 20000)

        if n_agents > 1 and random.random() < 0.05:
            # remove agent
            agent_skills.pop(random.randint(0, n_agents - 1))
            n_agents -= 1
        if 8 > n_agents > 0 and random.random() < 0.05:
            # add agent
            agent_skills.append(copy.deepcopy(agent_skills[random.randint(0, n_agents - 1)]))


        return agent_skills

    def make_game(self, agent_skills):
        return None

    def restore(self, **kwargs):
        pass



class EvolverNMMO(LambdaMuEvolver):
    def __init__(self, save_path, make_env, trainer, config, n_proc=12, n_pop=12,):
#       save_path = os.path.abspath(os.path.join('evo_experiment', 'evo_test'))
        super().__init__(save_path, n_proc=n_proc, n_pop=n_pop)
#       torch._C._cuda_init()
        # balance between skill and agent entropies
#       self.alpha = config.DIVERSITY_ALPHA
        self.make_env = make_env
        self.trainer = trainer
        self.map_width = config.TERRAIN_SIZE + 2 * config.TERRAIN_BORDER
        self.map_height = config.TERRAIN_SIZE + 2 * config.TERRAIN_BORDER
        self.n_tiles = len(TILE_TYPES)
        # how much has each individual's score varied over the past n simulations?
        # can tell us how informative further simulation will be
        # assumes task is stationary
       #self.score_var = {}
        # age at which individuals can die and reproduce
       #self.gens_alive = {}
       #self.last_scores = {}
        self.mature_age = 3

        self.state = {}
        self.done = {}
        # self.config   = config
        # config.RENDER = True
        map_arr = self.genRandMap()
        config = {'config': config, #'map_arr': map_arr
                }
        self.config = config
#       env = make_env(config)
#       self.obs = env.reset(map_arr=map_arr, idx=0)
#       self.population[0] = (env, None, 0)
#       self.genes[0] = map_arr
#       self.score_hists[0] = []
       #self.score_var[0] = None, []
       #self.gens_alive[0] = 0
       #self.last_scores[0] = 0

        self.skill_idxs = {}
        self.idx_skills = {}

    def evolve_generation(self):
        print('epoch {}'.format(self.n_epoch))
        population = self.population
        n_cull = int(self.n_pop * self.mu)
        n_parents = int(self.n_pop * self.lam)
        dead_hashes = []
        processes = {}
        n_proc = 0

        for g_hash, (game, score, age) in population.items():
            map_arr = self.genes[g_hash]
            if MULTIPROCESSING:
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
            else:
                # NB: specific to NMMO!!
                p = self.simulate_game(game, map_arr, self.n_sim_ticks, g_hash=g_hash)
                parent_conn, child_conn = None, None
                processes[g_hash] = p, parent_conn, child_conn
                for g_hash in self.population.keys():
                    with open(os.path.join('./evo_experiment', '{}'.format(self.config.EVO_DIR), 'env_{}_skills.json'.format(g_hash))) as f:
                        agent_skills = json.load(f)
                        p = self.update_entropy_skills(agent_skills, self.alpha)
                        raise Exception
                break

            n_proc += 1

        if n_proc % self.n_proc == 0:
            self.join_procs(processes)

        if len(processes) > 0:
            self.join_procs(processes)

        ranked_pop = sorted(
                [(g_hash, game, score, age)
                    for g_hash, (game, score, age) in self.population.items()],
                key=lambda tpl: tpl[2])
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
                    last_score = self.score_hists[g_hash][-1]
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

        if par_hashes:
            while dead_hashes:
                n_parent = j % len(par_hashes)
                par_hash = par_hashes[n_parent]
                # parent = population[par_hash]
                par_map = self.genes[par_hash]
                # par_game = parent[0]  # get game from (game, score, age) tuple
                g_hash = dead_hashes.pop()
                population.pop(g_hash)
               #self.score_var.pop(g_hash)
                child_map = self.mutate(par_map)
               #child_game = self.make_game(child_map)
                child_game = None
                population[g_hash] = (child_game, None, 0)
                self.genes[g_hash] = child_map
                self.score_hists[g_hash] = []
                j += 1

        if self.n_epoch % 10 == 0 or self.n_epoch == 2:
            self.save()
        self.n_epoch += 1


#       self.overlays = Overlays(env, self.model, trainer, config['config'])

    def make_game(self, child_map):
        config = self.config
       #config['map_arr'] = child_map
        game = self.make_env(config)

        return game


    def restore(self, trash_data=False):
        '''
        trash_data: to avoid undetermined weirdness when reloading
        '''

        for g_hash, (game, score, age) in self.population.items():
            if game is None:
               #self.config['map_arr'] = self.genes[g_hash]
                game = self.make_env(self.config)

                if trash_data:
                    pass
                   #score = None
                   #age = 0
                   #self.score_var[g_hash] = None, []
                   #self.score_hists[g_hash] = []
                self.population[g_hash] = (game, score, age)

        if self.trainer is None:

            # Create policies
            policies = createPolicies(self.config['config'])

            # Instantiate monolithic RLlib Trainer object.
            trainer = rlutils.EvoPPOTrainer(env="custom",
                                             path='experiment',
                                             config={
                                                 'num_workers': 1, # normally: 4
                                                 'num_gpus': 1,
                                                 'num_envs_per_worker': 1,
                                                 'train_batch_size': 100, # normally: 4000
                                                 'rollout_fragment_length':
                                                 100,
                                                 'sgd_minibatch_size': 100,  # normally: 128
                                                 'num_sgd_iter': 1,
                                                 'framework': 'torch',
                                                 'horizon': np.inf,
                                                 'soft_horizon': False,
                                                 'no_done_at_end': False,
                                                 'callbacks': LogCallbacks,
                                                 'env_config': {
                                                     'config':
                                                     self.config['config'],
                                                    #'map_arr': self.genRandMap(),
                                                    #'maps': self.genes,
                                                 },
                                                 'multiagent': {
                                                     "policies": policies,
                                                     "policy_mapping_fn":
                                                     mapPolicy
                                                 },
                                                 'model': {
                                                     'custom_model': 'test_model',
                                                     'custom_model_config': {
                                                         'config':
                                                         self.config['config']
                                                     }
                                                 },
                                             })

            # Print model size
            utils.modelSize(trainer.defaultModel())
            trainer.restore(self.config['config'].MODEL)
            self.trainer = trainer

    def infer(self):
        ''' Do inference, just on the top individual for now.'''

        best_score = None, -999

        g_hash = None

        for g_hash, (game, score, age) in self.population.items():
            if score and score > best_score[1] and age > self.mature_age:
                best_score = g_hash, score

        if not g_hash:
            raise Exception('No population found for inference.')
        game = self.population[g_hash][0]
        map_arr = self.genes[g_hash]
       #map_arr = self.genRandMap()
        self.run(game, map_arr)
        map_arr = self.genes[g_hash]


    def genRandMap(self):
        if self.n_epoch > 0:
            print('generating new random map when I probably should not be... \n\n')
        map_arr= np.random.randint(0, self.n_tiles,
                                    (self.map_width, self.map_height))
       #map_arr.fill(TILE_TYPES.index('forest'))
       #map_arr[20:-20, 20:-20]= TILE_TYPES.index('water')
        self.add_border(map_arr)

        return map_arr

    def add_border(self, map_arr):
        b= 9
        map_arr[0:b, :]= TILE_TYPES.index('lava')
        map_arr[:, 0:b]= TILE_TYPES.index('lava')
        map_arr[-b:, :]= TILE_TYPES.index('lava')
        map_arr[:, -b:]= TILE_TYPES.index('lava')

        return map_arr

    def mutate(self, map_arr):
        map_arr= map_arr.copy()

        for i in range(random.randint(0, self.n_mutate_actions)):
            x= random.randint(0, self.map_width - 1)
            y= random.randint(0, self.map_height - 1)
            t= random.randint(0, self.n_tiles - 1)
            map_arr[x, y]= t
        map_arr = self.add_border(map_arr)

        return map_arr

    def update_max_skills(self, ent):
        skills = ent.skills.packet()

        for s, v in skills.items():
            if s == 'level':
                continue
            exp= v['exp']

            if s not in self.max_skills:
                self.max_skills[s]= exp
            else:
                if exp > self.max_skills[s]:
                    self.max_skills[s]= exp

    def update_specialized_skills(self, ent):
        skills = ent.skills.packet()

        n_skills = 0
        max_skill = None, -999
        total_xp = 0

        for s, v in skills.items():
            if s == 'level':
                continue
            n_skills += 1
            xp= v['exp']

            if xp > max_skill[1]:
                max_skill = s, xp
            total_xp += xp

        max_xp = max_skill[1]
        total_xp -= max_xp
        # subtract average in other skills from top skill score
        # max_xp is always nonnegative
        max_xp -= (total_xp / (n_skills - 1))
        skill = max_skill[0]

        if skill not in self.max_skills:
            self.max_skills[skill] = max_xp
        else:
            if xp > self.max_skills[skill]:
                self.max_skills[skill] = max_xp

    def update_entropy_skills(self, skill_dict):
        agent_skills = None
        i = 0

        for agent_id, skills in skill_dict.items():

            skills = skills['skills']
            if agent_skills is None:
                agent_skills = [[None for j in range(len(skills)-4)] for i in range(len(skill_dict))]
            j = 0

            for s, v in skills.items():
                if s in ['level', 'constitution', 'hunting', 'fishing']:
                    continue

                if s not in self.skill_idxs:
                    skill_idx = j
                    self.skill_idxs[s] = skill_idx
                    self.idx_skills[skill_idx] = s
                else:
                    skill_idx = self.skill_idxs[s]
                exp_val = v['exp']
                if self.idx_skills[j] in ['woodcutting', 'mining']:
                    exp_val -= 1154
                agent_skills[i][j] = exp_val
                j += 1
            i += 1

        return calc_diversity_l2(agent_skills, self.alpha, self.skill_idxs)

    def simulate_game(self, game, map_arr, n_ticks, conn=None, g_hash=None):
        #       print('running simulation on this map: {}'.format(map_arr))
#       self.obs= game.reset(#map_arr=map_arr, 
#               idx=g_hash)
#       self.done= {}
#       #       self.overlays = Overlays(game, self.model, self.trainer, self.config['config'])

        score = 0

       #for i in range(n_ticks):
       #    print('tick', i)
        score += self.tick_game(game, g_hash=g_hash)
           #self.tick_game(game)

        # reward for diversity of skills developed during simulation
        # consider max skills of living agents

       #score += self.update_entropy_skills(game.desciples.values())
       #for ent in game.desciples.values():
       #   #self.update_max_skills(ent)
       #    self.update_specialized_skills(ent)
       #score += sum(list(self.max_skills.values()))

        if conn:
            conn.send(score)

        return score

    def run(self, game, map_arr):
        self.obs= game.reset(#map_arr=map_arr, 
                idx=0)
        self.game= game
        from forge.trinity.twistedserver import Application
        Application(game, self.tick).run()

    def evolve_generation(self):
        super().evolve_generation()

    def tick_game(self, game, g_hash=None):
        return self.tick(game=game, g_hash=g_hash)

    def tick(self, game=None, g_hash=None):
        # check if we are doing inference
        if game is None:
            game = self.game
            self.update_entropy_skills(game.desciples.values())

       #if self.n_tick > 15:
       #    self.obs = game.reset()
       #    self.done = {}
       #    self.n_tick = 0
        reward= 0
        # Remove dead agents

        for agentID in self.done:
            if self.done[agentID]:
                assert agentID in self.obs
                del self.obs[agentID]

               #if agentID in game.desciples:
               #    print('update max score during tick')
               #    ent= game.desciples[agentID]
               #    self.update_max_skills(ent)
                reward -= 1
        # Compute batch of actions
        FROZEN = False
        if FROZEN:
            actions, self.state, _= self.trainer.compute_actions(
                self.obs, state=self.state, policy_id='policy_0')

            # Compute overlay maps
            #           self.overlays.register(self.obs)

            # Step the environment
            self.obs, rewards, self.done, _= game.step(actions)
        else:
           #self.trainer.reset()
#           print(dir(self.trainer))
           #self.trainer.train(self.genes[g_hash])
            stats = self.trainer.train()
            print('evo map trainer stats', stats)

           #self.trainer.train()

        self.n_tick += 1

        return reward

    def save(self):
        save_file= open(self.evolver_path, 'wb')
        for g_hash in self.population:
            game, score, age= self.population[g_hash]
            # FIXME: omething weird is happening after reload. Not the maps though.
            # so for now, trash score history and re-calculate after reload
            self.population[g_hash]= None, score, age
           #self.population[g_hash]= None, score, age
        self.trainer= None
        self.game= None
        # map_arr = self.genes[g_hash]
        copyfile(self.evolver_path, self.evolver_path + '.bkp')
        pickle.dump(self, save_file)
        self.restore()


    # main()
# evolve(experiment_name, load, args)


def calc_diversity(agent_skills, alpha, skill_idxs=None):
    BASE_VAL = 0.0001
    # split between skill and agent entropy
    n_skills = len(agent_skills[0])
    n_pop = len(agent_skills)
    agent_sums = [sum(skills) for skills in agent_skills]
    i = 0

    for a in agent_sums:
        if a == 0:
            agent_sums[i] = BASE_VAL * n_skills
        i += 1
    skill_sums = [0 for i in range(n_skills)]

    for i in range(n_skills):

        for a_skills in agent_skills:
            skill_sums[i] += a_skills[i]

        if skill_sums[i] == 0:
            skill_sums[i] = BASE_VAL * n_pop

    skill_ents = []

    for i in range(n_skills):
        skill_ent = 0

        for j in range(n_pop):

            a_skill = agent_skills[j][i]

            if a_skill == 0:
                a_skill = BASE_VAL
            p = a_skill / skill_sums[i]

            if p == 0:
                skill_ent += 0
            else:
                skill_ent += p * np.log(p)
        skill_ent = skill_ent / (n_pop)
        skill_ents.append(skill_ent)

    agent_ents = []

    for j in range(n_pop):
        agent_ent = 0

        for i in range(n_skills):

            a_skill = agent_skills[j][i]

            if a_skill == 0:
                a_skill = BASE_VAL
            p = a_skill / agent_sums[j]

            if p == 0:
                agent_ent += 0
            else:
                agent_ent += p * np.log(p)
        agent_ent = agent_ent / (n_skills)
        agent_ents.append(agent_ent)
    agent_score = np.mean(agent_ents)
    skill_score = np.mean(skill_ents)
    score = (alpha * skill_score + (1 - alpha) * agent_score)
    score = score * 100
    print('agent skills:\n{}\n{}'.format(skill_idxs, np.array(agent_skills)))
    print('skill_ents:\n{}\nskill_mean:\n{}\nagent_ents:\n{}\nagent_mean:{}\nscore:\n{}\n'.format(
        np.array(skill_ents), skill_score, np.array(agent_ents), agent_score, score))

    return score

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
                        default='scratch',
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
