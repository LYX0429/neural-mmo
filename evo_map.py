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
from evolution.lambda_mu import LambdaMuEvolver

import numpy as np

import projekt
from forge.ethyr.torch import utils
from pcg import TILE_TYPES
from projekt import rlutils
from projekt.evaluator import Evaluator
from projekt.overlay import OverlayRegistry
from forge.blade.core.terrain import MapGenerator, Save
from forge.blade.lib import enums

import ray
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




def calc_diversity_l2(agent_skills):
    score = 0
    agent_skills = np.array(agent_skills)
    for a in agent_skills:
        for b in agent_skills:
            score += np.linalg.norm(a-b)
#   print('agent skills:\n{}'.format(np.array(agent_skills)))
#   print('score:\n{}\n'.format(
#       score))

    return score


class LogCallbacks(DefaultCallbacks):
   STEP_KEYS = 'env_step realm_step env_stim stim_process'.split()
   EPISODE_KEYS = ['env_reset']
   
   def init(self, episode):
      for key in LogCallbacks.STEP_KEYS + LogCallbacks.EPISODE_KEYS: 
         episode.hist_data[key] = []
      episode.hist_data['map_fitness'] = []

   def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
         policies: Dict[str, Policy],
         episode: MultiAgentEpisode, **kwargs):
      self.init(episode)
     #episode.hist_data['map_fitness'].append((base_env.envs[0].worldIdx, -1))

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
      episode.hist_data['map_fitness'].append((env.worldIdx, env.map_fitness))

   def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
      print("trainer.train() result: {} -> {} episodes".format(
         trainer, result["episodes_this_iter"]))
      # you can mutate the result dict to add new fields to return
      # result['something'] = True

#sz = 16 + 16
#scale = int(sz / 5)
#root = 'resource/maps/'


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





class EvolverNMMO(LambdaMuEvolver):
    def __init__(self, save_path, make_env, trainer, config, n_proc=12, n_pop=12,):
#       save_path = os.path.abspath(os.path.join('evo_experiment', 'evo_test'))
        super().__init__(save_path, n_proc=n_proc, n_pop=n_pop)
#       torch._C._cuda_init()
        # balance between skill and agent entropies
#       self.alpha = config.DIVERSITY_ALPHA
        self.make_env = make_env
        self.trainer = trainer
        self.map_width = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
        self.map_height = config.TERRAIN_SIZE#+ 2 * config.TERRAIN_BORDER
        # FIXME: hack: ignore orerock
        self.n_tiles = len(TILE_TYPES) - 1
        # how much has each individual's score varied over the past n simulations?
        # can tell us how informative further simulation will be
        # assumes task is stationary
       #self.score_var = {}
        # age at which individuals can die and reproduce
       #self.gens_alive = {}
       #self.last_scores = {}
        self.mature_age = config.MATURE_AGE

        self.state = {}
        self.done = {}
        # self.config   = config
        # config.RENDER = True
        self.map_generator = MapGenerator(config)
        config = {'config': config, #'map_arr': map_arr
                }
        self.config = config
        map_arr = self.genRandMap()
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

    def saveMaps(self, maps, mutated=None):
       #for i, map_arr in maps.items():
        if mutated is None:
            mutated = list(self.population.keys())
        for i in mutated:
           map_arr = maps[i]
           print('Generating map ' + str(i))
           path = os.path.join(self.save_path, 'maps', 'map' + str(i), '')
           try:
              os.mkdir(path)
           except FileExistsError:
               pass
           #  terrain, tiles = self.map_generator.grid(self.config['config'], seed)
           #  Save.np(tiles, path)
           #  if self.config['config'].TERRAIN_RENDER:
           #     Save.fractal(terrain, path+'fractal.png')
           #     Save.render(tiles, self.map_generator.textures, path+'map.png')

              #terrain = copy.deepcopy(terrain).astype(object)
              #for y in range(map_arr.shape[1]):
              #    for x in range(map_arr.shape[0]):
              #        tile_type = TILE_TYPES[map_arr[x, y]]
              #       #print(tile_type)
              #        texture = tex[tile_type]
              #       #print(texture)
              #        terrain[y, x] = texture
           Save.np(map_arr, path)
           if self.config['config'].TERRAIN_RENDER:
       #      Save.fractal(terrain, path+'fractal.png')
              Save.render(map_arr, self.map_generator.textures, path+'map.png')
           #fractal(terrain, path+'fractal.png')
           #render(tiles, path+'map.png')

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


       #self.overlays = Overlay(env, self.model, trainer, config['config'])

    def make_game(self, child_map):
        config = self.config
       #config['map_arr'] = child_map
        game = self.make_env(config)

        return game


    def restore(self, trash_data=False):
        '''
        trash_data: to avoid undetermined weirdness when reloading
        '''
#       for g_hash, (game, score, age) in self.population.items():
#           if game is None:
#              #self.config['map_arr'] = self.genes[g_hash]

#               if trash_data:
#                   pass
#                  #score = None
#                  #age = 0
#                  #self.score_var[g_hash] = None, []
#                  #self.score_hists[g_hash] = []
#               self.population[g_hash] = (game, score, age)

        if self.trainer is None:

            # Create policies
            policies = createPolicies(self.config['config'])

            # Instantiate monolithic RLlib Trainer object.
            trainer = rlutils.EvoPPOTrainer(env="custom",
                                             path='experiment',
                                             config={
                                                 'num_workers': 12, # normally: 4
                                                 'num_gpus_per_worker': 0.083,  # hack fix
                                                 'num_gpus': 1,
                                                 'num_envs_per_worker': 4,
                                                 'train_batch_size': 120, # normally: 4000
                                                 'rollout_fragment_length':
                                                 self.config['config'].MAX_STEPS + 3,
                                                 'sgd_minibatch_size': 110,  # normally: 128
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
#       game = self.population[g_hash][0]
#       game = self.make_env(self.config)
#       model    = self.trainer.get_policy('policy_0').model
        evaluator = Evaluator(self.config['config'], self.trainer)
        evaluator.render()
#       self.registry = OverlayRegistry(game, self.model, self.trainer, self.config['config'])
#       map_arr = self.genes[g_hash]
#      #map_arr = self.genRandMap()
#       self.run(game, map_arr)
#       map_arr = self.genes[g_hash]


    def genRandMap(self):
        if self.n_epoch > 0:
            print('generating new random map when I probably should not be... \n\n')
        # FIXME: hack: ignore lava
        map_arr= np.random.randint(1, self.n_tiles,
                                    (self.map_width, self.map_height))
       #map_arr.fill(TILE_TYPES.index('forest'))
       #map_arr[20:-20, 20:-20]= TILE_TYPES.index('water')
        self.add_border(map_arr)

        return map_arr

    def add_border(self, map_arr):
        b = self.config['config'].TERRAIN_BORDER
        # agents should not spawn and die immediately, as this may crash the env
        a = 2
        map_arr[b:b+a, :]= enums.Material.GRASS.value.index
        map_arr[:, b:b+a]= enums.Material.GRASS.value.index
        map_arr[:, -b-a:-b]= enums.Material.GRASS.value.index
        map_arr[-b-a:-b, :]= enums.Material.GRASS.value.index
        # the border must be lava
        map_arr[0:b, :]= enums.Material.LAVA.value.index
        map_arr[:, 0:b]= enums.Material.LAVA.value.index
        map_arr[-b:, :]= enums.Material.LAVA.value.index
        map_arr[:, -b:]= enums.Material.LAVA.value.index

        return map_arr

    def mutate(self, map_arr):
        map_arr= map_arr.copy()

        for i in range(random.randint(0, self.n_mutate_actions)):
            x= random.randint(0, self.map_width - 1)
            y= random.randint(0, self.map_height - 1)
            # FIXME: hack: ignore lava
            t= np.random.randint(1, self.n_tiles)
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
        agent_skills = [[] for _ in range(len(skill_dict))]
        i = 0

        for a_skills in skill_dict:
            j = 0
            for a_skill in a_skills:
                try:
                    val = float(a_skill)
                    agent_skills[i].append(val)
                    j += 1
                except:
                    pass
            i += 1

       #for agent_id, skills in skill_dict.items():
       #for skill, vals in skill_dict.items():

#      #    skills = skills['skills']
       #    if agent_skills is None:
       #        agent_skills = [[None for j in range(len(skills)-4)] for i in range(len(skill_dict))]
       #    j = 0

       #   #for s, v in skills.items():
       #       #if s in ['level', 'constitution', 'hunting', 'fishing']:
       #       #    continue

       #       #if s not in self.skill_idxs:
       #       #    skill_idx = j
       #       #    self.skill_idxs[s] = skill_idx
       #       #    self.idx_skills[skill_idx] = s
       #       #else:
       #       #    skill_idx = self.skill_idxs[s]
       #    exp_val = vals['exp']
       #       #if self.idx_skills[j] in ['woodcutting', 'mining']:
       #       #    exp_val -= 1154
       #    agent_skills[i][j] = exp_val
       #   #    j += 1
       #   #i += 1

        return calc_diversity_l2(agent_skills)

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
#       self.trainer.reset(self.trainer.workers.remote_workers())
        stats = self.trainer.train()
       #print('evo map trainer stats', stats)

        for g_hash, (game, score, age) in self.population.items():
            try:
                with open(os.path.join('./evo_experiment', '{}'.format(self.config['config'].EVO_DIR),
                    'map_{}_skills.csv'.format(g_hash)), newline='') as csv_skills:
                    skillsreader = csv.reader(csv_skills, delimiter=',', quotechar='|')
                    agent_skills = []
                    i = 0
                    for row in skillsreader:
                      # if i == 0:
                      #     if not self.skill_idxs:
                      #         for j, s_name in enumerate(row):
                      #             self.skill_idxs[s_name] = j
                      #     for j, s_name in enumerate(row):
                      #         skill_idxs[j] = s_name
                        if i > 0:
                            agent_skills.append(row)
                        i += 1
                    score = self.update_entropy_skills(agent_skills)
                assert csv_skills.closed
                self.population[g_hash] = (game, score, age)
            except FileNotFoundError:
                T()
                raise Exception
                score = None

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
        super().mutate_gen()

    def tick_game(self, game, g_hash=None):
        return self.tick(game=game, g_hash=g_hash)

    def tick(self, game=None, g_hash=None):
        # check if we are doing inference
        FROZEN = True
        if FROZEN:
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
                actions, self.state, _= self.trainer.compute_actions(
                    self.obs, state=self.state, policy_id='policy_0')

                # Compute overlay maps
                self.overlays.register(self.obs)

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

        reward = 0
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


