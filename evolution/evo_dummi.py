from pdb import set_trace as T
from evolution.evo_map import EvolverNMMO

class DummiNMMO(EvolverNMMO):
    def __init__(self, save_path, make_env, trainer, config, n_proc, n_pop, map_policy):
        super().__init__(save_path, make_env, trainer, config, n_proc, n_pop, map_policy)


    def evolve_generation(self):
        #     global_stats = self.global_stats
        #     self.send_genes(global_stats)
        maps = dict([(i, None) for i in range(self.config.N_EVO_MAPS)])
        stats = self.train_and_log(maps=maps)
#       #     stats = self.stats
#       #headers = ray.get(global_stats.get_headers.remote())
#       #     n_epis = train_stats['episodes_this_iter']

#       #     if n_epis == 0:
#       #        print('Missing simulation stats. I assume this is the 0th generation? Re-running the training step.')

#       #        return self.evolve_generation()

#       #     global_stats.reset.remote()

#       for g_hash in self.population.keys():
#           if g_hash not in stats:
#               print('Missing simulation stats for map {}. I assume this is the 0th generation, or re-load? Re-running the training step.'.format(g_hash))
#               stats = self.trainer.train(maps=maps)
#               #           stats = self.stats

#               break

#       for g_hash, (game, score, age) in self.population.items():
#           #print(self.config.SKILLS)
#           score = self.calc_fitness(stats[g_hash], verbose=False)
#           self.population[g_hash] = (game, score, age)

#       for g_hash, (game, score_t, age) in self.population.items():
#           # get score from latest simulation
#           # cull score history

#           #if len(self.score_hists[g_hash]) >= self.config.ROLLING_FITNESS:
#           #   self.score_hists[g_hash] = self.score_hists[g_hash][-self.config.ROLLING_FITNESS:]
#           #           while len(self.score_hists[g_hash]) >= self.config.ROLLING_FITNESS:
#           #              self.score_hists[g_hash].pop(0)
#           # hack

#           #if score_t is None:
#           #    score_t = 0
#           #else:
#           self.score_hists[g_hash].append(score_t)
#           self.score_hists[g_hash] = self.score_hists[g_hash][-self.config.ROLLING_FITNESS:]

#           if self.LEARNING_PROGRESS:
#               score = self.score_hists[-1] - self.score_hists[0]
#           else:
#               score = np.mean(self.score_hists[g_hash])
#           game, _, age = self.population[g_hash]
#           self.population[g_hash] = (game, score, age + 1)
#      #super().mutate_gen()
#       self.save()
