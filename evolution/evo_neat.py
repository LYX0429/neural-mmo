from evolution.evo_map import EvolverNMMO

class NeatNMMO(EvolverNMMO):
    def __init__(self, save_path, make_env, trainer, config, n_proc, n_pop, map_policy):
         super().__init__(self, save_path, make_env, trainer, config, n_proc, n_pop, map_policy)

         self.n_epoch = -1

         stats = neat.statistics.StatisticsReporter()
         self.neat_pop.add_reporter(stats)
         self.neat_pop.add_reporter(neat.reporting.StdOutReporter(True))

    def evolve(self):

         if self.n_epoch == -1:
            self.init_pop()
         else:
            self.map_generator = MapGenerator(self.config)
         winner = self.neat_pop.run(self.neat_eval_fitness, self.n_epochs)

    def saveMaps(self):
        if self.n_epoch == -1:
            self.n_epoch += 1

        super().saveMaps()


