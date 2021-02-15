from evolution.evo_map import EvolverNMMO
from evolution.evo_me import meNMMO
from evolution.evo_neat import NeatNMMO

def init_evolver(save_path, make_env, trainer, config, n_proc=12, n_pop=12, map_policy=None):
    if config.EVO_ALGO == 'Simple':
        return EvolverNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy)
    elif config.EVO_ALGO == 'Neat':
        return NeatNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy)
    elif config.EVO_ALGO == 'MAP-Elites':
        return meNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy)
