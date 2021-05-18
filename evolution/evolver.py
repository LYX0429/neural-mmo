from pdb import set_trace as T

from evolution.evo_dummi import DummiNMMO
from evolution.evo_map import EvolverNMMO
from evolution.qdpy_wrapper import meNMMO
from evolution.evo_neat import NeatNMMO


def init_evolver(save_path, make_env, trainer, config, n_proc=12, n_pop=12, map_policy=None, n_epochs=10000):
   #if config.PRETRAIN:
   #    return DummiNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy)
    if config.EVO_ALGO == 'Simple':
        return EvolverNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy, n_epochs=n_epochs)
    elif config.EVO_ALGO == 'Neat':
        return NeatNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy, n_epochs=n_epochs)
    elif config.EVO_ALGO in ['MAP-Elites', 'CMAES', 'CMAME']:
        return meNMMO(save_path, make_env, trainer, config, n_proc, n_pop, map_policy, n_epochs=n_epochs)
    else:
        raise Exception