'''
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from dead processes.
'''
import os
import sys
import copy
import json
import csv
import re
import argparse
import pickle
import itertools
import numpy as np
from pdb import set_trace as TT
import matplotlib
from matplotlib import pyplot as plt

from forge.blade.core.terrain import MapGenerator, Save
from plot_diversity import heatmap, annotate_heatmap
from projekt import config
from fire import Fire
from projekt.config import get_experiment_name
from evolution.diversity import get_div_calc, get_pop_stats
from evolution.utils import get_exp_shorthand


genomes = [
   'Baseline',
   'Simplex',
   'NCA',
#  'TileFlip',
#  'CPPN',
#  'Primitives',
#  'L-System',
#  'All',
]
generator_objectives = [
#   'MapTestText',
    'Lifespans',
#   'L2',
#   'Hull',
    'Differential',
#   'Sum',
#   'Discrete',
    'FarNearestNeighbor',
#   'CloseNearestNeighbor',
#   'InvL2',
]
skills = [
    'ALL',
#   'HARVEST',
#   'COMBAT',
#   'EXPLORATION',
]
algos = [
    'MAP-Elites',
#   'Simple',
#   'CMAES',
#   'CMAME',
#   'NEAT',
]
me_bin_sizes = [
#   [1,1],
    [50, 50],
#   [100,100],
]

# Are we running a PAIRED-type algorithm? If so, we use two policies, and reward the generator for maximizing the
# difference in terms of the generator_objective between the "protagonist" and "antagonist" policies.
PAIRED_bools = [
   True,
#  False
]

# TODO: use this variable in the eval command string. Formatting might be weird.
SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
DIV_CALCS = ['L2', 'Differential', 'Hull', 'Discrete', 'Sum']
global eval_args
global EVALUATION_HORIZON
global TERRAIN_BORDER  # Assuming this is the same for all experiments!
global MAP_GENERATOR  # Also tile-set
TERRAIN_BORDER = None

def launch_cmd(new_cmd, i):
   with open(sbatch_file, 'r') as f:
      content = f.read()
      job_name = 'nmmo_'
      if EVALUATE:
          job_name += 'eval_'
      job_name += str(i)
      content = re.sub('nmmo_(eval_)?\d+', job_name, content)
      content = re.sub('#SBATCH --time=\d+:', '#SBATCH --time={}:'.format(JOB_TIME), content)
      new_content = re.sub('python Forge.*', new_cmd, content)

   with open(sbatch_file, 'w') as f:
      f.write(new_content)
   if LOCAL:
      os.system(new_cmd)
      if not (opts.vis_maps or opts.vis_cross_eval):
         os.system('ray stop')
   else:
      os.system('sbatch {}'.format(sbatch_file))


def launch_batch(exp_name, preeval=False):
   global TERRAIN_BORDER
   global MAP_GENERATOR
   if LOCAL:
      default_config['n_generations'] = 1
      if EVALUATE or opts.render:
         NENT = 16
      else:
         NENT = 3
      N_EVALS = 2
   else:
      NENT = 16
      N_EVALS = 20
   N_PROC = opts.n_cpu
   N_EVO_MAPS = 12
   global EVALUATION_HORIZON
   if opts.multi_policy:
      EVALUATION_HORIZON = 500
   else:
      EVALUATION_HORIZON = 100
   launched_baseline = False
   i = 0
   global eval_args
   eval_args = "--EVALUATION_HORIZON {} --N_EVAL {} --NEW_EVAL --SKILLS \"['constitution', 'fishing', 'hunting', " \
               "'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]\" --NENT {} " \
               "--FITNESS_METRIC {} ".format(
      EVALUATION_HORIZON, N_EVALS, NENT, generator_objectives[0])

   settings_tpls = [i for i in itertools.product(genomes, generator_objectives, skills, algos, me_bin_sizes,
                                                 PAIRED_bools)]
   for (gene, fit_func, skillset, algo, me_bins, PAIRED_bool) in settings_tpls:
      if fit_func in ['Lifespans', 'Sum']:
         if skillset != 'ALL':
            continue
         skillset = 'NONE'

      if gene == 'Baseline':
         if launched_baseline:
            # Only launch one baseline, these other settings are irrelevant
            continue
         else:
            launched_baseline = True
      if algo != 'MAP-Elites' and not (np.array(me_bins) == 1).all():
         # If using MAP-Elites, ME bin sizes are irrelevant
         continue
      if (np.array(me_bins) == 1).all():
         # If we're doing a simple evolutionary strategy (lazily, through qdpy ME, then set 12 individuals per bin
         items_per_bin = 12
         feature_calc = None
      else:
         items_per_bin = 1
         feature_calc = 'map_entropy'

      if LOCAL:
         if fit_func == 'MapTestText':
            N_GENERATIONS = 100000
            if gene == 'All':
               EVO_SAVE_INTERVAL = 100
            else:
               EVO_SAVE_INTERVAL = 100
         else:
            N_GENERATIONS = 10000
            EVO_SAVE_INTERVAL = 10
      else:
         EVO_SAVE_INTERVAL = 500
         N_GENERATIONS = 10000

      # Write the config file with the desired settings
      exp_config = copy.deepcopy(default_config)
      exp_config.update({
         'N_GENERATIONS': N_GENERATIONS,
         'TERRAIN_SIZE': 70,
         'NENT': NENT,
         'GENOME': gene,
         'FITNESS_METRIC': fit_func,
         'EVO_ALGO': algo,
         'EVO_DIR': exp_name,
         'SKILLS': skillset,
         'ME_BIN_SIZES': me_bins,
         'ME_BOUNDS': [(0,100),(0,100)],
         'FEATURE_CALC': feature_calc,
         'ITEMS_PER_BIN': items_per_bin,
         'N_EVO_MAPS': N_EVO_MAPS,
         'N_PROC': N_PROC,
         'TERRAIN_RENDER': False,
         'EVO_SAVE_INTERVAL': EVO_SAVE_INTERVAL,
         'VIS_MAPS': opts.vis_maps,
         'PAIRED': PAIRED_bool,
         'NUM_GPUS': 1 if CUDA else 0,
         })
      if gene == 'Baseline':
         exp_config.update({
             'PRETRAIN': True,
         })

      print('Saving experiment config:\n{}'.format(exp_config))
      with open('configs/settings_{}.json'.format(i), 'w') as f:
         json.dump(exp_config, f, ensure_ascii=False, indent=4)

      # Edit the sbatch file to load the correct config file
      # Launch the experiment. It should load the saved settings

      if not preeval:
         assert not EVALUATE
         new_cmd = 'python ForgeEvo.py --load_arguments {}'.format(i)
         launch_cmd(new_cmd, i)
         i += 1

      else:
         evo_config = config.EvoNMMO
        #sys.argv = sys.argv[:1] + ['override']
        #Fire(config)
         for (k, v) in exp_config.items():
            setattr(evo_config, k, v)
        #   config.set(config, k, v)
         if TERRAIN_BORDER is None:
            TERRAIN_BORDER = evo_config.TERRAIN_BORDER
            MAP_GENERATOR = MapGenerator(evo_config)
         else:
            assert TERRAIN_BORDER == evo_config.TERRAIN_BORDER
         experiment_name = get_experiment_name(evo_config)
         experiment_names.append(experiment_name)
         experiment_configs.append({'PAIRED': PAIRED_bool})

                    #config.set(config, 'ROOT', re.sub('evo_experiment/.*/', 'evo_experiment/{}/'.format(experiment_name), config.ROOT))

#   if TRAIN_BASELINE:
#      # Finally, launch a baseline
#      with open(sbatch_file, 'r') as f:
#         content = f.read()
#         if not EVALUATE:
#            new_cmd = 'python Forge.py train --config TreeOrerock --MODEL current --TRAIN_HORIZON 100 --NUM_WORKERS 12 --NENT 16 #--TERRAIN_SIZE 70'
#         else:
#            assert preeval
#            new_cmd = 'python Forge.py evaluate -la {}'.format(i)
#         content = re.sub('nmmo\d*', 'nmmo00', content)
#         new_content = re.sub('python Forge.*', new_cmd, content)
#
#      with open(sbatch_file, 'w') as f:
#         f.write(new_content)
#
#      if not preeval:
#         if LOCAL:
#            os.system(new_cmd)
#            os.system('ray stop')
#         else:
#            os.system('sbatch {}'.format(sbatch_file))
#      else:
#         config = config.EvoNMMO
##        experiment_names.append(get_experiment_name(config))


def launch_cross_eval(experiment_names, experiment_configs, vis_only=False, render=False, vis_cross_eval=False):
   """Launch a batch of evaluations, evaluating player models on generated maps from different experiments.
   If not just visualizing, run each evaluation (cartesian product of set of experiments with itself), then return.
   Otherwise, load data from past evaluations to generate visualizations of individual evaluations and/or of comparisons
   between them."""
   n = 0
   model_exp_names = experiment_names
   map_exp_names = experiment_names
   # We will use these heatmaps to visualize performance between generator-agent pairs over the set of experiments
   mean_lifespans = np.zeros((len(model_exp_names), len(map_exp_names)))
   std_lifespans = np.zeros((len(model_exp_names), len(map_exp_names) + 1))  # also take std of each model's average performance
   mean_skills = np.zeros((len(SKILLS), len(model_exp_names), len(map_exp_names)))
   div_scores = np.zeros((len(DIV_CALCS), len(model_exp_names), len(map_exp_names)))
   div_scores[:] = np.nan
   mean_skills[:] = np.nan
   mean_lifespans[:] = np.nan
   if opts.multi_policy:
      mean_survivors = np.zeros((len(map_exp_names), len(map_exp_names)), dtype=np.float)
   for (j, map_exp_name) in enumerate(map_exp_names):
      try:
         with open(os.path.join('evo_experiment', map_exp_name, 'ME_archive.p'), "rb") as f:
            archive = pickle.load(f)
      except FileNotFoundError as fnf:
         print(fnf)
         print('skipping eval with map from: {}'.format(map_exp_name))
         continue
      best_ind = archive['container'].best
      infer_idx, best_fitness = best_ind.idx, best_ind.fitness
      map_path = os.path.join('evo_experiment', map_exp_name, 'maps', 'map' + str(infer_idx), '')
      map_arr = best_ind.chromosome.map_arr
      Save.np(map_arr, map_path)
      png_path = os.path.join('evo_experiment', map_exp_name, 'maps', 'map' + str(infer_idx) + '.png')
      Save.render(map_arr[TERRAIN_BORDER:-TERRAIN_BORDER, TERRAIN_BORDER:-TERRAIN_BORDER], MAP_GENERATOR.textures, png_path)
      if vis_only:
         txt_verb = 'Visualizing past inference'
      elif vis_cross_eval:
         txt_verb = 'Collecting data for cross-eval visualization'
      else:
         txt_verb = 'Inferring'
      print('{} on map {}, with fitness {}, and age {}.'.format(txt_verb, infer_idx, best_fitness, best_ind.age))
      for (i, (model_exp_name, model_config)) in enumerate(zip(model_exp_names, experiment_configs)):
         l_eval_args = '--config TreeOrerock --MAP {} --INFER_IDX \"{}\" '.format(map_exp_name,
                                                                                          infer_idx)
         if opts.multi_policy:
            NPOLICIES = len(experiment_names)
            l_eval_args += '--MODEL {} '.format(str(model_exp_names).replace(' ', ''))
         else:
            NPOLICIES = 1
            l_eval_args += '--MODEL {} '.format(model_exp_name)
         NPOP = NPOLICIES
         #TODO: deal with PAIRED, and combinations of PAIRED and non-PAIRED experiments
         l_eval_args += '--NPOLICIES {} --NPOP {} --PAIRED {}'.format(NPOLICIES, NPOP, model_config['PAIRED'])

         if render:
            render_cmd = 'python Forge.py render {} {}'.format(l_eval_args, eval_args)
            assert LOCAL  # cannot render on SLURM
            assert not vis_only
            client_cmd = 'cd ../neural-mmo-client && ./UnityClient/neural-mmo-resources.x86_64&'
            os.system(client_cmd)
            print(render_cmd)
            os.system(render_cmd)
         elif not (vis_only or vis_cross_eval):
            eval_cmd = 'python Forge.py evaluate {} {} --EVO_DIR {}'.format(l_eval_args, eval_args, EXP_NAME)
            print(eval_cmd)
            launch_cmd(eval_cmd, n)
            n += 1
         else:
            global EVALUATION_HORIZON
            if opts.multi_policy:
               model_exp_folder = 'multi_policy'
               model_name = str([get_exp_shorthand(m) for m in model_exp_names])
            else:
               model_name = get_exp_shorthand(model_exp_name)
               model_exp_folder = model_exp_name
            map_exp_folder = map_exp_name
            eval_data_path = os.path.join(
               'eval_experiment',
               map_exp_folder,
               str(infer_idx),
               model_exp_folder,
               'MODEL_{}_MAP_{}_ID{}_{}steps eval.npy'.format(
                  model_name,
                  get_exp_shorthand(map_exp_name),
                  infer_idx,
                  EVALUATION_HORIZON
               ),
            )
            try:
               data = np.load(eval_data_path, allow_pickle=True)
            except FileNotFoundError as fnf:
               print(fnf)
               print('Skipping. Missing eval data at: {}'.format(eval_data_path))
               continue
            # get the mean lifespan of each eval episode
            evals_mean_lifespans = [np.mean(get_pop_stats(data_i['lifespans'], pop=None)) for data_i in data]
            # take the mean lifespan over these episodes
            mean_lifespans[i, j] = np.mean(evals_mean_lifespans)
            # std over episodes
            std_lifespans[i, j] = np.std(evals_mean_lifespans)
            # get the mean agent skill vector of each eval episode
            evals_mean_skills = np.vstack([get_pop_stats(data_i['skills'],pop=None).mean(axis=0) for data_i in data])
            for k in range(len(SKILLS)):
               mean_skills[k, i, j] = np.mean(evals_mean_skills[:, k])
            for (k, div_calc_name) in enumerate(DIV_CALCS):
               evals_div_scores = [get_div_calc(div_calc_name)(data_i) for data_i in data]
               div_scores[k, i, j] = np.mean(evals_div_scores)
            if opts.multi_policy:
               model_name_idxs = {get_exp_shorthand(r): i for (i, r) in enumerate(model_exp_names)}
               multi_eval_data_path = eval_data_path.replace('eval.npy', 'multi_eval.npy')
               survivors = np.load(multi_eval_data_path, allow_pickle=True).item()
               for survivor_name, n_survivors in survivors.items():
                  mean_survivors[model_name_idxs[survivor_name], j] = n_survivors.mean()
         # TODO:
         # get std of model's mean lifespan over all maps
#        std_lifespans[i, j+1] =
         if opts.multi_policy:  # don't need to iterate through models since we pit them against each other during the same episode
            break
   TT()
   if vis_cross_eval or vis_only:  # might as well do cross-eval vis if visualizing individual evals I guess
      print("Visualizing cross-evaluation.")
      # NOTE: this is placeholder code, valid only for the current batch of experiments which varies along the "genome" , "generator_objective" and "PAIRED" dimensions exclusively. Expand crappy get_exp_shorthand function if we need more.
      # TODO: annotate the heatmap with labels more fancily, i.e. use the lists of hyperparams to create concise (hierarchical?) axis labels.
      row_labels = []
      col_labels = []

      for r in model_exp_names:
         row_labels.append(get_exp_shorthand(r))

      for c in map_exp_names:
         col_labels.append(get_exp_shorthand(c))
      cross_eval_heatmap(mean_lifespans, row_labels, col_labels, "lifespans", "mean lifespan [ticks]", errors=std_lifespans)
      for (k, skill_name) in enumerate(SKILLS):
         cross_eval_heatmap(mean_skills[k], row_labels, col_labels, skill_name, "mean {} [xp]".format(skill_name))
      for (k, div_calc_name) in enumerate(DIV_CALCS):
         cross_eval_heatmap(div_scores[k], row_labels, col_labels, "{} diversity".format(div_calc_name), "{} diversity".format(div_calc_name))
      if opts.multi_policy:
         cross_eval_heatmap(mean_survivors, row_labels, col_labels, "mean survivors", "")

def cross_eval_heatmap(data, row_labels, col_labels, title, cbarlabel, errors=None):
   fig, ax = plt.subplots()
   fig.set_figheight(15)
   fig.set_figwidth(15)

   # Remove empty rows and columns
   i = 0
   for (row_label, data_row) in zip(row_labels, data):
      if np.isnan(data_row).all():
         row_labels = row_labels[:i] + row_labels[i+1:]
         data = np.vstack((data[:i], data[i+1:]))
         continue
      i += 1
   i = 0
   for (col_label, data_col) in zip(col_labels, data.T):
      if np.isnan(data_col).all():
         col_labels = col_labels[:i] + col_labels[i + 1:]
         data = (np.vstack((data.T[:i], data.T[i + 1:]))).T
         continue
      i += 1

   # Add col. with averages over each row (each model)
   col_labels += ['mean']
   data = np.hstack((data, np.expand_dims(data.mean(axis=1), 1)))

   im, cbar = heatmap(data, row_labels, col_labels, ax=ax,
                      cmap="YlGn", cbarlabel=cbarlabel)
   texts = annotate_heatmap(im, valfmt="{x:.1f}")
   ax.set_title(title)

#  fig.tight_layout(rect=[1,0,1,0])
   fig.tight_layout(pad=3)
#  plt.show()
   ax.set_ylabel('model')
   ax.set_xlabel('map')
   plt.savefig(os.path.join(
      'eval_experiment',
      '{}.png'.format(title),
   ))
   plt.close()


if __name__ == '__main__':
   opts = argparse.ArgumentParser(
      description='Launch a batch of experiments/evaluations for evo-pcgrl')

   opts.add_argument(
       '-ex',
       '--experiment_name',
       help='A name to be shared by the batch of experiments.',
       default='0',
   )
   opts.add_argument(
       '-ev',
       '--evaluate',
       help='Cross-evaluate a batch of joint map-evolution, agent-learning experiments, looking at the behavior of all '
            'agent models on all ("best") maps.',
       action='store_true',
   )
   opts.add_argument(
       '-l',
       '--local',
       help='Run the batch script on a local machine (evolving for a minimal number of generations, or running full evaluations sequentially).',
       action='store_true',
   )
   opts.add_argument(
      '-bl',
      '--train_baseline',
      help='Train a baseline on Perlin noise-generated maps.',
      action='store_true',
   )
   opts.add_argument(
      '--cpu',
      help='Do not use GPU (only applies to SLURM, not recommended for default, big neural networks).',
      action='store_true',
   )
   opts.add_argument(
      '--n_cpu',
      help='How many parallel processes ray should use.',
      type=int,
      default=12,
   )
   opts.add_argument(
      '--vis_cross_eval',
      help='Visualize the results of cross-evaluation. (No new evaluations.)',
      action='store_true',
   )
   opts.add_argument(
      '--vis_evals',
      help='Visualize the results of individual evaluations and cross-evaluation. (No new evaluations.)',
      action='store_true',
   )
   opts.add_argument(
      '--vis_maps',
      help='Save and visualize evolved maps, and plot their fitness.',
      action='store_true'
   )
   opts.add_argument(
      '--render',
      help='Render an episode in unity.',
      action='store_true'
   )
   opts.add_argument(
      '-mp',
      '--multi-policy',
      help='Evaluate all policies on each map simultaneously, to allow for inter-policy competition.',
      action='store_true',
   )
   opts = opts.parse_args()
   EXP_NAME = opts.experiment_name
   EVALUATE = opts.evaluate
   LOCAL = opts.local
   TRAIN_BASELINE = opts.train_baseline
   CUDA = not opts.cpu and not opts.vis_maps and not EVALUATE
   VIS_CROSS_EVAL = opts.vis_cross_eval
   VIS_EVALS = opts.vis_evals
   RENDER = opts.render
   if EVALUATE or opts.vis_maps:
      JOB_TIME = 12
   elif CUDA:
      JOB_TIME = 48  # NYU HPC Greene limits number of gpu jobs otherwise
   else:
      pass
#     JOB_TIME = 120  # never use CPU-only for training anyway

   if CUDA:
      sbatch_file = 'evo_train.sh'
   else:
      sbatch_file = 'evo_train_cpu.sh'
   if LOCAL:
      print('Testing locally.')
   else:
      print('Launching batch of experiments on SLURM.')
   with open('configs/default_settings.json', 'r') as f:
      default_config = json.load(f)
   print('Loaded default config:\n{}'.format(default_config))

   if EVALUATE or RENDER and not opts.vis_maps:
      experiment_names = []
      experiment_configs = []
      # just get the names and configs of experiments in which we are interested (no actual evaluations are run)
      launch_batch(EXP_NAME, preeval=True)
      if RENDER:
         print('rendering experiments: {}\n KeyboardInterrupt (Ctrl+c) to render next.'.format(experiment_names))
         launch_cross_eval(experiment_names, vis_only=False, render=True, experiment_configs=experiment_configs)
      else:
         if not (VIS_CROSS_EVAL or VIS_EVALS):
            print('cross evaluating experiments: {}'.format(experiment_names))
            # only launch these cross evaluations if we need to
            launch_cross_eval(experiment_names, experiment_configs=experiment_configs, vis_only=False)
         # otherwise just load up old data to visualize results
         if VIS_EVALS:
            # visualize individual evaluations.
            launch_cross_eval(experiment_names, experiment_configs=experiment_configs, vis_only=True)
         elif VIS_CROSS_EVAL or LOCAL:  # elif since vis_only also prompts cross-eval visualization
            # visualize cross-evaluation tables
            launch_cross_eval(experiment_names, experiment_configs=experiment_configs, vis_only=False, vis_cross_eval=True)
   else:
      # Launch a batch of joint map-evolution and agent-training experiments (maybe also a baseline agent-training experiment on a fixed set of maps).
      launch_batch(EXP_NAME)
