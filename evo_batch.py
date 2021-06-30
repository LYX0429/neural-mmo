'''
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from dead processes.
'''
import os
import sys
import copy
import json
import re
import argparse
import numpy as np
from pdb import set_trace as TT
import matplotlib
from matplotlib import pyplot as plt

import projekt
from fire import Fire
from ForgeEvo import get_experiment_name
from evolution.diversity import get_div_calc

genomes = [
#   'Random',
#   'CPPN',
#   'Pattern',
#   'Simplex',
#   'CA',
#   'LSystem',
#   'All',
    'Baseline',
]
fitness_funcs = [
    'Lifespans',
#   'L2',
#   'Hull',
#   'Differential',
#   'Sum',
#   'Discrete',
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
    [100,100],
]

EVALUATION_HORIZON = 100
if False:
    N_EVALS = 1
else:
    N_EVALS = 20
# TODO: use this variable in the eval command string. Formatting might be weird.
SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
eval_args = "--EVALUATION_HORIZON {} --N_EVAL {} --NEW_EVAL --SKILLS \"['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]\"".format(EVALUATION_HORIZON, N_EVALS)
DIV_CALCS = ['L2', 'Differential', 'Hull', 'Discrete', 'Sum']


def launch_cmd(new_cmd, i):
   with open(sbatch_file, 'r') as f:
      content = f.read()
      content_0 = re.sub('nmmo\d*', 'nmmo{}'.format(i), content)
      new_content = re.sub('python Forge.*', new_cmd, content_0)

   with open(sbatch_file, 'w') as f:
      f.write(new_content)
   if LOCAL:
      os.system(new_cmd)
      os.system('ray stop')
   else:
      os.system('sbatch {}'.format(sbatch_file))


def launch_batch(exp_name, preeval=False):


   if LOCAL:
      default_config['n_generations'] = 1
   launched_baseline = False
   i = 0

   for gene in genomes:
      for fit_func in fitness_funcs:
         for skillset in skills:
            if fit_func in ['Lifespans', 'Sum']:
               if skillset != 'ALL':
                  continue
               skillset = 'NONE'

            for algo in algos:
               for me_bins in me_bin_sizes:
                  if gene == 'Baseline' and launched_baseline:
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



                  # Write the config file with the desired settings
                  exp_config = copy.deepcopy(default_config)
                  exp_config.update({
                     'N_GENERATIONS': 10000,
                     'TERRAIN_SIZE': 70,
                     'NENT': 16,
                     'GENOME': gene,
                     'FITNESS_METRIC': fit_func,
                     'EVO_ALGO': algo,
                     'EVO_DIR': exp_name,
                     'SKILLS': skillset,
                     'ME_BIN_SIZES': me_bins,
                     'ME_BOUNDS': [(0,100),(0,100)],
                     'FEATURE_CALC': feature_calc,
                     'ITEMS_PER_BIN': items_per_bin,
                     'N_EVO_MAPS': 48,
                     'N_PROC': 48,
                     'TERRAIN_RENDER': False,
                     'EVO_SAVE_INTERVAL': 300,
                     })
                  if gene == 'Baseline':
                     exp_config.update({
                         'PRETRAIN': True,
                     })

                  if CUDA:
                     exp_config.update({
                        'N_EVO_MAPS': 12,
                        'N_PROC': 12,
                     })

                  if LOCAL:
                     exp_config.update({
                        'N_GENERATIONS': 100,
                        'N_EVO_MAPS': 12,
                        'N_PROC': 12,
                        'EVO_SAVE_INTERVAL': 10,
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
                     config = projekt.config.EvoNMMO
                    #sys.argv = sys.argv[:1] + ['override']
                    #Fire(config)
                     for (k, v) in exp_config.items():
                        setattr(config, k, v)
                    #   config.set(config, k, v)
                     experiment_name = get_experiment_name(config)
                     experiment_names.append(experiment_name)

                    #config.set(config, 'ROOT', re.sub('evo_experiment/.*/', 'evo_experiment/{}/'.format(experiment_name), config.ROOT))

   if TRAIN_BASELINE:
      # Finally, launch a baseline
      with open(sbatch_file, 'r') as f:
         content = f.read()
         if not EVALUATE:
            new_cmd = 'python Forge.py train --config TreeOrerock --MODEL current --TRAIN_HORIZON 100 --NUM_WORKERS 12 --NENT 16 --TERRAIN_SIZE 70'
         else:
            assert preeval
            new_cmd = 'python Forge.py evaluate -la {}'.format(i)
         content = re.sub('nmmo\d*', 'nmmo00', content)
         new_content = re.sub('python Forge.*', new_cmd, content)

      with open(sbatch_file, 'w') as f:
         f.write(new_content)

      if not preeval:
         if LOCAL:
            os.system(new_cmd)
            os.system('ray stop')
         else:
            os.system('sbatch {}'.format(sbatch_file))
      else:
         config = projekt.config.EvoNMMO
#        experiment_names.append(get_experiment_name(config))

def launch_cross_eval(experiment_names, vis_only=False):
   n = 0
   model_exp_names = experiment_names + ['current']
   map_exp_names = experiment_names + ['PCG']
   mean_lifespans = np.zeros((len(model_exp_names), len(map_exp_names)))
   mean_skills = np.zeros((len(SKILLS), len(model_exp_names), len(map_exp_names)))
   div_scores = np.zeros((len(DIV_CALCS), len(model_exp_names), len(map_exp_names)))
   for (i, model_exp_name) in enumerate(model_exp_names):
      for (j, map_exp_name) in enumerate(map_exp_names):
         # TODO: select best map and from saved archive and evaluate on this one
         infer_idx = None
         if map_exp_name == 'PCG':
            infer_idx = 0
         elif 'Pattern' in map_exp_name:
            infer_idx = "(10, 6, 0)"
         elif 'CPPN' in map_exp_name:
            infer_idx = "(10, 8, 0)"
         elif 'Random' in map_exp_name:
            infer_idx = "(18, 17, 0)"
         elif 'Simplex' in map_exp_name:
            infer_idx = "(10, 5, 0)"
         if not vis_only:
            new_cmd = 'python Forge.py evaluate --config TreeOrerock --MODEL {} --MAP {} --INFER_IDX \"{}\" {} --EVO_DIR'.format(model_exp_name, map_exp_name, infer_idx, eval_args, EXP_NAME)
            print(new_cmd)
            launch_cmd(new_cmd, n)
         else:
            eval_data_path = os.path.join(
               'eval_experiment',
               map_exp_name,
               str(infer_idx),
               model_exp_name,
               'MODEL_{}_MAP_{}_ID{}_{}steps eval.npy'.format(
                  model_exp_name,
                  map_exp_name,
                  infer_idx,
                  EVALUATION_HORIZON
               ),
            )
            data = np.load(eval_data_path, allow_pickle=True)
            mean_lifespans[i, j] = np.mean(data[0]['lifespans'])
            # TODO: will this work for more than one episode of evaluation??? No???
            for k in range(len(SKILLS)):
               mean_skill_arr = np.vstack(data[0]['skills'])
               mean_skills[k, i, j] = np.mean(mean_skill_arr[:, k])
            for (k, div_calc_name) in enumerate(DIV_CALCS):
              #skill_arr = np.vstack(data[0]['skills'])
              #div_scores[k, i, j] = get_div_calc(div_calc_name)(skill_arr)
               div_scores[k, i, j] = get_div_calc(div_calc_name)(data[0])
            n += 1
   if vis_only:
      # NOTE: this is placeholder code, valid only for the current batch of experiments which varies along the "genome" dimension exclusively.
      # TODO: annotate the heatmap with labels more fancily, i.e. use the lists of hyperparams to create concise (hierarchical?) axis labels.
      row_labels = []
      col_labels = []
      def get_genome_name(exp_name):
         if 'CPPN' in exp_name:
            return 'CPPN'
         elif 'Pattern' in exp_name:
            return 'Pattern'
         elif 'Random' in exp_name:
            return 'Random'
         elif 'Simplex' in exp_name:
            return 'Simplex'
         else:
            return exp_name

      for r in model_exp_names:
         row_labels.append(get_genome_name(r))

      for c in map_exp_names:
         col_labels.append(get_genome_name(c))
      cross_eval_heatmap(mean_lifespans, row_labels, col_labels, "lifespans", "mean lifespan [ticks]")
      for (k, skill_name) in enumerate(SKILLS):
         cross_eval_heatmap(mean_skills[k], row_labels, col_labels, skill_name, "mean {} [xp]".format(skill_name))
      for (k, div_calc_name) in enumerate(DIV_CALCS):
         cross_eval_heatmap(div_scores[k], row_labels, col_labels, "{} diversity".format(div_calc_name), "{} diversity".format(div_calc_name))

def cross_eval_heatmap(data, row_labels, col_labels, title, cbarlabel):
   fig, ax = plt.subplots()

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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
   """
   Create a heatmap from a numpy array and two lists of labels.

   Parameters
   ----------
   data
       A 2D numpy array of shape (N, M).
   row_labels
       A list or array of length N with the labels for the rows.
   col_labels
       A list or array of length M with the labels for the columns.
   ax
       A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
       not provided, use current axes or create a new one.  Optional.
   cbar_kw
       A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
   cbarlabel
       The label for the colorbar.  Optional.
   **kwargs
       All other arguments are forwarded to `imshow`.
   """

   if not ax:
      ax = plt.gca()

   # Plot the heatmap
   im = ax.imshow(data, **kwargs)

   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
   cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

   # We want to show all ticks...
   ax.set_xticks(np.arange(data.shape[1]))
   ax.set_yticks(np.arange(data.shape[0]))
   # ... and label them with the respective list entries.
   ax.set_xticklabels(col_labels)
   ax.set_yticklabels(row_labels)

   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False,
                  labeltop=True, labelbottom=False)

   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")

   # Turn spines off and create white grid.
  #ax.spines[:].set_visible(False)

   ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
   ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
   ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
   ax.tick_params(which="minor", bottom=False, left=False)

   return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
   """
   A function to annotate a heatmap.

   Parameters
   ----------
   im
       The AxesImage to be labeled.
   data
       Data used to annotate.  If None, the image's data is used.  Optional.
   valfmt
       The format of the annotations inside the heatmap.  This should either
       use the string format method, e.g. "$ {x:.2f}", or be a
       `matplotlib.ticker.Formatter`.  Optional.
   textcolors
       A pair of colors.  The first is used for values below a threshold,
       the second for those above.  Optional.
   threshold
       Value in data units according to which the colors from textcolors are
       applied.  If None (the default) uses the middle of the colormap as
       separation.  Optional.
   **kwargs
       All other arguments are forwarded to each call to `text` used to create
       the text labels.
   """

   if not isinstance(data, (list, np.ndarray)):
      data = im.get_array()

   # Normalize the threshold to the images color range.
   if threshold is not None:
      threshold = im.norm(threshold)
   else:
      threshold = im.norm(data.max()) / 2.

   # Set default alignment to center, but allow it to be
   # overwritten by textkw.
   kw = dict(horizontalalignment="center",
             verticalalignment="center")
   kw.update(textkw)

   # Get the formatter in case a string is supplied
   if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

   # Loop over the data and create a `Text` for each "pixel".
   # Change the text's color depending on the data.
   texts = []
   for i in range(data.shape[0]):
      for j in range(data.shape[1]):
         kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
         text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
         texts.append(text)

   return texts

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
      '--gpu',
      help='Use GPU (only applies to SLURM).',
      action='store_true',
   )
   opts.add_argument(
      '--vis_cross_eval',
      help='Visualize the results of cross-evaluation. (No new evaluations.)',
      action='store_true',
   )
   opts = opts.parse_args()
   EXP_NAME = opts.experiment_name
   EVALUATE = opts.evaluate
   LOCAL = opts.local
   TRAIN_BASELINE = opts.train_baseline
   CUDA = opts.gpu
   VIS_CROSS_EVAL = opts.vis_cross_eval

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

   if EVALUATE:
      experiment_names = []
      # just get the names of experiments in which we are interested (no actual evaluations are run)
      launch_batch(EXP_NAME, preeval=True)
      print('cross evaluating experiments: {}'.format(experiment_names))
      if not VIS_CROSS_EVAL:
         # only launch these cross evaluations if we need to
         launch_cross_eval(experiment_names, vis_only=False)
      # otherwise just load up old data to visualize results
      launch_cross_eval(experiment_names, vis_only=True)
   else:
      # Launch a batch of joint map-evolution and agent-training experiments (maybe also a baseline agent-training experiment on a fixed set of maps).
      launch_batch(EXP_NAME)
