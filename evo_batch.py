'''
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from dead processes.
'''
import os
import copy
import json
import re
import argparse
import pickle
import itertools
import numpy as np
from pdb import set_trace as TT
import matplotlib
from matplotlib import pyplot as plt

from forge.blade.core.terrain import MapGenerator, Save
from evolution.plot_diversity import heatmap, annotate_heatmap
from projekt import config
from projekt.config import get_experiment_name
from evolution.diversity import get_div_calc, get_pop_stats
from evolution.utils import get_exp_shorthand, get_eval_map_inds


##### HYPER-PARAMETERS #####

genomes = [
#  'Baseline',
#  'RiverBottleneckBaseline',
#  'ResourceNichesBaseline',
#  'BottleneckedResourceNichesBaseline',
#  'LabyrinthBaseline',
#  'Simplex',
   'NCA',
#  'TileFlip',
   'CPPN',
#  'Primitives',
   'L-System',
#  'All',
]
generator_objectives = [
#  'MapTestText',
   'Lifespans',
#  'L2',
#  'Hull',
#  'Differential',
#  'Sum',
#  'Discrete',
#  'FarNearestNeighbor',
#  'CloseNearestNeighbor',
#  'InvL2',
#  'AdversityDiversity',
#  'AdversityDiversityTrgs',
   'Achievement'

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
#  True,
   False
]
adv_div_ratios = [.5]
# adv_div_ratios = np.arange(0, 1.01, 1/6)  # this gets stretched to [-1, 1] and used to shrink one agent or the either

#adv_div_trgs = np.arange(0, 1.01, 1/5)
#adv_div_trgs = [
#  0,
#  1/5,
#  2/5,
#  3/5,
#  4/5,
#  1,
#  ]
#adv_div_trgs = itertools.product(adv_div_trgs, adv_div_trgs)

# For "AdversityDiversityTrgs" -- how long should agents live, how diverse should they be
adv_trgs = [
   0,
   1/5,
   2/5,
   3/5,
   4/5,
   1,
]
div_trgs = [
   0,
   1/5,
   2/5,
   3/5,
   4/5,
   1,
]
adv_div_trgs = itertools.product(adv_trgs, div_trgs)

##########################



# TODO: use this variable in the eval command string. Formatting might be weird.
SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
DIV_CALCS = ['L2', 'Differential', 'Hull',
             #'Discrete',
             'FarNearestNeighbor',
            'Sum']
global eval_args
global EVALUATION_HORIZON
global TERRAIN_BORDER  # Assuming this is the same for all experiments!
global MAP_GENERATOR  # Also tile-set
global N_EVAL_MAPS
global N_MAP_EVALS
TERRAIN_BORDER = None
MAP_GENERATOR = None

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


def launch_batch(exp_name, get_exp_info_only=False):
   exp_names = []
   exp_configs = []
   global TERRAIN_BORDER
   global MAP_GENERATOR
   global N_EVAL_MAPS
   global N_MAP_EVALS
   if LOCAL:
      default_config['n_generations'] = 1
      if EVALUATE or opts.render:
         NENT = 16
      else:
         NENT = 3
      #FIXME: we're overwriting a variable from original NMMO here. Will this be a problem?
      N_EVAL_MAPS = 4  # How many maps to evaluate on. This must always be divisible by 2
      N_MAP_EVALS = 2  # How many times to evaluate on each map
   else:
      NENT = 16
      N_EVAL_MAPS = 4
      N_MAP_EVALS = 2
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
   eval_args = "--EVALUATION_HORIZON {} --N_EVAL {} --N_EVAL_MAPS {} --NEW_EVAL --SKILLS \"['constitution', 'fishing', 'hunting', " \
               "'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]\" --NENT {} " \
               "--FITNESS_METRIC {} ".format(
      EVALUATION_HORIZON, N_MAP_EVALS, N_EVAL_MAPS, NENT, generator_objectives[0])

   settings_tpls = [i for i in itertools.product(genomes, generator_objectives, skills, algos, me_bin_sizes,
                                                 PAIRED_bools)]
   for (gene, gen_obj, skillset, algo, me_bins, PAIRED_bool) in settings_tpls:
      if gen_obj in ['Lifespans', 'Sum']:
         if skillset != 'ALL':
            continue
         skillset = 'NONE'

#     if gene == 'Baseline':
#        if launched_baseline:
#           # Only launch one baseline, these other settings are irrelevant
#           # FIXME: but now you're going to get redundant baselines with different names across batch runs if you're
#           #  not careful (and I am not careful)
#           continue
#        else:
#           launched_baseline = True
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
         if gen_obj == 'MapTestText':
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

      def launch_experiment(i):
         # Write the config file with the desired settings
         exp_config = copy.deepcopy(default_config)
         root = os.path.dirname(os.path.abspath(__file__)) + "/evo_experiment/experiment-name_0/maps/map"
         exp_config.update({
            'ROOT': root,
            'N_GENERATIONS': N_GENERATIONS,
            'TERRAIN_SIZE': 70,
            'NENT': NENT,
            'GENOME': gene,
            'FITNESS_METRIC': gen_obj,
            'EVO_ALGO': algo,
            'EVO_DIR': exp_name,
            'SKILLS': skillset,
            'ME_BIN_SIZES': me_bins,
            'ME_BOUNDS': [(0, 100), (0, 100)],
            'FEATURE_CALC': feature_calc,
            'ITEMS_PER_BIN': items_per_bin,
            'N_EVO_MAPS': N_EVO_MAPS,
            'N_PROC': N_PROC,
            'TERRAIN_RENDER': False,
            'EVO_SAVE_INTERVAL': EVO_SAVE_INTERVAL,
            'VIS_MAPS': opts.vis_maps,
            'RENDER': RENDER,
            'EVALUATE': EVALUATE,
            'PAIRED': PAIRED_bool,
            'NUM_GPUS': 1 if CUDA else 0,
            'ADVERSITY_DIVERSITY_RATIO': adv_div_ratio,
            'ADVERSITY_DIVERSITY_TRGS': adv_div_trg,
         })
         #     if gene == 'Baseline':
         #        exp_config.update({
         #            'PRETRAIN': True,
         #        })

         print('Saving experiment config:\n{}'.format(exp_config))
         with open('configs/settings_{}.json'.format(i), 'w') as f:
            json.dump(exp_config, f, ensure_ascii=False, indent=4)

         # Edit the sbatch file to load the correct config file
         # Launch the experiment. It should load the saved settings
         new_cmd = 'python ForgeEvo.py --load_arguments {}'.format(i)
         exp_configs.append(exp_config)
         if not get_exp_info_only:
            launch_cmd(new_cmd, i)

      adv_div_ratio = 0.5  # dummi var
      adv_div_trg = (1, 1)  # dummi var
      if gen_obj in ['AdversityDiversity', 'AdversityDiversityTrgs']:
         for adv_div_ratio in adv_div_ratios:
            if gen_obj == 'AdversityDiversityTrgs':
               for adv_div_trg in adv_div_trgs:
                  launch_experiment(i)
                  i += 1
            else:
               launch_experiment(i)
               i += 1
      else:
         launch_experiment(i)
         i += 1
   return exp_configs


def launch_cross_eval(experiment_names, experiment_configs, vis_only=False, render=False, vis_cross_eval=False):
   """Launch a batch of evaluations, evaluating player models on generated maps from different experiments.
   If not just visualizing, run each evaluation (cartesian product of set of experiments with itself), then return.
   Otherwise, load data from past evaluations to generate visualizations of individual evaluations and/or of comparisons
   between them."""
   global MAP_GENERATOR
   model_exp_names, model_exp_configs = experiment_names, experiment_configs
   map_exp_names, map_exp_configs = experiment_names, experiment_configs
   # TODO: The dimensions of these arrays are NOT to be fucked up. Attach them to an enum type class or something
   # We will use these heatmaps to visualize performance between generator-agent pairs over the set of experiments
   mean_lifespans = np.zeros((len(model_exp_names), len(map_exp_names), N_MAP_EVALS, N_EVAL_MAPS))
#  std_lifespans = np.zeros((len(model_exp_names), len(map_exp_names) + 1, N_MAP_EVALS, N_EVAL_MAPS))  # also take std of each model's average performance
   mean_skills = np.zeros((len(SKILLS), len(model_exp_names), len(map_exp_names), N_MAP_EVALS, N_EVAL_MAPS))
   div_scores = np.zeros((len(DIV_CALCS), len(model_exp_names), len(map_exp_names), N_MAP_EVALS, N_EVAL_MAPS))
   div_scores[:] = np.nan
   mean_skills[:] = np.nan
   mean_lifespans[:] = np.nan
   if opts.multi_policy:
      mean_survivors = np.zeros((len(map_exp_names), len(map_exp_names), N_MAP_EVALS, N_EVAL_MAPS), dtype=np.float)
   if vis_only:
      txt_verb = 'Visualizing past inference'
   elif vis_cross_eval:
      txt_verb = 'Collecting data for cross-eval visualization'
   else:
      txt_verb = 'Inferring'
   # FIXME: why tf is this MapGenerator breaking if I move it inside the loop?
   if MAP_GENERATOR is None:
      MAP_GENERATOR = MapGenerator(map_exp_configs[0])

   def collect_eval_data():
      n = 0
      for (gen_i, (map_exp_name, map_exp_config)) in enumerate(zip(map_exp_names, map_exp_configs)):
         if opts.eval_baseline_maps_only:
            if 'Baseline' not in map_exp_config.GENOME:
               continue
         TERRAIN_BORDER = map_exp_config.TERRAIN_BORDER
         # For each experiment from which we are evaluating generated maps, load up its map archive in order to select
         # these evaluation maps
         print('{} from evaluation on maps from experiment: {}'.format(txt_verb, map_exp_name))
         try:
            with open(os.path.join('evo_experiment', map_exp_name, 'ME_archive.p'), "rb") as f:
               map_archive = pickle.load(f)
         except FileNotFoundError as fnf:
            print(fnf)
            print('skipping eval with map from: {}'.format(map_exp_name))
            continue
         # best_ind = archive['container'].best
         eval_inds = get_eval_map_inds(map_archive, n_inds=N_EVAL_MAPS)
         # Evaluate on a handful of elite maps
#        for map_i, eval_map in enumerate(eval_inds):
#        infer_idx, best_fitness = eval_map.idx, eval_map.fitness
         infer_idxs, best_fitnesses = [map.idx for map in eval_inds], [map.fitness for map in eval_inds]
         for eval_map in eval_inds:
            map_path = os.path.join('evo_experiment', map_exp_name, 'maps', 'map' + str(eval_map.idx), '')
            map_arr = eval_map.chromosome.map_arr
            # Saving just in case we haven't already
            Save.np(map_arr, map_path)
#           png_path = os.path.join('evo_experiment', map_exp_name, 'maps', 'map' + str(infer_idx) + '.png')
#           Save.render(map_arr[TERRAIN_BORDER:-TERRAIN_BORDER, TERRAIN_BORDER:-TERRAIN_BORDER], MAP_GENERATOR.textures, png_path)
         print('{} on maps {}, with fitness scores {}, and ages {}.'.format(txt_verb, infer_idxs, best_fitnesses,
                                                                     [map.age for map in eval_inds]))
         for (mdl_i, (model_exp_name, model_config)) in enumerate(zip(model_exp_names, experiment_configs)):
            l_eval_args = '--config TreeOrerock --MAP {} '.format(map_exp_name,
                                                                                             infer_idxs)
            if opts.multi_policy:
               NPOLICIES = len(experiment_names)
               l_eval_args += '--MODEL {} '.format(str(model_exp_names).replace(' ', ''))
            else:
               NPOLICIES = 1
               l_eval_args += '--MODEL {} '.format(model_exp_name)
            NPOP = NPOLICIES
            l_eval_args += '--NPOLICIES {} --NPOP {} --PAIRED {}'.format(NPOLICIES, NPOP, model_config.PAIRED)

            if render:
               for infer_idx in infer_idxs:
                  l_eval_args_i = l_eval_args + ' --INFER_IDXS \"{}\" '.format(infer_idx)
                  render_cmd = 'python Forge.py render {} {}'.format(l_eval_args_i, eval_args)
                  assert LOCAL  # cannot render on SLURM
                  assert not vis_only
                  # Launch the client as a background process
                  client_cmd = './neural-mmo-client/UnityClient/neural-mmo-resources.x86_64&'
                  os.system(client_cmd)
                  print(render_cmd)
                  os.system(render_cmd)
            elif not (vis_only or vis_cross_eval):
               l_eval_args_i = l_eval_args + ' --INFER_IDXS \"{}\" '.format(infer_idxs)
               eval_cmd = 'python Forge.py evaluate {} {} --EVO_DIR {}'.format(l_eval_args_i, eval_args, EXP_NAME)
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
               try:
                  map_eval_data = []
                  for infer_idx in infer_idxs:
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
                     data = np.load(eval_data_path, allow_pickle=True)
                     map_eval_data.append(data)
               except FileNotFoundError as fnf:
                  # print(fnf)
                  print('Skipping. Missing eval data at: {}'.format(eval_data_path))
                  continue
               # FIXME: this is a tad gnarly. Could we do this more cleanly over different maps?
               for map_i, data in enumerate(map_eval_data):
                  # how many eval episodes will we use for data collection? can collect fewer than saved for fast iteration
                  n_evals_data = min(N_MAP_EVALS, len(data))
                  # get the mean lifespan of each eval episode
                  evals_mean_lifespans = [np.mean(get_pop_stats(data[i]['lifespans'], pop=None))
                                          for i in range(n_evals_data)]
                  # take the mean lifespan over these episodes
                  mean_lifespans[mdl_i, gen_i, :, map_i] = evals_mean_lifespans
                  # std over episodes
                  # std_lifespans[mdl_i, gen_i, map_i] = np.std(evals_mean_lifespans)
                  # get the mean agent skill vector of each eval episode
                  evals_mean_skills = np.vstack([get_pop_stats(data_i['skills'],pop=None).mean(axis=0) for data_i in data])
                  for s_i in range(len(SKILLS)):
                     mean_skills[s_i, mdl_i, gen_i, :, map_i] = evals_mean_skills[0:n_evals_data, s_i]
                  for (s_i, div_calc_name) in enumerate(DIV_CALCS):
                     evals_div_scores = [get_div_calc(div_calc_name)(data_i) for data_i in data[:n_evals_data]]
                     div_scores[s_i, mdl_i, gen_i, 0:n_evals_data, map_i] = evals_div_scores
                  if opts.multi_policy:
                     model_name_idxs = {get_exp_shorthand(r): i for (i, r) in enumerate(model_exp_names)}
                     multi_eval_data_path = eval_data_path.replace('eval.npy', 'multi_eval.npy')
                     survivors = np.load(multi_eval_data_path, allow_pickle=True)
                     for map_survivors in survivors:
                        for model_name, n_survivors in map_survivors.items():
                           mean_survivors[model_name_idxs[model_name], gen_i, 0:n_evals_data, map_i] = n_survivors
            # TODO: get std of model's mean lifespan over all maps
   #        std_lifespans[i, j+1] =
            if opts.multi_policy:  # don't need to iterate through models since we pit them against each other during the same episode
               break
      ret = (mean_lifespans, mean_skills, div_scores)
      if opts.multi_policy:
         ret = (*ret, mean_survivors)
      return ret

   cross_eval_data_path = os.path.join('eval_experiment', 'cross_eval_data.npy')
   if not opts.re_render_cross_vis:
      ret = collect_eval_data()
      np.save(cross_eval_data_path, ret)
   else:
      ret = np.load(cross_eval_data_path, allow_pickle=True)
      # FIXME: messy
      if opts.multi_policy:
         mean_lifespans, mean_skills, div_scores, mean_survivors = ret
      else:
         mean_lifespans, mean_skills, div_scores = ret

   if vis_cross_eval or vis_only:  # might as well do cross-eval vis if visualizing individual evals I guess
      print("Visualizing cross-evaluation.")
      # NOTE: this is placeholder code, valid only for the current batch of experiments which varies along the "genome" , "generator_objective" and "PAIRED" dimensions exclusively. Expand crappy get_exp_shorthand function if we need more.
      # TODO: annotate the heatmap with labels more fancily, i.e. use the lists of hyperparams to create concise (hierarchical?) axis labels.

      def get_meanstd(data):
         '''Funky function for getting mean, standard deviation of our data'''
         # TODO: these indices should be global variables or something like that
         # This gets the mean over evaluations (-2) and maps (-1)
         mean_model_mapgen = np.nanmean(data, axis=(-2, -1))
         # We want the standard deviation over evaluations. So we get the mean on maps first
         std_model_mapgen = np.nanmean(data, axis=-1).std(axis=-1)

         # add a column looking at the mean performance of each model over all maps
         mean_model = np.nanmean(data, axis=(-3, -1))  # work around missing generators/maps
         mean_model = np.mean(mean_model, axis=-1, keepdims=True)  # and evals (careful though)

         # standard deviation in this column is calculated a little differently: by getting the aggregate score of each model
         # model over all maps, then looking at *this* random variable's standard deviation over evals

         # this is the mean over generators and maps (not evals!)
         aggr_model = np.nanmean(data, axis=(-3, -1))
         # std over evals
         std_model = aggr_model.std(axis=-1, keepdims=True)

         means = np.concatenate((mean_model_mapgen, mean_model), axis=-1)
         stds = np.concatenate((std_model_mapgen, std_model), axis=-1)

         return means, stds

      # mean and standard deviation of lifespans over maps and evals
      mean_mapgen_lifespans, std_mapgen_lifespans = get_meanstd(mean_lifespans)
      # Repeat this averaging logic for other stats
      mean_mapgen_div_scores, std_mapgen_divscores = get_meanstd(div_scores)
      mean_mapgen_skills, std_mapgen_skills = get_meanstd(mean_skills)
      if opts.multi_policy:
         mean_mapgen_survivors, std_mapgen_survivors = get_meanstd(mean_survivors)

      row_labels = []
      col_labels = []

      for r in model_exp_names:
         row_labels.append(get_exp_shorthand(r))

      for c in map_exp_names:
         col_labels.append(get_exp_shorthand(c))
      col_labels.append('mean')
      cross_eval_heatmap(mean_mapgen_lifespans, row_labels, col_labels, "lifespans", "mean lifespan [ticks]",
                         errors=std_mapgen_lifespans)
      for (s_i, skill_name) in enumerate(SKILLS):
         cross_eval_heatmap(mean_mapgen_skills[s_i], row_labels, col_labels, skill_name,
                            "mean {} [xp]".format(skill_name), errors=std_mapgen_skills[s_i])
      for (d_i, div_calc_name) in enumerate(DIV_CALCS):
         cross_eval_heatmap(mean_mapgen_div_scores[d_i], row_labels, col_labels, "{} diversity".format(div_calc_name),
                            "{} diversity".format(div_calc_name), errors=std_mapgen_divscores[d_i])
      if opts.multi_policy:
         cross_eval_heatmap(mean_mapgen_survivors, row_labels, col_labels, "mean survivors", "",
                            errors=std_mapgen_survivors)

def cross_eval_heatmap(data, row_labels, col_labels, title, cbarlabel, errors=None):
   fig, ax = plt.subplots()

   # Remove empty rows and columns
   i = 0
   for (row_label, data_row) in zip(row_labels, data):
      if np.isnan(data_row).all():
         row_labels = row_labels[:i] + row_labels[i+1:]
         data = np.vstack((data[:i], data[i+1:]))
         assert np.isnan(errors[i]).all()
         errors = np.vstack((errors[:i], errors[i+1:]))
         continue
      i += 1
   i = 0
   for (col_label, data_col) in zip(col_labels, data.T):
      if np.isnan(data_col).all():
         col_labels = col_labels[:i] + col_labels[i + 1:]
         data = (np.vstack((data.T[:i], data.T[i + 1:]))).T
         assert np.isnan(errors.T[i]).all()
         errors = (np.vstack((errors.T[:i], errors.T[i+1:]))).T
         continue
      i += 1

#  fig.set_figheight(1.5*len(col_labels))
#  fig.set_figwidth(1.0*len(row_labels))
   fig.set_figwidth(20)
   fig.set_figheight(20)

   # Add col. with averages over each row (each model)
#  col_labels += ['mean']
#  data = np.hstack((data, np.expand_dims(data.mean(axis=1), 1)))

   im, cbar = heatmap(data, row_labels, col_labels, ax=ax,
                      cmap="YlGn", cbarlabel=cbarlabel)

   class CellFormatter(object):
      def __init__(self, errors):
         self.errors = errors
      def func(self, x, pos):
         if np.isnan(x) or np.isnan(errors[pos]):
#           print(x, errors[pos])

            # Turns out the data entry is "masked" while the error entry is nan
#           assert np.isnan(x) and np.isnan(errors[pos])
#           if not np.isnan(x) and np.isnan(errors[pos]):
#              TT()
            return '--'
         x_str = "{:.1f}".format(x)
         err = errors[pos]
         x_str = x_str + "  ± {:.1f}".format(err)
         return x_str
   cf = CellFormatter(errors)

   texts = annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(cf.func))
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
   opts.add_argument(
      '--eval_baseline_maps_only',
      help='Only use baseline experiments for evaluation maps.',
      action='store_true',
   )
   opts.add_argument(
      '--re-render_cross_vis',
      help='Re-render the heatmaps resulting from the last cross-visualization. For iterating on the way we render '
           'these cross-vis graphics.',
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
      JOB_TIME = 3
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

   if (EVALUATE or RENDER or VIS_EVALS or VIS_CROSS_EVAL) and not opts.vis_maps:
      # just get the names and configs of experiments in which we are interested (no actual evaluations are run)
      exp_dicts = launch_batch(EXP_NAME, get_exp_info_only=True)
      experiment_configs = [config.EvoNMMO() for ec in exp_dicts]
      [ec.set(*i) for ec, ecd in zip(experiment_configs, exp_dicts) for i in ecd.items()]
      experiment_names = [get_experiment_name(ec) for ec in experiment_configs]
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
