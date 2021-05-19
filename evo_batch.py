'''
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from dead processes.
'''
import os
import copy
import json
import re
import argparse
import numpy as np
from pdb import set_trace as TT

genomes = [
    'Random',
#   'CPPN',
#   'Pattern',
#   'CA',
#   'LSystem',
#   'All',
]
fitness_funcs = [
#   'Lifespans',
    'L2',
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
    [20,20],
]

def launch_batch(exp_name):
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

   if LOCAL:
       default_config['n_generations'] = 1
   i = 0

   for gene in genomes:
      for fit_func in fitness_funcs:
         for skillset in skills:
            if fit_func in ['Lifespans', 'Sum']:
               if skillset != 'ALL':
                  continue
               skillset = ['NONE']

            for algo in algos:
               for me_bins in me_bin_sizes:
                  if algo != 'MAP-Elites' and not (np.array(me_bins) == 1).all():
                     continue
                  if (np.array(me_bins) == 1).all():
                     items_per_bin = 12
                     feature_calc = None
                  else:
                     items_per_bin = 1
                     feature_calc = 'map_entropy'


                  # Edit the sbatch file to load the correct config file
                  with open(sbatch_file, 'r') as f:
                     content = f.read()
                     if not EVALUATE:
                        new_cmd = 'python ForgeEvo.py --load_arguments {}'.format(i)
                     else:
                        new_cmd = 'python Forge.py evaluate -la {}'.format(i)
                     content_0 = re.sub('nmmo\d*', 'nmmo{}'.format(i), content)
                     new_content = re.sub('python Forge.*', new_cmd, content_0)

                  with open(sbatch_file, 'w') as f:
                     f.write(new_content)
                  # Write the config file with the desired settings
                  exp_config = copy.deepcopy(default_config)
                  exp_config.update({
                     'N_GENERATIONS': 100000,
                     'TERRAIN_SIZE': 70,
                     'NENT': 16,
                  })
                  if not EVALUATE:
                     exp_config.update({
                        'GENOME': gene,
                        'FITNESS_METRIC': fit_func,
                        'EVO_ALGO': algo,
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
                  if EVALUATE:
                     # TODO: use function to get experiment names based on parameters so that we can cross-evaluate among the batch (all models on all maps)
                     exp_config.update({
                        'EVALUATION_HORIZON': 100,
                        'N_EVAL': 1,
                        'NEW_EVAL': True,
                     })
                  if CUDA:
                     exp_config.update({
                        'N_EVO_MAPS': 12,
                        'N_PROC': 12,
                     })

                  if LOCAL:
                     exp_config.update({
                        'N_GENERATIONS': 100,
                        'N_EVO_MAPS': 8,
                        'N_PROC': 8,
                        'EVO_SAVE_INTERVAL': 10,
                     })
                  print('Saving experiment config:\n{}'.format(exp_config))
                  with open('configs/settings_{}.json'.format(i), 'w') as f:
                     json.dump(exp_config, f, ensure_ascii=False, indent=4)
                  # Launch the experiment. It should load the saved settings

                  if LOCAL:
                     os.system('python ForgeEvo.py --load_arguments {}'.format(i))
                     os.system('ray stop')
                  else:
                     os.system('sbatch {}'.format(sbatch_file))
                  i += 1

   if TRAIN_BASELINE:
      # Finally, launch a baseline
      with open(sbatch_file, 'r') as f:
         content = f.read()
         if not EVALUATE:
            new_cmd = 'python Forge.py train --config TreeOrerock --MODEL None --TRAIN_HORIZON 100 --NUM_WORKERS 12 --NENT 16 --TERRAIN_SIZE 70'
         else:
            new_cmd = 'python Forge.py evaluate -la {}'.format(i)
         content = re.sub('nmmo\d*', 'nmmo00', content)
         new_content = re.sub('python Forge.*', new_cmd, content)

      with open(sbatch_file, 'w') as f:
         f.write(new_content)

      if LOCAL:
         os.system(new_cmd)
         os.system('ray stop')
      else:
         os.system('sbatch {}'.format(sbatch_file))

if __name__ == '__main__':
   opts = argparse.ArgumentParser(
      description='Launch a batch of experiments/evaluations for evo-pcgrl')

   opts.add_argument(
       '-ex',
       '--experiment_name',
       help='A name to be shared by the batch of experiments.',
       default='test_0',
   )
   opts.add_argument(
       '-ev',
       '--evaluate',
       help='Evaluate a batch of evolution experiments.',
       action='store_true',
   )
   opts.add_argument(
       '-t',
       '--local',
       help='Test the batch script, i.e. run it on a local machine and evolve for minimal number of generations.',
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
   opts = opts.parse_args()
   EXP_NAME = opts.experiment_name
   EVALUATE = opts.evaluate
   LOCAL = opts.local
   TRAIN_BASELINE = opts.train_baseline
   CUDA = opts.gpu

   launch_batch(EXP_NAME)
