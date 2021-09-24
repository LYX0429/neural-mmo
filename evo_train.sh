#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=nmmo0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo0_%j.out

cd /scratch/se2161/neural-mmo || exit

conda init bash
conda activate nmmo0

export TUNE_RESULT_DIR='./evo_experiment/'
python Forge.py evaluate --config TreeOrerock --MAP fit-Lifespans_skills-NONE_gene-Simplex_algo-MAP-Elites_PAIRED_0 --INFER_IDX "(5, 4, 0)" --MODEL fit-Lifespans_skills-NONE_gene-Simplex_algo-MAP-Elites_PAIRED_0 --NPOLICIES 1 --NPOP 1 --PAIRED True --EVALUATION_HORIZON 100 --N_EVAL 1 --NEW_EVAL --SKILLS "['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]" --NENT 16 --FITNESS_METRIC Lifespans  --EVO_DIR 0

# gotta leave a trailing line here apparently

