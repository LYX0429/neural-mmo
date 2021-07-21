#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=nmmo2
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo2_%j.out

cd /scratch/se2161/neural-mmo || exit

#conda init bash
source activate
conda activate nmmo

export TUNE_RESULT_DIR='./evo_experiment/'
python Forge.py evaluate --config treeorerock --model fit-L2_skills-ALL_gene-Random_algo-MAP-Elites_0 --map fit-L2_skills-ALL_gene-Random_algo-MAP-Elites_0 --infer_idx "(18, 17, 0)" --EVALUATION_HORIZON 100 --N_EVAL 20 --NEW_EVAL --SKILLS "['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]"

