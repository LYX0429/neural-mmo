#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=nmmo0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo0_%j.out

cd /scratch/se2161/neural-mmo || exit

#conda init bash
source activate

export TUNE_RESULT_DIR='./evo_experiment/'
python Forge.py evaluate --config TreeOrerock --MODEL fit-L2_skills-ALL_gene-CA_algo-MAP-Elites_2 --MAP fit-L2_skills-ALL_gene-CA_algo-MAP-Elites_2 --INFER_IDX "(36, 32, 0)" --EVALUATION_HORIZON 100 --N_EVAL 2 --NEW_EVAL --SKILLS "['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]" --NENT 16 --EVO_DIR 2

#make onespawn_div_combat_pair_prims_ES
#make paired_ES
#make div_all_pair_prims_ES
#make div_all_pair_tile_ES
#make div_all_pair_cppn_ES
#make pretrain_vanilla