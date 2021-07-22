#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=nmmo0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo0_%j.out

cd /scratch/se2161/neural-mmo || exit

#conda init bash
source activate

export TUNE_RESULT_DIR='./evo_experiment/'
python Forge.py evaluate --config TreeOrerock --MAP fit-Differential_skills-ALL_gene-CA_algo-MAP-Elites_0 --INFER_IDX "(43, 32, 0)" --MODEL ['fit-Differential_skills-ALL_gene-Simplex_algo-MAP-Elites_0','fit-Differential_skills-ALL_gene-CA_algo-MAP-Elites_0'] --NPOLICIES 2 --NPOP 2 --EVALUATION_HORIZON 500 --N_EVAL 2 --NEW_EVAL --SKILLS "['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]" --NENT 16 --EVO_DIR 0

#make onespawn_div_combat_pair_prims_ES
#make paired_ES
#make div_all_pair_prims_ES
#make div_all_pair_tile_ES
#make div_all_pair_cppn_ES
#make pretrain_vanilla
