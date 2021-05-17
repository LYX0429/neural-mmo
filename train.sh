#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=nmmo
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo%j.out

cd /scratch/se2161/neural-mmo

#conda init bash
source activate

exp_name='div_all_pair_prims_ES_0'
export TUNE_RESULT_DIR='./evo_experiment/'${exp_name}
#make onespawn_div_combat_pair_prims_ES
#make paired_ES
#make div_all_pair_prims_ES
#make div_all_pair_tile_ES
make div_all_pair_cppn_ES
#make pretrain_vanilla
