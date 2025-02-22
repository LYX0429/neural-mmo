#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=nmmo_0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yl8616@nyu.edu
#SBATCH --output=nmmo_0_%j.out

#cd /scratch/se2161/neural-mmo || exit

conda init bash
conda activate nmmo

export TUNE_RESULT_DIR='./evo_experiment/'
python ForgeEvo.py --load_arguments 0

# gotta leave a trailing line here apparently
