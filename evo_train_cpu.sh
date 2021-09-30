#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=nmmo_0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=nmmo_0_%j.out

cd /scratch/se2161/neural-mmo || exit

#conda init bash
source activate
conda activate nmmo

export TUNE_RESULT_DIR='./evo_experiment/'
python ForgeEvo.py --load_arguments 0

# need a blank space say the elders

