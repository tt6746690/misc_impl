#!/bin/bash
#SBATCH --job-name=tsboard
#SBATCH --output=/data/vision/polina/scratch/wpq/misc_impl/scripts/tsboard.out

#SBATCH --time=1000-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=chili
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB

source ~/.bashrc
conda init zsh
conda activate misc_impl

cd /data/vision/polina/scratch/wpq/misc_impl/

tensorboard --logdir=$logdir --port=8890
