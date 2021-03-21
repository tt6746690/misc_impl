#!/bin/bash
#SBATCH --job-name=jpt_mi
#SBATCH --output=/data/vision/polina/scratch/wpq/github/misc_impl/scripts/jupyter.out

#SBATCH --time=1000-00:00:00
#SBATCH --partition=gpu
##SBATCH --gres=gpu:2
#SBATCH --nodelist=perilla
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1GB

source ~/.bashrc
conda init zsh
conda activate misc_impl

export XDG_RUNTIME_DIR=""

cd /data/vision/polina/scratch/wpq/github/misc_impl

jupyter notebook --no-browser --port=8898
