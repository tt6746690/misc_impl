#!/bin/bash
#SBATCH --job-name=jupyt
#SBATCH --output=/data/vision/polina/scratch/wpq/medical_vision/results/jupyter.out

#SBATCH --time=1000:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=chili
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5GB

source ~/.bashrc
conda init zsh
conda activate pseg

export XDG_RUNTIME_DIR=""

cd /data/vision/polina/scratch/wpq/medical_vision/notebooks

jupyter notebook --no-browser --port=8889