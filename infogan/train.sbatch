#!/bin/bash

#SBATCH --job-name=infogan
#SBATCH --output=/data/vision/polina/scratch/wpq/misc_impl/infogan/train.out

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3GB

#SBATCH --array=1

if [[ ${SLURM_ARRAY_TASK_ID} == "" ]]; then
  SLURM_ARRAY_TASK_ID=1
fi

source ~/.zshrc ||:
conda init zsh
conda activate misc_impl

cd /data/vision/polina/scratch/wpq/misc_impl/infogan

python infogan.py \
   --seed $SLURM_ARRAY_TASK_ID \
   --model_name infogan_seed=$SLURM_ARRAY_TASK_ID \
   --gpu_id=$SLURM_JOB_GPUS