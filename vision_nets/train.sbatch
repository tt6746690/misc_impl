#!/bin/bash

#SBATCH --job-name=vision_nets
#SBATCH --output=/data/vision/polina/scratch/wpq/misc_impl/vision_nets/train.out

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3GB

#SBATCH --array=1-2

if [[ ${SLURM_ARRAY_TASK_ID} == "" ]]; then
  SLURM_ARRAY_TASK_ID=1
fi

source ~/.zshrc ||:
conda init zsh
conda activate misc_impl

cd /data/vision/polina/scratch/wpq/misc_impl/vision_nets

if [[ ${SLURM_ARRAY_TASK_ID} == "1" ]]; then
  python train_resgan.py \
   --model_name resgan_use_sn=True \
   --gpu_id=$SLURM_JOB_GPUS
   --use_sn
fi


if [[ ${SLURM_ARRAY_TASK_ID} == "2" ]]; then
    python train_resgan.py \
    --model_name resgan_use_sn=False \
    --gpu_id=$SLURM_JOB_GPUS
fi
