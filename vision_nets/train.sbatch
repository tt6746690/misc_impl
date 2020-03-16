#!/bin/bash

#SBATCH --job-name=v_nets
#SBATCH --output=/data/vision/polina/scratch/wpq/misc_impl/vision_nets/train.out

#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3GB

#SBATCH --array=1-3

if [[ ${SLURM_ARRAY_TASK_ID} == "" ]]; then
  SLURM_ARRAY_TASK_ID=1
fi

source ~/.zshrc ||:
conda init zsh
conda activate misc_impl

cd /data/vision/polina/scratch/wpq/misc_impl/vision_nets


## Compare spectral normalization 
#
# if [[ ${SLURM_ARRAY_TASK_ID} == "1" ]]; then
#   python train_resgan.py \
#    --model_name resgan_use_sn=True \
#    --gpu_id=$SLURM_JOB_GPUS
#    --use_sn
# fi
# if [[ ${SLURM_ARRAY_TASK_ID} == "2" ]]; then
#     python train_resgan.py \
#     --model_name resgan_use_sn=False \
#     --gpu_id=$SLURM_JOB_GPUS
# fi

## Testing conditioning methods
#
if [[ ${SLURM_ARRAY_TASK_ID} == "1" ]]; then
  python train_resgan.py \
    --model_name resgan_conditional_both \
    --use_sn \
    --conditional_D \
    --conditional_G \
    --seed=1
fi
if [[ ${SLURM_ARRAY_TASK_ID} == "2" ]]; then
  python train_resgan.py \
    --model_name resgan_conditional_D_concat_G \
    --use_sn \
    --conditional_D \
    --include_c_in_z \
    --seed=1
fi
if [[ ${SLURM_ARRAY_TASK_ID} == "3" ]]; then
  python train_resgan.py \
    --model_name resgan_conditional_D \
    --use_sn \
    --conditional_D \
    --seed=1
fi
