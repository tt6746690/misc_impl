
nb_start:
	scancel --name tsboard
	sbatch --export=logdir=./infogan/logs ../scripts/tensorboard.sbatch

	sleep 0.5
	squeue -u wpq
	

train:
	sbatch train.sbatch

	sleep 0.5
	squeue -u wpq


local:
	python infogan.py \
		--seed 1 \
		--model_name train_local \
		--gpu_id=1