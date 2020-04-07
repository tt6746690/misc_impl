
restart_tsboard:
	scancel --name tsboard
	sbatch --export=logdir=./pe/logs ../scripts/tensorboard.sbatch

	sleep 0.5
	squeue -u wpq
	

train:
	sbatch train.sbatch

	sleep 0.5
	squeue -u wpq


local:
	python acgan.py \
		--seed 1 \
		--model_name train_local \
		--gpu_id=1



clean:
	rm -r logs ||:
	rm -r models ||:
	rm -r figures ||:

logs_clean:
	rm -r logs ||:

gif:
	mkdir assets ||:
	
	# delay 10 -> 100/10 = 10fps
	# delay 20 -> 100/20 = 5fps

	convert -delay 10 -loop 0 `ls -v figures/cgan_jpt/*.png` assets/pe_training.gif
	convert assets/pe_training.gif -coalesce -resize 224x224 -layers OptimizeFrame assets/pe_training.gif
