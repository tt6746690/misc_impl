
nb_start:
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

gif:
	mkdir gifs ||:
	
	# delay 10 -> 100/10 = 10fps
	# delay 20 -> 100/20 = 5fps



	convert -delay 10 -loop 0 `ls -v figures/mnist_cgan/*.png` gifs/mnist_cgan_training.gif
	convert gifs/mnist_cgan_training.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/mnist_cgan_training.gif
