
restart_tsboard:
	scancel --name tsboard
	sbatch --export=logdir=./opt/logs ../scripts/tensorboard.sbatch

	sleep 0.5
	squeue -u wpq


clean:
	rm -r logs ||:
	rm -r models ||:
	rm -r figures ||:

logs_clean:
	rm -r logs ||:

gif:
	mkdir gifs ||:
	
	# delay 10 -> 100/10 = 10fps
	# delay 20 -> 100/20 = 5fps

	convert -delay 10 -loop 0 `ls -v figures/mnist_cgan/*.png` gifs/mnist_cgan_training.gif
	convert gifs/mnist_cgan_training.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/mnist_cgan_training.gif
