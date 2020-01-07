tsb:
	cd .. && sbatch --export=logdir=dcgan/logs scripts/tensorboard.sbatch

train:
	sbatch train.sbatch 

clean:
	rm -r figures/ ||:
	rm -r models/  ||:
	rm -r logs/    ||: