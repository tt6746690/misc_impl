tsb:
	cd .. && sbatch --export=logdir=dcgan/logs scripts/tensorboard.sbatch

train:
	sbatch train.sbatch 

clean:
	rm -r figures/ ||:
	rm -r models/  ||:
	rm -r logs/    ||:
	rm -r gifs/	   ||:

gif:
	mkdir gifs ||:
	
	# delay 10 -> 100/10 = 10fps

	convert -delay 5 -loop 0 `ls -v figures/dcgan_seed=1/dcgan*.png`	gifs/dcgan_stronger_G.gif
	convert gifs/dcgan_stronger_G.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/dcgan_stronger_G_small.gif