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
	# delay 20 -> 100/20 = 5fps

	convert -delay 5 -loop 0 `ls -v figures/dcgan_seed=1/dcgan*.png`	gifs/dcgan_stronger_G.gif
	convert gifs/dcgan_stronger_G.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/dcgan_stronger_G_small.gif

	convert -delay 50 -loop 0 `ls -v ../../pytorch-examples/dcgan/out/fake*.png`	gifs/dcgan_example_impl.gif

	convert -delay 10 -loop 0 `ls -v figures/single_dim_varying_z/*.png` gifs/single_dim_varying_z_0.05.gif
	convert -delay 10 -loop 0 `ls -v figures/single_dim_varying_z_labeled/*.png` gifs/single_dim_varying_labeled_z_0.05.gif
	convert gifs/single_dim_varying_z_0.05.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/single_dim_varying_z_0.05_small.gif


	convert -delay 10 -loop 0 `ls -v figures/all_dim_varying_z/*.png` gifs/all_dim_varying_z_1.gif
	convert -delay 10 -loop 0 `ls -v figures/all_dim_varying_z_labeled/*.png` gifs/all_dim_varying_labeled_z_1.gif
	convert gifs/all_dim_varying_z_1.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/all_dim_varying_z_1_small.gif

	convert -delay 10 -loop 0 `ls -v figures/interpolate/*.png` gifs/interpolate.gif
	convert gifs/interpolate.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/interpolate_small.gif