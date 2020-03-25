
nb_start:
	scancel --name tsboard
	sbatch --export=logdir=./cgan/logs ../scripts/tensorboard.sbatch

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
	mkdir assets ||:
	
	# delay 10 -> 100/10 = 10fps
	# delay 20 -> 100/20 = 5fps

	# convert -delay 10 -loop 0 `ls -v figures/resgan_use_sn=False/*.png` assets/resgan_use_sn=False.gif
	# convert assets/resgan_use_sn=False.gif -coalesce -resize 224x224 -layers OptimizeFrame assets/resgan_use_sn=False.gif

	# convert -delay 10 -loop 0 `ls -v figures/resgan_use_sn=True/*.png` assets/resgan_use_sn=True.gif
	# convert assets/resgan_use_sn=True.gif -coalesce -resize 224x224 -layers OptimizeFrame assets/resgan_use_sn=True.gif


	convert -delay 10 -loop 0 `ls -v figures/resgan_conditional_both/*.png` assets/resgan_conditional_both.gif
	convert assets/resgan_conditional_both.gif -coalesce -resize 224x224 -layers OptimizeFrame assets/resgan_conditional_both.gif


	convert -delay 10 -loop 0 `ls -v figures/resgan_conditional_D_concat_G/*.png` assets/resgan_conditional_D_concat_G.gif
	convert assets/resgan_conditional_D_concat_G.gif -coalesce -resize 224x224 -layers OptimizeFrame assets/resgan_conditional_D_concat_G.gif



assets:

	mkdir assets ||:

	cp figures/resgan_use_sn=True/resgan_use_sn=True_fake_samples_epoch=0_it=99.png assets/resgan_use_sn=True_epoch=0.png
	cp figures/resgan_use_sn=True/resgan_use_sn=True_fake_samples_epoch=1_it=99.png assets/resgan_use_sn=True_epoch=1.png
	cp figures/resgan_use_sn=True/resgan_use_sn=True_fake_samples_epoch=2_it=99.png assets/resgan_use_sn=True_epoch=2.png
	cp figures/resgan_use_sn=True/resgan_use_sn=True_fake_samples_epoch=3_it=99.png assets/resgan_use_sn=True_epoch=3.png
	cp figures/resgan_use_sn=True/resgan_use_sn=True_fake_samples_epoch=4_it=99.png assets/resgan_use_sn=True_epoch=4.png

	cp figures/resgan_use_sn=False/resgan_use_sn=False_fake_samples_epoch=0_it=99.png assets/resgan_use_sn=False_epoch=0.png
	cp figures/resgan_use_sn=False/resgan_use_sn=False_fake_samples_epoch=1_it=99.png assets/resgan_use_sn=False_epoch=1.png
	cp figures/resgan_use_sn=False/resgan_use_sn=False_fake_samples_epoch=2_it=99.png assets/resgan_use_sn=False_epoch=2.png
	cp figures/resgan_use_sn=False/resgan_use_sn=False_fake_samples_epoch=3_it=99.png assets/resgan_use_sn=False_epoch=3.png
	cp figures/resgan_use_sn=False/resgan_use_sn=False_fake_samples_epoch=4_it=99.png assets/resgan_use_sn=False_epoch=4.png



	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=0_it=199.png assets/resgan_conditional_both_epoch=0.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=3_it=199.png assets/resgan_conditional_both_epoch=3.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=6_it=199.png assets/resgan_conditional_both_epoch=6.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=9_it=199.png assets/resgan_conditional_both_epoch=9.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=12_it=199.png assets/resgan_conditional_both_epoch=12.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=15_it=199.png assets/resgan_conditional_both_epoch=15.png
	cp figures/resgan_conditional_both/resgan_conditional_both_epoch=18_it=199.png assets/resgan_conditional_both_epoch=18.png


	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=0_it=199.png assets/resgan_conditional_D_concat_G_epoch=0.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=3_it=199.png assets/resgan_conditional_D_concat_G_epoch=3.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=6_it=199.png assets/resgan_conditional_D_concat_G_epoch=6.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=9_it=199.png assets/resgan_conditional_D_concat_G_epoch=9.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=12_it=199.png assets/resgan_conditional_D_concat_G_epoch=12.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=15_it=199.png assets/resgan_conditional_D_concat_G_epoch=15.png
	cp figures/resgan_conditional_D_concat_G/resgan_conditional_D_concat_G_epoch=18_it=199.png assets/resgan_conditional_D_concat_G_epoch=18.png