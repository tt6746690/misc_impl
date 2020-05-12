



RSYNC = rsync
RSYNCTAGS = --archive --update --verbose --info=progress2
HOST_FOLDER = $(HOME)/github/misc_impl/knockoff
REMOTE = wpq@chili.csail.mit.edu
REMOTE_FOLDER = /data/vision/polina/scratch/wpq/misc_impl/knockoff

sync:
	$(RSYNC) $(RSYNCTAGS) $(HOST_FOLDER)/ $(REMOTE):$(REMOTE_FOLDER)
syncr:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(REMOTE_FOLDER)/ $(HOST_FOLDER)



nb_start:
	scancel --name tsboard
	sbatch --export=logdir=./knockoff/logs ../scripts/tensorboard.sbatch

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



clean:
	rm -r logs ||:
	rm -r models ||:
	rm -r figures ||:

gif:
	mkdir gifs ||:
	
	convert -delay 10 -loop 0 `ls -v figures/infogan_seed=3_2020.02.14-02:07:54/*.png` gifs/visualize_training.gif
	convert gifs/visualize_training.gif -coalesce -resize 224x224 -layers OptimizeFrame gifs/visualize_training.gif