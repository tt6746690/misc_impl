

nbconvert:
	jupyter nbconvert --to script note_apx.ipynb

test:
	/Users/wpq/opt/miniconda3/envs/misc_impl/bin/python3 gpax_test.py -v

pytest:
	python -m pytest -v -n 2 --durations=10

pytest_time:
	python -m pytest -v --durations=10
	python -m pytest --collect-only


install_jax:
	# pip install --upgrade jax==0.2.3 jaxlib==0.1.56+cuda100 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	# pip install flax==0.2.2
	# https://pypi.org/project/flax/#history
	# flax >0.3.0 requires jax>0.2.6

	pip install --upgrade jax jaxlib==0.1.66+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	pip install flax
	

autoremove:
	pip install pip-autoremove
	pip-autoremove jax jaxlib flax

uninstall_jax:
	pip install pip-autoremove
	pip-autoremove jax jaxlib -y

rsync_assets_r2l:
	rsync -a wasabi.csail.mit.edu:/data/vision/polina/scratch/wpq/github/misc_impl/gp/summary/assets/ summary/assets/