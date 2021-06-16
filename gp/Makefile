

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
	pip install --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	pip install flax

autoremove:
	pip install pip-autoremove
	pip-autoremove jax jaxlib flax

uninstall_jax:
	pip install pip-autoremove
	pip-autoremove jax jaxlib -y
	