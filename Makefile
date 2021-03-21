
## conda 

remove_env:
	conda env remove --name misc_impl

create_env:
	conda env create -f misc_impl.yml

update_env:
	conda env update -f misc_impl.yml --prune

pip_install:
	pip3 install -r requirements.txt

notebook:
	conda activate misc_impl
	cd notebook
	jupyter notebook --no-browser --port=8888

create_ipykernel:
	# set kernels in jupyter for an env
	python -m ipykernel install --user --name misc_impl --display-name "Python (misc_impl)"

remove_ipykernel:
	jupyter kernelspec uninstall misc_impl

update_jaxlib:
	pip install --upgrade jax jaxlib==0.1.64+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
