
## conda 

create_env:
	conda env create -f misc_impl.yml

update_env:
	conda env update -f misc_impl.yml --prune

notebook:
	conda activate misc_impl
	cd notebook
	jupyter notebook --no-browser --port=8888

create_ipykernel:
	# set kernels in jupyter for an env
	python -m ipykernel install --user --name misc_impl --display-name "Python (misc_impl)"