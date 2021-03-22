
## conda 

remove_env:
	conda env remove --name misc_impl

create_env:
	conda env create -f misc_impl.yml

update_env:
	conda env update -f misc_impl.yml --prune

pip_install:
	pip install -r requirements.txt

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

install_cuda:
	# Install cuda helper: https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
	sudo cp cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
	sudo apt-get update
	sudo apt-get -y install cuda

	# driver+software: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-nvidia-driver-and-cuda-software
	#

	# post-install: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
	# 