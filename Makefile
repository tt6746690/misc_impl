
## conda 

create_env:
	conda env create -f misc_impl.yml

update_env:
	conda env update -f misc_impl.yml --prune

notebook:
	conda activate misc_impl
	cd notebook
	jupyter notebook --no-browser --port=8888


## gif

vae_gif:
	mkdir vae/gifs ||:
	
	# delay 4 -> 100/4 = 25fps
	convert -delay 5 -loop 0 `ls -v vae/figure/latent_space_vae_epochs=*.png` 		        vae/gifs/latent_space_vae.gif
	convert -delay 5 -loop 0 `ls -v vae/figure/latent_sample_decoded_vae_epochs=*.png`      vae/gifs/latent_sample_decoded_vae.gif
	convert -delay 5 -loop 0 `ls -v vae/figure/decode_along_a_lattice_vae_c=3*.png` 		vae/gifs/decode_along_a_lattice_vae_c=3.gif

	convert -delay 5 -loop 0 `ls -v vae/figure/latent_space_cvae_epochs=*.png` 		        vae/gifs/latent_space_cvae.gif
	convert -delay 5 -loop 0 `ls -v vae/figure/latent_sample_decoded_cvae_epochs=*.png`     vae/gifs/latent_sample_decoded_cvae.gif
	convert -delay 5 -loop 0 `ls -v vae/figure/decode_along_a_lattice_cvae_c=3*.png` 		vae/gifs/decode_along_a_lattice_cvae_c=3.gif
	
