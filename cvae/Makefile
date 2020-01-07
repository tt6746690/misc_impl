

## gif

vae_gif:
	mkdir vae/gifs ||:
	
	# delay 4 -> 100/4 = 25fps
	convert -delay 5 -loop 0 `ls -v figure/latent_space_vae_epochs=*.png` 		        gifs/latent_space_vae.gif
	convert -delay 5 -loop 0 `ls -v figure/latent_sample_decoded_vae_epochs=*.png`      gifs/latent_sample_decoded_vae.gif
	convert -delay 5 -loop 0 `ls -v figure/decode_along_a_lattice_vae_c=3*.png` 		gifs/decode_along_a_lattice_vae_c=3.gif

	convert -delay 5 -loop 0 `ls -v figure/latent_space_cvae_epochs=*.png` 		        gifs/latent_space_cvae.gif
	convert -delay 5 -loop 0 `ls -v figure/latent_sample_decoded_cvae_epochs=*.png`     gifs/latent_sample_decoded_cvae.gif
	convert -delay 5 -loop 0 `ls -v figure/decode_along_a_lattice_cvae_c=3*.png` 		gifs/decode_along_a_lattice_cvae_c=3.gif
	
