


+ implementation for
    + [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    + [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)



## VAE 


![](gifs/latent_space_vae.gif)
![](gifs/latent_sample_decoded_vae.gif)
![](gifs/decode_along_a_lattice_vae_c=3.gif)



## CVAE

![](gifs/latent_space_cvae.gif)
![](gifs/latent_sample_decoded_cvae.gif)
![](gifs/decode_along_a_lattice_cvae_c=3.gif)



## Todos


+ interpolate between class labels for cvae
+ 


## tutorials
 

+ pytorch sequential tips
    + https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
+ cvae reference impl   
    + https://github.com/jojonki/AutoEncoders/blob/master/cvae.ipynb
    + https://github.com/Prasanna1991/pytorch-vae/blob/master/cvae.py
+ a nice blog post for vaes 
    + https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html

## Take-aways


+ initialization mattered a lot
    + if initialize linear layer weights to [0,1] (instead of [-std,std] where std=1/sqrt(weights.size(0))), will results in mean, covariance being too large. loss on prior distribution too large, hard to converge
+ bernoulli loss
    + should use sum of pixel-wise loss, instead of mean of pixel-wise loss
+ numerical issues with log Gaussian pdf
    + order of evaluation mattered 


## Questions

+ approximate posterior should be Gaussian, by construction. Why would the true posterior p_z|x looks like Gaussian as well ?
+ in cvae, is it customary to replace recognition network `q_z|x,y` with prior network `p_z|x` ?