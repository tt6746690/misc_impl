

+ implementation for
    + [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)



<p align="center">
  <img src="gifs/dcgan_stronger_G_small.gif">
</p>


## take-aways

+ initialization mattered !
    + the normal distribution initialization worked better than default initialization
+ the number of feature layers for conv layers mattered 
    + changing from 128 -> 64 worked on MNIST
+ a stronger (more feature layers) generator definitely helped with convergence


## tutorials

+ https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
    + large kernel more filters
    + look at the gradients !
        + want generator conv weights to have large gradients early in training
    + flip labels (generated=1, data=0)
+ https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
    + various variants of cost function
    + impl tips
        + scale between -1,1, use tanh as output layer for generator
        + add noise to real/generated image before feeding into discrimiantor