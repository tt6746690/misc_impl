

+ implementation for 
    + [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)

## tutorials

+ blog post on variational bound on MI https://www.inference.vc/infogan-variational-bound-on-mutual-information-twice/
+ there is a paper on variational lower bound of MI very recently https://arxiv.org/pdf/1905.06922.pdf
+ parameter tuning mattered https://github.com/taeoh-kim/Pytorch_InfoGAN

## things didn't work 

+ expect lower bound on MI to approach `H((c1,c2)) ~= 3.7` for 1 `c_1 ~ Cat({1/10})` `c_2 ~ N(0,1)`, but this never happend, instead loss for Q approach 0
+ network stopped generating better images after ~8 epochs, after which the generated image becomes quite noisy

#### Visualize Generator during Training

<p align="center">
  <img width='224', height='224' src="gifs/visualize_training.gif">
</p>
 
+ varying categorical latent code along column; varying continuous latent code along row

