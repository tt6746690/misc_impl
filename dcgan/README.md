

+ implementation for
    + [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

```
sbatch --export=logdir=dcgan/logs scripts/tensorboard.sbatch
```


## take-aways

+ initialization mattered !
    + the normal distribution initialization worked better than default initialization
+ the number of feature layers for conv layers mattered 
    + changing from 128 -> 64 worked on MNIST