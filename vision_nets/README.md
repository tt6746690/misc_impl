


## Vision NetWorks


+ torchvision models 
    + https://github.com/pytorch/vision/tree/master/torchvision/models


## GAN w/ residual block 


#### with/without spectral normalization


<p align="center">
    <img width='200', height='200' src="assets/resgan_use_sn=True.gif">
    <img width='200', height='200' src="assets/resgan_use_sn=True_epoch=0.png">
    <img width='200', height='200' src="assets/resgan_use_sn=True_epoch=2.png">
    <img width='200', height='200' src="assets/resgan_use_sn=True_epoch=4.png">
</p>


<p align="center">
    <img width='200', height='200' src="assets/resgan_use_sn=False.gif">
    <img width='200', height='200' src="assets/resgan_use_sn=False_epoch=0.png">
    <img width='200', height='200' src="assets/resgan_use_sn=False_epoch=2.png">
    <img width='200', height='200' src="assets/resgan_use_sn=False_epoch=4.png">
</p>


#### Compare conditional generator (both use conditional discriminator w/ projection)


##### Conditional Batchnorm

<p align="center">
    <img width='200', height='200' src="assets/resgan_conditional_both.gif">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=0.png">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=3.png">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=6.png">
</p>
<p align="center">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=9.png">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=12.png">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=15.png">
    <img width='200', height='200' src="assets/resgan_conditional_both_epoch=18.png">
</p>

##### Concat c to z

<p align="center">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G.gif">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=0.png">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=3.png">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=6.png">
</p>
<p align="center">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=9.png">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=12.png">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=15.png">
    <img width='200', height='200' src="assets/resgan_conditional_D_concat_G_epoch=18.png">
</p>

