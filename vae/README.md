

## tutorials
 

+ pytorch sequential tips
    + https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict

## Take-aways


+ initialization mattered a lot
    + if initialize linear layer weights to [0,1] (instead of [-std,std] where std=1/sqrt(weights.size(0))), will results in mean, covariance being too large. loss on prior distribution too large, hard to converge
+ bernoulli loss
    + should use sum of pixel-wise loss, instead of mean of pixel-wise loss
+ numerical issues with log Gaussian pdf
    + order of evaluation mattered 


## Questions

+ approximate posterior should be Gaussian, by construction. Why would the true posterior p_z|x looks like Gaussian as well ?
