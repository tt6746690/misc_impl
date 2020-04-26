
+ implementation for 
    + [Panning for Gold: Model-X Knockoffs for High-dimensional Controlled Variable Selection](https://arxiv.org/abs/1610.02351) 



## resources


+ knockoffgan https://github.com/firmai/tsgan/blob/master/alg/knockoffgan/KnockoffGAN.py
+ invase https://github.com/jsyoon0823/INVASE
+ glmnet_python https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html#References

## Setup

```
## Add exception for Gatekeeper
#  https://osxdaily.com/2015/07/15/add-remove-gatekeeper-app-command-line-mac-os-x/
find . -name "*.mexmaci64" -print -exec spctl --add {} \;
## glmnet_matlab precompiled to work with MATLAB2020a
#  https://github.com/growlix/glmnet_matlab
```