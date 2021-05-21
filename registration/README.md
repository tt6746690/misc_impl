

+ code
    + unc https://github.com/uncbiag/registration https://github.com/uncbiag/registration
    + lddmm-ot https://github.com/jeanfeydy/lddmm-ot/projects
        + seems out of date
    + geomloss https://www.kernel-operations.io/geomloss/
        + related to OT multiscale etc ... but had to understand i thin k
    + pylddmm simple impl https://github.com/SteffenCzolbe/pyLDDMM
    + POT https://pythonot.github.io/quickstart.html




ideas
+ visualize outliers by plotting dual potential 
    + see which ones have higher potential, suggest ways to improve
+ interdomain gp for modeling velocity + momentum field ... 
    + is there benefit for doing this ? 
+ parameter estimation 
    + is there parameter to the discrete ot distance ? 
+ dataset 
    + https://github.com/gpeyre/2015-SIGGRAPH-convolutional-ot/blob/master/data/images/faces/face4.png
    + https://dabi.temple.edu/external/shape/MPEG7/dataset.html
+ grid idea
    + warped coordinate space use grid + sinkhorn iteration can be implemented efficiently using fft. is it possible to evolve the grid back in time (should be possible, since we use diffeomorphism) 