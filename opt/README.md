






## Resources

+ Optim.jl nonconvex optimization https://julianlsolvers.github.io/Optim.jl/stable/#
+ GD impl 
    + matlab https://github.com/hiroyuki-kasai/GDLibrary
+ NAG blog
    + strongly convex proof https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
+ NAG impl 
    + https://github.com/GRYE/Nesterov-accelerated-gradient-descent/blob/master/nesterov_method.py
    + https://github.com/idc9/optimization_algos/blob/master/opt_algos/accelerated_gradient_descent.py
+ setup julia https://github.com/mitmath/julia-mit
+ slides on impl http://www.princeton.edu/~yc5/ele522_optimization/lectures/accelerated_gradient.pdf

## Setup

```
# Install julia v1.0.5 @ https://julialang.org/downloads/

# Open julia repl
$ julia

# Enter package mode
julia> ]

# Activate package 
(v0.7) pkg> activate .

# Download dependencies
(opt) pkg> instantiate

# Update package and precompile modules
(opt) pkg> update; precompile

# Back to julia repl, and start hacking!
julia> using opt

# condigure PyCall to use conda envs
julia> ENV["CONDA_JL_HOME"] = "/data/vision/polina/shared_software/miniconda3/envs/misc_impl
(opt) pkg> build Conda
```

```
# IJulia https://github.com/JuliaLang/IJulia.jl
# Install additional julia kernel, i.e. pass cmd args
using IJulia
installkernel("Julia nodeps", "--depwarn=no")
installkernel("Julia (4 threads)", env=Dict("JULIA_NUM_THREADS"=>"4"))
```