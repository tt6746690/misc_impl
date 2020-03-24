






## Resources

+ NAG impl 
    + https://github.com/GRYE/Nesterov-accelerated-gradient-descent/blob/master/nesterov_method.py
    + https://github.com/idc9/optimization_algos/blob/master/opt_algos/accelerated_gradient_descent.py
+ setup julia https://github.com/mitmath/julia-mit
+ slides on impl http://www.princeton.edu/~yc5/ele522_optimization/lectures/accelerated_gradient.pdf


```
# 1. Install julia v1.0.5 @ https://julialang.org/downloads/

# 2. Open julia repl
$ julia

# 3. Enter package mode
julia> ]

# 4. Activate package 
(v0.7) pkg> activate .

# 5. Download dependencies
(opt) pkg> instantiate

# 6. Update package and precompile modules
(opt) pkg> update; precompile

# 7. Back to julia repl, and start hacking!
julia> using opt
```

```
# IJulia https://github.com/JuliaLang/IJulia.jl
# Install additional julia kernel, i.e. pass cmd args
using IJulia
installkernel("Julia nodeps", "--depwarn=no")
installkernel("Julia (4 threads)", env=Dict("JULIA_NUM_THREADS"=>"4"))
```