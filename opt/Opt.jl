module Opt

import LinearAlgebra: norm

include("gd.jl")
export GradientDescentState, GradientDescent

include("test_func.jl")
export rosenbrock, rosenbrock_grad!, rosenbrock_hess!

end