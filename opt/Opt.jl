module Opt

import Base
import LinearAlgebra: norm, dot

include("gd.jl")
export GradientDescentState, gradient_descent

include("gd_barzilaiborwein.jl")
export GradientDescentBarzilaiBorweinState, gradient_descent_barzilaiborwein

include("test_func.jl")
export rosenbrock, rosenbrock_grad!, rosenbrock_hess!

end