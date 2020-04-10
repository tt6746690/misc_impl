module Opt

import Base
import LinearAlgebra: norm, dot, pinv
import SparseArrays: spdiagm, sprand, spzeros

include("gd.jl")
export GradientDescentState, gradient_descent

include("gd_barzilaiborwein.jl")
export GradientDescentBarzilaiBorweinState, gradient_descent_barzilaiborwein

include("gd_nesterov.jl")
export GradientDescentNesterovState, gradient_descent_nesterov

include("newton.jl")
export NewtonMethodState, newton_method

include("test_func.jl")
export rosenbrock, hard_leastsquares_problem


end