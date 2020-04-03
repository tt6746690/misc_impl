using Pkg; Pkg.activate(".")
using PyPlot
using LinearAlgebra
using RDatasets

# Install Dependencies
#
# Pkg.add("LIBSVM")
# Pkg.add("RDatasets")
# Pkg.precompile()

include("./src/Opt.jl")
import .Opt

############################
# Data
############################

iris = dataset("datasets", "iris")

categories = Dict(
    "setosa"     => 0,
    "versicolor" => -1,
    "virginica"  => 1
)

Y = convert(Vector, iris[:Species])
X = convert(Array, iris[:, 1:4])
X = [X ones(size(X,1),1)]

# Binary Classification
is = findall(y -> y!="setosa", Y)
X, Y = X[is,:], Y[is]
Y = map((x -> categories[x]), Y)

X_train, X_test = X[1:2:end,:], X[2:2:end,:]
Y_train, Y_test = Y[1:2:end], Y[2:2:end,:]

@show size(X_train), size(X_test)


############################
# problem setup
############################

function svm(X, Y, λ)
    
    function f(x)
        0.5norm(x[1:end-1])^2 + λ*sum(max.(0, 1 .- Y.*(X*x)))
    end

    function grad!(g, x)
        is = findall(z -> z>0, 1 .- Y.*(X*x))
        g[1:end-1] .= x[1:end-1] .- λ*dropdims(sum(Y[is].*X[is,1:end-1], dims=1), dims=1)
        g[end:end] .= -λ*sum(Y[is], dims=1)
    end
    
    return f, grad!
end


λ = 1
f, grad! = svm(X_train,Y_train,λ)

############################
# optimization
############################

n = size(X,2)-1
x0 = rand(n+1)
α = 0.001
n_iterations = 1000

fs = zeros(n_iterations)
grad_norm = zeros(n_iterations)

function access_state(state)
    fs[state.k] = state.f
    grad_norm[state.k] = norm(state.g)
    if mod(state.k, 100) == 0
        ŷ = map(y -> (y > 0) ? 1 : -1, X_test*state.x)
        println("(k=$(state.k)) \t misclassify: $(sum(Y_test .!= ŷ)) / $(size(Y_test,1))")
    end
end

Opt.gradient_descent(x0, f, grad!;
    α=α, n_iterations = n_iterations, access_state=access_state)


############################
# plot
############################

figure(figsize=(5,4))
ax = subplot()
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
plot(1:n_iterations, fs)
ylabel("ℓ")
xlabel("k")
title("SVM on Iris dataset")
savefig("summary/assets/svm_subgradient.png")