mutable struct GradientDescentState
    # iteration
    k::Int64
    # iterates
    x::Array{Float64, 1}
    # f(x)
    f::Float64
    # ∇f(x)
    g::Array{Float64, 1}

    function GradientDescentState(n)
        new(0, zeros(n), 0, zeros(n))
    end
end

function gradient_descent(
    x0::Array{Float64, 1},
    f,
    grad!;
    α,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    s = GradientDescentState(size(x, 1))
    
    for k in 1:n_iterations

        s.k, s.x = k, x
        s.f = f(x)
        grad!(s.g, x)

        access_state(s)

        x = s.x + α*(-s.g)

        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(x)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end