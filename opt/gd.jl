mutable struct GradientDescentState
    # iteration
    k::Int64
    # iterates
    x::AbstractVector{Float64}
    # f(x)
    f::Float64
    # ∇f(x)
    g::AbstractVector{Float64}

    function GradientDescentState(x0)
        n = size(x0, 1)
        x = similar(x0)
        g = similar(x0)
        new(0, x, 0, g)
    end
end

function gradient_descent(
    x0::AbstractVector{Float64},
    f,
    grad!;
    α,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    s = GradientDescentState(x0)
    
    for k in 1:n_iterations
        
        s.k, s.f = k, f(x)
        s.x .= x
        grad!(s.g, x)

        access_state(s)

        @. x = s.x + α*(-s.g)

        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(x)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end