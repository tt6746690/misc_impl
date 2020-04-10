mutable struct NewtonMethodState
    # iteration
    k::Int64
    # iterates
    x::AbstractVector{Float64}
    # f(x)
    f::Float64
    # ∇f(x)
    g::AbstractVector{Float64}
    # ∇²f(x)
    H::AbstractMatrix{Float64}

    function NewtonMethodState(x0)
        n = size(x0, 1)
        x = similar(x0)
        g = similar(x0)
        H = repeat(x0', outer=size(x0))
        new(0, x, 0, g, H)
    end
end

function newton_method(
    x0::AbstractVector{Float64},
    f,
    grad!,
    hess!;
    α = 1,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    s = NewtonState(x0)
    
    for k in 1:n_iterations
        
        s.k, s.f = k, f(x)
        s.x .= x
        grad!(s.g, x)
        hess!(s.H, x)

        access_state(s)

        @. x = s.x + α*(-pinv(s.H)*s.g)

        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(xᵏ)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end