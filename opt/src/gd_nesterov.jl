mutable struct GradientDescentNesterovState
    # iteration
    k::Int64
    # iterates xᵏ → x*, {yᵏ}
    x::AbstractVector{Float64}
    y::AbstractVector{Float64}
    # f(xᵏ)
    f::Float64
    # ∇f(yᵏ)
    g::AbstractVector{Float64}

    function GradientDescentNesterovState(x0)
        n = size(x0, 1)
        x = similar(x0)
        y = similar(x0)
        g = similar(x0)
        new(0, x, y, 0, g)
    end
end

function gradient_descent_nesterov(
    x0::AbstractVector{Float64},
    f,
    grad!;
    α,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    y = copy(x0)
    s = GradientDescentNesterovState(x0)
    
    for k in 1:n_iterations
    
        s.k, s.f = k, f(x)
        s.x .= x
        s.y .= y
        grad!(s.g, y)
    
        access_state(s)
    
        @. x = s.y + α*(-s.g)
        @. y = x + (k-1)/(k+2)*(x - s.x)
    
        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(yᵏ)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end