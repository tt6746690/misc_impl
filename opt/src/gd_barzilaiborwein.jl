mutable struct GradientDescentBarzilaiBorweinState
    # iteration
    k::Int64
    # iterates
    x_prev::AbstractVector{Float64}
    x::AbstractVector{Float64}
    # f(x)
    f::Float64
    # ∇f(x)
    g_prev::AbstractVector{Float64}
    g::AbstractVector{Float64}
    # stepsize α
    α::Float64

    function GradientDescentBarzilaiBorweinState(x0)
        n = size(x0, 1)
        x, x_prev = similar(x0), similar(x0)
        g, g_prev = similar(x0), similar(x0)
        new(0, x_prev, x, 0, g_prev, g, 0.)
    end
end

function Base.show(io::IO, s::GradientDescentBarzilaiBorweinState)
    print("k=$(s.k) \t f(x)=$(round(s.f,sigdigits=5)) \t α=$(s.α)")
end


function gradient_descent_barzilaiborwein(
    x0::AbstractVector{Float64},
    f,
    grad!;
    α₀ = 0.00001,
    barzilaiborwein_type = 1,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8,
    varargs...)

    x = copy(x0)
    g = copy(x0); grad!(g, x)
    u, v = zeros(size(x0)), zeros(size(x0))
    
    s = GradientDescentBarzilaiBorweinState(x0)
    grad!(s.g, x)
    
    for k in 1:n_iterations
    
        s.k, s.f = k, f(x)
        @. s.x_prev = s.x
        @. s.x = x
        @. s.g_prev = s.g
        @. s.g = g
    
        if k == 1
            # should do line search
            s.α = α₀
        else
            u = s.x - s.x_prev
            v = s.g - s.g_prev
            s.α = (barzilaiborwein_type == 1) ?
                dot(u, v) / norm(v)^2 :
                norm(u)^2 / dot(u, v)
        end
    
        access_state(s)        
    
        @. x = s.x + s.α*(-s.g)
        grad!(g, x)
    
        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(xᵏ)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end