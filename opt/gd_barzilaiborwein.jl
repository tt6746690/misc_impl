mutable struct GradientDescentBarzilaiBorweinState
    # iteration
    k::Int64
    # iterates
    x_prev::Array{Float64, 1}
    x::Array{Float64, 1}
    # f(x)
    f::Float64
    # ∇f(x)
    g_prev::Array{Float64, 1}
    g::Array{Float64, 1}
    # stepsize α
    α::Float64

    function GradientDescentBarzilaiBorweinState(n)
        new(0, zeros(n), zeros(n), 0., zeros(n), zeros(n), 0.)
    end
end

function Base.show(io::IO, s::GradientDescentBarzilaiBorweinState)
    print("k=$(s.k) \t f(x)=$(round(s.f,sigdigits=5)) \t α=$(s.α)")
end


function gradient_descent_barzilaiborwein(
    x0::Array{Float64, 1},
    f,
    grad!;
    α₀ = 0.00001,
    barzilaiborwein_type = 1,
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    g = copy(x0); grad!(g, x)
    u, v = zeros(size(x0)), zeros(size(x0))
    
    s = GradientDescentBarzilaiBorweinState(size(x, 1))
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
    
        x = s.x + s.α*(-s.g)
        grad!(g, x)
    
        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(x)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end