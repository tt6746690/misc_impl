
mutable struct GradientDescentState
    # iteration
    k::Int64
    # iterates
    x::Array{Float64, 1}
    # f(x)
    f::Float64
    # ∇f(x)
    g::Array{Float64, 1}

    function GradientDescentState(n, x0)
        new(0, x0, 0, zeros(n))
    end
    function GradientDescentState(n)
        new(0, zeros(n), 0, zeros(n))
    end
end


function GradientDescent(
    x0::Array{Float64, 1},
    f,
    grad!,
    alpha;
    n_iterations = nothing,
    access_state = (state -> nothing),
    g_abstol = 1e-8)

    x = copy(x0)
    s = Opt.GradientDescentState(2, x0)
    
    for k in 1:n_iterations
        
        s.k, s.x, s.f = k, x, f(x)
        grad!(s.g, x)
        access_state(s)        

        x = s.x + alpha*(-s.g)

        if norm(s.g) <= g_abstol
            println("Terminate at k=$k: |∇f(x)| = $(norm(s.g)) <= $g_abstol")
            break
        end
    end
end