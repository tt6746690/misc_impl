function rosenbrock(dim)
    dim == 2 || error("only supports 2d Rosenbrock for now")

    function f(x)
        # global min: [1 1]
       100*(x[2] - x[1]^2)^2 + (1. - x[1])^2 
    end
    
    function grad!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end
    
    function hess!(H, x)
        H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
        H[1, 2] = -400.0 * x[1]
        H[2, 1] = -400.0 * x[1]
        H[2, 2] = 200.0
    end
    
    return f, grad!, hess!
end


function hard_leastsquares_problem(n)
    D = spdiagm(n, n+1, 0 => -ones(n-1), 1 => ones(n-1))
    b = sprand(n+1, 0.5)
    
    function f(x)
        """ f(x) = ½‖Dᵀx-b‖² where 
                D ∈ ℝ^n×(n+1) is differencing matrix 
                b ∈ ℝ^(n+1) is offset vector
        """
        (1/2.)*norm(D'*x - b)^2
    end
    
    DDᵀ = D*D'
    Db = D*b
    
    function grad!(g, x)
        """ ∇f(x) = D(Dᵀx -b) """
        g .= DDᵀ * x - Db
    end

    dense_DDᵀ = Array(DDᵀ)
    
    function hess!(h, x)
        h .= dense_DDᵀ
    end
    
    return f, grad!, hess!
end