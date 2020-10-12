function mlogit_with_Z(θ, X, Z, y)
    α = θ[1:end-1]
    γ = θ[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigα = [reshape(α,K,J-1) zeros(K)]
    
    P = exp.(X*bigα .+ Z*γ) ./ sum.(eachrow(exp.(X*bigα .+ Z*γ)))
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end
