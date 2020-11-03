using JuMP, Ipopt, Optim, LineSearches, LinearAlgebra, SparseArrays, Distributions, DataFrames, CSV, HTTP

function dense_hessian(hessian_sparsity, V, n)
    I = [i for (i,j) in hessian_sparsity]
    J = [j for (i,j) in hessian_sparsity]
    raw = sparse(I, J, V, n, n)
    return Matrix(raw + raw' - sparse(diagm(0=>diag(raw))))
end

function wrapper()
    # Let's read in the data from PS8
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS8-factor/nlsy.csv"
    df = CSV.read(HTTP.get(url).body)
    X = [df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr ones(size(df,1),1)]
    y = df.logwage

    # first let's do unconstrained optimization
    function reg_mle(θ, X, y)
        # first K elements are the coefficients of the outcome equation
        β = θ[1:end-1]
        # last element is the variance (stdev)
        σ = θ[end]
        # now build the likelihood
        loglike = -sum(-.5 .* ( log(2*π) .+ log(σ^2) .+ ( (y .- X*β)./σ ).^2 ) )
        # more intuitive way? (but JuMP can't use pdf's from Distributions.jl)
        #loglike = -sum( log(1 ./ sqrt(σ^2)) .+ logpdf.(Normal(0,1),(y .- X*β)./sqrt(σ^2)) )
        return loglike
    end

    # run the optimizer for MLE
    svals = vcat(X\y,.5);
    td = TwiceDifferentiable(th -> reg_mle(th, X, y), svals; autodiff = :forward)
    θ̂_optim_ad = optimize(td, svals, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_mle_optim_ad = θ̂_optim_ad.minimizer
    loglikeval = θ̂_optim_ad.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_mle_optim_ad)
    θ̂_mle_optim_ad_se = sqrt.(diag(inv(H)))
    # store results in a data frame
    results = DataFrame(coef_mle = vcat(vec(θ̂_mle_optim_ad),-loglikeval), se_mle = vcat(vec(θ̂_mle_optim_ad_se),missing), coef_ols = vcat(X\y,missing,missing) )

    # now use JuMP (objective function + optimization call is all in one function here)
    function jump_mle(θ₀, X, y)
        # define the model
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, β[j=1:size(X,2)], start = θ₀[j])
        @variable(model, σ, start = θ₀[end])
        @NLobjective(model, Max, sum(-.5 * ( log(2*π) + log(σ^2) + ((y[i] - sum(X[i,j]*β[j] for j in 1:size(X,2)) )/σ)^2 ) for i in 1:size(X,1) ) )
        # optimize the model
        JuMP.optimize!(model)
        # return parameter estimates
        coef_jump = vcat(JuMP.value.(β), JuMP.value(σ), JuMP.objective_value(model) )
        # return Hessian for SEs
        values = coef_jump[1:end-1]
        MOI = JuMP.MathOptInterface
        d = JuMP.NLPEvaluator(model)
        MOI.initialize(d, [:Hess])
        hessian_sparsity = MOI.hessian_lagrangian_structure(d)
        V = zeros(length(hessian_sparsity))
        MOI.eval_hessian_lagrangian(d, V, values, 1.0, Float64[])
        H = dense_hessian(hessian_sparsity, V, length(values))
        se_jump = sqrt.(diag(inv(-H)))
        return coef_jump, se_jump
    end
    jump_coefs,jump_se = jump_mle(svals, X, y)
    results.coef_jump = jump_coefs
    results.se_jump = vcat(jump_se,missing)


    # now let's use JuMP to estimate a constrained optimization
    function jump_cns_mle(θ₀, X, y)
        # define the model
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, β[j=1:size(X,2)], start = θ₀[j])
        @variable(model, σ, start = θ₀[end])
        @constraint(model, β[2] == .16)
        @NLobjective(model, Max, sum(-.5 * ( log(2*π) + log(σ^2) + ((y[i] - sum(X[i,j]*β[j] for j in 1:size(X,2)) )/σ)^2 ) for i in 1:size(X,1) ) )
        # optimize the model
        JuMP.optimize!(model)
        # return parameter estimates
        coef_jump = vcat(JuMP.value.(β), JuMP.value(σ), JuMP.objective_value(model) )
        # return Hessian for SEs
        values = coef_jump[1:end-1]
        MOI = JuMP.MathOptInterface
        d = JuMP.NLPEvaluator(model)
        MOI.initialize(d, [:Hess])
        hessian_sparsity = MOI.hessian_lagrangian_structure(d)
        V = zeros(length(hessian_sparsity))
        MOI.eval_hessian_lagrangian(d, V, values, 1.0, Float64[])
        H = dense_hessian(hessian_sparsity, V, length(values))
        se_jump = sqrt.(diag(inv(-H)))
        return coef_jump, se_jump
    end
    jump_cns_coefs,jump_cns_se = jump_cns_mle(svals, X, y)
    results.coef_jump_cns = jump_cns_coefs
    results.se_jump_cns = vcat(jump_cns_se,missing)
    @show results


    # can we use optim for the same constrained optimization?
    function cns_reg_mle(θ, cns_mat, X, y)
        # first K elements are the coefficients of the outcome equation
        β = θ[1:end-1]
        # last element is the variance (stdev)
        σ = θ[end]
        # impose constraint
        for r in size(cns_mat,1)
            insert!(β,convert(Int64,cns_mat[r,1]),cns_mat[r,2])
        end
        # now build the likelihood
        loglike = -sum(-.5 .* ( log(2*π) .+ log(σ^2) .+ ( (y .- X*β)./σ ).^2 ) )
        # more intuitive way? (but JuMP can't use pdf's from Distributions.jl)
        #loglike = -sum( log(1 ./ sqrt(σ^2)) .+ logpdf.(Normal(0,1),(y .- X*β)./sqrt(σ^2)) )
        return loglike
    end

    # run the optimizer for MLE
    # first, set up constraints
    cns_mat = hcat(2,.16)
    svals = vcat(X\y,.5)
    # constraints are treated as data, so take them out of starting values (they get added back in inside the obj function)
    deleteat!(svals, convert(Int64,cns_mat[1,1]))
    td = TwiceDifferentiable(th -> cns_reg_mle(th, cns_mat, X, y), svals; autodiff = :forward)
    θ̂_optim_ad = optimize(td, svals, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_mle_optim_ad = θ̂_optim_ad.minimizer
    loglikeval = θ̂_optim_ad.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_mle_optim_ad)
    θ̂_mle_optim_ad_se = sqrt.(diag(inv(H)))
    # add back in constraint in both estimates and SEs
    insert!(θ̂_mle_optim_ad   ,convert(Int64,cns_mat[1,1]),cns_mat[1,2])
    insert!(θ̂_mle_optim_ad_se,convert(Int64,cns_mat[1,1]),0)
    # store results in a data frame
    results.coef_optim_cns = vcat(θ̂_mle_optim_ad,-loglikeval)
    results.se_optim_cns   = vcat(θ̂_mle_optim_ad_se,missing)
    @show results


    # now let's use JuMP to estimate another constrained optimization
    function jump_cns2_mle(θ₀, X, y)
        # define the model
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, β[j=1:size(X,2)], start = θ₀[j])
        @variable(model, σ, start = θ₀[end])
        @constraint(model, β[2] == .16)
        @constraint(model, β[4] == 1+2*β[3])
        @NLobjective(model, Max, sum(-.5 * ( log(2*π) + log(σ^2) + ((y[i] - sum(X[i,j]*β[j] for j in 1:size(X,2)) )/σ)^2 ) for i in 1:size(X,1) ) )
        # optimize the model
        JuMP.optimize!(model)
        # return parameter estimates
        coef_jump = vcat(JuMP.value.(β), JuMP.value(σ), JuMP.objective_value(model) )
        # return Hessian for SEs
        values = coef_jump[1:end-1]
        MOI = JuMP.MathOptInterface
        d = JuMP.NLPEvaluator(model)
        MOI.initialize(d, [:Hess])
        hessian_sparsity = MOI.hessian_lagrangian_structure(d)
        V = zeros(length(hessian_sparsity))
        MOI.eval_hessian_lagrangian(d, V, values, 1.0, Float64[])
        H = dense_hessian(hessian_sparsity, V, length(values))
        se_jump = sqrt.(diag(inv(-H)))
        return coef_jump, se_jump
    end
    jump_cns2_coefs,jump_cns2_se = jump_cns2_mle(svals, X, y)
    results.coef_jump_cns2 = jump_cns2_coefs
    results.se_jump_cns2 = vcat(jump_cns2_se,missing)
    @show results


    # can we use optim for the same constrained optimization?
    function cns2_reg_mle(θ, cns_mat, X, y)
        # first K elements are the coefficients of the outcome equation
        β = θ[1:end-1]
        # last element is the variance (stdev)
        σ = θ[end]
        # impose constraints
        for r in 1:size(cns_mat,1)
            idx1 = convert(Int64,cns_mat[r,1])
            idx2 = convert(Int64,cns_mat[r,2])
            if cns_mat[r,3]==0
                insert!(β,idx1,cns_mat[r,5])
            else
                insert!(β,idx1,cns_mat[r,5]+cns_mat[r,4]*β[idx2])
            end
        end
        # now build the likelihood
        loglike = -sum(-.5 .* ( log(2*π) .+ log(σ^2) .+ ( (y .- X*β)./σ ).^2 ) )
        # more intuitive way? (but JuMP can't use pdf's from Distributions.jl)
        #loglike = -sum( log(1 ./ sqrt(σ^2)) .+ logpdf.(Normal(0,1),(y .- X*β)./sqrt(σ^2)) )
        return loglike
    end

    # run the optimizer for MLE
    # now we need additional information: whether the constraint is "type 1" (set equal to fixed value) or "type 2" (set equal to another parameter)
    # first, set up constraints
    #     Type 1  Restricting one parameter ("parmA") to equal a fixed value
    #     Type 2  Restricting one parameter, parmA, to equal another ("parmB"),
    #             potentially multiplied by some real number q and addd to
    #             some constant m, e.g. parmA = m + q*parmB.
    #   
    #   RESTRMAT follows a very specific format. It is an R-by-5 matrix, 
    #   where R is the number of restrictions. The role of each of the four 
    #   columns is as follows
    #
    #   Column 1  The index of parmA
    #   Column 2  The index of parmB (zero if type 1 restriction)
    #   Column 3  Binary vector w   here 0 indciates a type 1 restriction (parmA
    #               set equal to fixed value) and 1 indicates a type 2 
    #               restriction (parmA set equal to parmB)
    #   Column 4  If a type 1 restriction, 0. If     a type 2 restriction, any 
    #               real number q such that parmA = q*parmB.
    #   Column 5  If a type 1 restriction, the fixed value. If a type 2
    #                restriction, any real number m such that parmA = m+q*parmB.
    #   NOTE: parmA should always be a later index than parmB
    cns_mat2 = [2 0 0 0 .16;
                4 3 1 2 1]
    svals = vcat(X\y,.5)
    # constraints are treated as data, so take them out of starting values (they get added back in inside the obj function)
    for r=1:size(cns_mat2,1)
        deleteat!(svals, convert(Int64,cns_mat2[r,1]))
    end
    td = TwiceDifferentiable(th -> cns2_reg_mle(th, cns_mat2, X, y), svals; autodiff = :forward)
    θ̂_optim_ad = optimize(td, svals, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_mle_optim_ad = θ̂_optim_ad.minimizer
    loglikeval = θ̂_optim_ad.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_mle_optim_ad)
    θ̂_mle_optim_ad_se = sqrt.(diag(inv(H)))
    println(θ̂_mle_optim_ad)
    # add back in constraint in both estimates and SEs
    for r in 1:size(cns_mat2,1)
        idx1 = convert(Int64,cns_mat2[r,1])
        idx2 = convert(Int64,cns_mat2[r,2])
        if cns_mat2[r,3]==0
            insert!(θ̂_mle_optim_ad,idx1,cns_mat2[r,5])
            insert!(θ̂_mle_optim_ad_se,idx1,0)
        else
            insert!(θ̂_mle_optim_ad,idx1,cns_mat2[r,5]+cns_mat2[r,4]*θ̂_mle_optim_ad[idx2])
            insert!(θ̂_mle_optim_ad_se,idx1,cns_mat2[r,5]+cns_mat2[r,4]*θ̂_mle_optim_ad_se[idx2]) # this is wrong
        end
        println(θ̂_mle_optim_ad)
    end
    # store results in a data frame
    results.coef_optim_cns2 = vcat(θ̂_mle_optim_ad,-loglikeval)
    results.se_optim_cns2   = vcat(θ̂_mle_optim_ad_se,missing)
    @show results





    # Analytical gradient
    @views @inline function asclogit(bstart::Vector,Y::Array,X::Array,Z::Array,J::Int64,baseAlt::Int64=J,W::Array=ones(length(Y)))

    ## error checking
    @assert ((!isempty(X) || !isempty(Z)) && !isempty(Y))    "You must supply data to the model"
    @assert (ndims(Y)==1 && size(Y,2)==1)                    "Y must be a 1-D Array"
    @assert (minimum(Y)==1 && maximum(Y)==J) "Y should contain integers numbered consecutively from 1 through J"
    if !isempty(X)
        @assert ndims(X)==2          "X must be a 2-dimensional matrix"
        @assert size(X,1)==size(Y,1) "The 1st dimension of X should equal the number of observations in Y"
    end
    if !isempty(Z)
        @assert ndims(Z)==3          "Z must be a 3-dimensional tensor"
        @assert size(Z,1)==size(Y,1) "The 1st dimension of Z should equal the number of observations in Y"
        @assert size(Z,3)==J         "The 3rd dimension of Z should equal the number of choice alternatives"
    end

    K1 = size(X,2)
    K2 = size(Z,2)
    jdx = setdiff(1:J,baseAlt)

    function f(b)
        T = promote_type(promote_type(promote_type(eltype(X),eltype(b)),eltype(Z)),eltype(W))
        num   = zeros(T,size(Y))
        dem   = zeros(T,size(Y))
        temp  = zeros(T,size(Y))
        numer =  ones(T,size(Y,1),J)
        P     = zeros(T,size(Y,1),J)
        ℓ     =  zero(T)
        b2 = b[K1*(J-1)+1:K1*(J-1)+K2]

        k = 1
        for j in 1:J
            if j != baseAlt
                temp       .= X*b[(k-1)*K1+1:k*K1] .+ (Z[:,:,j].-Z[:,:,baseAlt])*b2
                num        .= (Y.==j).*temp.+num
                dem        .+= exp.(temp)
                numer[:,j] .=  exp.(temp)
                k += 1
            end
        end
        dem.+=1
        P   .=numer./(1 .+ sum(numer;dims=2))

        ℓ = -W'*(num.-log.(dem))
    end

    function g!(G,b)
        T     = promote_type(promote_type(promote_type(eltype(X),eltype(b)),eltype(Z)),eltype(W))
        numer = zeros(T,size(Y,1),J)
        P     = zeros(T,size(Y,1),J)
        numg  = zeros(T,K2)
        demg  = zeros(T,K2)
        b2    = b[K1*(J-1)+1:K1*(J-1)+K2]
                                                                                                                                 
        G .= T(0)
        k = 1
        for j in 1:J
            if j != baseAlt
                numer[:,j] .= exp.( X*b[(k-1)*K1+1:k*K1] .+ (Z[:,:,j].-Z[:,:,baseAlt])*b2 )
                k += 1
            end
        end
        P   .=numer./(1 .+ sum(numer;dims=2))

        k = 1
        for j in 1:J
            if j != baseAlt
                G[(k-1)*K1+1:k*K1] .= -X'*(W.*((Y.==j).-P[:,j]))
                k += 1
            end
        end

        for j in 1:J
            if j != baseAlt
                numg .-= (Z[:,:,j].-Z[:,:,baseAlt])'*(W.*(Y.==j))
                demg .-= (Z[:,:,j].-Z[:,:,baseAlt])'*(W.*P[:,j])
            end
        end
        G[K1*(J-1)+1:K1*(J-1)+K2] .= numg.-demg
        return nothing
    end

    td = TwiceDifferentiable(f, g!, bstart, autodiff = :forwarddiff)
    rs = optimize(td, bstart, LBFGS(; linesearch = LineSearches.BackTracking()), Optim.Options(iterations=100_000,g_tol=1e-8,f_tol=1e-8,x_tol=1e-8,show_trace=true))
    β  = Optim.minimizer(rs)
    ℓ  = Optim.minimum(rs)*(-1)
    H  = Optim.hessian!(td, β)
    g  = Optim.gradient!(td, β)
    se = sqrt.(diag(inv(H)))

    return β,se,ℓ,g
    end

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    dff = CSV.read(HTTP.get(url).body)
    XX = [dff.age dff.white dff.collgrad]
    ZZ = cat(dff.elnwage1, dff.elnwage2, dff.elnwage3, dff.elnwage4, 
             dff.elnwage5, dff.elnwage6, dff.elnwage7, dff.elnwage8; dims=3)
    yy = dff.occ_code
    J  = 8
    startvals = [2*rand(7*size(XX,2)).-1; .1]
    β,se,ℓ,g = asclogit(startvals,yy,XX,ZZ,J,J,ones(length(yy)))
    ans = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477]
    dfr = DataFrame(β=β,se=se,answer=ans)
    # time: about 15 seconds
    @show dfr

    @views @inline function mlogit_with_Z(theta, X, Z, y)
        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end
    ZZ = cat(dff.elnwage1, dff.elnwage2, dff.elnwage3, dff.elnwage4, 
             dff.elnwage5, dff.elnwage6, dff.elnwage7, dff.elnwage8; dims=2)
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, XX, ZZ, yy), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(; linesearch = LineSearches.BackTracking()), Optim.Options(iterations=100_000,g_tol=1e-8,f_tol=1e-8,x_tol=1e-8,show_trace=true))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    results = DataFrame(coef = theta_hat_mle_ad, se = theta_hat_mle_ad_se)

    # time: about 70 seconds



    return nothing
end

wrapper()

