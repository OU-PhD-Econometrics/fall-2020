using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, ForwardDiff, FreqTables, Distributions

# read in data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# estimate multionomial logit model
include("mlogit_with_Z.jl")
θ_start = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477]
td = TwiceDifferentiable(b -> mlogit_with_Z(b, X, Z, y), θ_start; autodiff = :forward);
θ̂_optim_ad = optimize(td, θ_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50));
θ̂_mle_ad = θ̂_optim_ad.minimizer
H  = Optim.hessian!(td, θ̂_mle_ad)
θ̂_mle_ad_se = sqrt.(diag(inv(H)))

# get baseline predicted probabilities
function plogit(θ, X, Z, J)
    α = θ[1:end-1]
    γ = θ[end]
    K = size(X,2)
    bigα = [reshape(α,K,J-1) zeros(K)]
    P = exp.(X*bigα .+ Z*γ) ./ sum.(eachrow(exp.(X*bigα .+ Z*γ)))
    return P
end
println(θ̂_mle_ad)
P = plogit(θ̂_mle_ad, X, Z, length(unique(y)))

# compare with data
modelfitdf = DataFrame(data_freq = 100*convert(Array, prop(freqtable(df,:occ_code))), P = 100*vec(mean(P'; dims=2)))
display(modelfitdf)

# now run a counterfactual:
# suppose people don't care about wages when choosing occupation
# this means setting θ̂_mle_ad[end] = 0
θ̂_cfl = deepcopy(θ̂_mle_ad)
θ̂_cfl[end] = 0
P_cfl = plogit(θ̂_cfl, X, Z, length(unique(y)))
modelfitdf.P_cfl = 100*vec(mean(P_cfl'; dims=2))
modelfitdf.diff = modelfitdf.P_cfl .- modelfitdf.P
modelfitdf.ew   = vec(mean(Z'; dims=2))
display(modelfitdf)

# now do parametric bootstrap
Random.seed!(1234)
invHfix = UpperTriangular(inv(H)) .- Diagonal(inv(H)) .+ UpperTriangular(inv(H))' # this code makes inv(H) truly symmetric (not just numerically symmetric)
d = MvNormal(θ̂_mle_ad,invHfix)
B = 1000
cfl_bs = zeros(8,B)
for b = 1:B
    θ_draw = rand(d)
    Pb = plogit(θ_draw, X, Z, length(unique(y)))
    Pbs = 100*vec(mean(Pb'; dims=2))
    θ_draw[end] = 0
    P_cflb = plogit(θ_draw, X, Z, length(unique(y)))
    Pcfl_bs = 100*vec(mean(P_cflb'; dims=2))
    cfl_bs[:,b] = Pcfl_bs .- Pbs
end
bsdiffL = deepcopy(cfl_bs[:,1])
bsdiffU = deepcopy(cfl_bs[:,1])
for i=1:size(cfl_bs,1)
    bsdiffL[i] = quantile(cfl_bs[i,:],.025)
    bsdiffU[i] = quantile(cfl_bs[i,:],.975)
end
modelfitdf.bs_diffL = bsdiffL
modelfitdf.bs_diffU = bsdiffU
display(modelfitdf)
