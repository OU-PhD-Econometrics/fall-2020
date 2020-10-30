# Load required packages
using Random, Statistics, LinearAlgebra, DataFrames, DataFramesMeta, CSV, FixedEffectModels, MixedModels

# Tell Julia where to-be-called functions are
include("corrcov.jl")
include("estimatelearning.jl")
include("initlearning.jl")
include("estimatelearningcore.jl")

# read in data that was created in Stata
df = CSV.read("nlswlearn.csv")
ID = df.idcode
N = 4711
T = 15
S = 1
y = df.ln_wage
x = [ones(N*T) df.exper df.exper.^2 df.collgrad df.race.==1]
Choice = df.Choice

# Estimate learning model
bstart,sig,Delta,DeltaCorr = estimatelearning(ID,N,T,S,y,x,Choice); 
#ID is a N*T x 1 vector, 
#N is an integer,
#T is an integer,
#S should be set to 1 for now
#y is a N*T x J matrix of outcomes (with 999 where outcomes are unobserved),
#x is a N*T x K x J covariate tensor
#Choice is a N*T x 1 vector of integers (1, ..., J+1) where J+1 refers to "not receive a signal"

# print estimates
println("sqrt Delta = ",sqrt(Delta))
println("sqrt sig = ",sqrt(sig))
println("betahat = ",bstart)

# compare with FE/RE
dfuse = df[df.ln_wage.!=999,:]
# FE
@show reg(dfuse, @formula(ln_wage ~ 1 + exper*exper + collgrad + race1 + fe(idcode)), Vcov.cluster(:idcode))
# RE
categorical!(dfuse, :idcode)
fm1 = fit(MixedModel, @formula(ln_wage ~ 1 + exper*exper + collgrad + race1 + (1|idcode)), dfuse)

# recover beliefs
sig_eps = .092187
df = @transform(df, signal = :ln_wage .- coef(fm1)[1] .- coef(fm1)[2].*:exper .- coef(fm1)[3]*:collgrad .- coef(fm1)[4].*:race1 .- coef(fm1)[5].*:exper.^2,
                    priorEbelief = zeros(length(:ln_wage)),
                    postrEbelief = zeros(length(:ln_wage)),
                    priorVbelief = 0.106297*ones(length(:ln_wage)),
                    postrVbelief = 0.106297*ones(length(:ln_wage)))

for i = 1:N
    for t=1:T
        rowt = (i-1)*T+t
        row1 = (i-1)*T+t+1
        if df.ln_wage[rowt]==999
            df.signal[rowt] = 0
            df.postrEbelief[rowt] = df.priorEbelief[rowt]
            df.postrVbelief[rowt] = df.priorVbelief[rowt]
        else
            df.postrEbelief[rowt] = df.priorEbelief[rowt]*(sig_eps./(sig_eps + df.priorVbelief[rowt])) + df.signal[rowt]*(df.priorVbelief[rowt])./(sig_eps + df.priorVbelief[rowt])
            df.postrVbelief[rowt] = df.priorVbelief[rowt]*(sig_eps./(sig_eps + df.priorVbelief[rowt]))
        end
        if t<T
            df.priorEbelief[row1] = df.postrEbelief[rowt]
            df.priorVbelief[row1] = df.postrVbelief[rowt]
        end
    end
end
# 
