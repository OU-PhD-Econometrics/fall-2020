#Worked with Kasra and Waleed#

cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\PS2-optimization-intro")
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables, Optim, HTTP, GLM

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6: Wrapping Around a Function#
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function ProblemSet_2()
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, LBFGS())

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3: Using Optim for Logit Regression#
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)
    loglike_function=[d[i]*log((1+exp((X*alpha)[i])))+(1-d[i])*log(1-(1+exp((X*alpha)[i]))) for i in 1:size(d,1)]
    # P_[i1]=1+exp(X*alpha) #
    loglike=-sum(loglike_function)
    return loglike
end

alpha_loglike = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(alpha_loglike.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4: Using GLM for Logit Regression#
#:::::::::::::::::::::::::::::::::::::::::::::::::::
df.white = df.race.==1
α_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(α_glm)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5: Multinomial Logistic Regression#
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mnlogit(alpha, x, d)
    d=[i==j ? 1 : 0 for i in d , j in 1:6]
    loglike_function2=[log(exp.(x[i,:]'*alpha[:,j])/(1+sum(exp.(x[i,:]'*alpha)))) for i in 1:size(x,1) ,j in 1:6]
    loglike_multi=-sum(d.*loglike_function2)
    return loglike_multi
end

alpha_mnlog = optimize(b -> mnlogit(b, X, y), rand(size(X,2),6), LBFGS(), Optim.Options(g_tol=1e-7, iterations=100_000, show_trace=true))
println(alpha_mnlog.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6: Wrapping Around a Function#
#:::::::::::::::::::::::::::::::::::::::::::::::::::
end
ProblemSet_2()

                            #The End#
