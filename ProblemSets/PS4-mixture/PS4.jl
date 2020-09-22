using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables,Distributions
include("lgwt.jl")
#data
function PS4()
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code;

#Question 1
#from his solution:
function mlogit_with_Z(theta, X, Z, y)
        
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

#values from PS3
startvals=[0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243];
#look at this to see why you might need this: http://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/autodiff/
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward);

theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))

#Hessian from ps3
theta_hat_mle_ad = theta_hat_optim_ad.minimizer
println(theta_hat_mle_ad)
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([theta_hat_mle_ad theta_hat_mle_ad_se])

#question 2
# now it is positive which makes more sense because increase in log wage should increase utility

#question 3
#part a



#part b
#1
nodes, weights = lgwt(7,-10,10);
d=Normal(0,2)
sum(weights.*pdf.(d,nodes))
println(sum((nodes.^2).*weights.*pdf.(d,nodes)))

#2
nodes, weights = lgwt(10,-10,10);
sum(weights.*pdf.(d,nodes))
println(sum((nodes.^2).*weights.*pdf.(d,nodes)))

#3
# it is close, with more quadrature points it will converge to 4

#Part C
#1
d=Normal(0,2)
N=1000000
x=rand(Uniform(-10, 10 ),N)

println((20/N)*sum(x.^2 .*(pdf.(d,x))))

#2
println((20/N)*sum(x .*(pdf.(d,x))))

#3
println((20/N)*sum(pdf.(d,x)))



#4
#the difference is around 0.2 which is high in my opinion

#Question 4
function mxlogit_with_quadrature(theta, X, Z, y,rn)
        
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
        
        nodes, weights = lgwt(rn,-4,4);
        d=Normal(0,1)
        
    
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = sum(exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma).*weights.*pdf.(d,nodes))
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        
        loglike = -sum( log.(P.^bigY) )
        
        return loglike
    end



#Question 5
function mxlogit_with_simulation(theta, X, Z, y,rn)
        
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
        
        
        d=Normal(0,1)
        rx=rand(Uniform(0,1 ),rn)
    
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = sum(exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma).*(pdf.(d,x)))
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        
        loglike = -sum( log.(P.^bigY) )
        
        return loglike
    end


end

PS4()