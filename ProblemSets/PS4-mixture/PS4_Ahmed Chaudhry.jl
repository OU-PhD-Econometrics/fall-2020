# Worked with Kasra and Waleed #

cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\PS4-mixture")
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables, Optim, HTTP, GLM, ForwardDiff

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: Multinomial logit with Alternate Specific Covariates
#:::::::::::::::::::::::::::::::::::::::::::::::::::

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body)

X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code
# Using PS3's code for Multinomial Logit #
function mlogit(alpha, X, Z, y)
    beta = alpha[1:end-1] # Indexing the parameter so coefficient on X is the first element
    gamma = alpha[end] # Indexing the parameter so coefficient on Z is the last element
    K = size(X,2) # K is the no. of covariates in X, which in this case are 3
    J = length(unique(y)) # J is the no. of coefficients, which are 8 in this case (So J-1 = 7)
    N = length(y) # N is a vector with the same length as y (not sure why we are defining N)
    bigY = zeros(N,J) # Simply defining the matrix of dependent variables Y, which are going to be dummies in case of Multinomial Logit

    for j=1:J # Simply defining j to be something, next step, j represents column
        bigY[:,j] = y.==j # Making a dummies for each of the corresponding occupation in y
    end
    bigBeta = [reshape(beta,K,J-1) zeros(K)] # Reshaping into a matrix with K rows and J-1 columns
    # Adding a column of zeros in the end to make it conform with bigY and also to normalize the parameters of the last occupation to be zero

    T = promote_type(eltype(X),eltype(alpha)) # To get the standard errors we promote the type
    num   = zeros(T,N,J) # Define numerator T-type , N-rows, J-columns
    dem   = zeros(T,N) # Define denominator T-type , N-rows
    for j=1:J
        num[:,j] = exp.(X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma) # Numerator Likelihood function
        dem .+= num[:,j] # Denominator Likelihood function: Simply the Sum of all the numerators
    end

    P = num./repeat(dem,1,J) # Putting together numerator and denominator

    loglike = -sum( bigY.*log.(P) ) # Since Optim minimizes the function, we take its negative to maximize

    return loglike # This is used for storing the function parameters
end

startvals = [2*rand(7*size(X,2)).-1; .1]
td = TwiceDifferentiable(alpha -> mlogit(alpha, X, Z, y), startvals; autodiff = :forward)

# Use Optim #
α_hat_multinomial = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
α_hat_multinomial_ad = α_hat_multinomial.minimizer

# Find Hessian #
H  = Optim.hessian!(td, α_hat_multinomial_ad) # Finding the Hessian Matrix
α_hat_multinomial_ad_se = sqrt.(diag(inv(H))) # Standard Errors of alpha_hat

println([α_hat_multinomial_ad α_hat_multinomial_ad_se])

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: Interpretation of gamma_hat
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# gamma_hat make more sense as compared to PS3 because it's sign is positive #

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3:
#:::::::::::::::::::::::::::::::::::::::::::::::::::

### Part (A) ###
using Distributions
include("lgwt.jl") # make sure the function gets read in

# define distribution
d = Normal(0,1) # mean=0, standard deviation=1

# get quadrature nodes and weights for 7 grid points, with +_ 4SD
nodes, weights = lgwt(7,-4,4)

# now compute the integral over the density and verify it's 1
sum(weights.*pdf.(d,nodes)) # d here is normal dist. as defined earlier #

# now compute the expectation and verify it's 0
sum(weights.*nodes.*pdf.(d,nodes)) # it seems as if the nodes here is x in: intergral x phi(x) dx #

### Part (B) ###

#Practice 1#
d = Normal(0,2)
nodes, weights = lgwt(7,-5,5)
sum(weights.*(nodes.^2).*pdf.(d,nodes))

#Practice 2#
nodes, weights = lgwt(10,-5,5)
sum(weights.*(nodes.^2).*pdf.(d,nodes))

# Comment: Increasing the quadrature points gives us better estimates (of variace of f(x)) #

### Part (C) ###

#Practice 1#
Random.seed!(1234)
d = Normal(0,2)
N = 1000000
nodes, weights = lgwt(10,-5,5)
x = rand(Uniform(-10,10), N)
A = (10-(-10))*(1/N)*sum((x.^2).*pdf(d,x)) # A is approximately equal to 4 #

#Practice 2#
A1 = (10-(-10))*(1/N)*sum((x).*pdf(d,x)) # A1 is approx. equal to 0 #
#Practice 3#
A2 = (10-(-10))*(1/N)*sum(pdf(d,x)) # A2 is approx. equal to 1 #

#Practice 4#
D = 1000
x = rand(Uniform(-10,10), D)

A = (10-(-10))*(1/D)*sum((x.^2).*pdf(d,x))

A1 = (10-(-10))*(1/D)*sum((x).*pdf(d,x))

A2 = (10-(-10))*(1/D)*sum(pdf(d,x))
# Reducing D to 1000 more likely less precise estimates #

### Part (D) ###

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: With Gauss-Lagendre Quadrature
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function mixedlogit_quadrature(theta, X, Z, y, rn)

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


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5: With Monte Carlo Simulations
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function mixedlogit_simulation(theta, X, Z, y,rn)

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

                                ### End ###
