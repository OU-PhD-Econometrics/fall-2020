# For this PS3, I primary used Prof.Ransom's code as a referecne point to write down my code
# Also, I tried to write explantation with each line of code to better understand what is going on
# Also Prof. Ransom mentioned that its extremely imp to visualize what is going on before writng a loop - This has helped me quite a bit

cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\PS3-gev")
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables, Optim, HTTP, GLM, ForwardDiff

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: All Wrapping Function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function problemset_3()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: Multinomial logit with Alternate Specific Covariates
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"

df = CSV.read(HTTP.get(url).body)

X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

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
#  Since it is a linear-log model, The coefficient gamma/100 represents the change in utility with a 1% increase in wage.

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: Nested Logit Model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function nested_logit(alpha, X, Z, y, nesting_structure)

    beta = alpha[1:end-3]
    lambda = alpha[end-2:end-1]
    gamma = alpha[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [repeat(beta[1:K],1,length(nesting_structure[1])) repeat(beta[K+1:2K],1,length(nesting_structure[2])) zeros(K)]
    # Betas here vary with the nest structure

    T = promote_type(eltype(X),eltype(alpha))
    num   = zeros(T,N,J)
    lidx  = zeros(T,N,J) # Linear Index: Numerator is divided by lambda in choice probabilities
    # lidx is a part of the numerator without the exponent
    dem   = zeros(T,N)

    for j=1:J
        if j in nesting_structure[1]
            lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1] )
        elseif j in nesting_structure[2]
            lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2] )
        else
            lidx[:,j] = exp.(zeros(N))
        end
    end

    for j=1:J
        if j in nesting_structure[1]
            num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[1][:]];dims=2).^(lambda[1]-1)
        elseif j in nesting_structure[2]
            num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[2][:]];dims=2).^(lambda[2]-1)
            # only sum up the colums of the nesting structure
        else
            num[:,j] = lidx[:,j]
        end

        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)

    loglike = -sum( bigY.*log.(P) )

    return loglike
end

nesting_structure = [[1 2 3], [4 5 6 7]]
startvals = [2*rand(2*size(X,2)).-1; 1; 1; .1]

td2 = TwiceDifferentiable(alpha -> nested_logit(alpha, X, Z, y, nesting_structure), startvals; autodiff = :forward)

# Using Optim
α_hat_nlogit = optimize(td2, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
α_hat_nlogit_ad = α_hat_nlogit.minimizer

# Estimating Hessian
H  = Optim.hessian!(td2, α_hat_nlogit_ad)
α_hat_nlogit_ad_se = sqrt.(diag(inv(H)))

println([α_hat_nlogit_ad α_hat_nlogit_ad_se])

return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: All Wrapping Function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
problemset_3()
