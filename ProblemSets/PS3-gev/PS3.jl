using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function ps3()
	#Question 1
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
	df = CSV.read(HTTP.get(url).body)
	X = [df.age df.white df.collgrad]
	Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
	df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
	y = df.occupation;
	

	#Question 2
	#pass the concatenated array
	C= [X Z]
	function mnlogit(alpha, x, d)
		c=[i==j ? 1 : 0 for i in d , j in 1:6]
		log_mat=[log(exp.(x[i,:]'*alpha[:,j])/(1+sum(exp.(x[i,:]'*alpha)))) for i in 1:size(x,1) ,j in 1:6]
		loglike=-sum(c.*log_mat)

		return loglike
	end

	beta_mnlog = optimize(b -> mnlogit(b, C, y), rand(C,6), LBFGS(), Optim.Options(g_tol=1e-7, iterations=100_000, show_trace=true))
	#println(beta_mnlog.minimizer)
	
	#Question 4

	df.white = df.race.==1
	α̂ = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
	println(α̂)

	#Question 5
	freqtable(df, :occupation) # note small number of obs in some occupations
	df = dropmissing(df, :occupation)
	df[df.occupation.>7,:occupation] .= 7
	freqtable(df, :occupation) # problem solved

	X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
	y = df.occupation

	function mnlogit(alpha, x, d)
		c=[i==j ? 1 : 0 for i in d , j in 1:6]
		log_mat=[log(exp.(x[i,:]'*alpha[:,j])/(1+sum(exp.(x[i,:]'*alpha)))) for i in 1:size(x,1) ,j in 1:6]
		loglike=-sum(c.*log_mat)

		return loglike
	end

	beta_mnlog = optimize(b -> mnlogit(b, X, y), rand(size(X,2),6), LBFGS(), Optim.Options(g_tol=1e-7, iterations=100_000, show_trace=true))
	println(beta_mnlog.minimizer)
end


ps2()
