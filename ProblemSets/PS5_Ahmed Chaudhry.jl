# Worked with Kasra and Waleed #
cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\PS5-ddc")
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables, Optim, HTTP, GLM, ForwardDiff, DataFramesMeta


function PS5()
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/
master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: Reshape Data into Long
#:::::::::::::::::::::::::::::::::::::::::::::::::::
df = @transform(df, bus_id = 1:size(df,1))

dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: Solving Using GLM
#:::::::::::::::::::::::::::::::::::::::::::::::::::

theta_hat = glm(@formula(Y ~ Odometer + Branded ), df_long, Binomial(), LogitLink())
println(theta_hat)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: Dynamic Estimation
#:::::::::::::::::::::::::::::::::::::::::::::::::::

### Part (A) ###

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)

Y=df[:,r"Y"]
Xst=df[:,r"Xst"]
Odo=df[:,r"Odo"]
Zst=df.Zst
B=df.Branded

### Part (B) ###

zval,zbin,xval,xbin,xtran = create_grids()

### Part (C) ###

@views @inbounds function myfun(initial_theta)
    N=size(xtran,1)
    T=size(Y,2)
    FV=zeros(N,2,T+1)
    beta=0.9
    for t in T:-1:1
    for b in 0:1
    for z in 1:zbin
    for x in 1:xbin
    row=x+(z-1)*xbin
    v1=initial_theta[1]+initial_theta[2]*xval[x]+initial_theta[3]*b+ beta*xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
    v0=beta*xtran[1+(z-1)*xbin,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
    FV[row,b+1,t]=beta*log(exp(v1)+exp(v0))
    end
    end
    end
    end

### Part (D) to Part (G)###
    loglike=0
    #P=zeros(size(Y,1),T,2)
    for y in size(Y,1)
        for t in 1:T
            r1=1+ (Zst[y]-1)*xbin
            r0=Xst[y,t]+ (Zst[y]-1)*xbin
            U=initial_theta[1] + initial_theta[2]*Xst[y,t] + initial_theta[3]*B[y] +(xtran[r1,:].-xtran[r0, :])'*FV[r0:r0+xbin-1, B[y]+1, t+1]
            P=exp(U)/(1+exp(U))
            loglike+=Y[y,t]log(P)+(Y[y,t]-1)log(1-P)
        end
    end
    return -loglike
end

optimizer_ = optimize(theta -> myfun(theta), initial_theta, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))

end
