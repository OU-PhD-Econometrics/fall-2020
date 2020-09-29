# I was not able to solve this, hopfully, I will submit the complete solution tomorrow after seeing how Dr.Ransom does it #

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

Y = df[1 : 20]
X_odo = df[21 : 40]
X_st = df[43 : 62]

### Part (B) ###

zval,zbin,xval,xbin,xtran = create_grids()

### Part (E) ###
@views @inbounds function partE()

### Part (E) ###

beta=0.9
T=20



end

end
