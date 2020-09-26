using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data


# [Your turn]


# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])





#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# [Your turn]



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# [ Load in the data ]

Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

# [ convert other data frame columns to matrices]




#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::
zval,zbin,xval,xbin,xtran = create_grids()


