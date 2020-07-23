using CSV, DataFrames, Statistics, GLM
df = CSV.read("Data/slides3data.csv"; missingstrings=["NA"])
size(df)

describe(df)

N = size(df,1)
β = [1.65,.4,.06,-.0002]
σ = .4;
df.exper = df.age .- ( 18*(1 .- df.collgrad) .+ 22*df.collgrad )
df.lwsim = β[1] .+ β[2]*df.collgrad .+ β[3]*df.exper .+ β[4]*df.exper.^2 .+ σ*randn(N)
df.lw    = log.(df.wage)

describe(df;cols=[:lw,:lwsim])

β̂ = lm(@formula(lw ~ collgrad + exper + exper^2), df[df.employed.==1,:])
df.elwage = predict(β̂, df) # generates expected log wage for all observations
r2(β̂)                               # reports R2
sqrt(deviance(β̂)/dof_residual(β̂))  # reports root mean squared error

α̂ = glm(@formula(collgrad ~ parent_college + efc), df, Binomial(), LogitLink())
γ̂ = glm(@formula(employed ~ elwage + numkids), df, Binomial(), LogitLink())

df_cfl     = deepcopy(df)
df_cfl.efc = df.efc .- 1         # change value of efc to be $1,000 less
df.basesch = predict(α̂, df)     # generates expected log wage for all observations
df.cflsch  = predict(α̂, df_cfl) # generates expected log wage for all observations

describe(df;cols=[:basesch,:cflsch])
