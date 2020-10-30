function initlearning(N,T,J,Choice,y,x)

    bigN    = zeros(Int64,J)
    bstart  = zeros(size(x,2),J)
    resid   = Array{Float64}[]
    abilsub = Array{Float64}[]
    Resid   = zeros(size(y))
    Csum    = zeros(N*S,J)
    tresid  = zeros(N*S,J)
    for j=1:J
        bigN[j]                  = sum(Choice.==j) # later sum needs to be weighted by PTypesl
        flag                     = y[:,j].!=999
        bstart[:,j]              = x[flag,:,j]\y[flag,j]
        push!(resid  ,y[flag,j].-x[flag,:,j]*bstart[:,j])
        push!(abilsub,y[flag,j].-x[flag,:,j]*bstart[:,j])
        Resid[vec(Choice.==j),j] = resid[j]
        Csum[:,j]                = (sum(reshape(Choice.==j,(T,N*S));dims=1))'
        tresid[:,j]              = (sum(reshape(Resid[:,j],(T,N*S));dims=1))'
    end
    
    abil = tresid./(Csum.+eps())
    abil = kron(abil,ones(T,1))

    Psi1 = deepcopy(Csum)
    
    for j=1:J
        resid[j]               .+= abil[vec(Choice.==j),j]
        Resid[vec(Choice.==j),j] = resid[j]
    end
    
    sig2=zeros(500,J)
    cov2=zeros(500,convert(Int64,(J+1)*J/2))

    sigtemp = zeros(J,1)
    covtemp = zeros(J)

    return bigN,bstart,resid,abilsub,Resid,Csum,tresid,Psi1,sig2,cov2,sigtemp,covtemp
end
