function estimatelearningcore(y,x,Choice,bstart,N,T,S,J,BigN,Delta,sig,Resid,resid,tresid,Psi1,abilsub,Csum)
    abil=zeros(N*S,J)
    
    idelta   = inv(Delta)
    vtemp2   = zeros(J,J)
    vabil    = zeros(N*S,J)
    vabilw   = zeros(N*S,J)
    for j=1:J
        tresid[:,j] = (sum(reshape(Resid[:,j],(T,N*S));dims=1))'
    end
    
    for i=1:S*N
        psit=Psi1[i,:]
        Psi = zeros(J,J)
        for j=1:J
            Psi[j,j] = psit[j]./sig[j]
        end
        
        vtemp=inv(idelta.+Psi)
        
        vectres = zeros(J)
        for j=1:J
            vectres[j] = tresid[i,j]./sig[j]
        end
        temp=(vtemp*vectres)'
        abil[i,:]=temp
        
        for j=1:J
            vabil[i,j] = vtemp[j,j] # later need PTypesl[i]*vtemp[j,j]
        end
        
        vtemp2.+=(vtemp.+temp'*temp) # later need PTypesl[i]*(vtemp+temp'*temp)
    end
    
    Delta=vtemp2./N
    
    vabilw    = deepcopy(vabil)
    Abil      = kron(abil,ones(T,1))
    Vabil     = kron(vabil,ones(T,1))
    
    sigdem = zeros(J)
    for j=1:J
        abilsub[j] = Abil[vec(Choice.==j),j]
        sigdem[j]  = sum((resid[j].-abilsub[j]).^2) # later need sum(PTypesub[j].*(resid[j].-abilsub[j]).^2) 
    end
    
    sig=((sum(Csum.*vabilw;dims=1))'.+sigdem)./BigN
    
    for j=1:J
        flag                     = (y[:,j].!=999)
        bstart[:,j]              = x[flag,:,j]\(y[flag,j].-abilsub[j]) # later weight by PTypesub[j]
        resid[j]                 = y[flag,j].-x[flag,:,j]*bstart[:,j]
        Resid[vec(Choice.==j),j] = resid[j]
    end

    return bstart,Delta,sig,Resid,resid,tresid,abilsub 
end
