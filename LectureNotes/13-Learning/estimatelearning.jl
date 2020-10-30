function estimatelearning(ID,N,T,S,y,x,Choice)

    # PTypesl   = PTypes(:)
    # PTypeg    = PType(Cl==1)
    # PTypen    = PType(Cl==2)
    # PType4s   = PType(Cl==3)
    # PType4h   = PType(Cl==4)
    # PType2    = PType(Cl==5)

    iteration = 1
    J = convert(Int64,length(unique(Choice))-1)

    Delta = rand(J,J)
    Delta = 0.5.*(Delta+Delta')
    @assert Delta == Delta'

    sig = rand(J)

    BigN,bstart,resid,abilsub,Resid,Csum,tresid,Psi1,sig2,cov2,sigtemp,covtemp = initlearning(N,T,J,Choice,y,x)

    j=1

    while maximum(maximum(abs.(covtemp.-Delta)))>1e-4
        sigtemp=sig
        covtemp=Delta
        
        bstart,Delta,sig,Resid,resid,tresid,abilsub = estimatelearningcore(y,x,Choice,bstart,N,T,S,J,BigN,Delta,sig,Resid,resid,tresid,Psi1,abilsub,Csum)
        
        sig2[j,:]=sig'
        cov2[j,:]=LowerTriangular(Delta)[LowerTriangular(Delta).!=0]
        
        println(j)
        println(maximum(maximum(abs.(covtemp.-Delta))))
        j+=1
    end
    Delta_corr = corrcov(Delta)

    return bstart,sig,Delta,Delta_corr
end
