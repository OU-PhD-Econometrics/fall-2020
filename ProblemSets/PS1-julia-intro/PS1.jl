using Random, Distributions, JLD2, CSV, DataFrames,LinearAlgebra,Statistics,FreqTables,TexTables
#TexTables wasn't in the instructions but I think it is needed for freqtable
#q1,q2,q3 should run before declaration of q4 since q3 generates "nlsw88.jld"

Random.seed!(1234)
#Question 1
function q1()
    Random.seed!(1234)
    
    #1.a
    #1.a.i
    A=rand(Uniform(-5,10),(10,7))
    
    #1.a.ii
    B=rand(Normal(-2,15),(10,7))
    #1.a.iii
    C= [A[1:5,1:5] B[1:5,6:7]]
    #1.a.iv
    D=(A.<=0).*A
    
    #1.b
    length(A)
    #1.c
    unique(D)
    #1.d
    E=reshape(B,70,1)
    E=vec(B)
    #1.e
    F=cat(A, B;dims=3)
    #1.f
    F=permutedims(F,(3,1,2))
    #1.g
    G=kron(B,C)
    #kron(C,F) is not possible F is not vectorized properly. 
    #1.h
    JLD2.@save "matrixpractice.jld2" A B C D E F G
    #1.i
    JLD2.@save "firstmatrix.jld2" A B C D
    #1.j
    df=DataFrame(C)
    CSV.write("Cmatrix.csv",df)
    #1.k
    dfD=DataFrame(D)
    CSV.write("Dmatrix.dat",df,delim='\t')
    return A,B,C,D
end 



#Question 2
function q2(a,b,c)
    #2.a
    n=length(A)
    AB=zeros(size(A))
    
    for i=1:n
        AB[i]=A[i]*B[i]
    end
    AB2=A.*B

    #2.b
    n=length(C)
    Cprime=[]
    for i=1:n
        if -5<=C[i]<=5
            append!(Cprime,C[i])
        end
    end

    Cprime2=[x for x in C if -5<=x<=5]
    
    #2.c
    X=permutedims(
    cat(ones(15169,5),  
    [rand(Bernoulli(.75*(6-y)/5)) for x in 1:15169,y in 1:5], 
    [rand(Normal(15+y-1,5(y-1))) for x in 1:15169,y in 1:5],
    [rand(Normal(π*(6-y)/3,1/ℯ)) for x in 1:15169,y in 1:5],
    rand(Binomial(20,0.6),15169,5),
    rand(Binomial(20,0.5),15169,5);
    dims=3),(1,3,2))
    
    #2.d
    β=hcat([[1+.25*(t-1), log(t), -sqrt(t), ℯ^t-ℯ^(t+1), t, t/3] for t in 1:5]...) 
    
    #2.e
    Y=hcat([X[:,:,i]*β[:,i].+rand(Normal(0,.36)) for i in 1:5]...)
    return
end


#Question 3
function q3()
    #3.a
    #workspace() not supported on my version anymore
    df=CSV.read("nlsw88.csv"; header=true,missingstring="")
    JLD2.@save "nlsw88.jld" df

    #3.b
    describe(df)
    # the mean of neve_married is 0.104185 and married is {0,1}. so 10.4185% are never_married.

    #3.c
    tabulate(df,:race)
    #I think taubulate is in TexTables package but it wasn't listed in the required package list.

    #3.d
    summary_df=describe(df, :mean, :median, :std, :min, :max, :nunique, :q75, :q25)
    summary_df[!, :intq] = summary_df[:, :q75] - summary_df[:, :q25]
    summarystats=Matrix(summary_df[:,Not([:variable,:q25,:q75])])
    #from 3.b there are two observations missing.

    #3.e
    freqtable(df[:,:industry],df[:,:occupation])

    #3.f
    df_f=df[:,[:industry, :occupation , :wage]]
    gdf=groupby(df_f,[:industry, :occupation])
    gdf=combine(gdf, valuecols(gdf) .=> mean)
    freqtable(gdf,:wage_mean,:industry,:occupation);
    return
end

A,B,C,D=q1()
q2(A,B,C)
q3()




#Question4
function q4()
    #4.a
    JLD2.@load "firstmatrix.jld2" A B C D
    
    #4.b 4.c 4.e
    function matrixops(a,b)
        #calculate dot product, product and sum of elements for inputs.
        if size(a)==size(b)    
            dProduct=a.*b
            product=a'*b
            sprod=sum(a+b)
            return dProduct, product,sprod
        else
            println("inputs must have the same size.")
        end
    end
    #4.d
    matrixops(A,B)
    #4.f
    matrixops(C,D)
    #4.g
    JLD2.@load "nlsw88.jld"
    matrixops(convert(Array,df.ttl_exp),convert(Array,df.wage));
end
    

q4()