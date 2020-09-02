#Worked with Kasra#
                            #Question No.1#
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables

#part (l)#
function q1()
#Part (a)#
Random.seed!(1234)
A=rand(Uniform(-5,10),(10,7))
B=rand(Normal(-2,15),(10,7))
C=[A[1:5,1:5] B[1:5,6:7]]
D=(A.<=0).*A

#Part (b)#
length(A)

#Part (c)#
length(unique(D))

#Part (d)#
E=reshape(B,:,1)
E=vec(B) #Alternative: Trese command for vectorization

#Part (e)#
F=cat(A, B;dims=3)

#Part (f)#
F=permutedims(F,(3,1,2)) # 3 1 2 are dimentions

#Part (g)#
G=kron(B,C)
kron(C,F) #Kronecher product is not possible since dimentions of C & F do not match

#Part (h)#
cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\0 PS Solutions_Julia Directory")
JLD2.@save "matrixpractice.jld2" A B C D E F G

#Part (i)#
JLD2.@save "firstmatrix.jld2" A B C D

#Part (j)#
CC=convert(DataFrame, C) # df=DataFrame(C) --> Better code which Kasra told me
CSV.write("Cmatrix.csv", CC) # CSV.write("Cmatrix.csv", df)

#Part (k)#
DD=convert(DataFrame, D) # dfD=DataFrame(D) --> Better code by Kasra
CSV.write("Dmatrix.dat", DD, delim='\t') #CSV.write("Dmatrix.dat",df,delim='\t')

#Part (l) --> done
      return A,B,C,D
end


                            #Question No.2#
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables

#part (f)
function q2(a,b,c)
#Part (a)#
AB=zeros(size(A))
for i=1:length(A) #Alternatively define n=length(A)
       AB[i]=A[i]*B[i]
       end
AB2=A.*B

#Part (b)#
Cprime=[]
for i=1:length(C)
      if -5<=C[i]<=5
            append!(Cprime,C[i])    #Took help from Kasra to figure out the loops
      end
end
Cprime2=[i for i in C if -5<=i<=5]

#Part (c)#
X1=cat(ones(15169,5), [rand(Bernoulli(0.75*(6-t)/5)) for n in 1:15169,t in 1:5], [rand(Normal(15+t-1,5(t-1))) for n in 1:15169,t in 1:5], [rand(Normal(π*(6-t)/3,1/ℯ)) for n in 1:15169,t in 1:5], rand(Binomial(20,0.6),15169,5), rand(Binomial(20,0.5),15169,5);dims=3)
X=permutedims(X1,(1,3,2)) # Or replace X1 with the entire code of X1

#Part (d)#
β=hcat([[1+.25*(t-1), log(t), -sqrt(t), ℯ^t-ℯ^(t+1), t, t/3] for t in 1:5]...)
#Part (e)#
Y=hcat([X[:,:,i]*β[:,i].+rand(Normal(0,.36)) for i in 1:5]...)
#part(f)#
      return
end


                            #Question No.3#
#Cleared Julia Manually by restarting#
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables

#part (g)#
function q3()

#Part (a)#
cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\0 PS Solutions_Julia Directory")
df=CSV.read("nlsw88.csv", header=true, missingstring="")
JLD2.@save "nlsw88.jld" df

#Part (b)#
describe(df)
describe(df[:,:never_married])
# Since never_married is a binary variable which takes the value of 1 to represent
# people who never married; their precentage would be meean*100 = 10.4185% #
describe(df[:,:collgrad])
# By the same logic as above, the percentage of college grads would be 23.6866%#

#Part (c)#
tabulate(df,:race)
#Category 1 = 72.885% | Category 2 = 25.957% | Category 3 = 1.158%
#Discussed with Kasra and Waleed, and used TexTables package for tabulate command#

#Part (d)#
descriptives=describe(df, :mean, :median, :std, :min, :max, :nunique, :q75, :q25)
interquartile_range= descriptives[:,:q75] - descriptives[:,:q25]
#Cannot figure out how to find missing observations#

#Part (e)#
freqtable(df[:,:industry],df[:,:occupation])

#Part (f)#
df_f=df[:,[:industry, :occupation , :wage]]
gdf=groupby(df_f,[:industry, :occupation])
gdf=combine(gdf, valuecols(gdf) .=> mean)
freqtable(gdf,:wage_mean,:industry,:occupation);
#Part (g)#
      return
end
q1()=A,B,C,D
q2(A,B,C)
q3()


                              #Question No.4#
using Random, Distributions, JLD2, DataFrames, CSV, LinearAlgebra, Statistics, FreqTables, TexTables

#Part (h)#
function q4()
cd("D:\\Semester 3_Fall 2020\\ECON 6453 - Econometrics III\\ProblemSets\\0 PS Solutions_Julia Directory")

#Part (a)#
JLD2.@load "firstmatrix.jld2" A B C D

#Part (b+c+e)#
function matrixops(A,B)
      if if size(A)==size(B)
      AdotB=A.*B
      AtransB=A'*B
      sumAB=sum(A+B)
      return AdotB, AtransB, sumAB
else
      println("inputs must have the same size.")
end
end

#Part (d)#
matrixops(A,B)

#Part (f)#
matrixops(C,D)
#Does not work because inputs must have the same size.

#Part (g)#
JLD2.@load "nlsw88.jld"
matrixops(convert(Array,df.ttl_exp),convert(Array,df.wage));

#Part (h)#
      return
end
end
q4()
                              #THE END#
