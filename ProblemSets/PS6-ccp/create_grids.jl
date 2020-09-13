function create_grids()

    function xgrid(theta,xval)
        N      = length(xval)
        xub    = vcat(xval[2:N],Inf)
        xtran1  = zeros(N,N)
        xtran1c = zeros(N,N)
        lcdf   = zeros(N)
        for i=1:length(xval)
            xtran1[:,i]   = (xub[i] .>= xval).*(1 .- exp.(-theta*(xub[i] .- xval)) .- lcdf)
            lcdf        .+= xtran1[:,i]
            xtran1c[:,i] .+= lcdf
        end
        return xtran1, xtran1c
    end

    zval   = collect([.25:.01:1.25]...)
    zbin   = length(zval)
    xval   = collect([0:.125:25]...)
    xbin   = length(xval)
    tbin   = xbin*zbin
    xtran  = zeros(tbin,xbin)
    xtranc = zeros(xbin,xbin,xbin)
    for z=1:zbin
        xtran[(z-1)*xbin+1:z*xbin,:],xtranc[:,:,z] = xgrid(zval[z],xval)
    end

    return zval,zbin,xval,xbin,xtran
end
