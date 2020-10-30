function corrcov(C::AbstractMatrix)
    #should check if C is positive semidefinite

    sigma = sqrt.(diag(C))
    return C ./ (sigma*sigma')
end
