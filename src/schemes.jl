# -----------------------------------------------------
# This script gives the methods for maniupulating the
# Cartesian indices corresponding to multilevels that
# are either kept or cut in various schemes of
# "sparsification"
# -----------------------------------------------------

# Efficiency Criticality: LOW
# This is not evaluated very often

# Accuracy Criticality: N/A
# No float manipulation



# Gives the appropriate boolean cutoff corresponding
# to a given scheme, e.g. sparse basis, full basis,
# and in the future possibly the energy basis of
# Bungartz and Griebel

function cutoff(::Val{:sparse}, x::CartesianIndex{D}, n::Int) where {D}
    sum(x.I) > n+D
end

function cutoff(::Val{:full}, x::CartesianIndex{D}, n::Int) where {D}
    false
end

# Same functions but for anisotropic grids
# all grids where the multi-indices are bigger than the current level get cutoff
# index1 <=! n(1)+1, index2 <=! n(2)+1
function cutoff(::Val{:full}, x::CartesianIndex{D}, n::Vector{Int}) where {D}
    test = 0
    if sum(x.I) <= sum(n) + length(n) # eigentlich <=sum(n+1) aber n Vector
        for d in 1:D
            if x.I[d] > n[d]+1 # n+1 weil CartesianIndex bei 1 startet obwohl unsere theoretischen Levels bei 0 starten
                test += 1
            end
        end
        if test == 0
            return false
        end
    end
    return true
end

# cutoff for anisotropic grids (n is vector) only needed for full grids
# function cutoff(::Val{:sparse}, x::CartesianIndex{D}, n::Vector{Int}) where D
#     false
# end