# -----------------------------------------------------------
#
# Methods for DG interpolation using readable
# dictionary-style structs
#
# -----------------------------------------------------------

using HCubature

const REL_TOL = 1.0e-8
const ABS_TOL = 1.0e-12
const MAX_EVALS = 1000

# -----------------------------------------------------
# Here we have all our methods for
# taking a function and returning an appropriate list
# of DG coefficients.
#
# -----------------------------------------------------

# Efficiency criticality: HIGH

# -----------------------------------------------------
# 1-D Basis
# -----------------------------------------------------

function v(k::Int, level::Int, cell::Int, mode::Int, x::T) where T <: Real
    if level==0 # At the base level, we are not discontinuous, and we simply
                # use Legendre polynomials up to degree k-1 as a basis
        return LegendreP(mode-1,2*x-1)*sqrt(2.0)
    else
        return h(k, mode, (1<<level)*x - (2*cell-1)) * sqrt(one(T)*(1<<level))
        # Otherwise we use an appropriately shifted, scaled, and normalized
        # DG function
    end
end

function v(k::Int, level::Int, cell::Int, mode::Int)
    return (xs-> v(k,level,cell,mode,xs))
end

# -----------------------------------------------------
# Tensor Product Construction
# -----------------------------------------------------

# Returns the value of the function at x
function V(k::Int, level::NTuple{D, Int}, cell::CartesianIndex{D},
    mode::CartesianIndex{D}, xs::AbstractArray{T}) where {D,T}
    ans = one(T)
    for i = 1:D
        ans *= v(k, level[i], cell[i], mode[i], xs[i])
    end
    return ans
end
# Is there any fast way to precompute the ones I care about??

# Returns a function
function V(k, level::NTuple{D, Int}, cell::CartesianIndex{D},
    mode::CartesianIndex{D}) where D
    # return (xs-> V(k, level, cell, mode, xs))
    return function(xs) V(k, level, cell, mode, xs) end
end

# -----------------------------------------------------
# Methods for obtaining the coefficients
# -----------------------------------------------------

# Given a 1-D position and level, this tells us which cell
# that position belongs to, at that level resolution
function cell_index(x::Real,l::Int)
    if l <= 1
        return 1
    end
    if x >= 1
        return 2^(l-1)
    else
        return 1 + floor(Int, 2^(l-1) * x)
    end
end


# This takes an inner product, but since for higher levels our inner product
# is only concerned with a specific region in the grid, we restrict to that
# appropriately, depending on the level
function inner_product(f::Function, g::Function, lvl::NTuple{D, Int},
    cell::CartesianIndex{D}; rtol = REL_TOL, atol = ABS_TOL, maxevals = MAX_EVALS) where D

    _h(x) = f(x)*g(x)
    xmin = ntuple(i-> (cell[i]-1)/(1<<max(0,lvl[i]-1)), D)
    xmax = ntuple(i-> (cell[i])/(1<<max(0,lvl[i]-1)), D)
    val = hcubature(_h, xmin, xmax; rtol=rtol, atol=atol, maxevals=maxevals)[1]
    return val
end

# We obtain coefficients simply by doing inner products, it's easy :)
# Only hard part is inner product integrations can be slower than we want :(
function get_coefficient_DG(k::Int, lvl::NTuple{D, Int}, cell::CartesianIndex{D},
     mode::CartesianIndex{D},
     f::Function;
     rtol = REL_TOL,
     atol = ABS_TOL,
     maxevals = MAX_EVALS) where D

    return inner_product(f, V(k,lvl,cell,mode),lvl,cell;
                            rtol = rtol, atol = atol, maxevals=maxevals)
end



# -----------------------------------------------------
# Full or Sparse Galerkin Coefficients in n-D
# -----------------------------------------------------
function coeffs_DG(D::Int, k::Int, n::Int, f::Function;
                                        rtol = REL_TOL, atol = ABS_TOL,
                                        maxevals=MAX_EVALS,
                                        scheme="sparse")
    coeffs_DG(Val(D), k, n, f, rtol, atol, maxevals, Val(Symbol(scheme)))
end
function coeffs_DG(::Val{D}, k::Int, n::Int, f::Function,
                                        rtol, atol,
                                        maxevals,
                                        scheme::Val{Scheme}) where {D, Scheme}
    coeffs    = Dict{CartesianIndex{D}, Array{Array{Float64,D},D}}()
    modes    = ntuple(i-> k, D)
    ls        = ntuple(i->(n+1),D)

    for level in CartesianIndices(ls) #This really goes from 0 to l_i for each i
        cutoff(scheme, level, n) && continue

        cells = ntuple(i -> 1<<max(0, level[i]-2), D)
        level_coeffs = Array{Array{Float64,D}}(undef, cells)
        lvl = ntuple(i -> level[i]-1,D)
        for cell in CartesianIndices(cells)
            cell_coeffs = Array{Float64}(undef,modes)
            for mode in CartesianIndices(modes)
                cell_coeffs[mode] = get_coefficient_DG(k, lvl, cell, mode, f)
            end
            level_coeffs[cell]=cell_coeffs
        end
        coeffs[level] = level_coeffs
    end
    return coeffs
end
function coeffs_DG(D::Int, k::Int, n::Vector{Int}, f::Function;
        rtol = REL_TOL, atol = ABS_TOL,
        maxevals=MAX_EVALS,
        scheme="sparse")
    coeffs_DG(Val(D), k, n, f, rtol, atol, maxevals, Val(Symbol(scheme)))
end
function coeffs_DG(::Val{D}, k::Int, n::Vector{Int}, f::Function,
                    rtol, atol,
                    maxevals,
                    scheme::Val{Scheme}) where {D, Scheme}
    coeffs    = Dict{CartesianIndex{D}, Array{Array{Float64,D},D}}()
    modes     = ntuple(i-> k, D)
    #ls       = ntuple(i->(n+1),D)
    #ls       = (n1+1, n2+1) #falls D=2, Eingabevariablen n1::Int, n2::Int
    ls        = ntuple(i -> n[i]+1, D)

    for level in CartesianIndices(ls) #This really goes from 0 to l_i for each i
        cutoff(scheme, level, n) && continue #wenn scheme=full, dann cutoff=False -> nicht continue
                        # (Iterationsschritt wird nicht abgebrochen und mitm nächsten weitergemacht)

        cells = ntuple(i -> 1<<max(0, level[i]-2), D)
        level_coeffs = Array{Array{Float64,D}}(undef, cells)
        lvl = ntuple(i -> level[i]-1,D)
        for cell in CartesianIndices(cells)
            cell_coeffs = Array{Float64}(undef,modes)
            for mode in CartesianIndices(modes)
                cell_coeffs[mode] = get_coefficient_DG(k, lvl, cell, mode, f)
            end
            level_coeffs[cell]=cell_coeffs
        end
        coeffs[level] = level_coeffs
    end
    return coeffs
end

# -----------------------------------------------------------
# Reconstruction (full and sparse) in n-D from a Dict of
# coefficients
# -----------------------------------------------------------
function reconstruct_DG(coeffs::Dict{CartesianIndex{D}, <:AbstractArray{<:AbstractArray{T1, D}, D}}, xs::Array{T2, 1}) where {D, T1 <: Real, T2 <: Real}

    value    = zero(T2)
    k        = size(first(values(coeffs))[1])[1]
    modes    = ntuple(i-> k ,D)

    for key in keys(coeffs)
        level = ntuple(i->key[i]-1,D)
        cell = CartesianIndex{D}(ntuple(i->cell_index(xs[i],level[i]),D))
        coeff = coeffs[key][cell]::Array{T1,D}
        @inbounds for mode in CartesianIndices(modes)
            value += coeff[mode]*V(k, level, cell, mode, xs)
        end
    end
    return value
end
