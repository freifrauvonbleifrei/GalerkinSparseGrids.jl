# -----------------------------------------------------------
#
# Construction of multidimensional DG derivative matrices
# based on the 1-D derivative matrix
#
# -----------------------------------------------------------

# Efficiency criticality: HIGH
# Matrix computations are only performed once,
# but this can be the main bottleneck if not done right

# Accuracy criticality: HIGH
# Critical for accurate PDE evolution

function D_matrix(d::Int, k::Int, n::Int, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
        srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}; scheme = "sparse") where D
    D_matrix(Val(d), k, n, srefVD, srefDV, Val(Symbol(scheme)))
end

function D_matrix(::Val{d}, k::Int, n::Int, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
        srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}, scheme::Val{Scheme}) where {d, D, Scheme}

    # len = length(srefVD[:,1])
    len = length(srefVD)
    @assert length(srefVD[:,1]) == length(srefVD)
    V2D_1D = V2Dref(Val(1),k,n,Val(:sparse)) # Why not 'scheme'?
    D2V_1D = D2Vref(Val(1),k,n,Val(:sparse))
    I = Int[]; J = Int[]; V = Float64[]

    # 1-dimensional derivative matrix - this will give all coefficient info
    Dmat_1D = periodic_DLF_matrix(k, n)

    for j in 1:len
        lcm = srefVD[j]
        l = lcm[1][d]
        c = lcm[2][d]
        m = lcm[3][d]
        j_1D = D2V_1D[(CartesianIndex(l),CartesianIndex(c),CartesianIndex(m))]
        derivs = Dmat_1D[:, j_1D]::SparseVector{Float64,Int}

        for i_1D in derivs.nzind
            lcm_1D = V2D_1D[i_1D]
            # using tuple constructor independent of Julia's metaprogramming:
            # make_cartesian_index(d, arr1, arr2) takes arr::CartesianIndex{1}
            # and makes a new CartesianIndex{D} using arr2::CartesianIndex{D}
            # with the dth value replaced by arr1[1]
            level2 = make_cartesian_index(d, lcm_1D[1], lcm[1])
            cutoff(scheme, level2, n) && continue
            cell2 = make_cartesian_index(d, lcm_1D[2], lcm[2])
            mode2 = make_cartesian_index(d, lcm_1D[3], lcm[3])
            i = srefDV[(level2, cell2, mode2)]
            push!(I, i)
            push!(J, j)
            push!(V, derivs[i_1D])
        end
    end
    # dropzeros! does not seem helpful for this matrix:
    return sparse(I, J, V, len, len, +)
end

function D_matrix(D::Int, d::Int, k::Int, n::Int; scheme="sparse")
    VD = V2Dref(Val(D), k, n, Val(Symbol(scheme)))
    DV = D2Vref(Val(D), k, n, Val(Symbol(scheme)))
    return D_matrix(d, k, n, VD, DV; scheme=scheme)
end

# Same functions as above but for anisotropic grids
function D_matrix(d::Int, k::Int, n::Vector{Int}, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
                srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}; scheme = "full") where D
    D_matrix(Val(d), k, n, srefVD, srefDV, Val(Symbol(scheme)))
end

function D_matrix(::Val{d}, k::Int, n::Vector{Int}, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
                srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}, scheme::Val{Scheme}) where {d, D, Scheme}

    n_akt = n[d]

    len = length(srefVD)
    @assert length(srefVD[:,1]) == length(srefVD)
    
    V2D_1D = V2Dref(Val(1),k,n_akt,Val(:full)) 
    D2V_1D = D2Vref(Val(1),k,n_akt,Val(:full))

    if n_akt == 0
        num_cells = 1
    else
        num_cells = 2^(n_akt)
    end

    @assert length(D2V_1D) == num_cells*k "multidim_derivative.jl Z.95 length =/ num_cells*k"#*(n_akt+1)

    I = Int[]; J = Int[]; V = Float64[]

    # 1-dimensional derivative matrix - this will give all coefficient info
    Dmat_1D = periodic_DLF_matrix(k, n_akt)

    len_akt = length(V2D_1D)^length(n)
    @assert length(V2D_1D[:,1]) == length(V2D_1D)


    for j in 1:len
        lcm = srefVD[j]
        l = lcm[1][d]
        c = lcm[2][d]
        m = lcm[3][d]
        j_1D = D2V_1D[(CartesianIndex(l),CartesianIndex(c),CartesianIndex(m))]
        derivs = Dmat_1D[:, j_1D]::SparseVector{Float64,Int}

        for i_1D in derivs.nzind 
            lcm_1D = V2D_1D[i_1D]
            # using tuple constructor independent of Julia's metaprogramming:
            # make_cartesian_index(d, arr1, arr2) takes arr::CartesianIndex{1}
            # and makes a new CartesianIndex{D} using arr2::CartesianIndex{D}
            # with the dth value replaced by arr1[1]
            level2 = make_cartesian_index(d, lcm_1D[1], lcm[1])
            cutoff(scheme, level2, n) && continue 
            cell2 = make_cartesian_index(d, lcm_1D[2], lcm[2])
            mode2 = make_cartesian_index(d, lcm_1D[3], lcm[3])
            i = srefDV[(level2, cell2, mode2)]
            push!(I, i)
            push!(J, j)
            push!(V, derivs[i_1D])
        end
    end
    # dropzeros! does not seem helpful for this matrix:
    return sparse(I, J, V, len, len, +)
end

function D_matrix(D::Int, d::Int, k::Int, n::Vector{Int}; scheme="full")
    VD = V2Dref(Val(D), k, n, Val(Symbol(scheme)))
    DV = D2Vref(Val(D), k, n, Val(Symbol(scheme)))
    return D_matrix(d, k, n, VD, DV; scheme=scheme)
end

function grad_matrix(D::Int, k::Int, n::Int; scheme="sparse")
    return [D_matrix(D, d, k, n; scheme=scheme) for d in 1:D]
end

function laplacian_matrix(D::Int, k::Int, n::Int; scheme="sparse")
    len = get_size(Val(D), k, n, Val(Symbol(scheme)))
    lap = spzeros(len, len)
    for i in 1:D
        D_op = D_matrix(D, i, k, n; scheme=scheme) ##
        lap += D_op * D_op
    end
    return lap 
end

# Same function but for anisotropic grids
function laplacian_matrix(D::Int, k::Int, n::Vector{Int}; scheme="full")
    len = get_size(Val(D), k, n, Val(Symbol(scheme)))
    lap = spzeros(len, len)
    for i in 1:D
        D_op = D_matrix(D, i, k, n; scheme=scheme)
        lap += D_op * D_op
    end
    return lap
end