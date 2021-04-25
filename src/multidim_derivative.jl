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

# Same functions as above but for anisotropic grids, changed to default scheme 'full'
function D_matrix(d::Int, k::Int, n::Vector{Int}, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
                srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}; scheme = "full") where D
    D_matrix(Val(d), k, n, srefVD, srefDV, Val(Symbol(scheme)))
end

function D_matrix(::Val{d}, k::Int, n::Vector{Int}, srefVD::Array{NTuple{3, CartesianIndex{D}}, 1},
                srefDV::Dict{NTuple{3, CartesianIndex{D}}, Int}, scheme::Val{Scheme}) where {d, D, Scheme}

    # len = length(srefVD[:,1])
    len = length(srefVD)
    @assert length(srefVD[:,1]) == length(srefVD)
    #V2D_1D = V2Dref(Val(1),k,n,Val(:sparse)) # Why not 'scheme'?
    #D2V_1D = D2Vref(Val(1),k,n,Val(:sparse))

    I_all = Int[]
    J_all = Int[]
    V_all = Float64[]
    # len_all = 0
    len_all = Int[]

    for a in 1:length(n)

        srefVD = V2Dref(Val(D), k, n[a], scheme) # zuvor Val(Symbol(scheme))
        srefDV = D2Vref(Val(D), k, n[a], scheme) # zuvor Val(Symbol(scheme))
        len = length(srefVD)
        @assert length(srefVD[:,1]) == length(srefVD)

        # len_all += len
        append!(len_all, len)

        V2D_1D = V2Dref(Val(1),k,n[a],scheme)
        D2V_1D = D2Vref(Val(1),k,n[a],scheme)
        I = Int[]; J = Int[]; V = Float64[]

        # 1-dimensional derivative matrix - this will give all coefficient info
        Dmat_1D = periodic_DLF_matrix(k, n[a]) ##

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
                @assert cutoff(scheme, level2, n) == false
                cutoff(scheme, level2, n) && continue
                cell2 = make_cartesian_index(d, lcm_1D[2], lcm[2])
                mode2 = make_cartesian_index(d, lcm_1D[3], lcm[3])
                i = srefDV[(level2, cell2, mode2)]
                push!(I, i)
                push!(J, j)
                push!(V, derivs[i_1D])
            end
        end
        append!(I_all,I)
        append!(J_all,J)
        append!(V_all,V)
    end

    # len_avr = len_all / length(n)
    max_len = maximum(len_all)
    # println("I_all = ", I_all)
    # dropzeros! does not seem helpful for this matrix:
    return sparse(I_all, J_all, V_all, max_len, max_len, +)
end


# changed default scheme from sparse to full
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
    # len = get_size(Val(D), k, n, Val(Symbol(scheme)))
    # lap = spzeros(len, len)
    # # for i in 1:D
    # #     D_op = D_matrix(D, i, k, n; scheme=scheme)
    # #     lap += D_op * D_op
    # # end
    # max_n = argmax(n)
    # len = get_size(Val(D), k, n[max_n], Val(Symbol(scheme)))
    # lap = spzeros(len, len)
    # for i in 1:D
    #     lap += laplacian_matrix(D,k,n[i],scheme=scheme)
    # end
    # return lap
    lap_all = Float64[]
    for i in 1:D
        len = get_size(Val(D), k, n[i], Val(Symbol(scheme)))
        lap = spzeros(len, len)
        D_op = D_matrix(D, i, k, n[i]; scheme=scheme) ##
        lap += D_op * D_op
        lap_all
    end
    return lap_all ##auch wieder liste
end