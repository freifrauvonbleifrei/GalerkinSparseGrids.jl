# -----------------------------------------------------------
#
# Using boundary terms and integration (summation) by parts
# to construct more accurate derivative matrices for time
# evolution using Lax-Friedrichs-type fluxes
#
# All derivatives are computed here in pos basis first
# Then conjugated into hier basis
#
# -----------------------------------------------------------

# Efficiency criticality: MED
# Most computations time is taken up by functions called from
# other script files

# Accuracy criticality: LOW
# The accuracy is limited only by functions in other scripts
# and ODE.jl

using Plots
using ODE

# Construct the time evolution matrix for the wave equation
# given a Kaplacian and the initial position and velocity f0, v0
function wave_data(laplac::A, f0coeffs::AbstractArray{T, 1},
        v0coeffs::AbstractArray{T, 1}) where {A <: AbstractArray, T <: Real}

    @assert length(f0coeffs)==length(v0coeffs) "length(f0coeffs)=/length(v0coeffs)"
    len = length(f0coeffs)
    rows = rowvals(laplac)
    vals = nonzeros(laplac)

    I = Int[]
    J = Int[]
    V = T[]
    for col = 1:len
        for i in nzrange(laplac, col)
            row = rows[i]
            val = vals[i]
            push!(I,row+len)
            push!(J,col)
            push!(V,val)
        end
    end
    for i = 1:len
        push!(I, i)
        push!(J, i+len)
        push!(V, 1.0)
    end
    RHS = sparse(I, J, V, 2*len, 2*len, +)
    y0 = Array{T}([i<=len ? f0coeffs[i] : v0coeffs[i-len] for i in 1:2*len])
    return RHS, y0
end

# Evolves the wave equation from time t0 to t1
# given coefficients for an initial position profile of f0
# and an initial velocity profile of v0
function wave_evolve(D::Int, k::Int, n::Int,
                              f0coeffs::Array{T, 1}, v0coeffs::Array{T, 1},
                              time0::Real, time1::Real;
                              order="45", scheme="sparse", kwargs...) where T <: Real

    laplac   = laplacian_matrix(D, k, n; scheme=scheme)
    RHS, y0 = wave_data(laplac, f0coeffs, v0coeffs)
    if order == "45"
        soln = ode45((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "78"
        soln = ode78((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "1"
        soln = ODE.ode1((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "4"
        soln = ODE.ode4((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    else
        throw(ArgumentError(:order))
    end
    return soln
end

# Same function but for anisotropic grids
function wave_evolve(D::Int, k::Int, n::Vector{Int},
                    f0coeffs::Array{T, 1}, v0coeffs::Array{T, 1},
                    time0::Real, time1::Real;
                    order="45", scheme="full", kwargs...) where T <: Real
  
    laplac   = laplacian_matrix(D, k, n; scheme=scheme)
    RHS, y0 = wave_data(laplac, f0coeffs, v0coeffs)
    if order == "45"
        soln = ode45((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "78"
        soln = ode78((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "1"
        soln = ODE.ode1((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "4"
        soln = ODE.ode4((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    else
        throw(ArgumentError(:order))
    end
    return soln
end

# Evolves the wave equation from time t0 to t1
# given initial position profile f0 and velocity profile v0
function wave_evolve(D::Int, k::Int, n::Int,
                              f0::Function, v0::Function,
                              time0::Real, time1::Real;
                              order="45", scheme="sparse", kwargs...)

    f0coeffs = vcoeffs_DG(D, k, n, f0; scheme=scheme)
    v0coeffs = vcoeffs_DG(D, k, n, v0; scheme=scheme)
    return wave_evolve(D, k, n, f0coeffs, v0coeffs, time0, time1;
                        order=order, scheme=scheme, kwargs...)
end

# wave_evolve for anisotropic grids
function wave_evolve(D::Int, k::Int, n::Vector{Int},
                                f0::Function, v0::Function,
                                time0::Real, time1::Real;
                                order="45", scheme="full", kwargs...)

    f0coeffs = vcoeffs_DG(D, k, n, f0; scheme=scheme)
    v0coeffs = vcoeffs_DG(D, k, n, v0; scheme=scheme)
    return wave_evolve(D, k, n, f0coeffs, v0coeffs, time0, time1;
                         order=order, scheme=scheme, kwargs...)
end

# The reason we have a wave evolution in 1D is to test
# that the standard "position" and multiresolution "heirarchical"
# bases agree for wave PDE evolution
function wave_evolve_1D(k::Int, max_level::Int,
                            f0::Function, v0::Function,
                            time0::Real, time1::Real; basis="hier", order = "45",
                            kwargs...)
    if basis == "pos"
        f0coeffs = pos_vcoeffs_DG(k, max_level, f0)
        v0coeffs = pos_vcoeffs_DG(k, max_level, v0)
    elseif basis == "hier"
        f0coeffs = vcoeffs_DG(1, k, max_level, f0)
        v0coeffs = vcoeffs_DG(1, k, max_level, v0)
    elseif basis == "nodal"
        throw(MethodError("Nodal basis wave evolution not yet implemented"))
    elseif basis == "point"
        m2p = modal2points_1D(k, max_level)
        f0coeffs = m2p*vcoeffs_DG(1, k, max_level, f0)
        v0coeffs = m2p*vcoeffs_DG(1, k, max_level, v0)
    else
        throw(ArgumentError(:basis))
    end

    D_op = periodic_DLF_matrix(k, max_level; basis=basis)
    laplac= *(D_op,D_op)

    RHS, y0 = wave_data(laplac, f0coeffs, v0coeffs)
    if order == "45"
        soln = ode45((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "78"
        soln = ode78((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    elseif order == "1"
        soln = ODE.ode1((t,x)->*(RHS,x), y0, [time0,time1]; kwargs...)
    else
        throw(ArgumentError(:order))
    end
    return soln
end

# Evolve the Vlassov-Poisson (AKA collisionless Boltzmann) equation
# for a mass distribution in D dimensions
# -> in a 2*D-dimensional phase space
# D, k, n (good)
# f0_coeffs_modal (start with anything- like a gaussian) (good)
# F_point (Depends on the potential I pick. Use Isothermal Truncated)
# v_point (its a constant and simple vector for any potential) (good)
# time0, time1, order, scheme (good)
function vlasov_evolve(D::Int, k::Int, n::Int,
            m2n::SparseMatrixCSC{T, Int}, n2p::SparseMatrixCSC{T, Int},
            p2n::SparseMatrixCSC{T, Int}, n2m::SparseMatrixCSC{T, Int},
            f0_modal::Array{T,1}, F_point::Array{Array{T,1}, 1},
            time0::T, time1::T, nout::Int=2;
            order="45", scheme="sparse", kwargs...) where T <: Real

    # The grad matrix- using the same derivative operator as we did
    # for the wave equation.
    # We may need to apply filtering intermittently in the evolution
    Ds = grad_matrix(2*D, k, n; scheme=scheme)

    println("Writing operators...")
    flush(stdout)
    jldopen("vlasov.h5", "w") do f
        f["dimensions"] = D
        f["order"] = k
        f["levels"] = n
        function w(n::AbstractString, A::SparseMatrixCSC)
            f["$n.m"] = A.m
            f["$n.n"] = A.n
            f["$n.colptr"] = A.colptr
            f["$n.rowval"] = A.rowval
            f["$n.nzval"] = A.nzval
        end
        w("m2n", m2n)
        w("n2p", n2p)
        w("p2n", p2n)
        w("n2m", n2m)
        for (d, ds) in enumerate(Ds)
            w("Ds[$d]", ds)
        end
    end

    # Coeffs for the constant 1 in D-dim space
    one_1D = get_one_modal(1, k, n)
    # Coeffs for velocity in D-dim space
    v_modal_1D = get_xi_modal(1, 1, k, n)
    # Coeffs for velocity in 2*D-dim phase space - the tensor product of the above
    v_modal = [tensor_construct(2*D, k, n, [j-D == i ? v_modal_1D : one_1D for j in 1:2*D])
              for i in 1:D]
    v_point = [n2p * (m2n * v) for v in v_modal]

    function steprule(t::T, f_modal::Array{T, 1})
        println("    evaluating RHS at t=$t...")
        flush(stdout)
        # We want this in a modal basis for differentiation to work fast
        # Matrix multiplication is the bottle neck here.
        dfdxs_modal = [Ds[d] * f_modal for d in 1:D]
        dfdps_modal = [Ds[d] * f_modal for d in (D+1):(2*D)]

        dfdxs_point = [n2p * (m2n * x) for x in dfdxs_modal]
        dfdps_point = [n2p * (m2n * x) for x in dfdps_modal]

        # dH/dp * df/dx
        contrib1 = sum(v_point[d] .* dfdxs_point[d] for d in 1:D)
        # dH/dx * df/dp
        contrib2 = sum(F_point[d] .* dfdps_point[d] for d in 1:D)

        # df/dt = - (dH/dp * df/dx - dH/dx * df/dp)
        return n2m * (p2n * (- contrib1 + contrib2))
    end

    println("State vector size: $(length(f0_modal))")
    println("Operator sizes:")
    for d in 1:D
        println("    Ds[$d]: $(nnz(Ds[d]))")
    end
    println("    m2n:   $(nnz(m2n))")
    println("    n2p:   $(nnz(n2p))")
    println("    p2n:   $(nnz(p2n))")
    println("    n2m:   $(nnz(n2m))")
    println("Beginning PDE evolution...")
    flush(stdout)

    if order == "45"
        solver = ode45
    elseif order == "78"
        solver = ode78
    elseif order == "1"
        solver = ODE.ode1
    else
        throw(ArgumentError(:order))
    end
    soln = solver(steprule, f0_modal, range(time0, time1, length=nout); kwargs...)

    println("Writing solution...")
    flush(stdout)
    jldopen("vlasov.h5", "r+") do f
        f["times"] = soln[1]
        for (i,sol) in enumerate(soln[2])
            ifmt = @sprintf "%06d" i
            f["f_modal.$ifmt"] = sol
        end
    end

    println("Done.")
    flush(stdout)
end


function norm_squared(coeffs::AbstractArray{T},; basis = "hier") where T <: Real

    if basis=="hier" || basis=="pos"
        return sum(i^2 for i in coeffs)
    else
        throw(MethodError("Nodal and point basis norms not yet implemented"))
    end
end

function energy_func_1D(k, level, soln::Tuple{Array{T, 1}, Array{Array{T, 1}, 1}};
    basis = "hier") where T <: Real

    len            = length(soln[1])
    num_coeffs    = Int(round(length(soln[2][1])/2))
    times        = copy(soln[1])
    energies    = Array{T}(undef, len)
    D_op        = periodic_DLF_matrix(k, level; basis=basis)

    for i in 1:len
        ux   = D_op *  (@view soln[2][i][1:num_coeffs])
        udot = @view soln[2][i][num_coeffs+1:end]
        energies[i] = norm_squared(ux) + norm_squared(udot)

    end
    return (times, energies)
end


function energy_func(D::Int, k::Int, n::Int, soln::Tuple{Array{T, 1}, Array{Array{T, 1}, 1}};
        scheme = "sparse") where T <: Real

    len         = length(soln[1])
    num_coeffs    = div(length(soln[2][1]), 2)
    times         = copy(soln[1])
    energies    = Array{T}(undef, len)
    D_ops        = grad_matrix(D, k, n; scheme=scheme)

    for i in 1:len
        ux   = [D_ops[j] * (@view soln[2][i][1:num_coeffs]) for j in 1:D]
        udot = @view soln[2][i][num_coeffs+1:end]
        energies[i] = sum(norm_squared(ux[d]) for d in 1:D) + norm_squared(udot)
    end
    return (times, energies)
end
