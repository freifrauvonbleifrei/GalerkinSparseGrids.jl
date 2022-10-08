# -----------------------------------------------------------
#
# Some examples of using the combination technique to 
# approximate a scalar wave
#
# -----------------------------------------------------------

using GalerkinSparseGrids 
using Plots
using LinearAlgebra
using JLD2
using ODE

ENV["GKSwstype"]="nul"


# -----------------------------------------------------
# Evolves wave using hierarchical sparse grids with 
# a fixed time step scheme to allow for better
# comparison with combination technique
# -----------------------------------------------------
function isotropic_grid(D,k,n,wavenumber,t0,t1;scheme="sparse",phase=0.0,steps=100)
    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1)+phase)
    f0coeffs, v0coeffs = traveling_wave(k, n, wavenumber;scheme=scheme,phase)
    tstep = (t1-t0)/steps
    soln = []
    for t_start in t0:tstep:(t1-tstep)
        t_end = t_start + tstep
        soln = wave_evolve(D, k, n, f0coeffs, v0coeffs, t_start, t_end; scheme=scheme, order="4")
        coefficients = floor(Int,size(soln[2][end])[1]/2)
        f0coeffs = soln[2][end][1:coefficients]
        v0coeffs = soln[2][end][coefficients+1:end]
    end    
    dict = V2D(D, k, n, soln[2][end];scheme=scheme)
    iso_err = mcerr_rel(x->reconstruct_DG(dict, [x...]), truesoln, D)
    println("Monte Carlo Error on isotropic grid: ", iso_err, ", k=",k,", n=",n)
    return soln, iso_err
end


# -----------------------------------------------------
# Approximation error CT_end in 2D
# -----------------------------------------------------
function figure10a(D,k,n,wavenumber,t0,t1;phase=0.0,stepsize=0.01,steps=100)
    println("Figure 10a - Approximation error CT in 2D")

    sparsegrid_f = combine_end(D,k,n,wavenumber,0,0,t0,t1,true;phase=phase,steps=steps,order="4",recombine=false)[1]
    approx_CT = x -> reconstruct_DG(sparsegrid_f, [x...]) 
    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1)+phase)

    x = y = 0:stepsize:1

    CT_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    truesoln_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    for i in 1:Int(1/stepsize+1)
        for j in 1:Int(1/stepsize+1)
            CT_matrix[i,j] = approx_CT([x[i],y[j]])
            truesoln_matrix[i,j] = truesoln([x[i],y[j]])
        end
    end
    p0 = heatmap(10^5*(truesoln_matrix-CT_matrix),title="Approximation error CT in 2D \n n=$n, k=$k",
                    c=:balance,size=(500,500),
                    xticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    yticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    cbartitle=raw"$\times 10^{-5}$")
    savefig(p0,"./Figure10_CT_end_n=$(n)_k=$k.png")
    println("Finished Figure 10a - CT_end")
end


# -----------------------------------------------------
# Approximation error CT_between in 2D
# -----------------------------------------------------
function figure10b(D,k,n,wavenumber,t0,t1;phase=0.0,stepsize=0.01,steps=100)
    println("Figure 10b - Approximation error CT_between in 2D")
    
    sparsegrid_f = combine_between(D,k,n,wavenumber,0,0,t0,t1;phase,steps)[1]
    approx_CT_between = x -> reconstruct_DG(sparsegrid_f, [x...]) 
    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1)+phase)

    x = y = 0:stepsize:1

    CT_between_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    truesoln_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    for i in 1:Int(1/stepsize+1)
        for j in 1:Int(1/stepsize+1)
            CT_between_matrix[i,j] = approx_CT_between([x[i],y[j]])
            truesoln_matrix[i,j] = truesoln([x[i],y[j]])
        end
    end
    p1 = heatmap(10^5*(truesoln_matrix-CT_between_matrix),title="Approximation error CT_between in 2D \n n=$n, k=$k",
                    c=:balance,size=(500,500),
                    xticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    yticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    cbartitle=raw"$\times 10^{-5}$")
    savefig(p1,"./Figure10_CT_between_n=$(n)_k=$k.png")
    println("Finished Figure 10b - CT_between")
end


# -----------------------------------------------------
# Approximation error sparse grid in 2D
# -----------------------------------------------------
function figure10c(D,k,n,wavenumber,t0,t1;phase=0.0,stepsize=0.01,steps=100)
    println("Figure 10c - Approximation error sparse grid in 2D")

    soln = isotropic_grid(D, k, n, wavenumber, t0, t1; phase=phase, steps=steps)[1]
    dict = V2D(D, k, n, soln[2][end])
    approx = x -> reconstruct_DG(dict, [x...])

    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1) + phase)

    x = y = 0:stepsize:1

    sparse_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    truesoln_matrix = zeros(Int(1/stepsize+1),Int(1/stepsize+1))
    for i in 1:Int(1/stepsize+1)
        for j in 1:Int(1/stepsize+1)
            sparse_matrix[i,j] = approx([x[i],y[j]])
            truesoln_matrix[i,j] = truesoln([x[i],y[j]])
        end
    end
    p2 = heatmap(10^5*(truesoln_matrix-sparse_matrix),title="Approximation error SG in 2D \n n=$n, k=$k",
                    c=:balance,size=(500,500),
                    xticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    yticks=([1:20:101;],["0.0","0.2","0.4","0.6","0.8","1.0"]),
                    cbartitle=raw"$\times 10^{-5}$")
    savefig(p2,"./Figure10_SG_n=$(n)_k=$k.png")
    println("Finished Figure 10c - Sparse Grid")
end



# -----------------------------------------------------
# Main routine:
# -----------------------------------------------------

wavenumber = [1,2];
D = length(wavenumber);
k=5;
n=5;
phase = 0.4;
t0=0;
t1=0.54;
steps=200

figure10a(D,k,n,wavenumber,t0,t1;phase,steps)
figure10b(D,k,n,wavenumber,t0,t1;phase,steps)
figure10c(D,k,n,wavenumber,t0,t1;phase,steps)