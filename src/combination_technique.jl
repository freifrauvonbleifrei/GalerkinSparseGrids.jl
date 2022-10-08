using GalerkinSparseGrids

# Returns Dictionary with combination factors at relevant levels 
function get_coeffs(D,n)
    n_vec = fill(n,D)
    ls = ntuple(i->(n+1), D)
    coeffs = Dict()

    for q in 0:(D-1)
        factor = (-1)^q * binomial(D-1,q)

        for level in CartesianIndices(ls)
            cutoff(Val(:full),level,n_vec) && continue
            
            level_n = Vector{Int64}()
            for j in 1:D
                append!(level_n, level[j])
            end

            if sum(level_n) == n - q + D
                coeffs[level] = factor
            end
        end
    end
    return coeffs
end

# Combine grid solutions only after all solver time steps are done
# generate_coeffs is a boolean, deciding if f0coeffs and v0coeffs have to be initialised via traveling_wave (--> true) 
# or if they are already transfered in Dictionary form (--> false)
function combine_end(D,k,n,wavenumber,f_result,v_result,t0,t1,generate_coeffs;phase=0.0,steps=100,order="4",recombine=true)
    
    coeffs = get_coeffs(D,n)
   
    f_exact = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1)+phase)

    len = get_size(Val(D), k, n, Val(:full)) 
    vec = zeros(len)
    sparsegrid_f = V2D(D,k,n,vec,scheme="sparse")
    sparsegrid_v = V2D(D,k,n,vec,scheme="sparse")

    for (level,factor) in coeffs # only levels where combination coefficient != 0 are considered
        
        level_n = Vector{Int64}()
        for j in 1:D
            append!(level_n, level[j])
        end

        level_n = level_n.-1

        if generate_coeffs # 1. time step --> initial position f and velocity v are generatet by traveling_wave

            f0coeffs, v0coeffs = traveling_wave(k, level_n, wavenumber; scheme="full", phase)

        else # 2+ time step --> decombination is needed (divide current result onto component grids)
            
            len = get_size(Val(D), k, level_n, Val(:full))
            vec = zeros(len)
            subgrid_f = V2D(D,k,level_n,vec,scheme="full")
            subgrid_v = V2D(D,k,level_n,vec,scheme="full")

            for (key,value) in subgrid_f
                for c in eachindex(value), m in eachindex(value[c])
                    subgrid_f[key][c][m] = f_result[key][c][m]
                    subgrid_v[key][c][m] = v_result[key][c][m]
                end
            end

            f0coeffs = D2V(D,k,level_n,subgrid_f,scheme="full")
            v0coeffs = D2V(D,k,level_n,subgrid_v,scheme="full")
        end

        # Evolve function
        if order == "4"
            if recombine==false # only combine once after all solver steps are finished
                tstep = (t1-t0)/steps
                soln = []
                for t_start in t0:tstep:(t1-tstep)
                    t_end = t_start + tstep
                    soln = wave_evolve(D, k, level_n, f0coeffs, v0coeffs, t_start, t_end; order="4",scheme="full")
                    coefficients = floor(Int,size(soln[2][end])[1]/2)
                    f0coeffs = soln[2][end][1:coefficients]
                    v0coeffs = soln[2][end][coefficients+1:end]
                end
            else # combine after every solver step
                soln = wave_evolve(D, k, level_n, f0coeffs, v0coeffs, t0, t1; order="4", scheme="full")
            end
        else
            soln = wave_evolve(D, k, level_n, f0coeffs, v0coeffs, t0, t1; order="45", scheme="full")
        end

        coefficients = floor(Int,size(soln[2][end])[1]/2)
        dict_f = V2D(D,k,level_n,soln[2][end][1:coefficients],scheme="full")
        dict_v = V2D(D,k,level_n,soln[2][end][coefficients+1:end],scheme="full")
        
        for (key,value) in dict_f
            for c in eachindex(value), m in eachindex(value[c])
                sparsegrid_f[key][c][m] += factor * dict_f[key][c][m]
                sparsegrid_v[key][c][m] += factor * dict_v[key][c][m]
            end
        end
    end

    f_rep = x -> reconstruct_DG(sparsegrid_f, [x...])
    err = mcerr_rel(f_exact, f_rep, D)

    return sparsegrid_f, sparsegrid_v, err
end

# Combine grids after every solver time step
function combine_between(D,k,n,wavenumber,f_result,v_result,t0,t1;phase=0.0,steps=100)
    tstep = (t1-t0)/steps
    sparsegrid_f, sparsegrid_v = combine_end(D,k,n,wavenumber,f_result,v_result,t0,t0+tstep,true;phase=phase,recombine=true)
    err_end = NaN

    for t_start in (t0+tstep):tstep:(t1-tstep)
        t_end = t_start + tstep
        sparsegrid_f, sparsegrid_v, err_end = combine_end(D,k,n,wavenumber,sparsegrid_f,sparsegrid_v,t_start,t_end,false;phase=phase,order="4",recombine=true) #phase eig nicht mehr gebraucht, da kein generate_coeffs==true
    end

    return sparsegrid_f,sparsegrid_v,err_end
end

function compare(D,k,n,wavenumber,t0,t1;steps=10)
    println("D = ",D, ", k = ", k, ", n = ", n)
    
    ## Isotropic Grids
    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1))
    f0coeffs, v0coeffs = traveling_wave(k, n, wavenumber)
    tstep = (t1-t0)/steps
    soln = []
    for t_start in t0:tstep:(t1-tstep)
        t_end = t_start + tstep
        soln = wave_evolve(D, k, n, f0coeffs, v0coeffs, t_start, t_end; order="4")
        coefficients = floor(Int,size(soln[2][end])[1]/2)
        f0coeffs = soln[2][end][1:coefficients]
        v0coeffs = soln[2][end][coefficients+1:end]
    end
    dict = V2D(D, k, n, soln[2][end])
    iso_err = mcerr_rel(x->reconstruct_DG(dict, [x...]), truesoln, D)
    println("Monte Carlo Error on isotropic grid: ", iso_err)


    ## Anisotropic Grids

    # Combination after every time step
    aniso_err_btw = combine_between(D,k,n,wavenumber,0,0,t0,t1;steps=steps)[3]
    println("Monte Carlo Error on anisotropic grid (combined after every time step): ", aniso_err_btw)

    # Combination once after all time steps
    soln = combine_end(D,k,n,wavenumber,0,0,t0,t1,true;steps=steps,order="4",recombine=false)
    aniso_err_end = soln[3] 
    println("Monte Carlo Error on anisotropic grid (combined once after all time steps): ", aniso_err_end)
end