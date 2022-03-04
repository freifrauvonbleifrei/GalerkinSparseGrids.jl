using GalerkinSparseGrids
using LinearAlgebra
using Plots

function get_coeffs(D,n)

    n_vec = fill(n,D)
    ls = ntuple(i->(n+1), D)
    coeffs = Dict()

    for q in 0:(D-1)
        factor = (-1)^q * binomial(D-1,q)

        for level in CartesianIndices(ls)
            cutoff(Val(:full),level,n_vec) && continue # welcher cutoff? (entlang diagonale vs. rechteck links oben)
            # hier cutoff entlang Diagonale -> n::Int64
            
            #CartesianIndex als Vektor darstellen
            level_n = Vector{Int64}() #[]
            for j in 1:D
                append!(level_n, level[j])
            end

            if sum(level_n) == n - q + D
                coeffs[level] = factor
            end
        end
    end
    # println("Coefficients for CT: ", coeffs)
    return coeffs
end

function get_test_coeffs(D,n) #n als Array
    level = CartesianIndex(ntuple(i->(n[i]-1),D)) #funktioniert nur für gerade n?
    coeffs = Dict()
    coeffs[level] = 1
    return coeffs
end

# Ein Kombinationsschritt, wenn t0 und t1 ganzes Zeitintervall abdecken, dann Kombitechnik, wo erst nach allen Zeitschritten kombiniert wird
# generate_coeffs ist boolean, ob f0coeffs & v0coeffs neu initialisiert werden (über traveling_wave) (--> true) oder ob bereits Koeffizienten in Dictionary Form (f_result, v_result)
# übergeben wurden (--> false)
function combine_end(D,k,n,wavenumber,f_result,v_result,t0,t1,generate_coeffs,test)
    
    if test
        coeffs = get_test_coeffs(D,n)
    else
        coeffs = get_coeffs(D,n)
    end
   

    # f0 = x->sin(2*pi*x[1])*sin(2*pi*x[2])
    f_exact = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1))

    # # Convert n::Int to n::Vector
    # n_vec = fill(n,D)

    # Leere Vorlagen für soln erstellen
    len = get_size(Val(D), k, n, Val(:full)) # get_size mit n::Int #wieso n+D und nicht n+1? ##changed from n+D to n
    vec = zeros(len)
    sparsegrid_f = V2D(D,k,n,vec,scheme="sparse") # welches scheme? spzeros(len,len) oder vec von welcher Länge? #wieso n+D und nicht n+1?? ##changed from n+D to n
    sparsegrid_v = V2D(D,k,n,vec,scheme="sparse") #wieso n+D und nicht n+1? ##changed from n+D to n

    # println("sparsegrid_f: ",sparsegrid_f)
    # err_all = []
    # err_all_dict = []

    for (level,factor) in coeffs # geht nur über Level mit Koeffizienten != 0
        
        # CartesianIndex als Vektor darstellen
        level_n = Vector{Int64}() #[]
        for j in 1:D
            append!(level_n, level[j])
        end

        level_n = level_n.-1

        # Unterscheidung in 1. Zeitschritt und restliche Schritte
        if generate_coeffs # == true : 1. Zeitschritt --> initial position f und velocity v werden über traveling_wave generiert

            f0coeffs, v0coeffs = traveling_wave(k, level_n, wavenumber, scheme="full") #geht wsl nur, wenn f0 = x->sin(2*pi*x[1])*sin(2*pi*x[2]) verwendet wird
            # println("\nf0coeffs von travelling_wave bei level_n = ", level_n, ": ", f0coeffs)

        else # 2+ Zeitschritt --> benötigen zuerst noch Dekombinierung (Zwischenergebnis auf Gitter aufteilen)
            
            # Leere Vorlagen für subgrid erstellen
            len = get_size(Val(D), k, level_n, Val(:full)) # get_size mit n::Vector ## changed from level_n to level_n.-1
            # vec = Vector{Float64}(undef,len)
            vec = zeros(len)
            subgrid_f = V2D(D,k,level_n,vec,scheme="full") ## changed from level_n to level_n.-1
            subgrid_v = V2D(D,k,level_n,vec,scheme="full") ## changed from level_n to level_n.-1

            # println("\nsubgrid_f: ", subgrid_f)
            # println("\nf_result: ", f_result, "\n")

            # über subgrid_f iterieren
            for (key,value) in subgrid_f
                for c in eachindex(value), m in eachindex(value[c])
                    subgrid_f[key][c][m] = f_result[key][c][m]
                    subgrid_v[key][c][m] = v_result[key][c][m]
                end
            end

            # # über subgrid_v iterieren #alternativ: f und v in eine for schleife und vorsichtshalber assert subgrid_f = subgrid_v
            # for (key,value) in subgrid_v
            #     for c in eachindex(value), m in eachindex(value[c])
            #         subgrid_v[key][c][m] = v_result[key][c][m]
            #     end
            # end

            f0coeffs = D2V(D,k,level_n,subgrid_f,scheme="full") ## changed from level_n to level_n.-1
            v0coeffs = D2V(D,k,level_n,subgrid_v,scheme="full") ## changed from level_n to level_n.-1
        end

        # Evolve function
        soln = wave_evolve(D, k, level_n, f0coeffs, v0coeffs, t0, t1; order="78", scheme="full") ## needs to be same level_n as f0coeffs
        dict_f = V2D(D,k,level_n,soln[2][1],scheme="full") ## changed from level_n to level_n.-1
        # println("dict_f", dict_f)
        dict_v = V2D(D,k,level_n,soln[2][2],scheme="full") ## changed from level_n to level_n.-1

        # über dict_f iterieren
        for (key,value) in dict_f #sparsegrid_f
            for c in eachindex(value), m in eachindex(value[c])
                sparsegrid_f[key][c][m] += factor * dict_f[key][c][m]
                sparsegrid_v[key][c][m] += factor * dict_v[key][c][m]
            end
        end

        # # über dict_v iterieren
        # for (key,value) in dict_v #sparsegrid_v 
        #     for c in eachindex(value), m in eachindex(value[c])
        #         sparsegrid_v[key][c][m] += factor * dict_v[key][c][m]
        #     end
        # end

        #reconstruct_DG
        f_rep_dict = x -> reconstruct_DG(dict_f, [x...])
        # f_rep = x -> reconstruct_DG(sparsegrid_f, [x...])

        #mcerr
        err_dict = mcerr(f_exact, f_rep_dict, D)
        # err = mcerr(f_exact, f_rep, D)
        # println("k = ", k, ", n = ", level_n, ", mcerr (dict) = ", err_dict)
        # println("k = ", k, ", n = ", level_n, ", mcerr (kombiniert) = ", err)
        # append!(err_all_dict, err_dict)
        # append!(err_all, err)
    end

    f_rep = x -> reconstruct_DG(sparsegrid_f, [x...])
    err = mcerr(f_exact, f_rep, D)
    println("k = ", k, ", n = ", n, ", mcerr = ", err)
    
    # x = 1:n;
    # plot(x,err_all[1:n],title="Monte Carlo Error",xlabel="n",ylabel="mcerr",label="k=$k")

    return sparsegrid_f, sparsegrid_v, err
end

# Gitter nach jedem Zeitschritt kombinieren
function combine_between(D,k,n,wavenumber,f_result,v_result,t0,t1,test)
    tstep = (t1-t0)/10
    sparsegrid_f, sparsegrid_v = combine_end(D,k,n,wavenumber,f_result,v_result,t0,t0+tstep,true,test)
    err_end = -1.0
    for t_start in (t0+tstep):tstep:(t1-tstep)
        t_end = t_start + tstep
        # println("\nAktuelles t_start = ", t_start)
        sparsegrid_f, sparsegrid_v, err_end = combine_end(D,k,n,wavenumber,sparsegrid_f,sparsegrid_v,t_start,t_end,false,test)
    end
    return err_end
end

function compare(D,k,n,wavenumber,f0,v0,t0,t1)
    println("D = ",D, ", k = ", k, ", n = ", n)
    
    ## Isotropic Grids
    truesoln = x -> cos(2*pi*(dot(wavenumber,x) - sqrt(dot(wavenumber,wavenumber))*t1))
    f0coeffs, v0coeffs = traveling_wave(k, n, wavenumber)
    soln = wave_evolve(D, k, n, f0coeffs, v0coeffs, t0, t1)
    dict = V2D(D, k, n, soln[2][end])
    iso_err = mcerr(x->reconstruct_DG(dict, [x...]), truesoln, D)
    println("Monte Carlo Error on isotropic grid: ", iso_err)


    ## Anisotropic Grids

    # Combination after every time step
    aniso_err_btw = combine_between(D,k,n,wavenumber,0,0,t0,t1,false) #test::boolean
    println("Monte Carlo Error on anisotropic grid (combined after every time step): ", aniso_err_btw)

    # Combination once after all time steps
    aniso_err_end = combine_end(D,k,n,wavenumber,f0,v0,t0,t1,true,false)[3] #generate_coeffs:boolean, test::boolean
    println("Monte Carlo Error on anisotropic grid (combined once after all time steps): ", aniso_err_end)
end

# Test:
println("\n\n-------------------------------------------------------------------")
# D = 2; # if D is changed, m has to be changed too
k = 2;
n = 2; # Gitter ist immer isotrop
wavenumber = [1,2];
D = length(wavenumber)

# f0 = x->sin(2*pi*x[1])*sin(2*pi*x[2])
# f0 = x->0
# v0 = x->0
t0 = 0
t1 = 0.54

# get_coeffs(D,n)
combine_end(D,k,n,wavenumber,0,0,t0,t1,true,false) #generate_coeffs,test #f0,v0 = 0, since coeffs are generated through travelling_wave
# println("\nerr2 = ", err2)
# combine_between(D,k,n,wavenumber,0,0,t0,t1,false) #test::boolean
# get_test_coeffs(D,n)
# compare(D,k,n,wavenumber,0,0,t0,t1)
# >>>>>>> df276d3b7c88a58037c2596400e2dd95b9eecb8d
