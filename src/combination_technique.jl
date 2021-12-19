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
    println("Coefficients for CT: ", coeffs)
    return coeffs
end

function get_test_coeffs(D,n)
    level = CartesianIndex(ntuple(i->(n),D))
    coeffs = Dict()
    coeffs[level] = 1
    return coeffs
end

# Ein Kombinationsschritt, wenn t0 und t1 ganzes Zeitintervall abdecken, dann Kombitechnik, wo erst nach allen Zeitschritten kombiniert wird
# generate_coeffs ist boolean, ob f0coeffs & v0coeffs neu initialisiert werden (über traveling_wave) (--> true) oder ob bereits Koeffizienten in Dictionary Form (f_result, v_result)
# übergeben wurden (--> false)
function combine_end(D,k,n,m,f_result,v_result,t0,t1,generate_coeffs)
    # coeffs = get_coeffs(D,n)
    coeffs = get_test_coeffs(D,n)

    f_exact = x->sin(2*pi*x[1])*sin(2*pi*x[2])

    # # Convert n::Int to n::Vector
    # n_vec = fill(n,D)

    # Leere Vorlagen für soln erstellen
    len = get_size(Val(D), k, n+D, Val(:full)) # get_size mit n::Int
    # vec = Vector{Float64}(0.,len)
    vec = zeros(len)
    sparsegrid_f = V2D(D,k,n+D,vec,scheme="sparse") # welches scheme? spzeros(len,len) oder vec von welcher Länge?
    # println("sparsegrid_f", sparsegrid_f)
    sparsegrid_v = V2D(D,k,n+D,vec,scheme="sparse")
    err_all = []
    err_all_dict = []

    for (level,factor) in coeffs # geht nur über Level mit Koeffizienten != 0
        
        # CartesianIndex als Vektor darstellen
        level_n = Vector{Int64}() #[]
        for j in 1:D
            append!(level_n, level[j])
        end

        # Unterscheidung in 1. Zeitschritt und restliche Schritte
        if generate_coeffs # == true : 1. Zeitschritt --> initial position f und velocity v werden über traveling_wave generiert

            f0coeffs, v0coeffs = traveling_wave(k, level_n, m, scheme="full") #geht wsl nur, wenn f0 = x->sin(2*pi*x[1])*sin(2*pi*x[2]) verwendet wird

        else # 2+ Zeitschritt --> benötigen zuerst noch Dekombinierung (Zwischenergebnis auf Gitter aufteilen)
            # println("1")
            # Leere Vorlagen für subgrid erstellen
            len = get_size(Val(D), k, level_n, Val(:full)) # get_size mit n::Vector
            # vec = Vector{Float64}(undef,len)
            vec = zeros(len)
            subgrid_f = V2D(D,k,level_n,vec,scheme="full")
            subgrid_v = V2D(D,k,level_n,vec,scheme="full")
            # println("2")
            # über subgrid_f iterieren
            for (key,value) in subgrid_f
                for c in eachindex(value), m in eachindex(value[c])
                    subgrid_f[key][c][m] = f_result[key][c][m]
                end
            end
            # über subgrid_v iterieren #alternativ: f und v in eine for schleife und vorsichtshalber assert subgrid_f = subgrid_v
            for (key,value) in subgrid_v
                for c in eachindex(value), m in eachindex(value[c])
                    subgrid_v[key][c][m] = v_result[key][c][m]
                end
            end
            # println("3")
            f0coeffs = D2V(D,k,level_n,subgrid_f,scheme="full")
            v0coeffs = D2V(D,k,level_n,subgrid_v,scheme="full")
        end
        println("4: level_n = ", level_n)
        # Evolve function
        soln = wave_evolve(D, k, level_n, f0coeffs, v0coeffs, t0, t1; order="78", scheme="full")
        # println("5")
        dict_f = V2D(D,k,level_n,soln[2][1],scheme="full")
        # println("dict_f", dict_f)
        dict_v = V2D(D,k,level_n,soln[2][2],scheme="full")
        # println("6")
        # über dict_f iterieren
        for (key,value) in dict_f
            for c in eachindex(value), m in eachindex(value[c])
                sparsegrid_f[key][c][m] += factor * dict_f[key][c][m]
            end
        end
        # über dict_v iterieren
        for (key,value) in dict_v
            for c in eachindex(value), m in eachindex(value[c])
                sparsegrid_v[key][c][m] += factor * dict_v[key][c][m]
            end
        end
        # println("7")
        #reconstruct_DG
        f_rep_dict = x -> reconstruct_DG(dict_f, [x...])
        f_rep = x -> reconstruct_DG(sparsegrid_f, [x...])
        #mcerr
        f_exact = x -> cos(2*pi*(dot(m,x) - sqrt(dot(m,m))*t1))
        err_dict = mcerr(f_exact, f_rep_dict, D)
        err = mcerr(f_exact, f_rep, D)
        println("k = ", k, ", n = ", level_n, ", mcerr (dict) = ", err_dict)
        println("k = ", k, ", n = ", level_n, ", mcerr (kombiniert) = ", err)
        append!(err_all_dict, err_dict)
        append!(err_all, err)
        # println("8")
    end
    # println("9")
    f_rep = x -> reconstruct_DG(sparsegrid_f, [x...])
    err = mcerr(f_exact, f_rep, D)
    println("k = ", k, ", n = ", n, ", mcerr = ", err)
    # x = 1:n;
    # plot(x,err_all[1:n],title="Monte Carlo Error",xlabel="n",ylabel="mcerr",label="k=$k")
    return sparsegrid_f, sparsegrid_v
end

# Gitter nach jedem Zeitschritt kombinieren
function combine_between(D,k,n,m,f_result,v_result,t0,t1)
    tstep = t1-t0
    sparsegrid_f, sparsegrid_v = combine_end(D,k,n,m,f_result,v_result,t0,t0+tstep,true)
    for t_start in t0+tstep:tstep:(t1-tstep)
        t_end = t_start + tstep
        println("\nAktuelles t_start = ", t_start)
        sparsegrid_f, sparsegrid_v = combine_end(D,k,n,m,sparsegrid_f,sparsegrid_v,t_start,t_end,false)
    end
end

# Test:
println("\n\n-------------------------------------------------------------------")
D = 2;
k = 3;
n = 3; # Gitter ist immer isotrop
m = [1,2];
f0 = x->sin(2*pi*x[1])*sin(2*pi*x[2])
v0 = x->0
t0 = 0
t1 = 1

# get_coeffs(D,n)
combine_end(D,k,n,m,f0,v0,t0,t1,true)
# combine_between(D,k,n,m,0,0,t0,t1)
# get_test_coeffs(D,n)