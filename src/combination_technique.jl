# Sparse grids combination technique
function comb_formula(d,f)
    for q = 1:d-1
        (-1)^q*binomial(d-1,q)*sum(f)
    end
end