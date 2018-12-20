using GalerkinSparseGrids
using Test
using HCubature
using StaticArrays
using SparseArrays

#--------------------------------------
# Testing transformations between bases
#--------------------------------------

n = 5

@testset "transformations.jl" begin
	@info "Testing transformation between point and nodal bases... "
	@testset "Point to Nodal" begin
		for k in [2,3,5]
			p2n = transform_1D(k, n, "points", "nodal")
			n2p = transform_1D(k, n, "nodal", "points")
			@test norm(p2n*n2p-I) < 1e-15*(10^k)
			@test norm(n2p*p2n-I) < 1e-15*(10^k)
		end
	end 
	
	@info "Testing transformation between point and modal bases... "
	@testset "Point to Modal" begin
		for k in [2,3,5]
			p2m = transform_1D(k, n, "points", "modal")
			m2p = transform_1D(k, n, "modal", "points")
			@test norm(p2m*m2p-I) < 1e-15*(10^k)
			@test norm(m2p*p2m-I) < 1e-15*(10^k)
		end
	end

	@info "Testing transformation between nodal and modal bases... " 
	@testset "Nodal to Modal" begin
		for k in [2,3,5]
			n2m = transform_1D(k, n, "nodal", "modal")
			m2n = transform_1D(k, n, "modal", "nodal")
			@test norm(n2m*m2n-I) < 1e-15*(10^k)
			@test norm(m2n*n2m-I) < 1e-15*(10^k)
		end
	end

	@info "Testing transformation from position to modal bases... "
	@testset "Position to Modal" begin
		for k in [2,3,5]
			pos2m = transform_1D(k, n, "pos", "modal")
			m2pos = transform_1D(k, n, "modal", "pos")
			@test norm(pos2m*m2pos-I) < 1e-15*(10^k)
			@test norm(m2pos*pos2m-I) < 1e-15*(10^k)
		end
	end
end