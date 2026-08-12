using Test
using LinearAlgebra
using HadamardProduct

@testset "HadamardProduct" begin
    include("hadamard.jl")
    include("tensoroperations.jl")
    include("customtensor.jl")
    include("sparsearray.jl")
end
