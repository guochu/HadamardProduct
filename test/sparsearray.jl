# Tests for Hadamard products involving higher-order sparse tensors from
# SparseArrayKit.jl. Unlike the SparseArrays stdlib, `SparseArray{T,N}` supports
# arbitrary dimensionality, so all three cases (pointwise, partially shared, outer)
# can be exercised with 3D/4D outputs.
using Random
using SparseArrayKit
import HadamardProduct

# --- extend the internal tensor interfaces so that Hadamard products produce
# --- SparseArrays instead of the default dense Arrays. This mirrors how a downstream
# --- package would hook a custom tensor type into `@hadamard`.
function HadamardProduct.hadamardproduct_type(
        TC, A::SparseArray, pA::Index2Tuple, conjA::Bool,
        B::SparseArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple
    )
    return SparseArray{TC, HadamardProduct.numind(pAB)}
end
HadamardProduct.tensoralloc(::Type{SparseArray{T,N}}, dims) where {T,N} =
    SparseArray{T}(undef, dims)

# fill a SparseArray with random nonzeros
function fillsprand!(A::SparseArray, p::Float64)
    for I in CartesianIndices(A)
        rand() < p && (A[I] = rand())
    end
    return A
end

@testset "SparseArrayKit sparse tensors" begin
    @testset "pointwise products (3D)" begin
        A = fillsprand!(SparseArray{Float64}(undef, (3, 4, 5)), 0.4)
        B = fillsprand!(SparseArray{Float64}(undef, (3, 4, 5)), 0.4)
        C = SparseArray{Float64}(undef, (3, 4, 5))
        @hadamard C[i, j, k] = A[i, j, k] * B[i, j, k]
        ref = collect(A) .* collect(B)
        @test C ≈ ref
        @test C isa SparseArray{Float64, 3}
        # the result stays sparse: nonzeros only where both inputs are nonzero
        @test nonzero_length(C) == count(!iszero, ref)

        # accumulation
        D = SparseArray{Float64}(undef, (3, 4, 5))
        @hadamard D[i, j, k] += A[i, j, k] * B[i, j, k]
        @test D ≈ ref
    end

    @testset "partially shared products (3D x 2D -> 4D)" begin
        # C[i, j, k, l] = A[i, k, l] * B[l, j]
        A = fillsprand!(SparseArray{Float64}(undef, (2, 3, 4)), 0.4)
        B = fillsprand!(SparseArray{Float64}(undef, (4, 5)), 0.4)
        C = SparseArray{Float64}(undef, (2, 5, 3, 4))
        @hadamard C[i, j, k, l] = A[i, k, l] * B[l, j]
        ref = zeros(2, 5, 3, 4)
        for i in 1:2, j in 1:5, k in 1:3, l in 1:4
            ref[i, j, k, l] = A[i, k, l] * B[l, j]
        end
        @test C ≈ ref
        @test C isa SparseArray{Float64, 4}
        @test nonzero_length(C) == count(!iszero, ref)

        # accumulation into a nonempty sparse output exercises the β=1 branch
        C0 = fillsprand!(SparseArray{Float64}(undef, (2, 5, 3, 4)), 0.2)
        C0before = collect(C0)
        @hadamard C0[i, j, k, l] += A[i, k, l] * B[l, j]
        @test C0 ≈ C0before + ref
    end

    @testset "outer products (2D x 2D -> 4D)" begin
        # C[i, j, k, l] = A[i, k] * B[j, l]
        A = fillsprand!(SparseArray{Float64}(undef, (2, 3)), 0.4)
        B = fillsprand!(SparseArray{Float64}(undef, (4, 5)), 0.4)
        C = SparseArray{Float64}(undef, (2, 4, 3, 5))
        @hadamard C[i, j, k, l] = A[i, k] * B[j, l]
        ref = zeros(2, 4, 3, 5)
        for i in 1:2, j in 1:4, k in 1:3, l in 1:5
            ref[i, j, k, l] = A[i, k] * B[j, l]
        end
        @test C ≈ ref
        @test C isa SparseArray{Float64, 4}
        # outer product: nonzeros = product of input nonzeros
        @test nonzero_length(C) == nonzero_length(A) * nonzero_length(B)
    end

    @testset "definition mode allocates a SparseArray" begin
        A = fillsprand!(SparseArray{Float64}(undef, (2, 3, 4)), 0.4)
        B = fillsprand!(SparseArray{Float64}(undef, (4, 5)), 0.4)
        # the `:=` form uses the extended `tensoralloc`, so the result is a SparseArray
        @hadamard C[i, j, k, l] := A[i, k, l] * B[l, j]
        @test C isa SparseArray{Float64, 4}
        ref = zeros(2, 5, 3, 4)
        for i in 1:2, j in 1:5, k in 1:3, l in 1:4
            ref[i, j, k, l] = A[i, k, l] * B[l, j]
        end
        @test C ≈ ref

        # the function based API produces SparseArrays as well
        C2 = hadamardproduct((:i, :j, :k, :l), A, (:i, :k, :l), B, (:l, :j))
        @test C2 isa SparseArray{Float64, 4}
        @test C2 ≈ ref
    end

    @testset "complex scalars and conjugation" begin
        A = fillsprand!(SparseArray{ComplexF64}(undef, (3, 4)), 0.4)
        B = fillsprand!(SparseArray{ComplexF64}(undef, (3, 4)), 0.4)
        C = SparseArray{ComplexF64}(undef, (3, 4))
        @hadamard C[i, j] = conj(A[i, j]) * B[i, j]
        ref = conj.(collect(A)) .* collect(B)
        @test C ≈ ref
    end
end
