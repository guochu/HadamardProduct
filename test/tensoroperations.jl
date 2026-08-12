# Cross-check the Hadamard product against an independent reference implementation, based on
# the delta (identity matrix) equivalence used for debugging:
#
#     C[i,k,j] = A[i,k] * B[k,j]   (Hadamard: shared index k combined pointwise)
#       ⟺      C[i,k,j] = Σ_{k'} A[i,k'] * I[k',k] * B[k',j]
#
# i.e. the shared indices are duplicated (k -> k') and pinned by identity matrices, which
# turns the pointwise product into an equivalent tensor contraction. The reference below
# evaluates that contraction with explicit loops and identity matrices, and is therefore
# completely independent of the broadcast-based kernel.
#
# Note: `TensorOperations.@tensor` cannot express the pointwise (shared) indices directly,
# since any index appearing twice on the right hand side is treated as a contraction index
# that must be absent from the left hand side. It can however validate the pure outer
# product case (no shared indices), which is included below.
import TensorOperations

"""
    ref_hadamard(A, IA, B, IB, IC; conjA=false, conjB=false)

Reference implementation of the Hadamard product `C[IC] = A[IA] ⊙ B[IB]` via the delta
equivalence, evaluated with explicit loops. `IA`, `IB` and `IC` are index label iterables.
"""
function ref_hadamard(A, IA, B, IB, IC; conjA=false, conjB=false)
    oA = [l for l in IA if !(l in IB)]
    S = [l for l in IA if l in IB]
    oB = [l for l in IB if !(l in IA)]
    pA = [findfirst(==(l), IA) for l in vcat(oA, S)]
    pB = [findfirst(==(l), IB) for l in vcat(S, oB)]
    Av = permutedims(A, pA)          # dims: (oA..., S...)
    Bv = permutedims(B, pB)          # dims: (S..., oB...)
    nO, nS, nB = length(oA), length(S), length(oB)
    szS = size(Av)[nO+1:nO+nS]
    szC = (size(Av)[1:nO]..., szS..., size(Bv)[nS+1:nS+nB]...)
    TC = promote_type(eltype(A), eltype(B))
    Cnat = Array{TC}(undef, szC)
    for idx in CartesianIndices(Cnat)
        ioA, iS, ioB = Tuple(idx)[1:nO], Tuple(idx)[nO+1:nO+nS], Tuple(idx)[nO+nS+1:end]
        s = zero(TC)
        for iS′ in CartesianIndices(szS)
            a = conjA ? conj(Av[ioA..., Tuple(iS′)...]) : Av[ioA..., Tuple(iS′)...]
            b = conjB ? conj(Bv[Tuple(iS′)..., ioB...]) : Bv[Tuple(iS′)..., ioB...]
            s += a * (Tuple(iS′) == iS ? one(TC) : zero(TC)) * b
        end
        Cnat[idx] = s
    end
    pC = [findfirst(==(l), vcat(oA, S, oB)) for l in IC]
    return permutedims(Cnat, pC)
end

@testset "cross-check with TensorOperations" begin
    @testset "Float64" begin
        # outer product (no shared indices): also cross-checked with TensorOperations.@tensor
        A, B = rand(3), rand(4)
        @hadamard C[i, j] := A[i] * B[j]
        TensorOperations.@tensor Cref[i, j] := A[i] * B[j]
        @test C ≈ Cref
        @test C ≈ ref_hadamard(A, (:i,), B, (:j,), (:i, :j))

        # pointwise product (all indices shared)
        A, B = rand(3, 4), rand(3, 4)
        @hadamard C[i, j] := A[i, j] * B[i, j]
        @test C ≈ ref_hadamard(A, (:i, :j), B, (:i, :j), (:i, :j))

        # partially shared indices
        A, B = rand(2, 3), rand(3, 4)
        @hadamard C[i, k, j] := A[i, k] * B[k, j]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:k, :j), (:i, :k, :j))

        # shared indices of B in a different order
        A, B = rand(2, 3), rand(4, 3)
        @hadamard C[i, k, j] := A[i, k] * B[j, k]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:j, :k), (:i, :k, :j))

        # multiple shared indices
        A, B = rand(2, 3, 4), rand(4, 3, 5)
        @hadamard C[i, k, l, j] := A[i, k, l] * B[l, k, j]
        @test C ≈ ref_hadamard(A, (:i, :k, :l), B, (:l, :k, :j), (:i, :k, :l, :j))

        # permuted output indices
        A, B = rand(2, 3), rand(3, 4)
        @hadamard C[k, j, i] := A[i, k] * B[k, j]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:k, :j), (:k, :j, :i))
    end

    @testset "ComplexF64" begin
        # pointwise product
        A, B = rand(ComplexF64, 3, 4), rand(ComplexF64, 3, 4)
        @hadamard C[i, j] := A[i, j] * B[i, j]
        @test C ≈ ref_hadamard(A, (:i, :j), B, (:i, :j), (:i, :j))

        # partially shared indices
        A, B = rand(ComplexF64, 2, 3), rand(ComplexF64, 3, 4)
        @hadamard C[i, k, j] := A[i, k] * B[k, j]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:k, :j), (:i, :k, :j))

        # conjugation
        @hadamard C[i, k, j] := conj(A[i, k]) * B[k, j]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:k, :j), (:i, :k, :j); conjA=true)

        # permuted output indices with conjugation
        @hadamard C[k, j, i] := conj(A[i, k]) * B[k, j]
        @test C ≈ ref_hadamard(A, (:i, :k), B, (:k, :j), (:k, :j, :i); conjA=true)
    end
end