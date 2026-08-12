# ------------------------------------------------------------------------------------------
# Optimized implementation for strided arrays, using the fused broadcasting of Strided.jl.
#
# The views are created lazily: `StridedView(PermutedDimsArray(...))` reorders the
# dimensions (recomputing strides, no copy) and `sreshape` inserts the trivial dimensions
# needed for the broadcast alignment (again no copy). The actual computation is then
# carried out by Strided.jl's fused, cache-friendly broadcast kernel, which can take
# advantage of multiple threads.
#
# The three tensors are aligned in the natural index order `(oA..., s..., oB...)`, where
# `s` are the shared indices (combined pointwise) and `oA`/`oB` are the outer indices of
# A/B (combined as an outer product). A and B are padded with trivial (size-1) dimensions
# in the positions they do not occupy; the broadcast then aligns the shared dimensions
# and extends the trivial dimensions.
# ------------------------------------------------------------------------------------------
function hadamardproduct!(C::StridedArray, A::StridedArray, pA::Index2Tuple, conjA::Bool, B::StridedArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number = 1, β::Number = 0)
    checkhadamardproduct(C, A, pA, B, pB, pAB)

    oA, shA = pA
    shB, oB = pB
    nO, nS, nB = numout(pA), numin(pA), numin(pB)

    szA, szB = size(A), size(B)
    # dimensions of the aligned views in natural order (oA..., s..., oB...)
    dimsA = (ntuple(d -> szA[oA[d]], nO)..., ntuple(d -> szA[shA[d]], nS)..., ntuple(_ -> 1, nB)...)
    dimsB = (ntuple(_ -> 1, nO)..., ntuple(d -> szB[shB[d]], nS)..., ntuple(d -> szB[oB[d]], nB)...)

    # lazy strided views: permute to natural index order and insert trivial dimensions
    Av = sreshape(StridedView(PermutedDimsArray(A, linearize(pA))), dimsA)
    Bv = sreshape(StridedView(PermutedDimsArray(B, linearize(pB))), dimsB)
    Cv = StridedView(PermutedDimsArray(C, invperm(linearize(pAB))))

    if β == 0
        # C contains uninitialized memory, so it must not be read
        if α == 1
            @. Cv = _conj(Av, conjA) * _conj(Bv, conjB)
        else
            @. Cv = α * _conj(Av, conjA) * _conj(Bv, conjB)
        end
    else
        @. Cv = α * _conj(Av, conjA) * _conj(Bv, conjB) + β * Cv
    end
    return C
end
