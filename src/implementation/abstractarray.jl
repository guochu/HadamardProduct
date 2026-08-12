# ------------------------------------------------------------------------------------------
# General definitions for AbstractArray instances
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# Argument checking
# ------------------------------------------------------------------------------------------

"""
    argcheck_hadamardproduct(C, A, pA, B, pB, pAB)

Check that `C`, `A` and `pA`, and `B` and `pB` and `pAB` have compatible numbers of indices.
"""
function argcheck_hadamardproduct(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple, pAB::Index2Tuple)
    isperm(linearize(pA)) && length(linearize(pA)) == ndims(A) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
    isperm(linearize(pB)) && length(linearize(pB)) == ndims(B) ||
        throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
    numin(pA) == numout(pB) ||
        throw(IndexError("non-matching number of shared indices: $(numin(pA)) != $(numout(pB))"))
    ((numind(pAB) == numout(pA) + numin(pA) + numin(pB)) &&
     (numout(pA) + numin(pA) + numin(pB) == ndims(C))) ||
        throw(IndexError("non-matching number of output indices"))
    return nothing
end

"""
    dimcheck_hadamardproduct(C, A, pA, B, pB, pAB)

Check that `C`, `A` and `B` have compatible sizes for the Hadamard product specified by
`pA`, `pB` and `pAB`.
"""
function dimcheck_hadamardproduct(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple, pAB::Index2Tuple)
    szA, szB, szC = size(A), size(B), size(C)
    for k in 1:numin(pA)
        szA[pA[2][k]] == szB[pB[1][k]] ||
            throw(DimensionMismatch("non-matching sizes in shared dimensions: $(szA[pA[2][k]]) != $(szB[pB[1][k]])"))
    end
    size(C) == hadamardproduct_structure(A, pA, false, B, pB, false, pAB) ||
        throw(DimensionMismatch("non-matching sizes in output dimensions: $szC"))
    return nothing
end

"""
    checkhadamardproduct(C, A, pA, B, pB, pAB)

Check that `C`, `A` and `B` are compatible for the Hadamard product specified by `pA`, `pB`
and `pAB`.
"""
function checkhadamardproduct(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple, pAB::Index2Tuple)
    argcheck_hadamardproduct(C, A, pA, B, pB, pAB)
    dimcheck_hadamardproduct(C, A, pA, B, pB, pAB)
    return nothing
end

# ------------------------------------------------------------------------------------------
# Structure and type information
# ------------------------------------------------------------------------------------------

function hadamardproduct_structure(A::AbstractArray, pA::Index2Tuple, conjA::Bool, B::AbstractArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple)
    szA, szB = size(A), size(B)
    nO, nS = numout(pA), numin(pA)
    return map(linearize(pAB)) do n
        n <= nO + nS ? szA[n <= nO ? pA[1][n] : pA[2][n - nO]] : szB[pB[2][n - nO - nS]]
    end
end

function hadamardproduct_type(TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool, B::AbstractArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple)
    return Array{TC, numind(pAB)}
end

tensoralloc(::Type{Array{TC,N}}, structure::Tuple{Vararg{Int}}) where {TC,N} = Array{TC,N}(undef, structure)

# ------------------------------------------------------------------------------------------
# Default implementation
# ------------------------------------------------------------------------------------------

# apply conj if flag is set; used inside the fused broadcast
@inline _conj(x, conjflag::Bool) = conjflag ? conj(x) : x

"""
    hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB, α=1, β=0)

Default implementation for `AbstractArray` instances. The pointwise combination of the
shared indices is implemented by folding all shared dimensions into a single dimension and
broadcasting against a 3-dimensional view, such that the shared dimensions align and the
outer dimensions are combined as an outer product. All intermediate views are lazy, so no
temporary arrays are allocated.
"""
function hadamardproduct!(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, conjA::Bool, B::AbstractArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number = 1, β::Number = 0)
    checkhadamardproduct(C, A, pA, B, pB, pAB)

    oA, shA = pA
    shB, oB = pB
    nO, nS, nB = numout(pA), numin(pA), numin(pB)

    szA, szB, szC = size(A), size(B), size(C)
    ΠoA = prod((szA[i] for i in oA); init = 1) # product of the outer dimensions of A (1 for an empty collection)
    Πs = prod((szA[i] for i in shA); init = 1) # product of the shared dimensions
    ΠoB = prod((szB[i] for i in oB); init = 1) # product of the outer dimensions of B
    pC = invperm(linearize(pAB)) # natural index n of C lives at C dimension pC[n]
    ΠoA_c = prod((szC[i] for i in pC[1:nO]); init = 1)
    Πs_c = prod((szC[i] for i in pC[nO+1:nO+nS]); init = 1)
    ΠoB_c = prod((szC[i] for i in pC[nO+nS+1:nO+nS+nB]); init = 1)

    # lazy views with the natural index order (oA..., shA...) for A and (shB..., oB...) for B
    Av = PermutedDimsArray(A, linearize(pA))
    Bv = PermutedDimsArray(B, linearize(pB))
    Cv = PermutedDimsArray(C, pC)

    # fold the outer and shared dimensions of each tensor into a single dimension
    Am = reshape(Av, (ΠoA, Πs))
    Bm = reshape(Bv, (Πs, ΠoB))
    Cm = reshape(Cv, (ΠoA_c, Πs_c, ΠoB_c))

    # broadcast in 3 dimensions: shared dimensions align, outer dimensions are combined
    # as an outer product
    Am3 = reshape(Am, (size(Am, 1), size(Am, 2), 1))
    Bm3 = reshape(Bm, (1, size(Bm, 1), size(Bm, 2)))
    if β == 0
        # C contains uninitialized memory, so it must not be read
        if α == 1
            @. Cm = _conj(Am3, conjA) * _conj(Bm3, conjB)
        else
            @. Cm = α * _conj(Am3, conjA) * _conj(Bm3, conjB)
        end
    else
        @. Cm = α * _conj(Am3, conjA) * _conj(Bm3, conjB) + β * Cm
    end
    return C
end
