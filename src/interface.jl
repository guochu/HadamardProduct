#-------------------------------------------------------------------------------------------
# Function based API
#-------------------------------------------------------------------------------------------

"""
    hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB, α=1, β=0)

Compute `C = β*C + α * permutedims(A ⊙ B, pAB)`, where the "Hadamard product" `A ⊙ B` is
defined as follows. The indices of `A` are partitioned by `pA = (pA[1], pA[2])` into the
indices `pA[1]` that are combined with the indices of `B` in an outer product and the
indices `pA[2]` that are combined in a pointwise (Hadamard) fashion; analogously,
`pB = (pB[1], pB[2])` partitions the indices of `B` into the pointwise indices `pB[1]` and
the outer indices `pB[2]`. The `k`-th index of `pA[2]` is pointwise-multiplied with the
`k`-th index of `pB[1]` (their dimensions must match), and these shared indices are kept in
the output. The resulting tensor has the natural index order `(pA[1]..., pB[1]..., pB[2]...)`,
which is then permuted according to `pAB` to match the index order of `C`.

The operation `opA` acts as `identity` if `conjA` equals `false` and as `conj` if `conjA`
equals `true`; `opB` is determined by `conjB` analogously.

Special cases: if there are no shared indices (`pA[2] = pB[1] = ()`), this reduces to the
outer product; if all indices are shared (`pA[1] = pB[2] = ()`), this reduces to the
standard pointwise (Hadamard) product `C = α * opA(A) .* opB(B) + β*C`.

!!! warning
    The object `C` must not be aliased with `A` or `B`.

See also [`hadamardproduct`](@ref).
"""
function hadamardproduct! end
# insert default α and β arguments
function hadamardproduct!(C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple)
    return hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB, 1, 0)
end

"""
    hadamardproduct([IC], A, IA, [conjA], B, IB, [conjB], [α=1])
    hadamardproduct(A, pA, conjA, B, pB, conjB, pAB, α=1) # expert mode

Return the Hadamard product of two tensors `A` and `B`, where the iterables `IA` and `IB`
denote how the tensor data should be combined. More specifically, indices that appear in
both `IA` and `IB` are combined in a pointwise (Hadamard) fashion, while indices that
appear in only one of the two are combined as an outer product; all indices are kept in the
result. The indices of the result can be ordered by specifying the optional argument `IC`,
which defaults to the natural order `(oA..., shared..., oB...)`. Note that every index
label should appear exactly once in `IA` or `IB` separately, and exactly once in `IC`,
which should contain exactly the union of `IA` and `IB`.

Optionally, the symbols `conjA` and `conjB` can be used to specify that the input tensors
should be conjugated, and `α` can be used to scale the result.

See also [`hadamardproduct!`](@ref).
"""
function hadamardproduct end

function hadamardproduct(
        IC, A, IA, conjA::Bool, B, IB, conjB::Bool, α::Number = 1
    )
    pA, pB, pAB = hadamard_indices(IA, IB, IC)
    return hadamardproduct(A, pA, conjA, B, pB, conjB, pAB, α)
end
# default `IC`
function hadamardproduct(A, IA, conjA::Bool, B, IB, conjB::Bool, α::Number = 1)
    return hadamardproduct(unique(vcat(collect(IA), collect(IB))), A, IA, conjA, B, IB, conjB, α)
end
# default `conjA` and `conjB`
function hadamardproduct(IC, A, IA, B, IB, α::Number = 1)
    return hadamardproduct(IC, A, IA, false, B, IB, false, α)
end
# default `IC`, `conjA` and `conjB`
function hadamardproduct(A, IA, B, IB, α::Number = 1)
    return hadamardproduct(unique(vcat(collect(IA), collect(IB))), A, IA, false, B, IB, false, α)
end
# expert mode
function hadamardproduct(
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number = 1
    )
    TC = promote_hadamard(scalartype(A), scalartype(B), scalartype(α))
    C = tensoralloc_hadamard(TC, A, pA, conjA, B, pB, conjB, pAB)
    return hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB, α, 0)
end

#-------------------------------------------------------------------------------------------
# Internal tensor interfaces
#-------------------------------------------------------------------------------------------

"""
    scalartype(A)

Obtain the scalar type of a tensor-like object `A`. The default implementation returns
`eltype(A)`; tensor types with a different notion of scalar type can override this method.
"""
scalartype(A) = eltype(A)

"""
    promote_hadamard(TA, TB, Tα)

Obtain the scalar type of the result of a Hadamard product of tensors with scalar types
`TA`, `TB` and a scalar factor of type `Tα`.
"""
promote_hadamard(TA, TB, Tα) = promote_type(TA, TB, Tα)

"""
    hadamardproduct_structure(A, pA, conjA, B, pB, conjB, pAB)

Obtain the structure information of `C`, where `C` would be the output of
`hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB)`. For array-like tensors, this is the
tuple of dimensions of `C` in the index order of `C` itself.
"""
function hadamardproduct_structure end

"""
    hadamardproduct_type(TC, A, pA, conjA, B, pB, conjB, pAB)

Obtain the type information of `C`, where `C` would be the output of
`hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB)` with scalar type `TC`.
"""
function hadamardproduct_type end

"""
    tensoralloc(ttype, structure)

Allocate memory for a tensor of type `ttype` and structure `structure`.
"""
tensoralloc(ttype, structure) = ttype(undef, structure)

"""
    tensorfree!(C)

Provide a hint that the allocated memory of `C` can be released.
"""
tensorfree!(C) = nothing

"""
    tensoralloc_hadamard(TC, A, pA, conjA, B, pB, conjB, pAB)

Allocate a tensor `C` of scalar type `TC` that would be the result of
`hadamardproduct!(C, A, pA, conjA, B, pB, conjB, pAB)`.
"""
function tensoralloc_hadamard(TC, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple)
    ttype = hadamardproduct_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = hadamardproduct_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure)
end

"""
    checkhadamardproduct(C, A, pA, B, pB, pAB)

Verify whether `C`, `A` and `B` are compatible for being combined in a Hadamard product as
specified by `pA`, `pB` and `pAB`, and throws an error if not.
"""
function checkhadamardproduct end
