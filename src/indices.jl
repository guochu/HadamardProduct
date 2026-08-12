#-------------------------------------------------------------------------------------------
# Index types
#-------------------------------------------------------------------------------------------

"""
    IndexError

Exception thrown when the index specifications of a Hadamard product are inconsistent.
"""
struct IndexError{S <: AbstractString} <: Exception
    msg::S
end
Base.showerror(io::IO, e::IndexError) = print(io, "IndexError: ", e.msg)

"""
    IndexTuple

Alias for a tuple of integers, used to denote a (sequence of) index positions.
"""
const IndexTuple = Tuple{Vararg{Int}}

"""
    Index2Tuple

Tuple of two `IndexTuple`s. An `Index2Tuple` is used to encode a permutation of the indices
of a tensor, partitioned into two groups. For `hadamardproduct!`, the tuple
`pA = (pA[1], pA[2])` partitions the indices of `A` into the indices that are combined in an
outer product (`pA[1]`) and the indices that are combined in a pointwise fashion with the
corresponding indices of `B` (`pA[2]`).
"""
const Index2Tuple = Tuple{IndexTuple, IndexTuple}

"""
    linearize(t::Index2Tuple)

Flatten an `Index2Tuple` into a single `IndexTuple`.
"""
linearize(t::Index2Tuple) = (t[1]..., t[2]...)

numind(t::Index2Tuple) = length(t[1]) + length(t[2])
numout(t::Index2Tuple) = length(t[1])
numin(t::Index2Tuple) = length(t[2])

#-------------------------------------------------------------------------------------------
# Index labels -> index permutations
#-------------------------------------------------------------------------------------------

# position of label `l` in the iterable `I`; validation ensures it is always present
@inline _findpos(l, I) = findfirst(isequal(l), I)::Int

"""
    hadamard_indices(IA, IB, IC) -> (pA, pB, pAB)

Given iterables of index labels `IA` and `IB` (denoting the indices of two tensors `A` and
`B`) and `IC` (denoting the indices of the output `C`), compute the `Index2Tuple`s

- `pA = (pA[1], pA[2])`: positions in `IA` of the outer indices of `A` and of the shared
  (pointwise) indices of `A` (in the order they appear in `IA`),
- `pB = (pB[1], pB[2])`: positions in `IB` of the shared (pointwise) indices of `B` and of
  the outer indices of `B`,
- `pAB`: permutation such that the `k`-th index of `C` corresponds to the
  `linearize(pAB)[k]`-th index of the natural output order `(pA[1]..., pB[1]..., pB[2]...)`.

The `k`-th index of `pA[2]` is combined in a pointwise fashion with the `k`-th index of
`pB[1]`. The indices of `C` must be exactly the union of the indices of `A` and `B`, with
each index label appearing exactly once.
"""
function hadamard_indices(IA, IB, IC)
    # basic validation
    allunique(IA) || throw(IndexError("indices of A should be unique: $IA"))
    allunique(IB) || throw(IndexError("indices of B should be unique: $IB"))
    allunique(IC) || throw(IndexError("indices of C should be unique: $IC"))
    IA = collect(IA)
    IB = collect(IB)
    IC = collect(IC)

    # validation of output indices
    uAB = unique(vcat(IA, IB))
    (length(uAB) == length(IC) && all(l -> l in IC, uAB)) ||
        throw(IndexError("output indices $IC should be a permutation of the union of input indices $uAB"))

    oA = [l for l in IA if !(l in IB)] # outer indices of A, in the order of IA
    sh = [l for l in IA if l in IB] # shared indices, in the order of IA
    oB = [l for l in IB if !(l in IA)] # outer indices of B, in the order of IB

    pA = (Tuple(_findpos(l, IA) for l in oA), Tuple(_findpos(l, IA) for l in sh))
    pB = (Tuple(_findpos(l, IB) for l in sh), Tuple(_findpos(l, IB) for l in oB))

    natural = (oA..., sh..., oB...)
    pAB = (Tuple(_findpos(l, natural) for l in IC), ())

    return pA, pB, pAB
end
