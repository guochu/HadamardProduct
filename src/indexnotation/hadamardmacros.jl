#-------------------------------------------------------------------------------------------
# Macro based API
#-------------------------------------------------------------------------------------------

"""
    @hadamard [C[i,j,k] :=] A[i,j] * B[j,k]
    @hadamard [C[i,j,k] =]  A[i,j] * B[j,k]
    @hadamard [C[i,j,k] +=] A[i,j] * B[j,k]

Compute the Hadamard product of two tensors `A` and `B` using index notation: indices that
appear in both tensors are combined in a pointwise (Hadamard) fashion and are kept in the
output, while indices that appear in only one of the two tensors are combined as an outer
product.

- `C[i,j,k] := A[i,j] * B[j,k]` allocates a new tensor `C` whose indices are `i, j, k`,
- `C[i,j,k] = A[i,j] * B[j,k]` writes into the existing tensor `C`,
- `C[i,j,k] += A[i,j] * B[j,k]` accumulates into the existing tensor `C`.

The index labels can be `Symbol`s or `Integer`s, and the result indices can be written in
any order, i.e. the output indices are permuted accordingly. Tensors can be conjugated with
`conj(A[i,j])`, multiplied by a scalar factor (e.g. `2 * A[i,j] * B[j,k]` or
`c * A[i,j] * B[j,k]`), and products can be combined in linear combinations.

The macro lowers to calls of the function based API, in particular [`hadamardproduct!`](@ref),
and works with any tensor type that implements the internal tensor interfaces
([`scalartype`](@ref), [`tensoralloc`](@ref), [`hadamardproduct!`](@ref), ...).
"""
macro hadamard(ex)
    return esc(parse_hadamard(ex))
end
