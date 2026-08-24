# Verify that user-defined tensor types can hook into the internal tensor interfaces
# (`scalartype`, `tensoralloc`, `tensorfree!`, `tensoralloc_hadamard`,
# `hadamardproduct_type`, ...), so that both the function based API and the `@hadamard`
# macro operate on custom tensor objects.
import HadamardProduct

# a minimal custom tensor type: wraps an `Array` and carries a tag that is used to verify
# that the allocation hooks are actually invoked
struct TaggedTensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end
# the implicit outer constructor `TaggedTensor(data)` is used to wrap arrays

Base.size(t::TaggedTensor) = size(t.data)
Base.axes(t::TaggedTensor) = axes(t.data)
Base.getindex(t::TaggedTensor, I...) = t.data[I...]
Base.setindex!(t::TaggedTensor, v, I...) = (t.data[I...] = v; t)

# extend the internal interfaces: the output of a Hadamard product should be allocated as a
# `TaggedTensor` as well, and the hooks should record every allocation/free
HadamardProduct.scalartype(t::TaggedTensor) = scalartype(t.data)
function HadamardProduct.hadamardproduct_type(
        TC, A::TaggedTensor, pA::Index2Tuple, conjA::Bool,
        B::TaggedTensor, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple
    )
    return TaggedTensor{TC, HadamardProduct.numind(pAB)}
end

const tagalloc_log = Int[]
const tagfree_log = Int[]
const tagpc_log = Vector{Any}()
function HadamardProduct.tensoralloc(::Type{TaggedTensor{TC,N}}, structure) where {TC,N}
    push!(tagalloc_log, N)
    return TaggedTensor(Array{TC,N}(undef, structure))
end
function HadamardProduct.tensorfree!(t::TaggedTensor)
    push!(tagfree_log, 1)
    return nothing
end
# the extended 9-argument version of `hadamardproduct_type` records the codomain/domain
# grouping `pC` forwarded by `tensoralloc_hadamard` when the left hand side of `@hadamard`
# uses a semicolon
function HadamardProduct.hadamardproduct_type(
        TC, A::TaggedTensor, pA::Index2Tuple, conjA::Bool,
        B::TaggedTensor, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, pC::Index2Tuple
    )
    push!(tagpc_log, pC)
    return TaggedTensor{TC, HadamardProduct.numind(pAB)}
end

@testset "custom tensor type integration" begin
    A = TaggedTensor(rand(2, 3))
    B = TaggedTensor(rand(3, 4))
    ref = reshape(A.data, 2, 3, 1) .* reshape(B.data, 1, 3, 4)

    # scalartype and index structure
    @test scalartype(A) == Float64

    pA, pB, pAB = HadamardProduct.hadamard_indices((:i, :k), (:k, :j), (:i, :k, :j))
    @test hadamardproduct_structure(A, pA, false, B, pB, false, pAB) == (2, 3, 4)

    # (a) expert mode: in-place into an existing TaggedTensor
    C = TaggedTensor(zeros(2, 3, 4))
    hadamardproduct!(C, A, pA, false, B, pB, false, pAB)
    @test C ≈ ref

    # (b) allocation interfaces produce TaggedTensor outputs
    @test hadamardproduct_type(Float64, A, pA, false, B, pB, false, pAB) ==
          TaggedTensor{Float64, 3}
    C2 = tensoralloc_hadamard(Float64, A, pA, false, B, pB, false, pAB)
    @test C2 isa TaggedTensor{Float64, 3}
    @test C2.data isa Array{Float64, 3}

    # (c) label based API allocates a TaggedTensor, invoking the `tensoralloc` hook
    empty!(tagalloc_log)
    C3 = hadamardproduct((:i, :k, :j), A, (:i, :k), B, (:k, :j))
    @test C3 isa TaggedTensor{Float64, 3}
    @test C3 ≈ ref
    @test !isempty(tagalloc_log)

    # (d) `@hadamard` macro: definition mode allocates a TaggedTensor
    empty!(tagalloc_log)
    @hadamard C4[i, k, j] := A[i, k] * B[k, j]
    @test C4 isa TaggedTensor{Float64, 3}
    @test C4 ≈ ref
    @test !isempty(tagalloc_log)

    # (e) `@hadamard` macro: assignment mode writes into an existing TaggedTensor
    C5 = TaggedTensor(zeros(2, 3, 4))
    @hadamard C5[i, k, j] = A[i, k] * B[k, j]
    @test C5 ≈ ref

    # (e') accumulation mode accumulates into the existing TaggedTensor
    @hadamard C5[i, k, j] += A[i, k] * B[k, j]
    @test C5 ≈ 2 * ref

    # (f) tensorfree! hook is available for custom tensors
    empty!(tagfree_log)
    tensorfree!(C5)
    @test tagfree_log == [1]

    # (g) semicolon grouping on the left hand side: the domain/codomain grouping `pC` is
    # forwarded to the extended `hadamardproduct_type`, and the result is unchanged
    empty!(tagpc_log)
    @hadamard C6[i; j k] := A[i, j] * B[j, k]
    @test C6 isa TaggedTensor{Float64, 3}
    @test C6 ≈ ref
    @test tagpc_log == [((1,), (2, 3))]

    # (h) semicolons on the right hand side are accepted and equivalent to commas
    @hadamard C7[i; j k] := A[i; j] * B[j; k]
    @test C7 ≈ ref
    # ... and so is the comma-then-semicolon form on the left hand side
    @hadamard C8[i, j; k] := A[i, j] * B[j, k]
    @test C8 ≈ ref

    # (i) assignment mode with a semicolon grouping on the left hand side
    C9 = TaggedTensor(zeros(2, 3, 4))
    @hadamard C9[i; j k] = A[i, j] * B[j, k]
    @test C9 ≈ ref
    @hadamard C9[i; j k] += A[i, j] * B[j, k]
    @test C9 ≈ 2 * ref

    # (j) multiple tensors on the right hand side are combined pairwise
    D = TaggedTensor(rand(4, 5))
    ref3 = reshape(A.data, 2, 3, 1, 1) .* reshape(B.data, 1, 3, 4, 1) .*
        reshape(D.data, 1, 1, 4, 5)
    @hadamard C10[i, j, k, l] := A[i, j] * B[j, k] * D[k, l]
    @test C10 isa TaggedTensor{Float64, 4}
    @test C10 ≈ ref3
    # accumulation with a three-tensor chain
    C11 = TaggedTensor(zeros(2, 3, 4, 5))
    @hadamard C11[i, j, k, l] = A[i, j] * B[j, k] * D[k, l]
    @test C11 ≈ ref3
    @hadamard C11[i, j, k, l] += A[i, j] * B[j, k] * D[k, l]
    @test C11 ≈ 2 * ref3

    # (k) parentheses change the grouping order of the chain
    @hadamard C12[i, j, k, l] := A[i, j] * (B[j, k] * D[k, l])
    @test C12 ≈ ref3
    @hadamard C13[i, j, k, l] := (A[i, j] * B[j, k]) * D[k, l]
    @test C13 ≈ ref3
    # a parenthesized group sharing an index with an outer tensor
    E = TaggedTensor(rand(3, 5))
    ref4 = reshape(A.data, 2, 3, 1, 1) .* reshape(B.data, 1, 3, 4, 1) .*
        reshape(E.data, 1, 3, 1, 5)
    @hadamard C14[i, j, k, l] := A[i, j] * (B[j, k] * E[j, l])
    @test C14 ≈ ref4
end
