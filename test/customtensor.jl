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
function HadamardProduct.tensoralloc(::Type{TaggedTensor{TC,N}}, structure) where {TC,N}
    push!(tagalloc_log, N)
    return TaggedTensor(Array{TC,N}(undef, structure))
end
function HadamardProduct.tensorfree!(t::TaggedTensor)
    push!(tagfree_log, 1)
    return nothing
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
end
