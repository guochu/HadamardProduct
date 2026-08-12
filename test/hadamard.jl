using Random
using HadamardProduct
using HadamardProduct: Index2Tuple, linearize, numind

Random.seed!(1234)

@testset "basic semantics" begin
    # no shared indices -> outer product
    A, B = rand(3), rand(4)
    @hadamard C[i, j] := A[i] * B[j]
    @test size(C) == (3, 4)
    @test C ≈ A .* reshape(B, 1, :)
    # equivalence with the function API
    @test C == hadamardproduct((:i, :j), A, (:i,), B, (:j,))

    # all indices shared -> pointwise product
    A, B = rand(3, 4), rand(3, 4)
    @hadamard C[i, j] := A[i, j] * B[i, j]
    @test size(C) == (3, 4)
    @test C ≈ A .* B

    # partially shared -> shared pointwise, unshared outer
    A, B = rand(2, 3), rand(3, 4)
    @hadamard C[i, k, j] := A[i, k] * B[k, j]
    @test size(C) == (2, 3, 4)
    @test C ≈ reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4)

    # multiple shared indices
    A, B = rand(2, 3, 4), rand(3, 4, 5)
    @hadamard C[i, k, l, j] := A[i, k, l] * B[k, l, j]
    @test size(C) == (2, 3, 4, 5)
    @test C ≈ reshape(A, 2, 3, 4, 1) .* reshape(B, 1, 3, 4, 5)

    # shared indices of B in a different order
    A, B = rand(3, 4), rand(4, 3)
    @hadamard C[i, j] := A[i, j] * B[j, i]
    @test C ≈ A .* B'

    # permutation of the output indices
    A, B = rand(2, 3), rand(3, 4)
    @hadamard C[j, k, i] := A[i, k] * B[k, j]
    @test C ≈ permutedims(reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4), (3, 2, 1))

    # integer index labels
    A, B = rand(2, 3), rand(3, 4)
    @hadamard C[1, 2, 3] := A[1, 2] * B[2, 3]
    @test C ≈ reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4)

    # scalar (0-dimensional) tensors
    a, b = fill(2.0), fill(3.0)
    @hadamard c[] := a[] * b[]
    @test c[] == 6.0
    @test hadamardproduct((), a, (), b, ())[] == 6.0
end

@testset "assignment forms and accumulation" begin
    A, B = rand(2, 3), rand(3, 4)
    ref = reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4)

    # := allocates a new tensor
    @hadamard C1[i, k, j] := A[i, k] * B[k, j]
    @test C1 ≈ ref

    # = writes into an existing tensor
    C2 = zeros(2, 3, 4)
    @hadamard C2[i, k, j] = A[i, k] * B[k, j]
    @test C2 ≈ ref

    # += accumulates
    C3 = zeros(2, 3, 4)
    @hadamard C3[i, k, j] += A[i, k] * B[k, j]
    @test C3 ≈ ref
    @hadamard C3[i, k, j] += A[i, k] * B[k, j]
    @test C3 ≈ 2 * ref
end

@testset "conj and scalar factors" begin
    A, B = rand(ComplexF64, 2, 3), rand(ComplexF64, 3, 4)
    ref = reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4)

    @hadamard C[i, k, j] := conj(A[i, k]) * B[k, j]
    @test C ≈ conj.(A) .* reshape(B, 1, 3, 4) .+ 0 .* reshape(A, 2, 3, 1)

    @hadamard C[i, k, j] := 2 * A[i, k] * B[k, j]
    @test C ≈ 2 * ref

    c = 3
    @hadamard C[i, k, j] := c * A[i, k] * B[k, j]
    @test C ≈ 3 * ref

    # scalar factor in an assignment form
    C = zeros(ComplexF64, 2, 3, 4)
    @hadamard C[i, k, j] = 2 * A[i, k] * B[k, j]
    @test C ≈ 2 * ref

    # sum of products
    A2, B2 = rand(ComplexF64, 2, 3), rand(ComplexF64, 3, 4)
    @hadamard C[i, k, j] := A[i, k] * B[k, j] + 2 * conj(A2[i, k]) * B2[k, j]
    @test C ≈ ref .+ 2 .* conj.(A2) .* reshape(B2, 1, 3, 4) .+ 0 .* reshape(A2, 2, 3, 1)

    # sum with a minus sign
    @hadamard C[i, k, j] := A[i, k] * B[k, j] - A2[i, k] * B2[k, j]
    @test C ≈ ref .- reshape(A2, 2, 3, 1) .* reshape(B2, 1, 3, 4)

    # sum via += with a definition
    @hadamard C[i, k, j] := A[i, k] * B[k, j] + A2[i, k] * B2[k, j]
    @test C ≈ ref .+ reshape(A2, 2, 3, 1) .* reshape(B2, 1, 3, 4)
end

@testset "function based API" begin
    A, B = rand(2, 3), rand(3, 4)
    ref = reshape(A, 2, 3, 1) .* reshape(B, 1, 3, 4)

    # label based
    C = hadamardproduct((:i, :k, :j), A, (:i, :k), B, (:k, :j))
    @test C ≈ ref
    # default output index order and conj
    C = hadamardproduct(A, (:i, :k), B, (:k, :j))
    @test C ≈ ref

    # expert mode
    pA = ((1,), (2,))
    pB = ((1,), (2,))
    pAB = ((1, 2, 3), ())
    C = zeros(2, 3, 4)
    hadamardproduct!(C, A, pA, false, B, pB, false, pAB)
    @test C ≈ ref

    # α and β accumulation
    hadamardproduct!(C, A, pA, false, B, pB, false, pAB, 2, 0)
    @test C ≈ 2 * ref
    hadamardproduct!(C, A, pA, false, B, pB, false, pAB, 2, 3)
    @test C ≈ 3 * (2 * ref) + 2 * ref

    # conj flags
    A = rand(ComplexF64, 2, 3)
    C = hadamardproduct((:i, :k, :j), A, (:i, :k), true, B, (:k, :j), false)
    @test C ≈ conj.(A) .* reshape(B, 1, 3, 4) .+ 0 .* reshape(A, 2, 3, 1)
end

@testset "errors" begin
    # non-matching shared dimensions (runtime dimension check)
    A, B = rand(2, 3), rand(4, 4)
    @test_throws DimensionMismatch @hadamard C[i, k, j] := A[i, k] * B[k, j]

    # non-matching shared dimensions via function API
    @test_throws DimensionMismatch hadamardproduct(A, (:i, :k), B, (:k, :j))

    # invalid index structures are detected during macro expansion; use the runtime
    # function `macroexpand` to trigger and capture these errors
    for ex in (
        :(C[i, j] := A[i, k] * B[k, j]),       # left hand side missing an index
        :(C[i, k, j, l] := A[i, k] * B[k, j]), # left hand side with a superfluous index
        :(C[i, j] := A[i, i] * B[i, j]),       # repeated index inside a single tensor
        :(C[i, j] := A[i, j]),                 # a single tensor term is not a product
        :(A[i, k] * B[k, j]),                  # not an assignment
    )
        full = Expr(:macrocall, Symbol("@hadamard"), LineNumberNode(0, nothing), ex)
        @test_throws Exception macroexpand(@__MODULE__, full)
    end
end
