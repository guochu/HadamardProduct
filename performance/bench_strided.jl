# Correctness + performance comparison: PermutedDimsArray-based kernel vs Strided-based kernel
using HadamardProduct
import HadamardProduct: checkhadamardproduct, Index2Tuple, numout, numin, linearize, _conj
using Strided: StridedView, sreshape

# ---- old kernel: lazy PermutedDimsArray + reshape views ----
function kernel_old!(C::Array, A::Array, pA::Index2Tuple, conjA::Bool, B::Array, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number = 1, β::Number = 0)
    oA, shA = pA
    shB, oB = pB
    nO, nS, nB = numout(pA), numin(pA), numin(pB)
    szA, szB, szC = size(A), size(B), size(C)
    ΠoA = prod((szA[i] for i in oA); init = 1)
    Πs = prod((szA[i] for i in shA); init = 1)
    ΠoB = prod((szB[i] for i in oB); init = 1)
    pC = invperm(linearize(pAB))
    ΠoA_c = prod((szC[i] for i in pC[1:nO]); init = 1)
    Πs_c = prod((szC[i] for i in pC[nO+1:nO+nS]); init = 1)
    ΠoB_c = prod((szC[i] for i in pC[nO+nS+1:nO+nS+nB]); init = 1)
    Av = PermutedDimsArray(A, linearize(pA))
    Bv = PermutedDimsArray(B, linearize(pB))
    Cv = PermutedDimsArray(C, pC)
    Am = reshape(Av, (ΠoA, Πs))
    Bm = reshape(Bv, (Πs, ΠoB))
    Cm = reshape(Cv, (ΠoA_c, Πs_c, ΠoB_c))
    Am3 = reshape(Am, (size(Am, 1), size(Am, 2), 1))
    Bm3 = reshape(Bm, (1, size(Bm, 1), size(Bm, 2)))
    if β == 0
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

# ---- correctness: the new strided kernel must agree with the old one ----
using Random; Random.seed!(1234)
function check_correctness()
    ok = true
    for (sA, sB, pA, pB, pAB) in (
        # outer: no shared indices, C[i,j,k,l] = A[i,k] * B[j,l]
        ((3, 4), (5, 6), ((1, 2), ()), ((), (1, 2)), ((1, 3, 2, 4), ())),
        # pointwise: fully shared, C[i,j] = A[i,j] * B[i,j]
        ((3, 4), (3, 4), ((), (1, 2)), ((1, 2), ()), ((1, 2), ())),
        # partially shared: C[i,k,j] = A[i,k] * B[k,j]
        ((2, 3), (3, 4), ((1,), (2,)), ((1,), (2,)), ((1, 2, 3), ())),
        # shared index with permuted C order: C[k,i,j] = A[i,k] * B[k,j]
        ((2, 3), (3, 4), ((1,), (2,)), ((1,), (2,)), ((2, 1, 3), ())),
    )
        A, B = rand(sA...), rand(sB...)
        C1 = zeros(HadamardProduct.hadamardproduct_structure(A, pA, false, B, pB, false, pAB))
        C2 = zeros(size(C1))
        kernel_old!(C1, A, pA, false, B, pB, false, pAB)
        hadamardproduct!(C2, A, pA, false, B, pB, false, pAB)
        ok &= C1 ≈ C2
        println("correctness (dims ", sA, " x ", sB, "): ", C1 ≈ C2)
        # with α and β and conj
        C3 = zeros(size(C1)); C3 .= rand(size(C1))
        C4 = copy(C3)
        kernel_old!(C3, A, pA, true, B, pB, true, pAB, 2.5, 0.7)
        hadamardproduct!(C4, A, pA, true, B, pB, true, pAB, 2.5, 0.7)
        ok &= C3 ≈ C4
        println("correctness (conj, α, β): ", C3 ≈ C4)
    end
    return ok
end
ok = check_correctness()

# ---- performance ----
function bench(f, n)
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

println("JULIA_NUM_THREADS = ", Threads.nthreads())
for (name, sA, sB, pA, pB, pAB) in (
    ("partial-shared 100x200 . 200x300", (100, 200), (200, 300), ((1,), (2,)), ((1,), (2,)), ((1, 2, 3), ())),
    ("pointwise 3D 200^3", (200, 200, 200), (200, 200, 200), ((), (1, 2, 3)), ((1, 2, 3), ()), ((1, 2, 3), ())),
    ("outer 2D x 2D -> 4D", (60, 60), (60, 60), ((1, 2), ()), ((), (1, 2)), ((1, 3, 2, 4), ())),
)
    A, B = rand(sA...), rand(sB...)
    shp = HadamardProduct.hadamardproduct_structure(A, pA, false, B, pB, false, pAB)
    nbytes = (prod(sA) + prod(sB) + prod(shp)) * 8 / 1e6
    C1 = zeros(shp); C2 = zeros(shp)
    f1() = kernel_old!(C1, A, pA, false, B, pB, false, pAB)
    f2() = hadamardproduct!(C2, A, pA, false, B, pB, false, pAB)
    f1(); f2()  # compile
    t1, t2 = bench(f1, 20), bench(f2, 20)
    println(rpad(name, 42), " old=", round(t1 * 1e6; digits = 1), "µs  strided=", round(t2 * 1e6; digits = 1),
            "µs  speedup=", round(t1 / t2; digits = 2), "x  (", round(nbytes / t2; digits = 1), " GB/s)")
end
println(ok ? "ALL OK" : "SOME FAILED")
