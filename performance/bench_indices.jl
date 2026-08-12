# Correctness + performance comparison: vector-based hadamard_indices vs TupleTools-based version
using TupleTools
using Base: tail

struct IndexError{S <: AbstractString} <: Exception
    msg::S
end
Base.showerror(io::IO, e::IndexError) = print(io, "IndexError: ", e.msg)

@inline _findpos(l, I) = findfirst(isequal(l), I)::Int

# ---- old version: repeated tuple <-> vector conversions ----
function hadamard_indices_old(IA, IB, IC)
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

# ---- new version: pure tuple operations with TupleTools ----
# elements of `t` for which `f` returns true, preserving order
@inline _filter(f, ::Tuple{}) = ()
@inline function _filter(f, t::Tuple)
    x, r = t[1], _filter(f, tail(t))
    return f(x) ? (x, r...) : r
end

# unique elements of `t`, keeping the first occurrence order
@inline _tupleunique(t::Tuple{}) = ()
@inline function _tupleunique(t::Tuple)
    x, r = t[1], _tupleunique(tail(t))
    return x in r ? r : (x, r...)
end

# positions of the elements of `a` in `b` (asserted present, so always ::Int)
@inline _indexin(a::Tuple, b::Tuple) = map(x -> x::Int, TupleTools.indexin(a, b))

function hadamard_indices_new(IA, IB, IC)
    # basic validation
    allunique(IA) || throw(IndexError("indices of A should be unique: $IA"))
    allunique(IB) || throw(IndexError("indices of B should be unique: $IB"))
    allunique(IC) || throw(IndexError("indices of C should be unique: $IC"))
    IA = Tuple(IA)
    IB = Tuple(IB)
    IC = Tuple(IC)

    # validation of output indices
    uAB = _tupleunique(TupleTools.vcat(IA, IB))
    (length(uAB) == length(IC) && all(l -> l in IC, uAB)) ||
        throw(IndexError("output indices $IC should be a permutation of the union of input indices $uAB"))

    # shared indices (in the order of IA) and outer indices of A and B
    sh = _filter(l -> l in IB, IA)
    oA = _filter(l -> !(l in IB), IA)
    oB = _filter(l -> !(l in IA), IB)

    pA = (_indexin(oA, IA), _indexin(sh, IA))
    pB = (_indexin(sh, IB), _indexin(oB, IB))

    natural = TupleTools.vcat(oA, sh, oB)
    pAB = (_indexin(IC, natural), ())

    return pA, pB, pAB
end

# ---- correctness ----
function check_correctness()
    ok = true
    for (IA, IB, IC) in (
        # outer
        ((:i, :k), (:j, :l), (:i, :k, :j, :l)),
        # pointwise
        ((:i, :k), (:i, :k), (:i, :k)),
        # partial
        ((:i, :k), (:k, :j), (:i, :k, :j)),
        ((:i, :k), (:k, :j), (:k, :i, :j)), # permuted C
        ((:a, :b, :c), (:c, :d), (:a, :b, :d, :c)),
        (("a", "b"), ("b", "c"), ("b", "a", "c")), # non-Symbol labels
        (1:3, (2, 4), (1, 2, 3, 4)), # Int labels
        ((:i,), (:i,), (:i,)), # single scalar index
    )
        o1 = hadamard_indices_old(IA, IB, IC)
        o2 = hadamard_indices_new(IA, IB, IC)
        ok &= o1 == o2
        println("correctness ", (IA, IB, IC), ": ", o1 == o2)
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
for (name, IA, IB, IC) in (
    # macro path: Vector{Symbol} inputs
    ("vector: partial 3 inds", [:i, :k], [:k, :j], [:i, :k, :j]),
    ("vector: outer 4 inds", [:i, :k], [:j, :l], [:i, :k, :j, :l]),
    ("vector: pointwise 2 inds", [:i, :k], [:i, :k], [:i, :k]),
    ("vector: 6 inds", [:i, :j, :k, :l], [:k, :l, :m, :n], [:i, :j, :m, :n, :k, :l]),
    # tuple path: NTuple inputs
    ("tuple: partial 3 inds", (:i, :k), (:k, :j), (:i, :k, :j)),
    ("tuple: outer 4 inds", (:i, :k), (:j, :l), (:i, :k, :j, :l)),
    ("tuple: 6 inds", (:i, :j, :k, :l), (:k, :l, :m, :n), (:i, :j, :m, :n, :k, :l)),
)
    f1() = hadamard_indices_old(IA, IB, IC)
    f2() = hadamard_indices_new(IA, IB, IC)
    f1(); f2() # compile
    t1, t2 = bench(f1, 200_000), bench(f2, 200_000)
    println(rpad(name, 30), " old=", round(t1 * 1e9; digits = 1), "ns  new=", round(t2 * 1e9; digits = 1),
            "ns  speedup=", round(t1 / t2; digits = 2), "x")
end
println(ok ? "ALL OK" : "SOME FAILED")
