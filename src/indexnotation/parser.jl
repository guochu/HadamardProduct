#-------------------------------------------------------------------------------------------
# Parser for the @hadamard macro
#-------------------------------------------------------------------------------------------

using Base.Meta: isexpr

# is this expression a tensor term `A[i,j]` (or `conj(A[i,j])`)?
istensorterm(ex) = isexpr(ex, :ref) ||
    (isexpr(ex, :call) && ex.args[1] == :conj && length(ex.args) == 2 && isexpr(ex.args[2], :ref))

"""
    TensorTerm

A parsed tensor term `A[i1,...,in]` or `conj(A[i1,...,in])`.
"""
struct TensorTerm
    object::Any # the tensor object expression (e.g. `A`)
    indices::Vector{Any} # the index labels
    conj::Bool # whether the tensor is conjugated
end

function _checklabels(labels)
    for l in labels
        (l isa Symbol || l isa Integer) ||
            throw(ArgumentError("index label should be a Symbol or an Integer, got $l"))
    end
    return nothing
end

function parse_tensorterm(ex)
    if isexpr(ex, :ref)
        length(ex.args) >= 1 ||
            throw(ArgumentError("invalid tensor term: $ex"))
        indices = collect(ex.args[2:end])
        _checklabels(indices)
        allunique(indices) ||
            throw(IndexError("indices of a tensor should be unique: $ex"))
        return TensorTerm(ex.args[1], indices, false)
    elseif isexpr(ex, :call) && ex.args[1] == :conj && length(ex.args) == 2
        t = parse_tensorterm(ex.args[2])
        return TensorTerm(t.object, t.indices, !t.conj)
    else
        throw(ArgumentError("expected a tensor term like `A[i,j]` or `conj(A[i,j])`, got: $ex"))
    end
end

# split an expression into a sum of terms, keeping track of the sign of each term
function sum_terms(ex)
    if isexpr(ex, :call) && ex.args[1] == :+
        return reduce(vcat, [sum_terms(a) for a in ex.args[2:end]])
    elseif isexpr(ex, :call) && ex.args[1] == :- && length(ex.args) == 3
        return vcat(sum_terms(ex.args[2]), [(-s, t) for (s, t) in sum_terms(ex.args[3])])
    elseif isexpr(ex, :call) && ex.args[1] == :- && length(ex.args) == 2
        return [(-s, t) for (s, t) in sum_terms(ex.args[2])]
    else
        return [(1, ex)]
    end
end

# node of a product tree: the Hadamard product of two sub-products, where a sub-product is
# either a `TensorTerm` (leaf) or another `ProdNode`. Parentheses in the source expression
# are preserved as the shape of the tree.
struct ProdNode
    left::Any # TensorTerm or ProdNode
    right::Any # TensorTerm or ProdNode
end

# multiply two scalar factor expressions into a single expression
mulscalar(a::Number, b::Number) = a * b
mulscalar(a, b) = Expr(:call, :*, a, b)

# combine a vector of product elements (tensor terms or sub-products) into a left-associative
# product tree
function combine_product(elements)
    length(elements) >= 2 ||
        throw(ArgumentError("expected a product of at least two tensor terms"))
    return foldl((l, r) -> ProdNode(l, r), elements)
end

# decompose a term `scalar * T1 * T2 * ...` into the scalar factor and a product tree; a
# parenthesized sub-expression like `(T2 * T3)` is kept as a nested `ProdNode`
function decompose_term(ex)
    factors = (isexpr(ex, :call) && ex.args[1] == :*) ? ex.args[2:end] : [ex]
    α = 1
    elements = Any[]
    for f in factors
        if istensorterm(f)
            push!(elements, parse_tensorterm(f))
        elseif isexpr(f, :call) && f.args[1] == :* && length(f.args) >= 3
            # parenthesized sub-product: `(T2 * T3)` parses as a nested `*` call
            αi, tree = decompose_term(f)
            push!(elements, tree)
            α = α == 1 ? αi : mulscalar(α, αi)
        else
            α = α == 1 ? f : mulscalar(α, f)
        end
    end
    return α, combine_product(elements)
end

# apply a sign (±1) to a scalar factor
function applysign(α::Number, sign::Int)
    return sign == 1 ? α : -α
end
function applysign(α, sign::Int)
    return sign == 1 ? α : Expr(:call, :-, α)
end
