#-------------------------------------------------------------------------------------------
# Parser for the @hadamard macro
#-------------------------------------------------------------------------------------------

using Base.Meta: isexpr

# is this expression a tensor term `A[i,j]` (or `conj(A[i,j])`)?
istensorterm(ex) = isexpr(ex, [:ref, :typed_vcat]) ||
    (isexpr(ex, :call) && ex.args[1] == :conj && length(ex.args) == 2 && istensorterm(ex.args[2]))

"""
    TensorTerm

A parsed tensor term `A[i1,...,in]` or `conj(A[i1,...,in])`. The indices may be separated
by a semicolon into a left (codomain) group `left` and a right (domain) group `right`,
following the convention of `@tensor` in TensorOperations.jl and TensorKit.jl, which is
useful for tensor types that distinguish between the two; for regular arrays the semicolon
has no effect and `indices == vcat(left, right)` is the full index sequence.
"""
struct TensorTerm
    object::Any # the tensor object expression (e.g. `A`)
    indices::Vector{Any} # all index labels, in order (left followed by right)
    left::Vector{Any} # the index labels before a semicolon (codomain)
    right::Vector{Any} # the index labels after a semicolon (domain)
    conj::Bool # whether the tensor is conjugated
end

function _checklabels(labels)
    for l in labels
        (l isa Symbol || l isa Integer) ||
            throw(ArgumentError("index label should be a Symbol or an Integer, got $l"))
    end
    return nothing
end

# decompose a tensor expression `A[i,j]`, `A[i j]`, `A[i,j;k]`, `A[i;j]`, `A[();j k]` or
# `A[(); (j,k)]` into the tensor object and the left (codomain) and right (domain) index
# groups, following the `@tensor` convention of TensorOperations.jl and TensorKit.jl
# (indices before the semicolon are codomain, indices after the semicolon are domain); a
# missing semicolon results in an empty right group
function decompose_tensorterm(ex)
    if isexpr(ex, [:ref, :typed_hcat])
        if length(ex.args) == 1
            return ex.args[1], Any[], Any[]
        elseif isexpr(ex.args[2], :parameters)
            return ex.args[1], collect(ex.args[3:end]), collect(ex.args[2].args)
        else
            return ex.args[1], collect(ex.args[2:end]), Any[]
        end
    elseif isexpr(ex, :typed_vcat)
        left = isexpr(ex.args[2], [:row, :tuple]) ? collect(ex.args[2].args) : Any[ex.args[2]]
        right = isexpr(ex.args[3], [:row, :tuple]) ? collect(ex.args[3].args) : Any[ex.args[3]]
        return ex.args[1], left, right
    end
    throw(ArgumentError("not a valid tensor term: $ex"))
end

function parse_tensorterm(ex)
    if istensorterm(ex) && !(isexpr(ex, :call))
        object, left, right = decompose_tensorterm(ex)
        indices = vcat(left, right)
        _checklabels(indices)
        allunique(indices) ||
            throw(IndexError("indices of a tensor should be unique: $ex"))
        return TensorTerm(object, indices, left, right, false)
    elseif isexpr(ex, :call) && ex.args[1] == :conj && length(ex.args) == 2
        t = parse_tensorterm(ex.args[2])
        return TensorTerm(t.object, t.indices, t.left, t.right, !t.conj)
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
