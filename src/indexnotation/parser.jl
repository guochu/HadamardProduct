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

# decompose a term `scalar * T1 * T2` into the scalar factor and the two tensor terms
function decompose_term(ex)
    factors = (isexpr(ex, :call) && ex.args[1] == :*) ? ex.args[2:end] : [ex]
    α = 1
    tensors = TensorTerm[]
    for f in factors
        if istensorterm(f)
            push!(tensors, parse_tensorterm(f))
        else
            α = α == 1 ? f : Expr(:call, :*, α, f)
        end
    end
    length(tensors) == 2 ||
        throw(ArgumentError("expected a product of exactly two tensor terms, got: $ex"))
    return α, tensors[1], tensors[2]
end

# apply a sign (±1) to a scalar factor
function applysign(α::Number, sign::Int)
    return sign == 1 ? α : -α
end
function applysign(α, sign::Int)
    return sign == 1 ? α : Expr(:call, :-, α)
end
