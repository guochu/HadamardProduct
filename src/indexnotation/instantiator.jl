#-------------------------------------------------------------------------------------------
# Instantiator for the @hadamard macro: generate hadamardproduct! calls
#-------------------------------------------------------------------------------------------

# functions that are used in expressions produced by `@hadamard` and should resolve to this
# package, even when the macro is used from another namespace
const hadamardfunctions = (:hadamardproduct!, :tensoralloc_hadamard, :scalartype, :promote_hadamard)

"""
    addhadamardfunctions(ex)

Fix references to HadamardProduct functions in namespaces where `@hadamard` is present but
the functions are not, by replacing the function name with a `GlobalRef`.
"""
function addhadamardfunctions(ex)
    if isexpr(ex, :call) && ex.args[1] in hadamardfunctions
        return Expr(
            ex.head, GlobalRef(HadamardProduct, ex.args[1]),
            (addhadamardfunctions(ex.args[i]) for i in 2:length(ex.args))...
        )
    elseif isa(ex, Expr)
        return Expr(ex.head, (addhadamardfunctions(e) for e in ex.args)...)
    else
        return ex
    end
end

# scalar type expression of a scalar factor
instantiate_scalartype(α::Number) = typeof(α)
instantiate_scalartype(α) = Expr(:call, :scalartype, α)

# scalar type expression of a product term `α * T1 * T2`
function instantiate_term_scalartype(α, A_term::TensorTerm, B_term::TensorTerm)
    return Expr(
        :call, :promote_hadamard,
        Expr(:call, :scalartype, A_term.object),
        Expr(:call, :scalartype, B_term.object),
        instantiate_scalartype(α)
    )
end

# bind a non-literal scalar factor to a gensym to avoid evaluating it twice
function bindscalar(out::Expr, α)
    if α isa Expr
        αsym = gensym("α")
        push!(out.args, Expr(:(=), αsym, α))
        return αsym
    else
        return α
    end
end

# generate a single `hadamardproduct!` call for a product term
function instantiate_hadamard(out::Expr, dst, β, α, A_term::TensorTerm, B_term::TensorTerm, lhs_indices)
    pA, pB, pAB = hadamard_indices(A_term.indices, B_term.indices, lhs_indices)
    push!(
        out.args,
        :(
            hadamardproduct!(
                $dst, $(A_term.object), $pA, $(A_term.conj),
                $(B_term.object), $pB, $(B_term.conj), $pAB, $α, $β
            )
        )
    )
    return nothing
end

"""
    parse_hadamard(ex)

Parse an expression of the form

```julia
@hadamard C[i,j,k] := A[i,j] * B[j,k]
```

into a `Expr(:block)` that computes the Hadamard product, using the function based API.
"""
function parse_hadamard(ex)
    mode = if isexpr(ex, :(:=))
        :definition
    elseif isexpr(ex, :(=))
        :assignment
    elseif isexpr(ex, :(+=))
        :accumulate
    else
        throw(ArgumentError("expected an assignment `C[i,j] = ...`, `C[i,j] := ...` or `C[i,j] += ...`, got: $ex"))
    end
    lhs = ex.args[1]
    rhs = ex.args[2]

    isexpr(lhs, :ref) ||
        throw(ArgumentError("left hand side should be a tensor like `C[i,j,k]`, got: $lhs"))
    dst = lhs.args[1]
    lhs_indices = collect(lhs.args[2:end])
    _checklabels(lhs_indices)
    allunique(lhs_indices) ||
        throw(IndexError("left hand side indices should be unique: $lhs"))

    terms = sum_terms(rhs)
    isempty(terms) && throw(ArgumentError("empty right hand side: $rhs"))

    out = Expr(:block)

    if mode == :definition
        # decompose all terms and bind their scalar factors
        bound = []
        for (sign, term) in terms
            α, t1, t2 = decompose_term(term)
            αs = bindscalar(out, applysign(α, sign))
            push!(bound, (αs, t1, t2))
        end
        # scalar type of the full right hand side, promoted over all terms
        TC = Expr(:call, :promote_type, [instantiate_term_scalartype(αs, t1, t2) for (αs, t1, t2) in bound]...)
        TCsym = gensym("TC")
        push!(out.args, Expr(:(=), TCsym, TC))
        # allocate the output using the first term, then accumulate all terms
        for (k, (αs, t1, t2)) in enumerate(bound)
            if k == 1
                pA, pB, pAB = hadamard_indices(t1.indices, t2.indices, lhs_indices)
                push!(
                    out.args,
                    :($dst = tensoralloc_hadamard($TCsym, $(t1.object), $pA, $(t1.conj), $(t2.object), $pB, $(t2.conj), $pAB))
                )
                push!(
                    out.args,
                    :($dst = hadamardproduct!($dst, $(t1.object), $pA, $(t1.conj), $(t2.object), $pB, $(t2.conj), $pAB, $αs, 0))
                )
            else
                instantiate_hadamard(out, dst, 1, αs, t1, t2, lhs_indices)
            end
        end
    else
        β0 = mode == :assignment ? 0 : 1
        for (k, (sign, term)) in enumerate(terms)
            α, t1, t2 = decompose_term(term)
            αs = bindscalar(out, applysign(α, sign))
            instantiate_hadamard(out, dst, k == 1 ? β0 : 1, αs, t1, t2, lhs_indices)
        end
    end
    return addhadamardfunctions(out)
end
