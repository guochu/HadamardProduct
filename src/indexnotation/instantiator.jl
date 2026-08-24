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

# scalar type expression of a product term `α * T1 * T2 * ...`
function instantiate_term_scalartype(α, tree)
    sts = term_scalartypes(tree)
    st = Expr(
        :call, :promote_hadamard,
        Expr(:call, :scalartype, sts[1].object),
        Expr(:call, :scalartype, sts[2].object),
        instantiate_scalartype(α)
    )
    for t in sts[3:end]
        st = Expr(
            :call, :promote_hadamard,
            st, Expr(:call, :scalartype, t.object), instantiate_scalartype(α)
        )
    end
    return st
end

# collect all `TensorTerm` leaves of a product tree, in depth-first order
term_scalartypes(t::TensorTerm) = [t]
term_scalartypes(t::ProdNode) = vcat(term_scalartypes(t.left), term_scalartypes(t.right))

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

# a sub-product is a plain product of two tensor terms, requiring no intermediate storage
isleafpair(t::ProdNode) = t.left isa TensorTerm && t.right isa TensorTerm

# generate the `hadamardproduct!` calls computing a sub-tree of a product tree into a fresh
# temporary with the natural index order; return the expression, its indices and conj flag
function instantiate_subtree(out::Expr, node, TCsym)
    if node isa TensorTerm
        return node.object, node.indices, node.conj
    end
    l_expr, l_indices, l_conj = instantiate_subtree(out, node.left, TCsym)
    r_expr, r_indices, r_conj = instantiate_subtree(out, node.right, TCsym)
    tmp_indices = unique(vcat(l_indices, r_indices))
    pA, pB, pAB = hadamard_indices(l_indices, r_indices, tmp_indices)
    tmp = gensym("tmp")
    push!(out.args, :($tmp = tensoralloc_hadamard($TCsym, $l_expr, $pA, $l_conj, $r_expr, $pB, $r_conj, $pAB)))
    push!(out.args, :(hadamardproduct!($tmp, $l_expr, $pA, $l_conj, $r_expr, $pB, $r_conj, $pAB, 1, 0)))
    return tmp, tmp_indices, false
end

# generate `hadamardproduct!` calls for a product term `α * T1 * T2 * ...`, computing
# `α * (T1 ⊙ T2) ⊙ ...` following the structure of the product tree `tree` (parentheses are
# preserved as intermediate results) into `dst`, accumulating with `β`. If `alloc_dst` is
# true, `dst` is allocated before the final step. Intermediate results are allocated with
# the promoted scalar type `TC` (or one computed from the term if `TC === nothing`). If the
# left hand side groups its indices with a semicolon, `pC` is the `Index2Tuple` of the
# positions of the left (codomain) and right (domain) indices of `dst` (following the
# `@tensor` convention of TensorOperations.jl and TensorKit.jl), and is forwarded to
# `tensoralloc_hadamard`.
function instantiate_hadamard(
        out::Expr, dst, β, α, tree, lhs_indices, alloc_dst::Bool, TC = nothing, pC = nothing
    )
    if TC === nothing && (alloc_dst || !isleafpair(tree))
        TCsym = gensym("TC")
        push!(out.args, Expr(:(=), TCsym, instantiate_term_scalartype(α, tree)))
    else
        TCsym = TC
    end

    l_expr, l_indices, l_conj = instantiate_subtree(out, tree.left, TCsym)
    r_expr, r_indices, r_conj = instantiate_subtree(out, tree.right, TCsym)
    pA, pB, pAB = hadamard_indices(l_indices, r_indices, lhs_indices)
    if alloc_dst
        if pC === nothing
            push!(out.args, :($dst = tensoralloc_hadamard($TCsym, $l_expr, $pA, $l_conj, $r_expr, $pB, $r_conj, $pAB)))
        else
            push!(out.args, :($dst = tensoralloc_hadamard($TCsym, $l_expr, $pA, $l_conj, $r_expr, $pB, $r_conj, $pAB, $pC)))
        end
    end
    push!(out.args, :(hadamardproduct!($dst, $l_expr, $pA, $l_conj, $r_expr, $pB, $r_conj, $pAB, $α, $β)))
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

    dst, lhs_left, lhs_right = decompose_tensorterm(lhs)
    lhs_indices = vcat(lhs_left, lhs_right)
    _checklabels(lhs_indices)
    allunique(lhs_indices) ||
        throw(IndexError("left hand side indices should be unique: $lhs"))
    # group the output indices into the left (codomain) and right (domain) part, following
    # the `@tensor` convention of TensorOperations.jl and TensorKit.jl; forwarded to
    # `tensoralloc_hadamard` for tensor types that distinguish the two groups
    pC = isempty(lhs_right) ? nothing : (
        Tuple(_findpos(l, lhs_indices) for l in lhs_left),
        Tuple(_findpos(l, lhs_indices) for l in lhs_right),
    )

    terms = sum_terms(rhs)
    isempty(terms) && throw(ArgumentError("empty right hand side: $rhs"))

    out = Expr(:block)

    if mode == :definition
        # decompose all terms and bind their scalar factors
        bound = []
        for (sign, term) in terms
            α, tensors = decompose_term(term)
            αs = bindscalar(out, applysign(α, sign))
            push!(bound, (αs, tensors))
        end
        # scalar type of the full right hand side, promoted over all terms
        TC = Expr(:call, :promote_type, [instantiate_term_scalartype(αs, tensors) for (αs, tensors) in bound]...)
        TCsym = gensym("TC")
        push!(out.args, Expr(:(=), TCsym, TC))
        # allocate the output using the first term, then accumulate all terms
        for (k, (αs, tensors)) in enumerate(bound)
            instantiate_hadamard(out, dst, k == 1 ? 0 : 1, αs, tensors, lhs_indices, k == 1, TCsym, pC)
        end
    else
        β0 = mode == :assignment ? 0 : 1
        for (k, (sign, term)) in enumerate(terms)
            α, tensors = decompose_term(term)
            αs = bindscalar(out, applysign(α, sign))
            instantiate_hadamard(out, dst, k == 1 ? β0 : 1, αs, tensors, lhs_indices, false, nothing, pC)
        end
    end
    return addhadamardfunctions(out)
end
