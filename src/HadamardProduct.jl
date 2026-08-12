module HadamardProduct

# Exports
#---------
# export macro API
export @hadamard

# export function based API
export hadamardproduct, hadamardproduct!
export hadamardproduct_structure, hadamardproduct_type, tensoralloc_hadamard
export tensoralloc, tensorfree!, scalartype
export checkhadamardproduct

# export index types
export IndexTuple, Index2Tuple, linearize

# Dependencies
#-------------
using Strided: StridedView, sreshape
using VectorInterface: scalartype

# Interface and index types
#---------------------------
include("indices.jl")
include("interface.jl")

# Implementations
#-----------------
include("implementation/abstractarray.jl")
include("implementation/strided.jl")

# Index notation via macros
#---------------------------
include("indexnotation/parser.jl")
include("indexnotation/instantiator.jl")
include("indexnotation/hadamardmacros.jl")

end # module
