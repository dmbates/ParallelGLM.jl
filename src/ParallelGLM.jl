module ParallelGLM
    using Distributions, Docile, StatsBase
    export GLM,                         # types
           IdentityLink,
           InverseLink,
           Link,
           LogLink,
           LogitLink,
           PGLM,
           SGLM,
           
           canonical,                   # functions
           devresid2,
           initμη!,
           invlink,
           link,
           updateXtW!,
           varfunc,
           μη

    include("link.jl")
    include("dist.jl")
    include("pglm.jl")
    include("irls.jl")

end # module
