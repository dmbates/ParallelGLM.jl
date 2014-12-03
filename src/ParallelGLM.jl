module ParallelGLM
    using Distributions, Docile, StatsBase
    export IdentityLink,                # types
           InverseLink,
           Link,
           LogLink,
           LogitLink,
           PGLM,
           
           canonical,                   # functions
           devresid2,
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
