module ParallelGLM
    using Distributions, StatsBase
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
    include("irls.jl")

end # module
