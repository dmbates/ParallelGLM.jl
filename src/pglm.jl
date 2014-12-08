abstract GLM

@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:FloatingPoint,D<:UnivariateDistribution,L<:Link} <: GLM
    Xt::SharedMatrix{T}                 # transposed model matrix
    XtWX::SharedArray{T,3}
    XtWr::SharedMatrix{T}
    wt::SharedVector{T}                 # prior case weights
    y::SharedVector{T}                  # observed response vector
    β::SharedVector{T}                  # base value of β
    βs::SharedVector{T}                 # value of β + s*δβ
    δβ::SharedVector{T}                 # increment
    η::SharedVector{T}                  # current linear predictor vector
    μ::SharedVector{T}                  # current mean vector
    dev::SharedVector{T}
    d::D
    l::L
    fit::Bool
end

function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                wt::SharedVector{T},
                                d::UnivariateDistribution,
                                l::Link)
    p,n = size(Xt)
    n == length(y) == length(wt) || throw(DimensionMismatch(""))
    pr = procs(y)
    Set(pr) == Set(procs(Xt)) == Set(procs(wt)) || error("SharedArrays must have same procs")
    β = Base.shmem_fill(zero(T),(p,);pids = pr)
    ntot = maximum(pr)
    g = PGLM(Xt,similar(y,(p,p,ntot)),similar(y,(p,ntot)),wt,y,
             β,copy(β),copy(β),similar(y),similar(y),similar(y,(ntot,)),
             d,l,false)
    initμη!(g)
    updateXtW!(g)
    copy!(g.β,g.δβ)
    g
end
function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},y::SharedVector{T},
                                d::UnivariateDistribution,l::Link)
    PGLM(Xt,y,fill!(similar(y),one(T)),d,l)
end
function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                d::UnivariateDistribution)
    PGLM(Xt,y,d,canonical(d))
end

type SGLM{T<:FloatingPoint,D<:UnivariateDistribution,L<:Link} <: GLM
    Xt::Matrix{T}                       # transposed model matrix
    XtWX::Matrix{T}
    wt::Vector{T}                    # prior case weights
    y::Vector{T}                     # observed response vector
    β::Vector{T}                     # base value of β
    βs::Vector{T}                    # value of β + s*δβ
    δβ::Vector{T}                    # increment
    η::Vector{T}                     # current linear predictor vector
    μ::Vector{T}                     # current mean vector
    d::D
    l::L
    fit::Bool
end
function SGLM{T<:FloatingPoint}(Xt::Matrix{T},
                                y::Vector{T},
                                wt::Vector{T},
                                d::UnivariateDistribution,
                                l::Link)
    p,n = size(Xt)
    n == length(y) == length(wt) || throw(DimensionMismatch(""))
    β = Array(T,(p,))
    g = SGLM(Xt,zeros(T,(p,p)),wt,y,β,Array(T,(p,)),Array(T,(p,)),
             similar(y),similar(y),d,l,false)
    initμη!(g)
    updateXtW!(g)
    copy!(g.β,g.δβ)
    g
end
function SGLM{T<:FloatingPoint}(Xt::Matrix{T},y::Vector{T},
                                d::UnivariateDistribution,l::Link)
    SGLM(Xt,y,fill!(similar(y),one(T)),d,l)
end
function SGLM{T<:FloatingPoint}(Xt::Matrix{T},
                                y::Vector{T},
                                d::UnivariateDistribution)
    SGLM(Xt,y,d,canonical(d))
end
