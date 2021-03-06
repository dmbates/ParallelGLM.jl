abstract GLM

@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:AbstractFloat,D<:UnivariateDistribution,L<:Link} <: GLM
    Xt::SharedMatrix{T}                 # transposed model matrix
    XtWX::SharedArray{T,3}
    XtWr::SharedMatrix{T}
    wt::SharedVector{T}                 # prior case weights
    y::SharedVector{T}                  # observed response vector
    β::Vector{T}                        # base value of β
    βs::Vector{T}                       # value of β + s*δβ
    δβ::Vector{T}                       # increment
    η::SharedVector{T}                  # current linear predictor vector
    μ::SharedVector{T}                  # current mean vector
    dev::SharedVector{T}
    d::D
    l::L
    fit::Bool
end

scvt(a::Array) = convert(SharedArray,a)

function PGLM{T<:AbstractFloat}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                wt::SharedVector{T},
                                d::UnivariateDistribution,
                                l::Link)
    p,n = size(Xt)
    n == length(y) == length(wt) || throw(DimensionMismatch(""))
    pr = procs(y)
    Set(pr) == Set(procs(Xt)) == Set(procs(wt)) || error("SharedArrays must have same procs")
    β = zeros(p)
    ntot = maximum(pr)
    g = PGLM(Xt,scvt(similar(y,(p,p,ntot))),scvt(similar(y,(p,ntot))),wt,y,
             β,copy(β),copy(β),scvt(similar(y)),scvt(similar(y)),scvt(similar(y,(ntot,))),
             d,l,false)
    initμη!(g)
    updateXtW!(g)
    copy!(g.β,g.δβ)
    g
end
function PGLM{T<:AbstractFloat}(Xt::SharedMatrix{T},y::SharedVector{T},
                                d::UnivariateDistribution,l::Link)
    PGLM(Xt,y,scvt(ones(y)),d,l)
end
function PGLM{T<:AbstractFloat}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                d::UnivariateDistribution)
    PGLM(Xt,y,d,canonical(d))
end
function PGLM{T<:AbstractFloat}(Xt::Matrix{T},y::Vector{T},d::UnivariateDistribution)
    PGLM(scvt(Xt),scvt(y),d)
end

type SGLM{T<:AbstractFloat,D<:UnivariateDistribution,L<:Link} <: GLM
    Xt::Matrix{T}                    # transposed model matrix
    XtW::Matrix{T}                   # weighted, transposed model matrix
    XtWX::Matrix{T}                  # weighted cross-product
    wt::Vector{T}                    # prior case weights
    wtres::Vector{T}                 # weighted residuals
    y::Vector{T}                     # observed response vector
    β::Vector{T}                     # base value of β
    βs::Vector{T}                    # value of β + s*δβ
    δβ::Vector{T}                    # increment
    η::Vector{T}                     # current linear predictor vector
    μ::Vector{T}                     # current mean vector
    d::D
    l::L
    blas::Bool                       # use BLAS.syrk! to evaluate XtWX
    fit::Bool
end
function SGLM{T<:AbstractFloat}(Xt::Matrix{T},
                                y::Vector{T},
                                wt::Vector{T},
                                d::UnivariateDistribution,
                                l::Link,
                                blas::Bool)
    p,n = size(Xt)
    XtW = blas ? similar(Xt) : similar(Xt,(0,0))
    n == length(y) == length(wt) || throw(DimensionMismatch(""))
    β = Array(T,(p,))
    g = SGLM(Xt,XtW,zeros(T,(p,p)),wt,similar(y),y,β,Array(T,(p,)),Array(T,(p,)),
             similar(y),similar(y),d,l,blas,false)
    initμη!(g)
    updateXtW!(g)
    copy!(g.β,g.δβ)
    g
end
function SGLM{T<:AbstractFloat}(Xt::Matrix{T},y::Vector{T},
                                d::UnivariateDistribution,l::Link,blas::Bool)
    SGLM(Xt,y,fill!(similar(y),one(T)),d,l,blas)
end
function SGLM{T<:AbstractFloat}(Xt::Matrix{T},
                                y::Vector{T},
                                d::UnivariateDistribution,
                                blas::Bool)
    SGLM(Xt,y,d,canonical(d),blas)
end
function SGLM{T<:AbstractFloat}(Xt::Matrix{T},
                                y::Vector{T},
                                d::UnivariateDistribution)
    SGLM(Xt,y,d,canonical(d),true)
end
