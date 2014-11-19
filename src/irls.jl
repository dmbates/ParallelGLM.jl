@doc """
Returns the canonical Link type for an instance of a distribution in the exponential family
""" ->
canonical(::Bernoulli) = LogitLink()
canonical(::Binomial) = LogitLink()
canonical(::Gamma) = InverseLink()
canonical(::Normal) = IdentityLink()
canonical(::Poisson) = LogLink()

@doc """
Evaluate `y*log(y/μ)` with the correct limit as `y` approaches zero from above
"""->
ylogydμ{T<:FloatingPoint}(y::T,μ::T) = y > zero(T) ? y*log(y/μ) : zero(T)

@doc """
Evaluate the variance function for distribution instance and value of `μ`
"""->
Base.var{T<:FloatingPoint}(::Bernoulli,μ::T) = μ*(one(T)-μ)
Base.var{T<:FloatingPoint}(::Binomial,μ::T) = μ*(one(T)-μ)
Base.var{T<:FloatingPoint}(::Gamma,μ::T) = abs2(μ)
Base.var{T<:FloatingPoint}(::Normal,μ::T) = one(T)
Base.var{T<:FloatingPoint}(::Poisson,μ::T) = μ

@doc """
Evaluate half the squared deviance residual for a distribution instance and values of `y` and `μ`
"""->
devresid2{T<:FloatingPoint}(::Bernoulli,y::T,μ::T) = ylogydμ(y,μ) + ylogydμ(one(T)-y,one(T)-μ)
devresid2{T<:FloatingPoint}(::Binomial,y::T,μ::T) = devresid2(Bernoulli(),y,μ)
devresid2{T<:FloatingPoint}(::Gamma,y::T,μ::T) = (y-μ)/μ - (y == zero(T) ? y : log(y/μ))
devresid2{T<:FloatingPoint}(::Normal,y::T,μ::T) = abs2(y-μ)
devresid2{T<:FloatingPoint}(::Poisson,y::T,μ::T) = ylogydμ(y,μ)-(y-μ)

@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:FloatingPoint,D<:UnivariateDistribution,L<:Link}
    Xt::SharedMatrix{T}
    wt::SharedVector{T}
    y::SharedVector{T}
    β::SharedVector{T}
    β₀::SharedVector{T}
    δβ::SharedVector{T}
    η::SharedVector{T}
    μ::SharedVector{T}
    d::D
    l::L
    canon::Bool
    fit::Bool
end

function PGLM{T<:FloatingPoint,D<:UnivariateDistribution,
              L<:Link}(Xt::SharedMatrix{T},
                       y::SharedVector{T},
                       wt::SharedVector{T},
                       d::D,
                       l::L)
    p,n = size(Xt)
    workrs = procs(y)
    n == length(y) || throw(DimensionMismatch(""))
    (lw = length(wt)) == 0 || lw == n || throw(DimensionMismatch(""))
    length(intersect(workrs,procs(Xt))) == length(workrs) ||
        error("worker procs for y and Xt are not identical")
    PGLM(Xt,wt,y,fill!(similar(y,(p,)),zero(T)),fill!(similar(y,(p,)),zero(T)),
         fill!(similar(y,(p,)),zero(T)),similar(y),similar(y),d,l,l==canonical(d),false)
end
function PGLM{T<:FloatingPoint,D<:UnivariateDistribution,
              L<:Link}(Xt::SharedMatrix{T},
                       y::SharedVector{T},
                       d::D,
                       l::L)
    PGLM(Xt,y,similar(y,(0,)),d,l)
end
function PGLM{T<:FloatingPoint,D<:UnivariateDistribution}(Xt::SharedMatrix{T},
                                                y::SharedVector{T},
                                                d::D)
    PGLM(Xt,y,d,canonical(d))
end

@doc """
Evaluate the sum of the squared deviance residuals on the local indices of `y`
""" ->
function locdev!{T<:FloatingPoint}(η::SharedVector{T},
                                   μ::SharedVector{T},
                                   Xt::SharedMatrix{T},
                                   y::SharedVector{T},
                                   wt::SharedVector{T},
                                   β::SharedVector{T},
                                   d::UnivariateDistribution,
                                   l::Link)
    dev = zero(T)
    usewt = length(wt) > 0
    @inbounds for j in localindexes(y)
        sm = zero(T)
        @simd for k in 1:length(β)
            sm += Xt[k,j]*β[k]
        end
        η[j] = sm
        μ[j] = invlink(l,sm)
        dr2 = devresid2(d,y[j],μ[j])
        dev += usewt ? wt[j] * dr2 : dr2
    end
    2dev
end

@doc """
Evaluate the deviance for `PGLM` model `g` using step factor `s`

This method also updates the `η` and `μ` members of `g`
"""->
function StatsBase.deviance{T<:FloatingPoint}(g::PGLM,s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.β[k] = g.β₀[k] + s * g.δβ[k]
    end
    dev = zeros(T,(length(procs(g.y)+1),))
    @sync for p in procs(g.y)
        @async dev[p] = remotecall_fetch(p,locdev!,g.η,g.μ,g.Xt,g.y,g.wt,g.β,g.d,g.l)
    end
    sum(dev)
end
