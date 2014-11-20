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
The Bernoulli and binomial variance function
"""->
bernvar(μ) = μ*(one(μ)-μ)

varfunc(::Bernoulli) = bernvar
varfunc(::Binomial) = bernvar
varfunc(::Gamma) = abs2
varfunc(::Normal) = one
varfunc(::Poisson) = id

@doc """
Evaluate half the squared deviance residual for a distribution instance and values of `y` and `μ`
"""->
devresid2(::Bernoulli) = (y,μ) -> 2*(ylogydμ(y,μ) + ylogydμ(one(y)-y,one(μ)-μ))
devresid2(::Binomial) = devresid2(Bernoulli())
devresid2(::Gamma) = (y,μ) -> (y-μ)/μ - (y == zero(T) ? y : log(y/μ))
devresid2(::Normal) = (y,μ) -> abs2(y-μ)
devresid2(::Poisson) = (y,μ) -> 2*(ylogydμ(y,μ)-(y-μ))

@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:FloatingPoint}
    Xt::SharedMatrix{T}
    wt::SharedVector{T}
    y::SharedVector{T}
    β::SharedVector{T}
    β₀::SharedVector{T}
    δβ::SharedVector{T}
    η::SharedVector{T}
    μ::SharedVector{T}
    dev::SharedVector{T}
    link::Function
    invlink::Function
    μη::Function
    varfunc::Function
    devresid2::Function
    canon::Bool
    fit::Bool
end

function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                wt::SharedVector{T},
                                d::UnivariateDistribution,
                                l::Link)
    p,n = size(Xt)
    n == length(y) || throw(DimensionMismatch(""))
    (lw = length(wt)) == 0 || lw == n || throw(DimensionMismatch(""))
    pr = procs(y)
    Set(pr) == Set(procs(Xt)) == Set(procs(wt)) || error("SharedArrays must have same procs")
    β = Base.shmem_fill(zero(T),(p,);pids = pr)
    ntot = maximum(pr)
    PGLM(Xt,wt,y,β,copy(β),copy(β),similar(y),similar(y),
         similar(y,(ntot,)),link(l),invlink(l),μη(l),
         varfunc(d),devresid2(d),l==canonical(d),false)
end
function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},y::SharedVector{T},
                                d::UnivariateDistribution,l::Link)
    PGLM(Xt,y,similar(y,(0,)),d,l)
end
function PGLM{T<:FloatingPoint}(Xt::SharedMatrix{T},
                                y::SharedVector{T},
                                d::UnivariateDistribution)
    PGLM(Xt,y,d,canonical(d))
end

@doc """
Evaluate the sum of the squared deviance residuals on the local indices of `y`
""" ->

function loc_dev!{T<:FloatingPoint}(g::PGLM{T})
    usewt = length(g.wt) > 0
    dev = zero(T)
    @inbounds for j in localindexes(g.y)
        sm = zero(T)
        @simd for k in 1:length(g.β)
            sm += g.Xt[k,j]*g.β[k]
        end
        g.η[j] = sm
        g.μ[j] = g.invlink(sm)
        dr2 = g.devresid2(g.y[j],g.μ[j])
        dev += usewt ? g.wt[j] * dr2 : dr2
    end
    g.dev[g.y.pidx] = dev
end

@doc """
Evaluate the deviance for `PGLM` model `g` using step factor `s`

This method also updates the `η` and `μ` members of `g`
"""->
function StatsBase.deviance{T<:FloatingPoint}(g::PGLM{T},s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.β[k] = g.β₀[k] + s * g.δβ[k]
    end
    fill!(g.dev,zero(T))
    @sync for p in procs(g.y)
        @async remotecall_wait(p,loc_dev!,g)
    end
    sum(g.dev)
end

function dev1proc{T<:FloatingPoint}(g::PGLM{T},s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.β[k] = g.β₀[k] + s * g.δβ[k]
    end
    usewt = length(g.wt) > 0
    dev = zero(T)
    for j in 1:length(g.y)
        sm = zero(T)
        @simd for k in 1:length(g.β)
            sm += g.Xt[k,j]*g.β[k]
        end
        g.η[j] = sm
        g.μ[j] = g.invlink(sm)
        dr2 = g.devresid2(g.y[j],g.μ[j])
        dev += usewt ? g.wt[j] * dr2 : dr2
    end
    dev
end

function dev1procinline{T<:FloatingPoint}(g::PGLM{T},s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.β[k] = g.β₀[k] + s * g.δβ[k]
    end
    usewt = length(g.wt) > 0
    dev = zero(T)
    for j in 1:length(g.y)
        sm = zero(T)
        @simd for k in 1:length(g.β)
            sm += g.Xt[k,j]*g.β[k]
        end
        g.η[j] = sm
        g.μ[j] = logistic(sm)
        dr2 = 2*(ylogydμ(g.y[j],g.μ[j]) + ylogydμ(one(T)-g.y[j],one(T)-g.μ[j]))
        dev += usewt ? g.wt[j] * dr2 : dr2
    end
    dev
end

### Create a macro to cause compilation of the loc_dev function with explicit substitution of
### linkinv and devresid2 for particular combinations of D and L
