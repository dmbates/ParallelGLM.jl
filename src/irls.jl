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

varfunc(::Bernoulli,μ) = μ*(one(μ)-μ)
varfunc(::Binomial,μ) = μ*(one(μ)-μ)
varfunc(::Gamma,μ) = abs2(μ)
varfunc(::Normal,μ) = one(μ)
varfunc(::Poisson,μ) = μ

two(y) = one(y) + one(y)
@doc """
Evaluate half the squared deviance residual for a distribution instance and values of `y` and `μ`
"""->
devresid2(::Bernoulli,y,μ) = two(y)*(ylogydμ(y,μ) + ylogydμ(one(y)-y,one(μ)-μ))
devresid2(::Binomial,y,μ) = devresid2(Bernoulli(),y,μ)
devresid2(::Gamma,y,μ) =  two(y)*((y-μ)/μ - (y == zero(y) ? y : log(y/μ)))
devresid2(::Normal,y,μ) = abs2(y-μ)
devresid2(::Poisson,y,μ) = two(y)*(ylogydμ(y,μ)-(y-μ))

@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:FloatingPoint,D<:UnivariateDistribution,L<:Link}
    Xt::SharedMatrix{T}
    XtWX::SharedArray{T,3}
    XtWr::SharedMatrix{T}
    wt::SharedVector{T}
    y::SharedVector{T}
    β::SharedVector{T}                  # base value of β
    βs::SharedVector{T}                 # value of β + s*̱δβ
    δβ::SharedVector{T}                 # increment
    η::SharedVector{T}
    μ::SharedVector{T}
    dev::SharedVector{T}
    d::D
    l::L
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
    PGLM(Xt,similar(y,(p,p,ntot)),similar(y,(p,ntot)),wt,y,
         β,copy(β),copy(β),similar(y),similar(y),similar(y,(ntot,)),
         d,l,l==canonical(d),false)
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

@doc """
Evaluate the sum of the squared deviance residuals on the local indices of `y`
""" ->

function loc_dev!{T<:FloatingPoint}(g::PGLM{T})
    dev = zero(T)
    @inbounds for j in localindexes(g.y)
        sm = zero(T)
        @simd for k in 1:length(g.βs)
            sm += g.Xt[k,j]*g.βs[k]
        end
        g.η[j] = sm
        g.μ[j] = invlink(g.l,sm)
        dr2 = devresid2(g.d,g.y[j],g.μ[j])
        dev += g.wt[j] * dr2
    end
    g.dev[g.y.pidx] = dev
end

Base.size(g::PGLM) = ((p,n) = size(g.Xt); (n,p,length(g.dev)))

@doc """
Evaluate the deviance for `PGLM` model `g` using step factor `s`

This method also updates the `η` and `μ` members of `g`
"""->
function StatsBase.deviance{T<:FloatingPoint}(g::PGLM{T},s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.βs[k] = g.β[k] + s * g.δβ[k]
    end
    fill!(g.dev,zero(T))
    @sync for p in procs(g.y)
        @async remotecall_wait(p,loc_dev!,g)
    end
    sum(g.dev)
end

function loc_updateXtW!{T<:FloatingPoint}(g::PGLM{T})
    n,p,npr = size(g)
    k = procs(g.y)[g.y.pidx]
    if g.canon
        @inbounds for ii in localindexes(g.y)
            W = g.wt[ii] * varfunc(g.d,g.μ[ii])
            for j in 1:p
                Wj = g.Xt[j,ii]
                g.XtWr[j,k] += Wj * (g.y[ii] - g.μ[ii])
                Wj *= W
                @simd for i in j:p
                    g.XtWX[i,j,k] += g.Xt[i,ii] * Wj
                end
            end
        end
    else
        for ii in localindexes(g.y)
            mueta = μη(g.l,g.η[ii])
            W = wt[ii] * abs2(mueta)/max(eps(T),varfunc(g.d,g.μ[ii]))
            for j in 1:p
                @simd for i in j:p
                    g.XtWX[i,j,k] += g.Xt[i,ii] * W * g.Xt[j,ii]
                end
                g.XtWr[j,k] += Xt[j,ii]*W*(g.y[ii] - g.μ[ii])/max(eps(T),mueta)
            end
        end
    end
end
function updateXtW!{T<:FloatingPoint}(g::PGLM{T})
    n,p,npr = size(g)
    fill!(g.XtWX,zero(T))
    fill!(g.XtWr,zero(T))
    @sync for pr in procs(g.y)
        @async remotecall_wait(pr,loc_updateXtW!,g)
    end
    for k in 2:npr
        for j in 1:p
            for i in j:p
                g.XtWX[i,j,1] += g.XtWX[i,j,k]
            end
            g.XtWr[j,1] += g.XtWr[j,k]
        end
    end
    A_ldiv_B!(cholfact!(view(sdata(g.XtWX),:,:,1),:L),copy!(sdata(g.δβ),view(sdata(g.XtWr),:,1)))
end

function update1!{T<:FloatingPoint}(g::PGLM{T})
    n,p,npr = size(g)
    usewt = length(g.wt) > 0
    fill!(g.XtWX,zero(T))
    fill!(g.XtWr,zero(T))
    @inbounds for ii in 1:n
        W = g.wt[ii] * varfunc(g.d,g.μ[ii])
        for j in 1:p
            Wj = g.Xt[j,ii]
            g.XtWr[j,1] += Wj * (g.y[ii] - g.μ[ii])
            Wj *= W
            @simd for i in j:p
                g.XtWX[i,j,1] += g.Xt[i,ii] * Wj
            end
        end
    end
    A_ldiv_B!(cholfact!(view(sdata(g.XtWX),:,:,1),:L),copy!(sdata(g.δβ),view(sdata(g.XtWr),:,1)))
end
    
