@doc "Representation of a generalized linear model using SharedArrays" ->
type PGLM{T<:FloatingPoint,D<:UnivariateDistribution,L<:Link}
    Xt::SharedMatrix{T}                 # transposed model matrix
    XtWX::SharedArray{T,3}
    XtWr::SharedMatrix{T}
    wt::SharedVector{T}                 # prior case weights
    y::SharedVector{T}                  # observed response vector
    β::SharedVector{T}                  # base value of β
    βs::SharedVector{T}                 # value of β + s*̱δβ
    δβ::SharedVector{T}                 # increment
    η::SharedVector{T}                  # current linear predictor vector
    μ::SharedVector{T}                  # current mean vector
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
        dev += g.wt[j] * devresid2(g.d,g.y[j],g.μ[j])
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
    pmap(loc_dev!,fill(g,nworkers()))
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
    pmap(loc_updateXtW!, fill(g,nworkers()))
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
    
function StatsBase.fit{T<:FloatingPoint}(g::PGLM{T};verbose::Bool=false, maxIter::Integer=30,
                                         minStepFac::Real=0.001, convTol::Real=1.e-6)
    g.fit && return g
    maxIter > zero(maxIter) || error("maxIter must be positive")
    zero(minStepFac) < minStepFac < one(minStepFac) || error("minStepFac must be in (0,1)")

    cvg = false
    devold = deviance(g,zero(T))
    for i in 1:maxIter
        updateXtW!(g)
        s = one(T)                      # step factor
        dev = deviance(g,s)
        while dev > devold
            s *= convert(T,0.5)         # halve the step factor
            s > minStepFac || error("step-halving failed at β₀ = $(g.β)")
            dev = deviance(g,s)
        end
        copy!(g.β,g.βs)
        crit = (devold - dev)/dev
        verbose && println("$i: $dev, $crit")
        if crit < convTol
            cvg = true
            break
        end
        devold = dev
    end
    cvg || error("failure to converge in $maxIter iterations")
    g.fit = true
    g
end
