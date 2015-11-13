"""
`loc_dev(g) -> deviance from local indices`

Evaluate the sum of the squared deviance residuals on the local indices of `y`
"""
function loc_dev!{T<:AbstractFloat}(g::PGLM{T})
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

"""
Evaluate the deviance for `PGLM` model `g` using step factor `s`

This method also updates the `η` and `μ` members of `g`
"""
function StatsBase.deviance{T<:AbstractFloat}(g::PGLM{T},s::T)
    @simd for k in 1:length(g.β)
        @inbounds g.βs[k] = g.β[k] + s * g.δβ[k]
    end
    fill!(g.dev,zero(T))
    pmap(loc_dev!,fill(g,nworkers()))
    sum(g.dev)
end
StatsBase.deviance{T<:AbstractFloat}(g::PGLM{T}) = deviance(g,zero(T))

"""
`loc_updateXtW(g) -> nothing`

local function to update the `XtW` array
"""
function loc_updateXtW!{T<:AbstractFloat}(g::PGLM{T})
    n,p,npr = size(g)
    k = procs(g.y)[g.y.pidx]
    if g.l == canonical(g.d)
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
            W = g.wt[ii] * abs2(mueta)/max(eps(T),varfunc(g.d,g.μ[ii]))
            for j in 1:p
                @simd for i in j:p
                    g.XtWX[i,j,k] += g.Xt[i,ii] * W * g.Xt[j,ii]
                end
                g.XtWr[j,k] += g.Xt[j,ii]*W*(g.y[ii] - g.μ[ii])/max(eps(T),mueta)
            end
        end
    end
end

"""
`updateXtW(g) -> δβ`

update the `XtW`, `XtWr` and `δβ` arrays in `g`
"""
function updateXtW!{T<:AbstractFloat}(g::PGLM{T})
    n,p,npr = size(g)
    fill!(g.XtWX,zero(T))
    fill!(g.XtWr,zero(T))
    pmap(loc_updateXtW!, fill(g,nworkers()))
    for k in procs(g.y)
        for j in 1:p
            for i in j:p
                g.XtWX[i,j,1] += g.XtWX[i,j,k]
            end
            g.δβ[j] += g.XtWr[j,k]
        end
    end
    A_ldiv_B!(cholfact!(sub(sdata(g.XtWX),:,:,1),:L),g.δβ)
end

function StatsBase.fit(g::GLM;verbose::Bool=false, maxIter::Integer=30,
                       minStepFac::Real=0.001, convTol::Real=1.e-6)
    g.fit && return g
    maxIter > zero(maxIter) || error("maxIter must be positive")
    zero(minStepFac) < minStepFac < one(minStepFac) || error("minStepFac must be in (0,1)")
    T = eltype(g.y)

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

function loc_initμη!(g::PGLM)
    for i in localindexes(g.y)
        g.μ[i] = mustart(g.d,g.y[i],g.wt[i])
        g.η[i] = link(g.l,g.μ[i])
    end
end

"""
`initμη(g) -> g`

Initialize the `μ` and `η` members of `g`
"""
initμη!(g::PGLM) = (pmap(loc_initμη!,fill(g,nworkers())), g)

function initμη!(g::SGLM)
    for i in 1:length(g.y)
        g.μ[i] = mustart(g.d,g.y[i],g.wt[i])
        g.η[i] = link(g.l,g.μ[i])
    end
    g
end

function updateXtW!{T<:AbstractFloat}(g::SGLM{T})
    p,n = size(g.Xt)
    fill!(g.XtWX,zero(T))
    fill!(g.δβ,zero(T))
    if g.l == canonical(g.d)
        @inbounds for ii in 1:length(g.y)
            W = sqrt(g.wt[ii] * varfunc(g.d,g.μ[ii]))
            g.wtres[ii] = g.y[ii] - g.μ[ii]
            @simd for j in 1:p
                g.XtW[j,ii] = W * g.Xt[j,ii]
            end
        end
        return A_ldiv_B!(cholfact(BLAS.syrk!('L','N',one(T),g.XtW,zero(T),g.XtWX),:L),
                         BLAS.gemv!('N',one(T),g.Xt,g.wtres,zero(T),g.δβ))
    end
    for ii in 1:length(g.y)
        mueta = μη(g.l,g.η[ii])
        W = g.wt[ii] * abs2(mueta)/max(eps(T),varfunc(g.d,g.μ[ii]))
        for j in 1:p
            @simd for i in j:p
                g.XtWX[i,j] += g.Xt[i,ii] * W * g.Xt[j,ii]
            end
            g.δβ[j] += g.Xt[j,ii]*W*(g.y[ii] - g.μ[ii])/max(eps(T),mueta)
        end
    end
    A_ldiv_B!(cholfact!(g.XtWX,:L),g.δβ)
end

"""
Evaluate the deviance for an `SGLM` model `g` using step factor `s`

This method also updates the `η` and `μ` members of `g`
"""
function StatsBase.deviance{T<:AbstractFloat}(g::SGLM{T},s::T)
    ## g.βs := g.β + s * g.δβ; g.η := g.Xt'*g.βs
    BLAS.gemv!('T',one(T),g.Xt,BLAS.axpy!(s,g.δβ,copy!(g.βs,g.β)),zero(T),g.η)
    dev = zero(T)
    @inbounds for j in 1:length(g.y)
        g.μ[j] = invlink(g.l,g.η[j])
        dev += g.wt[j] * devresid2(g.d,g.y[j],g.μ[j])
    end
    dev
end
