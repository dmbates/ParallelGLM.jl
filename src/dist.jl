## Methods for canonical, varfunc and devresid2 for distributions in the exponential family

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
ylogydμ{T<:AbstractFloat}(y::T,μ::T) = y > zero(T) ? y*log(y/μ) : zero(T)

varfunc(::Bernoulli,μ) = μ*(one(μ)-μ)
varfunc(::Binomial,μ) = μ*(one(μ)-μ)
varfunc(::Gamma,μ) = abs2(μ)
varfunc(::Normal,μ) = one(μ)
varfunc(::Poisson,μ) = μ

two(y) = one(y) + one(y)                # equivalent to convert(typeof(y),2)

@doc """
Evaluate the squared deviance residual for a distribution instance and values of `y` and `μ`
"""->
devresid2(::Bernoulli,y,μ) = two(y)*(ylogydμ(y,μ) + ylogydμ(one(y)-y,one(μ)-μ))
devresid2(::Binomial,y,μ) = devresid2(Bernoulli(),y,μ)
devresid2(::Gamma,y,μ) =  two(y)*((y-μ)/μ - (y == zero(y) ? y : log(y/μ)))
devresid2(::Normal,y,μ) = abs2(y-μ)
devresid2(::Poisson,y,μ) = two(y)*(ylogydμ(y,μ)-(y-μ))

@doc """
Initial μ value from the y and wt
""" ->
mustart{T<:AbstractFloat}(::Bernoulli,y::T,wt::T) = (wt*y + convert(T,0.5))/(wt + one(T))
mustart{T<:AbstractFloat}(::Binomial,y::T,wt::T) = (wt*y + convert(T,0.5))/(wt + one(T))
mustart(::Gamma,y,wt) = y
mustart(::Normal,y,wt) = y
mustart{T<:AbstractFloat}(::Poisson,y::T,wt::T) = convert(T,1.1)*y
