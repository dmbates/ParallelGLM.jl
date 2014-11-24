using Base.Test, StatsBase

addprocs(2)
@everywhere using Distributions, ParallelGLM

n = 100_000
srand(1234321)

const Xt = convert(SharedArray,vcat(ones(n)',randn(19,n)))
const βtrue = convert(SharedArray,randn(size(Xt,1)))
const ηtrue = Xt'*βtrue
const μtrue = similar(ηtrue);
const ll = LogitLink();
for i in 1:n
    μtrue[i] = invlink(ll,ηtrue[i])
end
const y = similar(μtrue)
for i in 1:n
    y[i] = rand() > μtrue[i] ? 0. : 1.
end

g = PGLM(Xt,y,Bernoulli());
fit(g; verbose=true);
