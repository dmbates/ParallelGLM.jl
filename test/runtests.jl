using Base.Test, StatsBase

addprocs(2)
@everywhere using Distributions, ParallelGLM

n = 100_000
y = map(round,Base.shmem_rand(n))  # random 0/1 Float64 values
Xt = convert(SharedArray,hcat(ones(n),rand(n))')
g = PGLM(Xt,y,Bernoulli())
deviance(g,1.0)                         # should be n*2*log(2.)
