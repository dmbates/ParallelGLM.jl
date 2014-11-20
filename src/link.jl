abstract Link

immutable IdentityLink <: Link end
immutable InverseLink <: Link end
immutable LogLink <: Link end
immutable LogitLink <: Link end

logit(μ) = log(μ/(one(μ)-μ))
logistic(η) = inv(one(η) + exp(-η))
logisticder(η) = (ee = exp(-η); ee/abs2(one(η)+ee))
id(x) = x
ninvabs2(η) = -inv(abs2(η))

link(::IdentityLink) = id
invlink(::IdentityLink) = id
μη(::IdentityLink) = one

link(::LogitLink) = logit
invlink(::LogitLink) = logistic
μη(::LogitLink) = logisticder

link(::LogLink) = log
invlink(::LogLink) = exp
μη(::LogLink) = exp

link(::InverseLink) = inv
invlink(::InverseLink) = inv
μη(::InverseLink) = ninvabs2

