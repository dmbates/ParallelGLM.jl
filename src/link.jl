## Definition of Link types and methods for link, invlink and μη, the derivative of μ w.r.t. η

abstract Link

immutable IdentityLink <: Link end
immutable InverseLink <: Link end
immutable LogLink <: Link end
immutable LogitLink <: Link end

link(::IdentityLink,μ) = μ
invlink(::IdentityLink,η) = η
μη(::IdentityLink,η) = one(η)

link(::LogitLink,μ) = log(μ/(one(μ)-μ))
invlink(::LogitLink,η) = inv(one(η) + exp(-η))
μη(::LogitLink,η) = (ee = exp(-η); ee/abs2(one(η)+ee))

link(::LogLink,μ) = log(μ)
invlink(::LogLink,η) = exp(η)
μη(::LogLink,η) = exp(η)

link(::InverseLink,μ) = inv(μ)
invlink(::InverseLink,η) = inv(η)
μη(::InverseLink,η) = -inv(abs2(η))

