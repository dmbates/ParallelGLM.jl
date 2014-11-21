abstract Link

type IdentityLink <: Link end
type InverseLink <: Link end
type LogLink <: Link end
type LogitLink <: Link end

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

