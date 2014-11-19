abstract Link

type IdentityLink <: Link end
type InverseLink <: Link end
type LogLink <: Link end
type LogitLink <: Link end

link{T<:FloatingPoint}(::IdentityLink,μ::T) = μ
invlink{T<:FloatingPoint}(::IdentityLink,η::T) = η
μη{T<:FloatingPoint}(::IdentityLink,η::T) = one(T)

link{T<:FloatingPoint}(::LogLink,μ::T) = log(μ)
invlink{T<:FloatingPoint}(::LogLink,η::T) = exp(η)
μη{T<:FloatingPoint}(::LogLink,η::T) = exp(η)

link{T<:FloatingPoint}(::LogitLink,μ::T) = log(μ/(one(T) - μ))
invlink{T<:FloatingPoint}(::LogitLink,η::T) = inv(one(T) + exp(-η))
μη{T<:FloatingPoint}(::LogitLink,η::T) = (μ = invlink(LogitLink(),η); μ*(one(T) - μ))  

function linkl!{T<:FloatingPoint}(η::SharedArray{T},l::Link,μ::SharedArray{T})
    for j in localindexes(μ)
        η[j] = link(l,μ)
    end
end
function invlinkl!{T<:FloatingPoint}(μ::SharedArray{T},l::Link,η::SharedArray{T})
    for j in localindexes(η)
        μ[j] = invlink(l,η)
    end
end
function μηl!{T<:FloatingPoint}(mueta::SharedArray{T},l::Link,η::SharedArray{T}) 
    for j in localindexes(η)
        mueta[j] = μη(l,η)
    end
end
function link!{T<:FloatingPoint}(η::SharedArray{T},l::Link,μ::SharedArray{T})
    length(η) == length(μ) || throw(DimensionMismatch(""))
    @sync for p in procs(μ)
         @async remotecall_wait(p,linkl!,η,l,μ)
    end
    η
end
link{T<:FloatingPoint}(l::Link,μ::SharedArray{T}) = link!(similar(μ),l,μ)
