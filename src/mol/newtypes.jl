import Base: ==, hash

mutable struct BundledAtomData
    name::String
    type::String
    resname::String
    mass::T
    charge::T
    
    α::Union{T, Nothing}
    β::Union{T, Nothing}
    δ::Union{T, Nothing}
    γ::Union{T, Nothing}

    σ::Union{T, Nothing}
    ϵ::Union{T, Nothing}
    A::Union{T, Nothing}
    B::Union{T, Nothing}
    C::Union{T, Nothing}
end

function ==(a::BundledAtomData, b::BundledAtomData)
    return a.name == b.name &&
           a.type == b.type &&
           a.resname == b.resname &&
           a.mass == b.mass &&
           a.charge == b.charge &&
           a.σ == b.σ &&
           a.ϵ == b.ϵ
end

function hash(a::BundledAtomData, h::UInt)
    return hash((a.name, a.type, a.resname, a.mass, a.charge, a.σ, a.ϵ), h)
end