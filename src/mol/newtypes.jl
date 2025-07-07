import Base: ==, hash

mutable struct BundledAtomData
    name::String
    type::String
    resname::String
    mass::Float32
    charge::Float32
    σ::Float32
    ϵ::Float32
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