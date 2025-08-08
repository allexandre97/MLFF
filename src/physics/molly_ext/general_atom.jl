struct GeneralAtom{T}
    index     :: Int
    atom_type :: Int
    mass      :: T
    charge    :: T
    σ_lj      :: T
    ϵ_lj      :: T
    σ_lj69    :: T
    ϵ_lj69    :: T
    σ_dexp    :: T
    ϵ_dexp    :: T
    σ_buff    :: T
    ϵ_buff    :: T
    A_buck    :: T
    B_buck    :: T
    C_buck    :: T
end

function Base.zero(::GeneralAtom{T}) where T
    Z = zero(Int)
    z = zero(T)
    return GeneralAtom(Z, Z, z, z, z, z, z, z, z, z, z, z, z, z, z)
end

function Base.zero(::Type{GeneralAtom})
    z = zero(T)
    return GeneralAtom(0, 0, z, z, z, z, z, z, z, z, z, z, z, z, z)
end

function +(a::GeneralAtom{T}, b::GeneralAtom{T}) where T
    GeneralAtom(
        a.index,
        a.atom_type,
        a.mass    + b.mass,
        a.charge  + b.charge,
        a.σ_lj    + b.σ_lj,
        a.ϵ_lj    + b.ϵ_lj,
        a.σ_lj69  + b.σ_lj69,
        a.ϵ_lj69  + b.ϵ_lj69,
        a.σ_dexp  + b.σ_dexp,
        a.ϵ_dexp  + b.ϵ_dexp,
        a.σ_buff  + b.σ_buff,
        a.ϵ_buff  + b.ϵ_buff,
        a.A_buck  + b.A_buck,
        a.B_buck  + b.B_buck,
        a.C_buck  + b.C_buck
    )
end

function *(i::Float32, a::GeneralAtom{T}) where {T}
    GeneralAtom(
        a.index,
        a.atom_type,
        a.mass * i,
        a.charge * i,
        i * a.σ_lj,
        i * a.ϵ_lj,
        i * a.σ_lj69,
        i * a.ϵ_lj69,
        i * a.σ_dexp,
        i * a.ϵ_dexp,
        i * a.σ_buff,
        i * a.ϵ_buff,
        i * a.A_buck,
        i * a.B_buck,
        i * a.C_buck
    )
end

function ChainRulesCore.rrule(::Type{<:GeneralAtom}, vs...)
    Y = GeneralAtom(vs...)
    function general_atom_pullback(ŷ)
        return NoTangent(), 
               NoTangent(), 
               NoTangent(), 
               NoTangent(), 
               ŷ.charge,
               ŷ.σ_lj,
               ŷ.ϵ_lj,
               ŷ.σ_lj69,
               ŷ.ϵ_lj69,
               ŷ.σ_dexp,
               ŷ.ϵ_dexp,
               ŷ.σ_buff,
               ŷ.ϵ_buff,
               ŷ.A_buck,
               ŷ.B_buck,
               ŷ.C_buck
    end
    return Y, general_atom_pullback
end