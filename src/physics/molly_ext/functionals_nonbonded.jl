##### LJ and Mie potential are already defined in Molly, we only need the rrules

function ChainRulesCore.rrule(TY::Type{<:LennardJones}, vs...)
    Y = TY(vs...)
    function LennardJones_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), Ȳ.weight_special
    end
    return Y, LennardJones_pullback
end

Base.:+(t::NamedTuple, i::LennardJones) = LennardJones(
    t.cutoff + i.cutoff,
    i.use_neighbors,
    i.shortcut,
    i.σ_mixing,
    i.ϵ_mixing,
    t.weight_special + i.weight_special
)

Base.:*(f::Float32, i::LennardJones) = LennardJones(
    i.cutoff,
    i.use_neighbors,
    i.shortcut,
    i.σ_mixing,
    i.ϵ_mixing,
    f * i.weight_special
)

function ChainRulesCore.rrule(TY::Type{<:Mie}, vs...)
    Y = TY(vs...)
    function Mie_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Mie_pullback
end

Base.:+(t::NamedTuple, i::Mie) = Mie(
    i.m,
    i.n,
    i.cutoff,
    i.use_neighbors,
    i.shortcut,
    i.σ_mixing,
    i.ϵ_mixing,
    t.weight_special + i.weight_special,
    i.mn_fac
)

Base.:*(f::Float32, i::Mie) = Mie(
    i.m,
    i.n,
    i.cutoff,
    i.use_neighbors,
    i.shortcut,
    i.σ_mixing,
    i.ϵ_mixing,
    f * i.weight_special,
    i.mn_fac
)


##### Double exponential

struct DoubleExponential{T, S, E, W}
    α::T
    β::T
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
    dist_cutoff::T
end

function Base.zero(i::DoubleExponential{T, S, E, W}) where {T, S, E, W}
    return DoubleExponential(zero(T), zero(T), i.σ_mixing, i.ϵ_mixing, zero(W), zero(T))
end

function Base.:+(i1::DoubleExponential, i2::DoubleExponential)
    return DoubleExponential(i1.α + i2.α, i1.β + i2.β, i1.σ_mixing, i1.ϵ_mixing,
                             i1.weight_special + i2.weight_special, i1.dist_cutoff + i2.dist_cutoff)
end

Base.:+(t::NamedTuple, i::DoubleExponential) = DoubleExponential(
    t.α + i.α,
    t.β + i.β,
    i.σ_mixing,
    i.ϵ_mixing,
    t.weight_special + i.weight_special,
    i.dist_cutoff
)

Base.:*(f::Float32, i::DoubleExponential) = DoubleExponential(
    f * i.α,
    f * i.β,
    i.σ_mixing,
    i.ϵ_mixing,
    f * i.weight_special,
    i.dist_cutoff
)

Molly.use_neighbors(inter::DoubleExponential) = true

function ChainRulesCore.rrule(TY::Type{<:DoubleExponential}, vs...)
    Y = TY(vs...)
    function DoubleExponential_pullback(Ȳ)
        return NoTangent(), Ȳ.α, Ȳ.β, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, DoubleExponential_pullback
end

##### Buffered 14, 7

struct Buffered147{T, S, E, W, D}
    δ::T
    γ::T
    σ_mixing::S
    ϵ_mixing::E 
    weight_special::W
    dist_cutoff::D
end

function Base.zero(i::Buffered147{T, S, E, W}) where {T, S, E, W}
    return Buffered147(zero(T), zero(T), i.σ_mixing, i.ϵ_mixing, zero(W), zero(T))
end

function Base.:+(i1::Buffered147, i2::Buffered147)
    return Buffered147(i1.δ + i2.δ, i1.γ + i2.γ, i1.σ_mixing, i1.ϵ_mixing,
                       i1.weight_special + i2.weight_special, i1.dist_cutoff)
end

Base.:+(t::NamedTuple, i::Buffered147) = Buffered147(
    t.δ + i.δ,
    t.γ + i.γ,
    i.σ_mixing,
    i.ϵ_mixing,
    t.weight_special + i.weight_special,
    i.dist_cutoff
)

Base.:*(f::Float32, i::Buffered147) = Buffered147(
    f * i.δ,
    f * i.γ,
    i.σ_mixing,
    i.ϵ_mixing,
    f * i.weight_special,
    i.dist_cutoff
)

Molly.use_neighbors(inter::Buffered147) = true

function ChainRulesCore.rrule(TY::Type{<:Buffered147}, vs...)
    Y = TY(vs...)
    function Buffered147_pullback(Ȳ)
        return NoTangent(), Ȳ.δ, Ȳ.γ, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buffered147_pullback
end

##### Buckingham

struct Buckingham{W, T}
    weight_special::W
    dist_cutoff::T
end 

Base.zero(::Buckingham{W, T}) where {W, T} = Buckingham(zero(W), zero(T))

Base.:+(i1::Buckingham, i2::Buckingham) = Buckingham(i1.weight_special + i2.weight_special,
                                                     i1.dist_cutoff + i2.dist_cutoff)

Base.:+(t::NamedTuple, i::Buckingham) = Buckingham(
    t.weight_special + i.weight_special,
    i.dist_cutoff
)

Base.:*(f::Float32, i::Buckingham) = Buckingham(
    f * i.weight_special,
    i.dist_cutoff
)

Molly.use_neighbors(::Buckingham) = true

function ChainRulesCore.rrule(TY::Type{<:Buckingham}, vs...)
    Y = TY(vs...)
    function Buckingham_pullback(Ȳ)
        return NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buckingham_pullback
end


##### Electrostatics

function ChainRulesCore.rrule(TY::Type{<:Coulomb}, vs...)
    Y = TY(vs...)
    function Coulomb_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.weight_special, Ȳ.coulomb_const
    end
    return Y, Coulomb_pullback
end

function ChainRulesCore.rrule(TY::Type{<:CoulombReactionField}, vs...)
    Y = TY(vs...)
    function CoulombReactionField_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.solvent_dielectric, NoTangent(),
                Ȳ.weight_special, Ȳ.coulomb_const
    end
    return Y, CoulombReactionField_pullback
end


Base.:+(::Nothing, ::Union{LennardJones, Mie, DoubleExponential, Buffered147, Buckingham}) = nothing