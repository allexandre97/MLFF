import Base: +
import Base: *

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

Molly.use_neighbors(inter::DoubleExponential) = true

@inline function Molly.potential_energy(inter::DoubleExponential, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...)
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        σ = inter.σ_mixing(atom_i, atom_j)
        ϵ = inter.ϵ_mixing(atom_i, atom_j)
        rm = σ * T(2^(1/6))
        α, β = inter.α, inter.β
        pe = ϵ * ((β * exp(α) / (α - β)) * exp(-α * r / rm) - (α * exp(β) / (α - β)) * exp(-β * r / rm))
        if special
            return pe
        else
            return pe * inter.weight_special
        end
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::DoubleExponential, vec_ij, atom_i, atom_j, force_units,
                             special, args...)
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        σ = inter.σ_mixing(atom_i, atom_j)
        ϵ = inter.ϵ_mixing(atom_i, atom_j)
        rm = σ * T(2^(1/6))
        α, β = inter.α, inter.β
        f = ϵ * (α * (β * exp(α) / (α - β)) * exp(-α * r / rm) - β * (α * exp(β) / (α - β)) * exp(-β * r / rm)) / rm
        fdr = f * normalize(vec_ij)
        if special
            return fdr
        else
            return fdr * inter.weight_special
        end
    else
        return zero(SVector{3, T})
    end
end

struct Buffered147{T, S, E, W}
    δ::T
    γ::T
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
    dist_cutoff::T
end

function Base.zero(i::Buffered147{T, S, E, W}) where {T, S, E, W}
    return Buffered147(zero(T), zero(T), i.σ_mixing, i.ϵ_mixing, zero(W), zero(T))
end

function Base.:+(i1::Buffered147, i2::Buffered147)
    return Buffered147(i1.δ + i2.δ, i1.γ + i2.γ, i1.σ_mixing, i1.ϵ_mixing,
                       i1.weight_special + i2.weight_special, i1.dist_cutoff + i2.dist_cutoff)
end

Molly.use_neighbors(inter::Buffered147) = true

@inline function Molly.potential_energy(inter::Buffered147, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...)
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        σ = inter.σ_mixing(atom_i, atom_j)
        ϵ = inter.ϵ_mixing(atom_i, atom_j)
        δ, γ = inter.δ, inter.γ
        rm = σ * T(2^(1/6))
        r_div_rm = r / rm
        pe = ϵ * ((1 + δ) / (r_div_rm + δ))^7 * (((1 + γ) / (r_div_rm^7 + γ)) - 2)
        if special
            return pe
        else
            return pe * inter.weight_special
        end
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::Buffered147, vec_ij, atom_i, atom_j, force_units,
                             special, args...)
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        σ = inter.σ_mixing(atom_i, atom_j)
        ϵ = inter.ϵ_mixing(atom_i, atom_j)
        δ, γ = inter.δ, inter.γ
        rm = σ * T(2^(1/6))
        r_div_rm = r / rm
        r_div_rm_6 = r_div_rm^6
        r_div_rm_7 = r_div_rm_6 * r_div_rm
        γ7_term = (1 + γ) / (r_div_rm_7 + γ)
        f = (7ϵ / rm) * ((1 + δ) / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
        fdr = f * normalize(vec_ij)
        if special
            return fdr
        else
            return fdr * inter.weight_special
        end
    else
        return zero(SVector{3, T})
    end
end

struct BuckinghamAtom{T}
    index::Int
    atom_type::T
    mass::T
    charge::T
    A::T
    B::T
    C::T
end

function Base.zero(::BuckinghamAtom{T}) where T
    z = zero(T)
    return BuckinghamAtom(0, z, z, z, z, z, z)
end

function Base.:+(a1::BuckinghamAtom{T}, a2::BuckinghamAtom{T}) where T
    return BuckinghamAtom(0, zero(T), a1.mass + a2.mass, a1.charge + a2.charge,
                          a1.A + a2.A, a1.B + a2.B, a1.C + a2.C)
end

struct Buckingham{W, T}
    weight_special::W
    dist_cutoff::T
end

Base.zero(::Buckingham{W, T}) where {W, T} = Buckingham(zero(W), zero(T))

Base.:+(i1::Buckingham, i2::Buckingham) = Buckingham(i1.weight_special + i2.weight_special,
                                                     i1.dist_cutoff + i2.dist_cutoff)

Molly.use_neighbors(::Buckingham) = true

@inline function Molly.potential_energy(inter::Buckingham, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...)
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    if r <= inter.dist_cutoff
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        pe = A * exp(-B * r) - (C / r)^6 # Modified this to hopefully make C a bit more well-behaved in training 
        if special
            return pe
        else
            return pe * inter.weight_special
        end
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::Buckingham, vec_ij, atom_i, atom_j, force_units,
                             special, args...)
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    if r <= inter.dist_cutoff
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        fdr = (A * B * exp(-B * r) / r - 6 * C / r2^4) * vec_ij
        if special
            return fdr
        else
            return fdr * inter.weight_special
        end
    else
        return zero(SVector{3, T})
    end
end

struct NNAtom{T}
    index::Int
    atom_type::Int
    mass::T
    charge::T
    params::Vector{T}
end

function Base.zero(a::NNAtom{T}) where T
    return NNAtom(0, 0, zero(T), zero(T), zero(a.params))
end

function Base.:+(a1::NNAtom, a2::NNAtom)
    return NNAtom(0, 0, a1.mass + a2.mass, a1.charge + a2.charge, a1.params .+ a2.params)
end

struct NNPairwise{T, W}
    params::Vector{T}
    weight_special::W
    dist_cutoff::T
end

Base.zero(i::NNPairwise{T, W}) where {T, W} = NNPairwise(zero(i.params), zero(W), zero(T))

Base.:+(i1::NNPairwise, i2::NNPairwise) = NNPairwise(
    i1.params .+ i2.params,
    i1.weight_special + i2.weight_special,
    i1.dist_cutoff + i2.dist_cutoff,
)

Molly.use_neighbors(::NNPairwise) = true

function mvp!(o, a, b)
    @assert size(a, 2) == length(b)
    # Using @inbounds here gave Enzyme error
    for i in axes(a, 1)
        s = zero(T)
        for j in axes(a, 2)
            s += a[i, j] * b[j]
        end
        o[i] = s
    end
    return o
end

@inline function Molly.potential_energy(inter::NNPairwise, vec_ij, atom_i, atom_j,
                                        energy_units, special, args...)
    r = sqrt(sum(abs2, vec_ij))
    @views if r <= inter.dist_cutoff
        inv_r = inv(r)
        # Using zeros and setting manually was faster but errored with loss function gradient
        inputs = vcat(atom_i.charge, atom_i.params, atom_j.charge, atom_j.params, inv_r)
        w1 = reshape(inter.params[1:(end-2*dim_hidden_pairwise)], dim_hidden_pairwise, length(inputs))
        b1 = inter.params[(end-2*dim_hidden_pairwise+1):(end-dim_hidden_pairwise)]
        w2 = reshape(inter.params[(end-dim_hidden_pairwise+1):end], 1, dim_hidden_pairwise)
        hl, out = zeros(T, size(w1, 1)), zeros(T, 1)
        mvp!(hl, w1, inputs)
        # Changing this activation also requires changing the force algorithm
        hl .= relu.(hl .+ b1)
        mvp!(out, w2, hl)
        pe_a = only(out)
        inputs[1] = atom_j.charge
        inputs[2:(nn_dim_atom + 1)] .= atom_j.params
        inputs[nn_dim_atom + 2] = atom_i.charge
        inputs[(nn_dim_atom + 3):(nn_dim_atom + 6)] .= atom_i.params
        mvp!(hl, w1, inputs)
        hl .= relu.(hl .+ b1)
        mvp!(out, w2, hl)
        pe_b = only(out)
        pe = (pe_a + pe_b) / 2
        if special
            return pe
        else
            return pe * inter.weight_special
        end
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::NNPairwise, vec_ij, atom_i, atom_j, force_units,
                             special, args...)
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    @views if r <= inter.dist_cutoff
        inv_r = inv(r)
        inputs = vcat(atom_i.charge, atom_i.params, atom_j.charge, atom_j.params, inv_r)
        w1 = reshape(inter.params[1:(end-2*dim_hidden_pairwise)], dim_hidden_pairwise, length(inputs))
        b1 = inter.params[(end-2*dim_hidden_pairwise+1):(end-dim_hidden_pairwise)]
        w2_flat = inter.params[(end-dim_hidden_pairwise+1):end]
        hl_unact = zeros(T, size(w1, 1))
        mvp!(hl_unact, w1, inputs)
        hl_unact .= hl_unact .+ b1
        dE_dinvr = zero(T)
        @inbounds for i in 1:dim_hidden_pairwise
            if hl_unact[i] > zero(T)
                dE_dinvr += w2_flat[i] * w1[i, end]
            end
        end
        inputs[1] = atom_j.charge
        inputs[2:(nn_dim_atom + 1)] .= atom_j.params
        inputs[nn_dim_atom + 2] = atom_i.charge
        inputs[(nn_dim_atom + 3):(nn_dim_atom + 6)] .= atom_i.params
        mvp!(hl_unact, w1, inputs)
        hl_unact .= hl_unact .+ b1
        @inbounds for i in 1:dim_hidden_pairwise
            if hl_unact[i] > zero(T)
                dE_dinvr += w2_flat[i] * w1[i, end]
            end
        end
        dE_r = dE_dinvr * -inv(r2) / 2
        fdr = (-dE_r / r) * vec_ij
        if special
            return fdr
        else
            return fdr * inter.weight_special
        end
    else
        return zero(SVector{3, T})
    end
end

########## CHAINRULE WRAPPERS FOR ZYGOTE ########## Is this really needed? Where?

function ChainRulesCore.rrule(TY::Type{<:Atom}, vs...)
    Y = TY(vs...)
    function Atom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.σ, Ȳ.ϵ
    end
    return Y, Atom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:BuckinghamAtom}, vs...)
    Y = TY(vs...)
    function BuckinghamAtom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.A, Ȳ.B, Ȳ.C
    end
    return Y, BuckinghamAtom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:NNAtom}, vs...)
    Y = TY(vs...)
    function NNAtom_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.params
    end
    return Y, NNAtom_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList2Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList2Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList2Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList3Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList3Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList3Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList4Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList4Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters,
               NoTangent()
    end
    return Y, InteractionList4Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicBond}, vs...)
    Y = TY(vs...)
    function HarmonicBond_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.r0
    end
    return Y, HarmonicBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:MorseBond}, vs...)
    Y = TY(vs...)
    function MorseBond_pullback(Ȳ)
        return NoTangent(), Ȳ.D, Ȳ.a, Ȳ.r0
    end
    return Y, MorseBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicAngle}, vs...)
    Y = TY(vs...)
    function HarmonicAngle_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.θ0
    end
    return Y, HarmonicAngle_pullback
end

function ChainRulesCore.rrule(TY::Type{<:UreyBradley}, vs...)
    Y = TY(vs...)
    function UreyBradley_pullback(Ȳ)
        return NoTangent(), Ȳ.kangle, Ȳ.θ0,  Ȳ.kbond, Ȳ.r0
    end
    return Y, UreyBradley_pullback
end

function ChainRulesCore.rrule(TY::Type{<:PeriodicTorsion}, vs...)
    Y = TY(vs...)
    function PeriodicTorsion_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.phases, Ȳ.ks, NoTangent()
    end
    return Y, PeriodicTorsion_pullback
end

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

function ChainRulesCore.rrule(TY::Type{<:LennardJones}, vs...)
    Y = TY(vs...)
    function LennardJones_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), Ȳ.weight_special
    end
    return Y, LennardJones_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Mie}, vs...)
    Y = TY(vs...)
    function Mie_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Mie_pullback
end

function ChainRulesCore.rrule(TY::Type{<:DoubleExponential}, vs...)
    Y = TY(vs...)
    function DoubleExponential_pullback(Ȳ)
        return NoTangent(), Ȳ.α, Ȳ.β, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, DoubleExponential_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Buffered147}, vs...)
    Y = TY(vs...)
    function Buffered147_pullback(Ȳ)
        return NoTangent(), Ȳ.δ, Ȳ.γ, NoTangent(), NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buffered147_pullback
end

function ChainRulesCore.rrule(TY::Type{<:Buckingham}, vs...)
    Y = TY(vs...)
    function Buckingham_pullback(Ȳ)
        return NoTangent(), Ȳ.weight_special, NoTangent()
    end
    return Y, Buckingham_pullback
end

function ChainRulesCore.rrule(TY::Type{<:NNPairwise}, vs...)
    Y = TY(vs...)
    function NNPairwise_pullback(Ȳ)
        return NoTangent(), Ȳ.params, Ȳ.weight_special, NoTangent()
    end
    return Y, NNPairwise_pullback
end

##########

########## WRAPPERS TO CALCULATE POTENTIAL ENERGIES ##########

function pe_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl,
                 sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    pe_vec = zeros(T, 1)

    pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
             sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    return pe_vec[1]
end

function pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
                  sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)

    
    pe = Molly.pairwise_pe(atoms, coords, velocities, boundary, neighbors, NoUnits, length(atoms),
                           (), pairwise_inters_nl, T, 0)
    pe += Molly.specific_pe(atoms, coords, velocities, boundary, NoUnits, (),
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, T, 0)
    pe_vec[1] = T(pe)
    return pe_vec
end

function ChainRulesCore.rrule(::typeof(pe_wrap), atoms, coords, velocities, boundary,
                              pairwise_inters_nl, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                              neighbors)
    Y = pe_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl, sils_2_atoms,
                sils_3_atoms, sils_4_atoms, neighbors)
    function pe_wrap_pullback(d_pe)
        d_atoms = zero.(atoms)
        d_coords = zero(coords)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        if vdw_functional_form == "nn"
            d_pairwise_inters_nl = zero.(pairwise_inters_nl)
            pair_enz = Enzyme.Duplicated(pairwise_inters_nl, d_pairwise_inters_nl)
        elseif length(pairwise_inters_nl) > 0
            pair_enz = Enzyme.Active(pairwise_inters_nl)
        else
            pair_enz = Enzyme.Const(pairwise_inters_nl)
        end
        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            pe_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(zeros(T, 1), [T(d_pe)]),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            pair_enz,
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Enzyme.Const(neighbors),
        )[1]
        pair_grad = (vdw_functional_form == "nn" ? d_pairwise_inters_nl : grads[6])
        return NoTangent(), d_atoms, d_coords, NoTangent(), NoTangent(), pair_grad,
               d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms, NoTangent()
    end
    return Y, pe_wrap_pullback
end

########## A GENERAL ATOM STRUCT TO ACCOMODATE THE DIFFERENT FUNCTIONAL FORMS ##########

struct GeneralAtom{T, A}
    index::Int
    type::Int
    mass::T
    charge::T
    atoms::A
end

function +(a::GeneralAtom{T, A}, b::GeneralAtom{T, A}) where {T, A}
    GeneralAtom(
        a.index,
        a.type,
        a.mass   + b.mass,
        a.charge + b.charge,
        a.atoms .+ b.atoms
    )
end

function *(i::Float32, a::Atom)
    Atom(
        a.index,
        a.atom_type,
        a.mass * i,
        a.charge * i,
        a.σ * i,
        a.ϵ * i
    )
end

function *(i::Float32, a::BuckinghamAtom)
    BuckinghamAtom(
        a.index,
        a.atom_type,
        a.mass * i,
        a.charge * i,
        a.A * i,
        a.B * i,
        a.C * i
    )
end

function *(i::Float32, a::GeneralAtom{T, A}) where {T, A}
    GeneralAtom(
        a.index,
        a.type,
        a.mass * i,
        a.charge * i,
        i .* a.atoms
    )
end

#Base.zero(::Type{GeneralAtom{T}}) = GeneralAtom{T}(0, 0, zero(T), zero(T), zero.())
Base.zero(x::GeneralAtom{T, A}) where {T, A} = GeneralAtom(0, 0, zero(T), zero(T), zero.(x.atoms))

function ChainRulesCore.rrule(TY::Type{<:GeneralAtom}, vs...)
    Y = TY(vs...)
    function pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.mass, Ȳ.charge, Ȳ.atoms
    end
    return Y, pullback
end


struct GroupInter{I}
    inters::I
    weights::SVector{5, T}
end

function +(a::GroupInter{I}, b::GroupInter{I}) where {I}
    GroupInter(
        a.inters .+ b.inters,
        a.weights + b.weights
    )
end

Base.zero(x::GroupInter{I}) where I = GroupInter(zero.(x.inters), zero.(x.weights))

Zygote.accum(a::NamedTuple{(:inters, :weights), <:Any}, b::GroupInter) = GroupInter(
    a.inters .+ b.inters,
    a.weights + b.weights
)

Base.:+(::Nothing, ::Union{LennardJones, Mie, DoubleExponential, Buffered147, Buckingham}) = nothing

function *(n::Float32, i::LennardJones)
    LennardJones(
        i.cutoff,
        i.use_neighbors,
        i.shortcut,
        i.σ_mixing,
        i.ϵ_mixing,
        n * i.weight_special
    )
end

function *(n::Float32, i::Mie)
    Mie(
        Int(n * i.m),
        Int(n * i.m),
        i.cutoff,
        i.use_neighbors,
        i.shortcut,
        i.σ_mixing,
        i.ϵ_mixing,
        i.weight_special,
        i.mn_fac
    )
end

function *(n::Float32, i::DoubleExponential)
    DoubleExponential(
        n * i.α,
        n * i.β,
        i.σ_mixing,
        i.ϵ_mixing,
        i.weight_special,
        i.dist_cutoff
    )
end

function *(n::Float32, i::Buffered147)
    Buffered147(
        n * i.δ,
        n * i.γ,
        i.σ_mixing,
        i.ϵ_mixing,
        i.weight_special,
        i.dist_cutoff
    )
end

function *(n::Float32, i::Buckingham)
    Buckingham(
        i.weight_special,
        i.dist_cutoff
    )
end

function ChainRulesCore.rrule(TY::Type{<:GroupInter}, vs...)
    Y = TY(vs...)
    function pullback(Ȳ)
        return NoTangent(), Ȳ.inters, Ȳ.weights
    end
    return Y, pullback
end

function Molly.force(inter::GroupInter{<:Tuple}, dr, a1::GeneralAtom, a2::GeneralAtom,
                     force_units, special, x1, x2, boundary, v1, v2, step_n)

    f_total = zero(SVector{3, T})

    for (idx, sub_inter) in enumerate(inter.inters)
        a1f = a1.atoms[idx]
        a2f = a2.atoms[idx]
        w = inter.weights[idx]

        f_total += w * Molly.force(sub_inter, dr, a1f, a2f, force_units, special, x1, x2, boundary, v1, v2, step_n)
    end

    return f_total
end

function Molly.potential_energy(inter::GroupInter, dr, a1::GeneralAtom, a2::GeneralAtom,
                                energy_units, special, x1, x2, boundary, v1, v2, step_n)
    pe_total = zero(T)

    #This stupid unroll is needed so Enzyme does not throw a tantrum
    w1 = inter.weights[1]
    a1f = a1.atoms[1]
    a2f = a2.atoms[1]
    pe_total += w1 + Molly.potential_energy(inter.inters[1], dr, a1f, a2f, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)
    
    w2 = inter.weights[2]
    a1f = a1.atoms[2]
    a2f = a2.atoms[2]
    pe_total += w2 + Molly.potential_energy(inter.inters[2], dr, a1f, a2f, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w3 = inter.weights[3]
    a1f = a1.atoms[3]
    a2f = a2.atoms[3]
    pe_total += w3 + Molly.potential_energy(inter.inters[3], dr, a1f, a2f, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w4 = inter.weights[4]
    a1f = a1.atoms[4]
    a2f = a2.atoms[4]
    pe_total += w4 + Molly.potential_energy(inter.inters[4], dr, a1f, a2f, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w5 = inter.weights[5]
    a1f = a1.atoms[5]
    a2f = a2.atoms[5]
    pe_total += w5 + Molly.potential_energy(inter.inters[5], dr, a1f, a2f, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    #= for (idx, sub_inter) in enumerate(inter.inters[i:i+1])
        #@show sub_inter
        w = inter.weights[idx+(i-1)]
        pe_total += w * Molly.potential_energy(sub_inter, dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)
    end
    #@show pe_total =#

    #= pe_total = sum(eachindex(inter.inters)) do idx
        sub_inter = inter.inters[idx]
        w = inter.weights[idx]
        return w * Molly.potential_energy(sub_inter, dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)
    end =#
    
    return pe_total
end

########## WRAPPERS TO CALCULATE FORCES ##########

function forces_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl,
                     sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    fs_nounits = zero(coords)
    forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, pairwise_inters_nl,
                 sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    return fs_nounits
end

function forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, pairwise_inters_nl,
                      sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    Molly.pairwise_forces!(fs_nounits, atoms, coords, velocities, boundary, neighbors, NoUnits,
                           length(atoms), (), pairwise_inters_nl, 0)
    Molly.specific_forces!(fs_nounits, atoms, coords, velocities, boundary, NoUnits, (),
                           sils_2_atoms, sils_3_atoms, sils_4_atoms, 0)
    return fs_nounits
end

duplicated_if_present(x, dx) = (length(x) > 0 ? Enzyme.Duplicated(x, dx) : Enzyme.Const(x))

function ChainRulesCore.rrule(::typeof(forces_wrap), atoms, coords, velocities, boundary,
                              pairwise_inters_nl, sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    Y = forces_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl, sils_2_atoms,
                    sils_3_atoms, sils_4_atoms, neighbors)
    function forces_wrap_pullback(d_fs_nounits)
        fs_nounits = zero(coords)
        d_atoms = zero.(atoms)
        d_coords = zero(coords)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        if vdw_functional_form == "nn"
            # Active fails here
            # Temp gives zero grad for weight_special, though that is set to 1 anyway
            d_pairwise_inters_nl = zero.(pairwise_inters_nl)
            pair_enz = Enzyme.Duplicated(pairwise_inters_nl, d_pairwise_inters_nl)
        elseif length(pairwise_inters_nl) > 0
            # Active required to get non-zero grads for weight_special etc.
            pair_enz = Enzyme.Active(pairwise_inters_nl)
        else
            pair_enz = Enzyme.Const(pairwise_inters_nl)
        end
        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            forces_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(fs_nounits, d_fs_nounits),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            pair_enz,
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Enzyme.Const(neighbors),
        )[1]
        pair_grad = (vdw_functional_form == "nn" ? d_pairwise_inters_nl : grads[6])

        return NoTangent(), d_atoms, d_coords, NoTangent(), NoTangent(), pair_grad,
               d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms, NoTangent()
    end
    return Y, forces_wrap_pullback
end