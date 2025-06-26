using Molly

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
    atom_type::Int
    mass::T
    charge::T
    A::T
    B::T
    C::T
end

function Base.zero(::BuckinghamAtom{T}) where T
    z = zero(T)
    return BuckinghamAtom(0, 0, z, z, z, z, z)
end

function Base.:+(a1::BuckinghamAtom, a2::BuckinghamAtom)
    return BuckinghamAtom(0, 0, a1.mass + a2.mass, a1.charge + a2.charge,
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
        pe = A * exp(-B * r) - C / r2^3
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