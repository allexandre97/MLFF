
##### Electrostatics

@inline function Molly.potential_energy(inter::Coulomb{C},
                                  dr,
                                  atom_i::GeneralAtom{T},
                                  atom_j::GeneralAtom{T},
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where {C, T}
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    pe = Molly.pe_cutoff(cutoff, inter, r, params)
    if special
        return T(pe * inter.weight_special)
    else
        return T(pe)
    end
end

@inline function Molly.potential_energy(inter::CoulombReactionField,
                                  dr,
                                  atom_i::GeneralAtom{T},
                                  atom_j::GeneralAtom{T},
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where T
    r = norm(dr)
    r2 = r*r
    if r > (inter.dist_cutoff)
        return ustrip(zero(dr[1])) * energy_units
    end

    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
        crf = (1 /  inter.dist_cutoff     ) * 0
    else
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) /
              (2 * inter.solvent_dielectric + 1))
        crf = (1 /  inter.dist_cutoff     ) * ((3 * inter.solvent_dielectric) /
              (2 * inter.solvent_dielectric + 1))
    end

    pe = T((ke * qi * qj) * (inv(r) + krf * r2 - crf))

    if special
        return T(pe * inter.weight_special)
    else
        return T(pe)
    end
end


##### Lennard Jones

@inline function vdw_potential(inter::LennardJones, atom::GeneralAtom{T}, r::Vector{T}) where {T}
    σ = atom.σ_lj
    ϵ = atom.ϵ_lj
    out = 4 * ϵ .* ((σ ./ r).^12 - (σ ./ r).^6)
    return T.(out)
end

@inline function Molly.potential_energy(inter::LennardJones,
                                  dr,
                                  atom_i::GeneralAtom{T},
                                  atom_j::GeneralAtom{T},
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where T
    

    σ = 0.5*(atom_i.σ_lj + atom_j.σ_lj) 
    ϵ = sqrt(atom_i.ϵ_lj * atom_j.ϵ_lj)

    cutoff = inter.cutoff
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    pe = Molly.pe_cutoff(cutoff, inter, r, params)

    if special
        return T(pe * inter.weight_special)
    else
        return T(pe)
    end
end

##### Mie

@inline function vdw_potential(inter::Mie, atom::GeneralAtom{T}, r::Vector{T}) where {T}
    σ = atom.σ_lj69
    ϵ = atom.ϵ_lj69
    m = inter.m
    n = inter.n
    C = (n / (n - m)) * (n / m) ^ (m / (n - m))
    out = C * ϵ .* ((σ ./ r).^n .- (σ ./ r).^m)
    return T.(out)
end

@inline function Molly.potential_energy(inter::Mie,
                                  dr,
                                  atom_i::GeneralAtom{T},
                                  atom_j::GeneralAtom{T},
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where T

    σ = 0.5*(atom_i.σ_lj69 + atom_j.σ_lj69) 
    ϵ = sqrt(atom_i.ϵ_lj69 * atom_j.ϵ_lj69)

    cutoff = inter.cutoff
    r = norm(dr)
    m = inter.m
    n = inter.n
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    pe = Molly.pe_cutoff(cutoff, inter, r, params)
    if special
        return T(pe * inter.weight_special)
    else
        return T(pe)
    end
end

##### Double Exponential

@inline function vdw_potential(inter::DoubleExponential, atom::GeneralAtom{T}, r::Vector{T}) where {T}
    α = inter.α
    β = inter.β
    σ = atom.σ_dexp
    ϵ = atom.ϵ_dexp
    rm = σ * T(2^(1 / 6))
    term1 = β * exp(α) / (α - β)
    term2 = α * exp(β) / (α - β)
    out = ϵ .* (term1 .* exp.(-α .* r ./ rm) .- term2 .* exp.(-β .* r ./ rm))
    return T.(out)
end

@inline function Molly.potential_energy(inter::DoubleExponential, 
                                        vec_ij, 
                                        atom_i::GeneralAtom{T}, 
                                        atom_j::GeneralAtom{T},
                                        energy_units, special, args...) where T


    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        
        σ = 0.5*(atom_i.σ_dexp + atom_j.σ_dexp) 
        ϵ = sqrt(atom_i.ϵ_dexp * atom_j.ϵ_dexp) 
        rm = σ * T(2^(1/6))
        α, β = inter.α, inter.β
        pe = ϵ * ((β * exp(α) / (α - β)) * exp(-α * r / rm) - (α * exp(β) / (α - β)) * exp(-β * r / rm))
        if special
            return T(pe)
        else
            return T(pe * inter.weight_special)
        end
    else
        return zero(T)
    end
end

##### Buffered 14, 7

@inline function vdw_potential(inter::Buffered147, atom::GeneralAtom{T}, r::Vector{T}) where {T}
    δ = inter.δ
    γ = inter.γ
    σ = atom.σ_buff
    ϵ = atom.ϵ_buff
    rm = σ * T(2^(1 / 6))
    
    out = ϵ .* ((1 + δ) ./ ((r ./ rm) .+ δ)).^7 .*
          ((1 + γ) ./ ((r ./ rm).^7 .+ γ) .- 2)
    return T.(out)
end

@inline function Molly.potential_energy(inter::Buffered147, 
                                        vec_ij, 
                                        atom_i::GeneralAtom{T}, 
                                        atom_j::GeneralAtom{T},
                                        energy_units, special, args...) where T
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        σ = 0.5*(atom_i.σ_buff + atom_j.σ_buff) 
        ϵ = sqrt(atom_i.ϵ_buff * atom_j.ϵ_buff) 
        δ, γ = inter.δ, inter.γ
        rm = σ * T(2^(1/6))
        r_div_rm = r / rm
        pe = ϵ * ((1 + δ) / (r_div_rm + δ))^7 * (((1 + γ) / (r_div_rm^7 + γ)) - 2)
        if special
            return T(pe)
        else
            return T(pe * inter.weight_special)
        end
    else
        return zero(T)
    end
end

##### Buckingham

@inline function vdw_potential(inter::Buckingham, atom::GeneralAtom{T}, r::Vector{T}) where {T}
    A = atom.A_buck
    B = atom.B_buck
    C = atom.C_buck
    out = A .* exp.(-B .* r) .- (C ./ r).^6
    return T.(out)
end

@inline function Molly.potential_energy(inter::Buckingham,
                                        vec_ij, 
                                        atom_i::GeneralAtom{T}, 
                                        atom_j::GeneralAtom{T},
                                        energy_units, special, args...) where T
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    if r <= inter.dist_cutoff
        A = (atom_i.A_buck + atom_j.A_buck) / 2
        B = (atom_i.B_buck + atom_j.B_buck) / 2
        C = (atom_i.C_buck + atom_j.C_buck) / 2
        pe = A * exp(-B * r) - (C / r)^6 # Modified this to hopefully make C a bit more well-behaved in training 
        if special
            return T(pe)
        else
            return T(pe * inter.weight_special)
        end
    else
        return zero(T)
    end
end


##### General Wrapper 

function pe_wrap(atoms, coords, velocities, boundary, pairwise_inters_nl,
                 sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    pe_vec = zeros(T, 1)
    
    pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
             sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    return pe_vec[1]
end

function pe_wrap!(pe_vec, atoms, coords, velocities, boundary, pairwise_inters_nl,
                  sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)

    
    pe = T(Molly.pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, NoUnits, length(atoms),
                                  (), pairwise_inters_nl, Val(T), Val(1), 0))

    pe += T(Molly.specific_pe(atoms, coords, velocities, boundary, NoUnits, (),
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, Val(T), 0))

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


function Molly.potential_energy(inter::NamedTuple, dr, a1::GeneralAtom{T}, a2::GeneralAtom{T},
                                energy_units, special, x1, x2, boundary, v1, v2, step_n) where T
    pe_total = zero(T)

    #This stupid unroll is needed so Enzyme does not throw a tantrum
    w1 = inter.weights[1]
    pe_total += w1 == zero(T) ? zero(T) : w1 * Molly.potential_energy(inter.inters[1], dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)
    
    w2 = inter.weights[2]
    pe_total += w2 == zero(T) ? zero(T) : w2 * Molly.potential_energy(inter.inters[2], dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w3 = inter.weights[3]
    pe_total += w3 == zero(T) ? zero(T) : w3 * Molly.potential_energy(inter.inters[3], dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w4 = inter.weights[4]
    pe_total += w4 == zero(T) ? zero(T) : w4 * Molly.potential_energy(inter.inters[4], dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    w5 = inter.weights[5]
    pe_total += w5 == zero(T) ? zero(T) : w5 * Molly.potential_energy(inter.inters[5], dr, a1, a2, energy_units, special,
                                               x1, x2, boundary, v1, v2, step_n)

    return pe_total
end

function calc_mean_U_gas(epoch_n, mol_id, feat_df, training_sim_dir, temp, models...)

    frame_is = ignore_derivatives() do
        shuffle(COND_SIM_FRAMES)[1:MODEL_PARAMS["training"]["enth_vap_gas_n_samples"]]
    end
    pe_sum = zero(T)
    n = 1
    for frame_i in frame_is

        coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
        
        _,
        _, pe, _, _,
        _, _, 
        _, _ = mol_to_preds(epoch_n, mol_id, feat_df, coords, boundary_inf, models...)
        
        pe_sum += pe
        n+=1
    end

    return pe_sum / MODEL_PARAMS["training"]["enth_vap_gas_n_samples"]
end

function pe_from_snapshot(
    epoch_n::Int,
    mol_id::String,
    args...
)

    sys, partial_charges, func_probs, weights_vdw, torsion_size, elements, mol_inds = mol_to_system(epoch_n, mol_id, args...)

    neighbors = ignore_derivatives() do
        return find_neighbors(sys; n_threads = 1)
    end

    # Get interaction lists separate depending on the number of atoms involves
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                        sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)

    return potential

end