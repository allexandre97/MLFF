# See Kovacs2023 and Magdau2023
function split_forces(fs, coords, molecule_inds, element_inds)
    forces_intra, forces_inter = zero(fs), zero(fs)
    split_forces!(forces_intra, forces_inter, fs, coords, molecule_inds, element_inds)
    return forces_intra, forces_inter
end

function split_forces!(forces_intra, forces_inter, fs, coords, molecule_inds, element_inds)
    n_molecules = maximum(molecule_inds)
    if n_molecules == 1
        forces_inter .= (zero(SVector{3, T}),)
        forces_intra .= fs
    else
        mol_f_trans = zeros(SVector{3, T}, n_molecules)
        mol_masses = zeros(n_molecules)
        for (f, mi, el) in zip(fs, molecule_inds, element_inds)
            mol_f_trans[mi] += f
            mol_masses[mi] += ATOMIC_MASSES[el]
        end

        forces_trans = zero(fs)
        mol_coms = zeros(SVector{3, T}, n_molecules)
        for (ai, (mi, el, coord)) in enumerate(zip(molecule_inds, element_inds, coords))
            atom_mass_frac = ATOMIC_MASSES[el] / mol_masses[mi]
            forces_trans[ai] = atom_mass_frac * mol_f_trans[mi]
            mol_coms[mi] += atom_mass_frac * coord
        end

        mol_torques = zeros(SVector{3, T}, n_molecules)
        for (f, mi, coord) in zip(fs, molecule_inds, coords)
            mol_torques[mi] += f × (coord - mol_coms[mi])
        end

        mol_Is = zeros(n_molecules, 3, 3)
        for (mi, el, coord) in zip(molecule_inds, element_inds, coords)
            atom_mass = ATOMIC_MASSES[el]
            coord_com = coord - mol_coms[mi]
            sqdist_com = sum(abs2, coord_com)
            for i in 1:3, j in 1:3
                if i == j
                    mol_Is[mi, i, j] += atom_mass * (sqdist_com - coord_com[i]^2)
                else
                    mol_Is[mi, i, j] += atom_mass * (-coord_com[i] * coord_com[j])
                end
            end
        end

        mol_inv_Is = zeros(SMatrix{3, 3, T}, n_molecules)
        for mi in 1:n_molecules
            if all(iszero, mol_Is[mi, :, :])
                mol_inv_Is[mi] = SMatrix{3, 3, T}(I)
            else
                mol_inv_Is[mi] = inv(SMatrix{3, 3, T}(mol_Is[mi, :, :]))
            end
        end

        forces_rot = zero(fs)
        for (ai, (mi, el, coord)) in enumerate(zip(molecule_inds, element_inds, coords))
            I_T_mvp = mol_inv_Is[mi] * mol_torques[mi]
            forces_rot[ai] = ATOMIC_MASSES[el] * (coord - mol_coms[mi]) × I_T_mvp
        end

        forces_inter .= forces_trans .+ forces_rot
        forces_intra .= fs .- forces_inter
    end
    return forces_intra, forces_inter
end

function ChainRulesCore.rrule(::typeof(split_forces), fs, coords, molecule_inds, element_inds)
    Y = split_forces(fs, coords, molecule_inds, element_inds)
    function split_forces_pullback(d_fs_both)
        d_coords = zero(coords)
        d_fs = zero(fs)

        df1 = typeof(d_fs_both[1]) == ZeroTangent ? zero(fs) : d_fs_both[1]
        df2 = typeof(d_fs_both[2]) == ZeroTangent ? zero(fs) : d_fs_both[2]

        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            split_forces!,
            Enzyme.Const,
            Enzyme.Duplicated(zero(fs), df1),
            Enzyme.Duplicated(zero(fs), df2),
            Enzyme.Duplicated(fs, d_fs),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(molecule_inds),
            Enzyme.Const(element_inds),
        )[1]
        return NoTangent(), d_fs, d_coords, NoTangent(), NoTangent()
    end
    return Y, split_forces_pullback
end

calc_RT(temp) = ustrip(u"kJ/mol", T(Unitful.R) * temp * u"K")
@non_differentiable calc_RT(args...)

function enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    # See https://docs.openforcefield.org/projects/evaluator/en/stable/properties/properties.html
    RT = calc_RT(temp)
    ΔH_vap = mean_U_gas - snapshot_U_liquid / n_molecules + RT
    return ΔH_vap
end

@inline function Molly.force(inter::Coulomb{C},
                       dr,
                       atom_i::GeneralAtom{T},
                       atom_j::GeneralAtom{T},
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where {C, T}
    r2 = sum(abs2, dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    f = Molly.force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return T.(f * dr * inter.weight_special)
    else
        return T.(f * dr)
    end
end

@inline function Molly.force(inter::LennardJones,
                       dr,
                       atom_i::GeneralAtom{T},
                       atom_j::GeneralAtom{T},
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T

    r = norm(dr)
    if r <= inter.cutoff.dist_cutoff

        σ = 0.5f0*(atom_i.σ_lj + atom_j.σ_lj)
        ϵ = sqrt(atom_i.ϵ_lj * atom_j.ϵ_lj)

        cutoff = inter.cutoff
        r2 = sum(abs2, dr)
        σ2 = σ^2
        params = (σ2, ϵ)

        f = Molly.force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
        if special
            return T.(f * dr * inter.weight_special)
        else
            return T.(f * dr)
        end
    else 
        return zero(SVector{3, T})
    end
end

@inline function Molly.force(inter::Mie,
                       dr,
                       atom_i::GeneralAtom{T},
                       atom_j::GeneralAtom{T},
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T

    r = norm(dr)
    if r <= inter.cutoff.dist_cutoff

        σ = 0.5f0*(atom_i.σ_lj69 + atom_j.σ_lj69)
        ϵ = sqrt(atom_i.ϵ_lj69 * atom_j.ϵ_lj69)

        cutoff = inter.cutoff
        r2 = sum(abs2, dr)
        r = √r2
        m = inter.m
        n = inter.n
        const_mn = inter.mn_fac * ϵ
        σ_r = σ / r
        params = (m, n, σ_r, const_mn)

        f = Molly.force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
        if special
            return T.(f * dr * inter.weight_special)
        else
            return T.(f * dr)
        end
    else
        return zero(SVector{3, T})
    end
end

@inline function Molly.force(inter::DoubleExponential,
                             vec_ij, 
                             atom_i::GeneralAtom{T}, 
                             atom_j::GeneralAtom{T}, 
                             force_units,
                             special, args...) where T
    r = norm(vec_ij)
    if r <= inter.dist_cutoff
        
        σ = 0.5*(atom_i.σ_dexp + atom_j.σ_dexp) 
        ϵ = sqrt(atom_i.ϵ_dexp * atom_j.ϵ_dexp) 

        rm = σ * T(2^(1/6))
        α, β = inter.α, inter.β
        f = ϵ * (α * (β * exp(α) / (α - β)) * exp(-α * r / rm) - β * (α * exp(β) / (α - β)) * exp(-β * r / rm)) / rm
        fdr = f * normalize(vec_ij)
        if special
            return T.(fdr)
        else
            return T.(fdr * inter.weight_special)
        end
    else
        return zero(SVector{3, T})
    end
end

#####

@inline function Molly.force(inter::Buffered147,
                             vec_ij, 
                             atom_i::GeneralAtom{T}, 
                             atom_j::GeneralAtom{T}, 
                             force_units,
                             special, args...) where T
    r = norm(vec_ij)
    if r <= inter.dist_cutoff

        σ = 0.5*(atom_i.σ_buff + atom_j.σ_buff) 
        ϵ = sqrt(atom_i.ϵ_buff * atom_j.ϵ_buff) 
        δ, γ = inter.δ, inter.γ
        rm = σ * T(2^(1/6))
        r_div_rm = r / rm
        r_div_rm_6 = r_div_rm^6
        r_div_rm_7 = r_div_rm_6 * r_div_rm
        γ7_term = (1 + γ) / (r_div_rm_7 + γ)
        f = (7ϵ / rm) * ((1 + δ) / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
        fdr = f * normalize(vec_ij)
        if special
            return T.(fdr)
        else
            return T.(fdr * inter.weight_special)
        end
    else
        return zero(SVector{3, T})
    end
end

#####

@inline function Molly.force(inter::Buckingham,
                             vec_ij, 
                             atom_i::GeneralAtom{T}, 
                             atom_j::GeneralAtom{T}, 
                             force_units,
                             special, args...)
    r2 = sum(abs2, vec_ij)
    r = sqrt(r2)
    if r <= inter.dist_cutoff
        A = (atom_i.A_buck + atom_j.A_buck) / 2
        B = (atom_i.B_buck + atom_j.B_buck) / 2
        C = (atom_i.C_buck + atom_j.C_buck) / 2
        fdr = (A * B * exp(-B * r) - 6 * (C^6) / r^7) * normalize(vec_ij) # Modified to match how we predict C now
        if special
            return T.(fdr)
        else
            return T.(fdr * inter.weight_special)
        end
    else
        return zero(SVector{3, T})
    end
end

function Molly.force(inter::NamedTuple,
                     dr, 
                     a1::GeneralAtom{T},
                     a2::GeneralAtom{T},
                     force_units, special, x1, x2, boundary, v1, v2, step_n) where T

    f_total = zero(SVector{3, T})
    for (idx, sub_inter) in enumerate(inter.inters)
        w = inter.weights[idx]

        f_total += T.(w * Molly.force(sub_inter, dr, a1, a2, force_units, special, x1, x2, boundary, v1, v2, step_n))
    end

    return f_total
end

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

Base.:+(::Nothing, ::DistanceCutoff{T, T, T}) where T = nothing
Base.:+(::Nothing, ::Float32) = zero(T)