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



@inline function Molly.force(inter::Coulomb{C},
                       dr,
                       atom_i::GeneralAtom{T},
                       atom_j::GeneralAtom{T},
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where {C, T}
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    f = Molly.force_cutoff(cutoff, inter, r, params)
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
        σ2 = σ^2
        params = (σ2, ϵ)

        f = Molly.force_cutoff(cutoff, inter, r, params)
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
        m = inter.m
        n = inter.n
        const_mn = inter.mn_fac * ϵ
        σ_r = σ / r
        params = (m, n, σ_r, const_mn)

        f = Molly.force_cutoff(cutoff, inter, r, params)
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
                             special, args...) where T
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

@inline function Molly.force(inter::NamedTuple,
                     dr, 
                     a1::GeneralAtom{T},
                     a2::GeneralAtom{T},
                     force_units, special, x1, x2, boundary, v1, v2, step_n) where T

    f_total = zero(SVector{3, T})
    for (idx, sub_inter) in enumerate(inter.inters)
        w = inter.weights[idx]

        f_total += w == zero(T) ? SVector{3, T}(0.0f0, 0.0f0, 0.0f0) : T.(w * Molly.force(sub_inter, dr, a1, a2, force_units, special, x1, x2, boundary, v1, v2, step_n))
    end

    return f_total
end

# --- pairwise forces wrappers using CompositeInter ---
@inline function pairwise_forces_wrap(atoms, coords, velocities, boundary,
                                      pairwise_inters, neighbors)
    fs_nounits = zero(coords)
    pairwise_forces_wrap!(fs_nounits, atoms, coords, velocities, boundary,
                          pairwise_inters, neighbors)
    fs_nounits
end

# Fast path when already wrapped
@inline function pairwise_forces_wrap!(fs_nounits, atoms, coords, velocities, boundary,
                                       pairwise_inters::PairwisePack, neighbors)
    Molly.pairwise_forces_loop!(fs_nounits, nothing, nothing, nothing, atoms, coords, velocities, boundary,
                           neighbors, NoUnits, length(atoms), (),
                           to_tuple(pairwise_inters), Val(1), Val(false), 0)
    fs_nounits
end

# --- rrule for the wrapper ---
function ChainRulesCore.rrule(::typeof(pairwise_forces_wrap),
                              atoms, coords, velocities, boundary,
                              pairwise_inters, neighbors)

    pack = pairwise_inters isa PairwisePack ? pairwise_inters : to_pack(pairwise_inters)
    Y = pairwise_forces_wrap(atoms, coords, velocities, boundary, pack, neighbors)

    function pairwise_forces_wrap_pullback(d_fs_nounits)
        fs_nounits = zero(coords)
        d_atoms  = zero.(atoms)
        d_coords = zero(coords)

        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            pairwise_forces_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(fs_nounits, d_fs_nounits),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            Enzyme.Active(pack),
            Enzyme.Const(neighbors)
        )[1]

        d_pack = grads[6]
        d_pairwise = if pairwise_inters isa PairwisePack
            d_pack
        else
            ((inters=(d_pack.lj, d_pack.mie, d_pack.dexp, d_pack.buff, d_pack.buck),
              weights=d_pack.weights),
             d_pack.coul)
        end

        # f, atoms, coords, velocities, boundary, pairwise_inters, neighbors
        return NoTangent(),
               d_atoms,
               d_coords,
               NoTangent(),      # velocities
               NoTangent(),      # boundary
               d_pairwise,
               NoTangent()       # neighbors
    end

    return Y, pairwise_forces_wrap_pullback
end

function specific_forces_wrap(atoms, coords, velocities, boundary, sils_2_atoms, sils_3_atoms, sils_4_atoms)
    fs_nounits = zero(coords)
    specific_forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, sils_2_atoms, sils_3_atoms, sils_4_atoms)
    return fs_nounits
end

function specific_forces_wrap!(fs_nounits, atoms, coords, velocities, boundary, sils_2_atoms, sils_3_atoms, sils_4_atoms)
    Molly.specific_forces!(fs_nounits, nothing, atoms, coords, velocities, boundary, NoUnits, (),
                           sils_2_atoms, sils_3_atoms, sils_4_atoms, Val(false), 0)
end

duplicated_if_present(x, dx) = (length(x) > 0 ? Enzyme.Duplicated(x, dx) : Enzyme.Const(x))

function ChainRulesCore.rrule(::typeof(specific_forces_wrap), atoms, coords, velocities, boundary,
                              sils_2_atoms, sils_3_atoms, sils_4_atoms)

    Y = specific_forces_wrap(atoms, coords, velocities, boundary, sils_2_atoms, sils_3_atoms, sils_4_atoms)

    function specific_forces_wrap_pullback(d_fs_nounits)

        fs_nounits = zero(coords)
        d_atoms = zero.(atoms)
        d_coords = zero(coords)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)

        grads = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            specific_forces_wrap!,
            Enzyme.Const,
            Enzyme.Duplicated(fs_nounits, d_fs_nounits),
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(coords, d_coords),
            Enzyme.Const(velocities),
            Enzyme.Const(boundary),
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms)
        )[1]

        return NoTangent(),
               d_atoms,
               d_coords,
               NoTangent(),
               NoTangent(),
               d_sils_2_atoms,
               d_sils_3_atoms,
               d_sils_4_atoms
    end
    return Y, specific_forces_wrap_pullback
end

# Generic pass-throughs
@inline Base.:+(::Nothing, y::T) where {T} = y
@inline Base.:+(x::T, ::Nothing) where {T} = x

# Optional: make DistanceCutoff explicit to avoid any ambiguity in that codepath
@inline Base.:+(::Nothing, y::DistanceCutoff{T1,T2}) where {T1,T2} = y
@inline Base.:+(x::DistanceCutoff{T1,T2}, ::Nothing) where {T1,T2} = x

# If you also hit Float32 explicitly in some sites, this is fine but not required:
@inline Base.:+(::Nothing, y::Float32) = y
@inline Base.:+(x::Float32, ::Nothing) = x