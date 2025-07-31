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

function calc_mean_U_gas(epoch_n, mol_id, feat_df, training_sim_dir, temp, models...)

    frame_is = ignore_derivatives() do
        shuffle(COND_SIM_FRAMES)[1:MODEL_PARAMS["training"]["enth_vap_gas_n_samples"]]
    end
    pe_sum = zero(T)
    n = 1
    for frame_i in frame_is
        coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
        _,_,pe,_,_,_,_,_ = mol_to_preds(epoch_n, mol_id, feat_df, coords, boundary, models...)
        pe_sum += pe
        n+=1
    end

    return pe_sum / MODEL_PARAMS["training"]["enth_vap_gas_n_samples"]
end

function vdw_potential(inter::LennardJones, atom::Atom, r::Vector{T}) where T

    return T.(4* atom.ϵ .* ((atom.σ ./ r).^12 - (atom.σ ./ r).^6))

end

function vdw_potential(inter::Mie, atom::Atom, r::Vector{T}) where T

    m, n = inter.m, inter.n
    C = (n/(n-m)) * (n/m) ^ (m/(n-m))
    return T.(C * atom.ϵ .* ((atom.σ ./ r).^n - (atom.σ ./ r).^m))

end

function vdw_potential(inter::DoubleExponential, atom::Atom, r::Vector{T}) where T

    α, β = inter.α, inter.β
    σ, ϵ = atom.σ, atom.ϵ

    rm = σ * T(2^(1/6))

    pe = ϵ * ((β * exp(α) / (α - β)) * exp.(-α .* r ./ rm) - (α * exp(β) / (α - β)) * exp.(-β .* r ./ rm))

    return T.(pe)

end

function vdw_potential(inter::Buffered147, atom::Atom, r::Vector{T}) where T

    
    δ, γ = inter.δ, inter.γ
    σ, ϵ = atom.σ, atom.ϵ

    rm = σ * T(2^(1/6))
    r_div_rm = r ./ rm

    pe = ϵ .* ((1 + δ) ./ (r_div_rm .+ δ)).^7 .* (((1 + γ) ./ (r_div_rm.^7 .+ γ)) .- 2)

    return T.(pe)

end

function vdw_potential(inter::Buckingham, atom::BuckinghamAtom, r::Vector{T}) where T
    
    A, B, C = atom.A, atom.B, atom.C

    pe = A .* exp.(-B .* r) .- C ./ r.^6

    return T.(pe)

end


# Lennard-Jones (12-6)
function vdw_rmin(::LennardJones, atom::Atom)
    return atom.σ * T(2)^(1/6)
end

# Mie (n-m)
function vdw_rmin(inter::Mie, atom::Atom)
    m, n = inter.m, inter.n
    return atom.σ * (n / m)^(1 / (n - m))
end

# Double Exponential
function vdw_rmin(::DoubleExponential, atom::Atom)
    return atom.σ * T(2)^(1/6)
end

# Buffered 14-7
using Roots
function vdw_rmin(inter::Buffered147, atom::Atom)
    δ, γ = inter.δ, inter.γ
    r_m = atom.σ * T(2)^(1/6)

    f(x) = 2x^14 + (4γ + (1 + γ) * δ) * x^7 + (γ^2 - γ)
    
    x_min = find_zero(f, T(1.0))  # initial guess x=1
    return r_m * x_min
end

using LambertW
# Buckingham (exp-6)
function vdw_rmin(::Buckingham, atom::BuckinghamAtom)
    A, B, C = atom.A, atom.B, atom.C

    z = -T(1) / 7 * (6 * C * B^6 / A)^(1 / 7)
    return -T(7) / B * lambertw(z)
end
