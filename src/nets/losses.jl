abs2_vec(x) = abs2.(x)
force_loss(fs, dft_fs)            = MODEL_PARAMS["training"]["loss_weight_force"] * mean(sqrt.(sum.(abs2_vec.(fs .- dft_fs))))
charge_loss(charges, dft_charges) = MODEL_PARAMS["training"]["loss_weight_charge"] * mean(abs2.(charges .- dft_charges))
vdw_params_loss(vdw_params_size)  = MODEL_PARAMS["training"]["loss_weight_vdw_params"] * -vdw_params_size
torsion_ks_loss(torsion_ks_size)  = MODEL_PARAMS["training"]["loss_weight_torsion_ks"] * torsion_ks_size
pe_loss(pe_diff, dft_pe_diff) = MODEL_PARAMS["training"]["loss_weight_energy"] * abs(pe_diff - dft_pe_diff)

ϵ_entropy = T(1e-8)

Ω_0          = T(MODEL_PARAMS["training"]["loss_weight_vdw_entropy_0"])
Ω_min        = T(MODEL_PARAMS["training"]["loss_weight_vdw_entropy_min"])
Ω_min_epoch  = T(MODEL_PARAMS["training"]["entropy_min_epoch"])
decay_rate_Ω = T(log(Ω_0 / Ω_min) / Ω_min_epoch)

entropy_loss(func_probs) = -mean(sum(func_probs .* log.(func_probs .+ ϵ_entropy); dims = 1))


#r = [T(_) for _ in LinRange(0.2, 0.4, 50)]
#= function vdw_params_regularisation(atoms, vdw_inters)

    loss::T = zero(T)
    
    for (atom_idx, atom) in enumerate(atoms)
        for (idx_i, (atype_i, inter_i)) in enumerate(zip(atom.atoms, vdw_inters))

            pot_i = vdw_potential(inter_i, atype_i, r)

            for (idx_j, (atype_j, inter_j)) in enumerate(zip(atom.atoms, vdw_inters))
                if idx_i == idx_j
                    continue
                end
                pot_j = vdw_potential(inter_j, atype_j, r)
                loss += mean(abs2.(pot_i .- pot_j))
            end
        end
        loss /= 10.0f0
    end
    return T(loss / length(atoms))
end =#

function vdw_params_regularisation(atoms, vdw_inters)

    loss::T = zero(T)
    
    for (atom_idx, atom) in enumerate(atoms)

        r = [T(_) for _ in LinRange(atom.atoms[1].σ, 2.0 * atom.atoms[1].σ, 50)]
        
        pot_lj   = vdw_potential(vdw_inters[1], atom.atoms[1], r)
        #pot_lj69 = vdw_potential(vdw_inters[2], atom.atoms[2], r)
        pot_dexp = vdw_potential(vdw_inters[3], atom.atoms[3], r)
        #pot_buff = vdw_potential(vdw_inters[4], atom.atoms[4], r)
        #pot_buck = vdw_potential(vdw_inters[5], atom.atoms[5], r)

        loss +=  1.0f0 * mean(abs2.(pot_dexp .- pot_lj)) #+
                #0.3333 * mean(abs2.(pot_dexp .- pot_lj)) +
                #0.3333 * mean(abs2.(pot_buff .- pot_lj)) +
                #0.25 * mean(abs2.(pot_buck .- pot_lj))

    end
    return T(loss / length(atoms))
end

#= function vdw_params_regularisation(atoms, vdw_inters)

    loss_r::T   = zero(T)
    loss_pot::T = zero(T)
    
    for (atom_idx, atom) in enumerate(atoms)

        min_r_lj   = vdw_rmin(vdw_inters[1], atom.atoms[1])
        min_r_lj69 = vdw_rmin(vdw_inters[2], atom.atoms[2])
        min_r_dexp = vdw_rmin(vdw_inters[3], atom.atoms[3])
        min_r_buff = vdw_rmin(vdw_inters[4], atom.atoms[4])
        min_r_buck = vdw_rmin(vdw_inters[5], atom.atoms[5])

        pot_lj   = vdw_potential(vdw_inters[1], atom.atoms[1], [min_r_lj])[1]
        pot_lj69 = vdw_potential(vdw_inters[2], atom.atoms[2], [min_r_lj69])[1]
        pot_dexp = vdw_potential(vdw_inters[3], atom.atoms[3], [min_r_dexp])[1]
        pot_buff = vdw_potential(vdw_inters[4], atom.atoms[4], [min_r_buff])[1]
        pot_buck = vdw_potential(vdw_inters[5], atom.atoms[5], [min_r_buck])[1]

        #= @show min_r_lj  
        @show min_r_lj69
        @show min_r_dexp
        @show min_r_buff
        @show min_r_buck =#

        loss_r += 0.25 * (min_r_lj69 - min_r_lj)^2 +
                  0.25 * (min_r_dexp - min_r_lj)^2 +
                  0.25 * (min_r_buff - min_r_lj)^2 +
                  0.25 * (min_r_buck - min_r_lj)^2

        loss_pot += 0.25 * (pot_lj69 - pot_lj)^2 +
                    0.25 * (pot_dexp - pot_lj)^2 +
                    0.25 * (pot_buff - pot_lj)^2 +
                    0.25 * (pot_buck - pot_lj)^2

    end

    return T((loss_r + 1e-3*loss_pot) / length(atoms))
end =#

function ChainRulesCore.rrule(
    ::typeof(vdw_params_regularisation),
    atoms, 
    vdw_inters
)
    Y = vdw_params_regularisation(atoms, vdw_inters)
    function pullback(ŷ)
        d_atoms      = zero.(atoms)
        d_vdw_inters = zero.(vdw_inters)
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            vdw_params_regularisation,
            Enzyme.Active,
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(vdw_inters, d_vdw_inters)
        )
        return NoTangent(), ŷ .* d_atoms, ŷ .* d_vdw_inters
    end
    return Y, pullback
end

function param_regularisation(models)
    s = sum(abs2, Flux.destructure(models[1:(end-1)])[1])
    # Global parameters excluded from regularisation except for NNPairwise NN params
    if vdw_functional_form == "nn"
        s += sum(abs2, Flux.destructure(models[end])[1][2:end])
    end
    return MODEL_PARAMS["training"]["loss_weight_regularisation"] * s
end

function store_string(store_id, str)
    if !isnothing(out_dir)
        open(joinpath(out_dir, "store_$store_id.txt"), "w") do of
            println(of, str)
        end
    end
end

Flux.@non_differentiable store_string(args...)

function enth_vap_loss(snapshot_U_liquid, mean_U_gas, temp, frame_i, repeat_i, n_molecules, mol_id)
    ΔH_vap = enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    ΔH_vap_exp = T(ENTH_VAP_EXP_DATA[mol_id](temp))
    loss_ΔH_vap = MODEL_PARAMS["training"]["loss_weight_enth_vap"] * abs(ΔH_vap - ΔH_vap_exp)
    ignore_derivatives() do
        if mol_id == "vapourisation_liquid_O" && temp == T(295.0) && frame_i == 101 && repeat_i == 1
            store_string("ΔHvap", "ΔHvap water $ΔH_vap, exp $ΔH_vap_exp, loss $loss_ΔH_vap")
        end
    end
    return loss_ΔH_vap
end

const pressure_enth_mixing = ustrip(T, u"kJ * mol^-1", 1.0u"bar" * 1.0u"nm^3" * Unitful.Na)

function enth_mixing_loss(pe_com, pe_1, pe_2, bound_com, bound_1, bound_2,
                          n_mols_com, n_mols_1, n_mols_2, mol_id, frame_i, repeat_i)
    u_mol_com = (pe_com + pressure_enth_mixing * Molly.volume(bound_com)) / n_mols_com
    u_mol_1   = (pe_1   + pressure_enth_mixing * Molly.volume(bound_1)  ) / n_mols_1
    u_mol_2   = (pe_2   + pressure_enth_mixing * Molly.volume(bound_2)  ) / n_mols_2
    ΔH_mix = u_mol_com - (u_mol_1 + u_mol_2) / 2 # β cancels out
    ΔH_mix_exp = enth_mixing_exp_data[mol_id]
    loss_ΔH_mix = loss_weight_enth_mixing * abs(ΔH_mix - ΔH_mix_exp)
    ignore_derivatives() do
        if mol_id == "mixing_combined_CCCCO_OC1=NCCC1" && frame_i == 101 && repeat_i == 1
            store_string("ΔHmix", "ΔHmix CCCCO_OC1=NCCC1 $ΔH_mix ($u_mol_com $u_mol_1 $u_mol_2), exp $ΔH_mix_exp, loss $loss_ΔH_mix")
        end
    end
    return loss_ΔH_mix
end