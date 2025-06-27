abs2_vec(x) = abs2.(x)
force_loss(fs, dft_fs)            = MODEL_PARAMS["training"]["loss_weight_force"] * mean(sqrt.(sum.(abs2_vec.(fs .- dft_fs))))
charge_loss(charges, dft_charges) = MODEL_PARAMS["training"]["loss_weight_charge"] * mean(abs2.(charges .- dft_charges))
vdw_params_loss(vdw_params_size)  = MODEL_PARAMS["training"]["loss_weight_vdw_params"] * -vdw_params_size
torsion_ks_loss(torsion_ks_size)  = MODEL_PARAMS["training"]["loss_weight_torsion_ks"] * torsion_ks_size
pe_loss(pe_diff, dft_pe_diff) = loss_weight_energy * abs(pe_diff - dft_pe_diff)

function param_regularisation(models)
    s = sum(abs2, Flux.destructure(models[1:(end-1)])[1])
    # Global parameters excluded from regularisation except for NNPairwise NN params
    if vdw_functional_form == "nn"
        s += sum(abs2, Flux.destructure(models[end])[1][2:end])
    end
    return MODEL_PARAMS["training"]["loss_weight_regularisation"] * s
end