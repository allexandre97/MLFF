function plot_training(plot_fp, models,
                       epochs_mean_fs_intra_train     , epochs_mean_fs_intra_val     ,
                       epochs_mean_fs_inter_train     , epochs_mean_fs_inter_val     ,
                       epochs_mean_pe_train           , epochs_mean_pe_val           ,
                       epochs_mean_charges_train      , epochs_mean_charges_val      ,
                       epochs_mean_vdw_params_train   , epochs_mean_vdw_params_val   ,
                       epochs_mean_torsion_ks_train   , epochs_mean_torsion_ks_val   ,
                       #= epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
                       epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems, =#
                       epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val     ,
                       epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val  ,
                       #= epochs_mean_J_coupling_train   , epochs_mean_J_coupling_val   ,
                       epochs_mean_chem_shift_train   , epochs_mean_chem_shift_val   , =#
                       epochs_loss_regularisation)
    f = Figure(size=(800, 600))
    ax = Axis(f[1, 1:2],
        title="Training progress",
        xlabel="Epoch",
        ylabel="Mean loss value",
    )
    plot_fs_inter_weight = round(10 / MODEL_PARAMS["training"]["loss_weight_force_inter"]; sigdigits=4)
    weighted_fs_inter_train      = epochs_mean_fs_inter_train      .* plot_fs_inter_weight
    weighted_fs_inter_val        = epochs_mean_fs_inter_val        .* plot_fs_inter_weight
    #= weighted_fs_inter_train_gems = epochs_mean_fs_inter_train_gems .* plot_fs_inter_weight
    weighted_fs_inter_val_gems   = epochs_mean_fs_inter_val_gems   .* plot_fs_inter_weight =#
    plot_pe_weight = round(1e-2 / MODEL_PARAMS["training"]["loss_weight_energy"]; sigdigits=4)
    weighted_pe_train = epochs_mean_pe_train .* plot_pe_weight
    weighted_pe_val   = epochs_mean_pe_val   .* plot_pe_weight
    plot_charge_weight = round(100 / MODEL_PARAMS["training"]["loss_weight_charge"]; sigdigits=4)
    weighted_charges_train = epochs_mean_charges_train .* plot_charge_weight
    weighted_charges_val   = epochs_mean_charges_val   .* plot_charge_weight
    plot_vdw_params_offset = 0.1
    weighted_vdw_params_train = epochs_mean_vdw_params_train .+ plot_vdw_params_offset
    weighted_vdw_params_val   = epochs_mean_vdw_params_val   .+ plot_vdw_params_offset
    plot_torsion_ks_weight = round(0.2 / MODEL_PARAMS["training"]["loss_weight_torsion_ks"]; sigdigits=4)
    weighted_torsion_ks_train = epochs_mean_torsion_ks_train .* plot_torsion_ks_weight
    weighted_torsion_ks_val   = epochs_mean_torsion_ks_val   .* plot_torsion_ks_weight
    plot_enth_vap_weight = round(0.05 / MODEL_PARAMS["training"]["loss_weight_enth_vap"]; sigdigits=4)
    weighted_enth_vap_train = epochs_mean_enth_vap_train .* plot_enth_vap_weight
    weighted_enth_vap_val   = epochs_mean_enth_vap_val   .* plot_enth_vap_weight
    plot_enth_mixing_weight = round(0.05 / MODEL_PARAMS["training"]["loss_weight_enth_mixing"]; sigdigits=4)
    weighted_enth_mixing_train = epochs_mean_enth_mixing_train .* plot_enth_mixing_weight
    weighted_enth_mixing_val   = epochs_mean_enth_mixing_val   .* plot_enth_mixing_weight
    #= plot_J_coupling_weight = round(0.02 / MODEL_PARAMS["training"]["loss_weight_J_coupling"]; sigdigits=4)
    weighted_J_coupling_train = epochs_mean_J_coupling_train .* plot_J_coupling_weight
    weighted_J_coupling_val   = epochs_mean_J_coupling_val   .* plot_J_coupling_weight
    plot_chem_shift_weight = round(0.2 / loss_weight_chem_shift; sigdigits=4)
    weighted_chem_shift_train = epochs_mean_chem_shift_train .* plot_chem_shift_weight
    weighted_chem_shift_val   = epochs_mean_chem_shift_val   .* plot_chem_shift_weight =#

    lines!(ax, epochs_mean_fs_intra_train,
           color=:blue, linestyle=:dash)
    lines!(ax, epochs_mean_fs_intra_val, label="Forces intra val",
           color=:blue)
    #= lines!(ax, epochs_mean_fs_intra_train_gems,
           color=:lightblue, linestyle=:dash)
    lines!(ax, epochs_mean_fs_intra_val_gems, label="Forces intra val GEMS",
           color=:lightblue) =#
    lines!(ax, weighted_fs_inter_train,
           color=:green, linestyle=:dash)
    lines!(ax, weighted_fs_inter_val, label="Forces inter val * $plot_fs_inter_weight",
           color=:green)
    #= lines!(ax, weighted_fs_inter_train_gems,
           color=:lightgreen, linestyle=:dash)
    lines!(ax, weighted_fs_inter_val_gems, label="Forces inter val GEMS * $plot_fs_inter_weight",
           color=:lightgreen) =#
    lines!(ax, weighted_pe_train,
           color=:red, linestyle=:dash)
    lines!(ax, weighted_pe_val, label="Energy val * $plot_pe_weight",
           color=:red)
    lines!(ax, weighted_charges_train,
           color=:orange, linestyle=:dash)
    lines!(ax, weighted_charges_val, label="Charge val * $plot_charge_weight",
           color=:orange)
    lines!(ax, weighted_enth_vap_train,
           color=:cyan, linestyle=:dash)
    lines!(ax, weighted_enth_vap_val, label="ΔHvap val * $plot_enth_vap_weight",
           color=:cyan)
    lines!(ax, weighted_enth_mixing_train,
           color=:magenta, linestyle=:dash)
    lines!(ax, weighted_enth_mixing_val, label="ΔHmix val * $plot_enth_mixing_weight",
           color=:magenta)
    #= lines!(ax, weighted_J_coupling_train,
           color=:brown, linestyle=:dash)
    lines!(ax, weighted_J_coupling_val, label="J-coupling val * $plot_J_coupling_weight",
           color=:brown)
    lines!(ax, weighted_chem_shift_train,
           color=:purple, linestyle=:dash)
    lines!(ax, weighted_chem_shift_val, label="Chemical shift val * $plot_chem_shift_weight",
           color=:purple) =#
    #=lines!(ax, weighted_vdw_params_train,
           color=:purple, linestyle=:dash)
    lines!(ax, weighted_vdw_params_val, label="vdw params val + $plot_vdw_params_offset",
           color=:purple)=#
    lines!(ax, weighted_torsion_ks_train,
           color=:pink, linestyle=:dash)
    lines!(ax, weighted_torsion_ks_val, label="Torsion ks val * $plot_torsion_ks_weight",
           color=:pink)
    #=lines!(ax, epochs_loss_regularisation, label="Regularisation",
           color=:pink)=#
    y_max = 1.2
    if MODEL_PARAMS["training"]["training_sims_first_epoch"] > 0
        lines!(ax, [MODEL_PARAMS["training"]["training_sims_first_epoch"], MODEL_PARAMS["training"]["training_sims_first_epoch"]], [0.0, y_max],
               color=:grey, label="Training simulations start")
    end
    xlims!(ax, low=0.0)
    ylims!(ax, low=0.0, high=y_max)
    f[1, 3] = Legend(f, ax; framevisible=false)

    #= mol_id = "val-val"
    coords = read_coordinates(h5read(spice_hdf5_fp, mol_id), 1)
    sys, _, _, _, _, _ = mol_to_system(mol_id, mol_features[mol_id], coords,
                                       boundary_inf, models...)
    θs = collect(-π:(π/100):(π+0.001))
    ind_ϕ, ind_ψ = 13, 43
    propers = sys.specific_inter_lists[3]
    @assert getindex.((propers.is, propers.js, propers.ks, propers.ls), ind_ϕ) == (5, 18, 8, 6)
    @assert getindex.((propers.is, propers.js, propers.ks, propers.ls), ind_ψ) == (8, 6, 19, 9)
    learned_pes_ϕ = torsion_pes(propers.inters[ind_ϕ], θs)
    learned_pes_ψ = torsion_pes(propers.inters[ind_ψ], θs)
    ff99SBildn_ϕ = PeriodicTorsion(
        periodicities=(3, 2),
        phases=(zero(T), zero(T)),
        ks=(T(1.75728), T(1.12968)),
    )
    ff99SBildn_ψ = PeriodicTorsion(
        periodicities=(3, 2, 1),
        phases=(T(π), T(π), T(π)),
        ks=(T(2.3012), T(6.61072), T(1.8828)),
    )
    ff99SBildn_pes_ϕ = torsion_pes(ff99SBildn_ϕ, θs)
    ff99SBildn_pes_ψ = torsion_pes(ff99SBildn_ψ, θs)

    ax1 = Axis(f[2, 1], xlabel="ϕ angle / radians", ylabel="Potential energy / kJ/mol")
    lines!(ax1, θs, learned_pes_ϕ, label="Learned potential")
    lines!(ax1, θs, ff99SBildn_pes_ϕ, label="ff99SBildn")
    ax2 = Axis(f[2, 2], xlabel="ψ angle / radians", ylabel="Potential energy / kJ/mol")
    lines!(ax2, θs, learned_pes_ψ, label="Learned potential")
    lines!(ax2, θs, ff99SBildn_pes_ψ, label="ff99SBildn")
    f[2, 3] = Legend(f, ax1; framevisible=false) =#
    save(plot_fp, f)
end