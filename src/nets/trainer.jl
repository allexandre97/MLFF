using Random
using Dates

include("../io/conformations.jl")

function train_epoch!(models, optims, epoch_n, conf_train, conf_test)

    time_start = now()

end

function train!(models, optims)

    epochs_mean_fs_intra_train     , epochs_mean_fs_intra_val      = T[], T[]
    epochs_mean_fs_inter_train     , epochs_mean_fs_inter_val      = T[], T[]
    epochs_mean_pe_train           , epochs_mean_pe_val            = T[], T[]
    epochs_mean_charges_train      , epochs_mean_charges_val       = T[], T[]
    epochs_mean_vdw_params_train   , epochs_mean_vdw_params_val    = T[], T[]
    epochs_mean_torsion_ks_train   , epochs_mean_torsion_ks_val    = T[], T[]
    epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems = T[], T[]
    epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems = T[], T[]
    epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val      = T[], T[]
    epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val   = T[], T[]
    epochs_mean_J_coupling_train   , epochs_mean_J_coupling_val    = T[], T[]
    epochs_loss_regularisation = T[]

    n_conf = size(CONF_DATAFRAME)[1]

    conf_pairs = gen_conf_pairs(CONF_DATAFRAME)

    return conf_pairs

    #= println(DataFrame(conf_pairs))

    n_conf_train = trunc(Int16, n_conf * MODEL_PARAMS["networks"]["train_size"])
    
    sample_indices = randperm(nrow(CONF_DATAFRAME))[1:n_conf_train]

    conf_train = CONF_DATAFRAME[sample_indices, :]
    conf_test  = CONF_DATAFRAME[setdiff(1:nrow(CONF_DATAFRAME), sample_indices), :]

    # For now simplified logic, must improve and simmilarize to JG code!
    starting_epoch = 1

    for epoch_n in starting_epoch:MODEL_PARAMS["networks"]["n_epochs"]
        train_epoch!(models, optims, epoch_n, conf_train, conf_test)

    end =#


end