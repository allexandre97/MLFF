using Random
using Dates

include("../io/conformations.jl")

function train_epoch!(models, optims, epoch_n, conf_train, conf_val, conf_test)
    #=
    Train for a given epoch of the training loop.
    TODO: Add the logic for the the rest of the datasets (GEMS, RNA)
    =#

    time_start = now()

    train_order = shuffle(gen_conf_pairs(conf_train))
    val_order   = gen_conf_pairs(conf_val)

    n_conf_train, n_conf_val = length(train_order), length(val_order)

    n_batches_train = cld(n_conf_train, MODEL_PARAMS["training"]["n_minibatch"])
    n_batches_val  = cld(n_conf_val, MODEL_PARAMS["training"]["n_minibatch"])

    train_order_cond, val_order_cond = shuffle(COND_MOL_TRAIN), shuffle(COND_MOL_VAL)
    time_wait_sims, time_spice, time_cond = zero(T), zero(T), zero(T)

    if !iszero(MODEL_PARAMS["training"]["training_sims_first_epoch"]) && epoch_n >= (MODEL_PARAMS["training"]["training_sims_first_epoch"] - 1)
        println("CACOLA")
    end

end

function split_vector(v::AbstractVector, p::Real; rng=Random.GLOBAL_RNG)
    N = length(v)
    n1 = floor(Int, p * N)          # number in first split
    perm = randperm(rng, N)         # a random ordering of 1:N
    i1 = perm[1:n1]                 # indices for first split
    i2 = perm[n1+1:end]             # indices for second split
    return v[i1], v[i2]
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

    #=
    TODO: Actually the train / (validation / test) split MUST be done on the  molecule
    types and NOT on the conformations. The idea behind this is that we want to leave
    some entire molecule observations out of the training data to make sure that the 
    model behaves in an extensible manner.
    =#

    #=
    Actually we will split the dataset in three chunks (90%/5%/5%):
        Train      --> The data to train the model
        Validation --> Data not used for training but to score internally different models with different hyper-parameters
            DOUBT: Should different models (different hyppars.) be evaluated with the same dataset ordering??
        Test       --> Data used at the very end to evaluate the best model 
    =#
    
    dataset_ids = 1:size(CONF_DATAFRAME)[1]
    train_ids, val_test_ids = split_vector(dataset_ids, MODEL_PARAMS["training"]["train_size"])

    conf_train    = CONF_DATAFRAME[train_ids,:]
    conf_val_test = CONF_DATAFRAME[val_test_ids,:]

    vt_dataset_ids = 1:size(conf_val_test)[1]
    val_ids, test_ids = split_vector(vt_dataset_ids, 0.5)

    conf_val = conf_val_test[val_ids,:]
    conf_test = conf_val_test[test_ids,:]


    # For now simplified logic, must improve and simmilarize to JG code!
    starting_epoch = 1

    for epoch_n in starting_epoch:MODEL_PARAMS["training"]["n_epochs"]
        train_epoch!(models, optims, epoch_n, conf_train, conf_val, conf_test)
    end


end