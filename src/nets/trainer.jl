using Random
using Dates
using Flux
using Zygote
using TimerOutputs

include("../io/conformations.jl")

const TO = TimerOutput()

function train_epoch!(models, optims, epoch_n, conf_train, conf_val, conf_test)
    #=
    Train for a given epoch of the training loop.
    TODO: Add the logic for the the rest of the datasets (GEMS, RNA)
    =#

    time_start = now()

    train_order = shuffle(gen_conf_pairs(conf_train))
    val_order   = gen_conf_pairs(conf_val)

    n_conf_pairs_train, n_conf_pairs_val = length(train_order), length(val_order)

    n_batches_train = cld(n_conf_pairs_train, MODEL_PARAMS["training"]["n_minibatch"])
    n_batches_val  = cld(n_conf_pairs_val, MODEL_PARAMS["training"]["n_minibatch"])

    train_order_cond, val_order_cond = shuffle(COND_MOL_TRAIN), shuffle(COND_MOL_VAL)
    time_wait_sims, time_spice, time_cond = zero(T), zero(T), zero(T)

    # Submits MD simulations given the FF prediction
    if !iszero(MODEL_PARAMS["training"]["training_sims_first_epoch"]) && epoch_n >= (MODEL_PARAMS["training"]["training_sims_first_epoch"] - 1)
        submit_dir  = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "epoch_$(epoch_n-1)")
        log_path    = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "epoch_$(epoch_n-1).log")
        ff_xml_path = joinpath(MODEL_PARAMS["paths"]["out_dir"], "ff_xml", "epoch_$(epoch_n-1).xml")
        # feats_to_xml method call
        # submit_sims method call
    end

    #=
    TODO: Is this the best way of keeping track of how simulations are running?? 
    =#
    if !iszero(MODEL_PARAMS["training"]["training_sims_first_epoch"]) && epoch_n >= MODEL_PARAMS["training"]["training_sims_first_epoch"]
        if epoch_n >= (MODEL_PARAMS["training"]["training_sims_first_epoch"] + 1)
            rm(joinpath(MODEL_PARAMS["paths"]["out_dir"], "epoch_$(epoch_n-3)"); recursive = true)
        end
        training_sim_dir = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "epoch_$(epoch_n-2)")
        training_sims_complete = false
        time_group = time()
        while !training_sims_complete
            if isfile(joinpath(training_sim_dir, "done.txt"))
                training_sims_complete = true
            else
                sleep(10)
            end
        end
        time_wait_sims += time() - time_group
        simulation_str = "used simulations from end of epoch $(epoch_n-2)"

    elseif MODEL_PARAMS["training"]["use_gaff_simulations"]
        training_sim_dir = joinpath("/lmb/home/alexandrebg/Documents/QuarantineScripts/JG/typing/condensed_data", "trajs_gaff")
        simulation_str = "used simulations from GAFF/TIP3P"

    else
        training_sim_dir = ""
        simulation_str = "did not use simulations"
    end

    #=
    The commented-out lines correspond to datasets I am still not using
    =#

    loss_sum_fs_intra_train, loss_sum_fs_intra_val = zero(T), zero(T)
    loss_sum_fs_inter_train, loss_sum_fs_inter_val = zero(T), zero(T)
    loss_sum_pe_train, loss_sum_pe_val = zero(T), zero(T)
    loss_sum_charges_train, loss_sum_charges_val = zero(T), zero(T)
    loss_sum_vdw_params_train, loss_sum_vdw_params_val = zero(T), zero(T)
    #loss_sum_torsion_ks_train, loss_sum_torsion_ks_val = zero(T), zero(T)

    #loss_sum_fs_intra_train_gems, loss_sum_fs_intra_val_gems = zero(T), zero(T)
    #loss_sum_fs_inter_train_gems, loss_sum_fs_inter_val_gems = zero(T), zero(T)
    
    loss_sum_enth_vap_train, loss_sum_enth_vap_val = zero(T), zero(T)
    loss_sum_enth_mixing_train, loss_sum_enth_mixing_val = zero(T), zero(T)
    
    #loss_sum_J_coupling_train, loss_sum_J_coupling_val = zero(T), zero(T)
    
    count_confs_train, count_confs_val = 0, 0
    count_confs_inter_train, count_confs_inter_val = 0, 0
    count_confs_pe_train, count_confs_pe_val = 0, 0
    count_confs_charges_train, count_confs_charges_val = 0, 0
    #count_confs_torsion_ks_train, count_confs_torsion_ks_val = 0, 0
    
    #count_confs_train_gems, count_confs_val_gems = 0, 0
    #count_confs_inter_train_gems, count_confs_inter_val_gems = 0, 0
    
    count_confs_enth_vap_train, count_confs_enth_vap_val = 0, 0
    count_confs_enth_mixing_train, count_confs_enth_mixing_val = 0, 0
    
    #count_confs_J_coupling_train, count_confs_J_coupling_val = 0, 0
    #loss_jc = zero(T)
    #grads_jc = convert(Vector{Any}, fill(nothing, length(models)))

    n_chunks = Threads.nthreads()
    println(n_chunks)
    if !isnothing(MODEL_PARAMS["paths"]["out_dir"])
        for store_id in ("val-val", "ΔHvap", "ΔHmix")
            rm(joinpath(MODEL_PARAMS["paths"]["out_dir"], "store_$store_id.txt"); force=true)
        end
    end

    # First we iterate over every batch of training data
    Flux.trainmode!(models)
    for batch_i in 1:n_batches_train
        
        # Getting the indices for the first and last conformations of batch
        start_i = (batch_i - 1) * MODEL_PARAMS["training"]["n_minibatch"] + 1
        end_i   = min(start_i + MODEL_PARAMS["training"]["n_minibatch"] - 1, n_conf_pairs_train)

        # Initialize vectors to store the losses of each chunk in parallel
        loss_intra_force_chunks = [T[] for _ in 1:n_chunks]
        loss_inter_force_chunks = [T[] for _ in 1:n_chunks]
        loss_pot_ener_chunks    = [T[] for _ in 1:n_chunks]
        loss_charges_chunks     = [T[] for _ in 1:n_chunks]
        loss_vdw_chunks         = [T[] for _ in 1:n_chunks]
        #loss_torsion_chunks = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for GEMS dataset
        loss_enth_vap_chunks    = [T[] for _ in 1:n_chunks]
        loss_enth_mix_chunks    = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for J couplings
        grads_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
        print_chunks = fill("", n_chunks)
        conf_data = read_conformation(train_order, start_i, end_i)

        #=
        Then we separate each batch into several chunks. Each chunk is worked
        on in parallel threads. This loop does just that.
        =#
        time_group = time()
        # The features for the SPICE dataset
        @timeit TO "SPICE" Threads.@threads for chunk_id in 1:n_chunks
            for i in (start_i - 1 + chunk_id):n_chunks:end_i

                # Read Conformation indices
                conf_i, conf_j, repeat_i = train_order[i]
                mol_id = CONF_DATAFRAME[conf_i,:mol_name]

                # Index dataframe for features
                feat_df = FEATURE_DATAFRAMES[1]
                feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]

                # Get the relevant conformation data
                coords_i, forces_i, energy_i, 
                charges_i, has_charges_i,
                coords_j, forces_j, energy_j,
                charges_j, has_charges_j,
                exceeds_force, pair_present = conf_data[i - start_i + 1]

                #TODO: Maybe it is a good idea to make logging modular
                if MODEL_PARAMS["training"]["verbose"]
                    print_chunks[chunk_id] *= "$mol_id conf $conf_i training - "
                end
                if exceeds_force
                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "max force exceeded! \n"
                    end
                    continue # We break out if the structure shows too much force
                end

                grads = Zygote.gradient(models...) do models...

                    force_loss_intra_sum, force_loss_inter_sum = zero(T), zero(T)
                    energy_loss_sum, charge_loss_sum, vdW_loss_sum, reg_loss_sum = zero(T), zero(T), zero(T), zero(T)
                    mol_to_preds(mol_id, feat_df, coords_i, boundary_inf, models...)
                    return 0 
                end

            end
        end

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