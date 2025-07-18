const TO = TimerOutput()

split_grad_safe(args...) = split(args...)
Flux.@non_differentiable split_grad_safe(args...)

multiply_grads(grads, x) = fmap(i -> (isnothing(i) ? nothing : i * x), grads)

accum_grads(x, y) = Zygote.accum(x, y)

# Fails on Dropout layer so define that special case
# This is type piracy but avoids defining all the methods
function Zygote.accum(::NamedTuple{(:p, :dims, :active, :rng), T},
                      ::NamedTuple{(:p, :dims, :active, :rng), T}) where T
    return nothing
end

Zygote.accum(::NamedTuple{(:p, :dims, :active, :rng), <:Any}, ::Nothing) = nothing
Zygote.accum(::Nothing, ::NamedTuple{(:p, :dims, :active, :rng), <:Any}) = nothing

check_no_nans(grads) = !any(g -> any(isnan, Flux.destructure(g)[1]), grads)

function fwd_and_loss(
    mol_id,
    feat_df,
    coords,
    dft_forces,
    dft_charges, has_charges,
    boundary_inf, 
    models
)
    # Forward pass and feat prediction
    sys,
    forces, potential, charges,
    vdw_size, torsion_size, 
    elements, mol_inds = mol_to_preds(mol_id, feat_df, coords, boundary_inf, models...)
    
    # Split the forces in inter and intramolecular contributions
    pred_force_intra, pred_force_inter = split_forces(forces, coords, mol_inds, elements)
    dft_force_intra, dft_force_inter   = split_forces(dft_forces, coords, mol_inds, elements)

    # Calculate the losses
    forces_loss_intra::T = force_loss(pred_force_intra, dft_force_intra)
    forces_loss_inter::T = T(MODEL_PARAMS["training"]["loss_weight_force_inter"]) * force_loss(pred_force_inter, dft_force_inter)
    charges_loss::T      = (has_charges ? charge_loss(charges, dft_charges) : zero(T))
    vdw_loss::T          = vdw_params_loss(vdw_size)
    torsions_loss::T     = torsion_ks_loss(torsion_size)
    reg_loss::T          = param_regularisation((models...,))
    
    return (
        sys,
        forces,
        potential,
        charges,
        vdw_size, torsion_size,
        elements, mol_inds,
        forces_loss_inter, forces_loss_intra,
        charges_loss, vdw_loss,
        torsions_loss, reg_loss
    )
    
end

function loss_update(
    chunk_id::Int,

    # Per-chunk loss arrays
    forces_intra_loss_chunks::Vector{Vector{T}},
    forces_inter_loss_chunks::Vector{Vector{T}},
    potential_loss_chunks::Vector{Vector{T}},
    charges_loss_chunks::Vector{Vector{T}},
    vdw_loss_chunks::Vector{Vector{T}},
    torsions_loss_chunks::Vector{Vector{T}},

    # Running totals
    forces_intra_loss_sum::T,
    forces_inter_loss_sum::T,
    potential_loss_sum::T,
    charges_loss_sum::T,
    vdw_loss_sum::T,
    torsions_loss_sum::T,
    reg_loss_sum::T,

    # Current loss values
    forces_loss_intra::T,
    forces_loss_inter::T,
    charges_loss::T,
    vdw_loss::T,
    torsions_loss::T,
    reg_loss::T,

    pair_present::Bool;

    # Optional keyword args
    epoch_n::Union{Nothing, Int} = nothing,
    pe_diff::Union{Nothing, T} = nothing,
    dft_pe_diff::Union{Nothing, T} = nothing,
    test_train::String = "train"
)

    # Check for NaNs in input loss values
    if any(isnan, (forces_loss_intra, forces_loss_inter, charges_loss, vdw_loss, torsions_loss, reg_loss))
        return false
    end

    # Compute potential energy loss if applicable
    loss_pe = zero(T)

    if pair_present && epoch_n !== nothing && pe_diff !== nothing && dft_pe_diff !== nothing
        loss_pe_unbound = pe_loss(pe_diff, dft_pe_diff)  # or conditional if needed

        if isnan(loss_pe_unbound)
            return false
        end

        loss_pe = loss_pe_unbound

    else
        loss_pe = zero(T)
    end

    # Push current losses into chunked arrays (ignoring gradients)
    ignore_derivatives() do
        push!(forces_intra_loss_chunks[chunk_id], forces_loss_intra)
        push!(forces_inter_loss_chunks[chunk_id], forces_loss_inter)
        push!(charges_loss_chunks[chunk_id], charges_loss)
        push!(vdw_loss_chunks[chunk_id], vdw_loss)
        push!(torsions_loss_chunks[chunk_id], torsions_loss)
        if pair_present
            push!(potential_loss_chunks[chunk_id], loss_pe)
        end
    end

    # Update cumulative loss values if in training mode
    if test_train == "train"
        forces_intra_loss_sum += forces_loss_intra
        forces_inter_loss_sum += forces_loss_inter
        charges_loss_sum      += charges_loss
        vdw_loss_sum          += vdw_loss
        torsions_loss_sum     += torsions_loss
        reg_loss_sum          += reg_loss
        if pair_present
            potential_loss_sum += loss_pe
        end
    end

    return true,
           forces_intra_loss_sum, forces_inter_loss_sum,
           potential_loss_sum, charges_loss_sum,
           vdw_loss_sum, torsions_loss_sum, reg_loss_sum
end


function train_epoch!(models, optims, epoch_n, conf_train, conf_val, conf_test,
    epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
    epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
    epochs_mean_pe_train, epochs_mean_pe_val,
    epochs_mean_charges_train, epochs_mean_charges_val,
    epochs_mean_vdw_params_train, epochs_mean_vdw_params_val,
    epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
    #= epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
    epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems, =#
    epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
    epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
    #= epochs_mean_J_coupling_train, epochs_mean_J_coupling_val,
    epochs_mean_chem_shift_train, epochs_mean_chem_shift_val, =#
    epochs_loss_regularisation)
    
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

    # shorthand for your config value
    first_epoch = MODEL_PARAMS["training"]["training_sims_first_epoch"]

    # ─── 1) Submit sims every epoch ≥ first_epoch ───────────────────────────
    if !iszero(first_epoch) && epoch_n ≥ first_epoch
        submit_epoch = epoch_n - 1
        submit_dir   = joinpath(MODEL_PARAMS["paths"]["out_dir"],
                                "training_sims", "epoch_$submit_epoch")
        log_path     = joinpath(MODEL_PARAMS["paths"]["out_dir"],
                                "training_sims", "epoch_$submit_epoch.log")
        ff_xml_path  = joinpath(MODEL_PARAMS["paths"]["out_dir"],
                                "ff_xml", "epoch_$submit_epoch.xml")

        unique_mols = unique(val[1] for val in COND_MOL_TRAIN)
        feats       = FEATURE_DATAFRAMES[3]

        for mol in unique_mols
            _ = features_to_xml(ff_xml_path, mol,
                                141, 295, feats, models...)
            run(`/lmb/home/alexandrebg/miniconda3/envs/rdkit/bin/python \
                        sim_training.py $submit_dir $ff_xml_path`)
        end
    end

    # ─── 2) Pick which simulations to use ─────────────────────────────────────────
    if !iszero(first_epoch) && epoch_n > first_epoch
        # a) Remove very-old folders once we’re safely past them
        if epoch_n ≥ first_epoch + 2
            old_epoch = epoch_n - 3
            rm(joinpath(MODEL_PARAMS["paths"]["out_dir"],
                        "training_sims", "epoch_$old_epoch");
            recursive=true)
        end

        use_epoch       = epoch_n - 2
        training_sim_dir = joinpath(MODEL_PARAMS["paths"]["out_dir"],
                                    "training_sims", "epoch_$use_epoch")
        done_file        = joinpath(training_sim_dir, "done.txt")
        error_file        = joinpath(training_sim_dir, "error.txt")

        # b) Wait (with timeout) for done.txt to appear
        poll_interval = 10.0                # seconds
        max_wait      = 100              
        elapsed       = 0.0
        time_group    = time()
        
        while !isfile(done_file) #&& elapsed < max_wait
            sleep(poll_interval)
            if isfile(error_file)
                break
            end
            elapsed += poll_interval
        end

        # c) Decide which data to use
        if isfile(done_file)
            time_wait_sims += time() - time_group
            simulation_str   = "used simulations from end of epoch $use_epoch"
        else
            if MODEL_PARAMS["training"]["use_gaff_simulations"]
                gaff_dir = joinpath(DATASETS_PATH, "condensed_data/trajs_gaff")
                gaff_dcd = joinpath(gaff_dir, "vapourisation_liquid", "O_295K.dcd")
                if isfile(gaff_dcd)
                    training_sim_dir = gaff_dir
                    simulation_str   = "fallback to GAFF/TIP3P simulations"
                else
                    training_sim_dir = ""
                    simulation_str   = "GAFF fallback requested but no GAFF sims found"
                end
            else
                training_sim_dir = ""
                simulation_str   = "no simulations available"
            end
        end

    # ─── 3) Before any sims kick in, optionally use GAFF ────────────────────
    elseif MODEL_PARAMS["training"]["use_gaff_simulations"]
        training_sim_dir = joinpath(DATASETS_PATH,
                                    "condensed_data", "trajs_gaff")
        simulation_str   = "used GAFF/TIP3P simulations (pre‐epoch $first_epoch)"

    # ─── 4) Finally, nothing to run or fall back on ───────────────────────────────
    else
        training_sim_dir = ""
        simulation_str   = "did not use simulations"
    end

    #=
    The commented-out lines correspond to datasets I am still not using
    =#

    forces_intra_sum_loss_train, forces_inter_sum_loss_train = zero(T), zero(T)
    forces_intra_sum_loss_val,   forces_inter_sum_loss_val   = zero(T), zero(T)
    
    potential_sum_loss_train = zero(T)
    potential_sum_loss_val   = zero(T)
    
    charges_sum_loss_train   = zero(T)
    charges_sum_loss_val     = zero(T)

    vdw_sum_loss_train       = zero(T)
    vdw_sum_loss_val         = zero(T)

    torsions_sum_loss_train  = zero(T)
    torsions_sum_loss_val    = zero(T)

    enth_vap_sum_loss_train  = zero(T)
    enth_vap_sum_loss_val    = zero(T)

    enth_mix_sum_loss_train  = zero(T)
    enth_mix_sum_loss_val    = zero(T)

    # TODO: Add these when the time comes
    #loss_sum_torsion_ks_train, loss_sum_torsion_ks_val = zero(T), zero(T)
    #loss_sum_fs_intra_train_gems, loss_sum_fs_intra_val_gems = zero(T), zero(T)
    #loss_sum_fs_inter_train_gems, loss_sum_fs_inter_val_gems = zero(T), zero(T)
    #loss_sum_J_coupling_train, loss_sum_J_coupling_val = zero(T), zero(T)
    
    count_train = 0
    count_val   = 0

    count_forces_intra_train, count_forces_inter_train = 0, 0
    count_forces_intra_val,   count_forces_inter_val   = 0, 0

    count_potential_train = 0
    count_potential_val   = 0

    count_charges_train   = 0
    count_charges_val     = 0

    count_vdw_train       = 0
    count_vdw_val         = 0

    count_torsions_train  = 0
    count_torsions_val    = 0

    count_vap_train       = 0
    count_vap_val         = 0

    count_mix_train       = 0
    count_mix_val         = 0

    # TODO: Add these when the time comes
    #count_confs_train_gems, count_confs_val_gems = 0, 0
    #count_confs_inter_train_gems, count_confs_inter_val_gems = 0, 0
    #count_confs_J_coupling_train, count_confs_J_coupling_val = 0, 0
    #loss_jc = zero(T)
    #grads_jc = convert(Vector{Any}, fill(nothing, length(models)))

    n_chunks = Threads.nthreads()
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
        forces_intra_loss_chunks  = [T[] for _ in 1:n_chunks]
        forces_inter_loss_chunks  = [T[] for _ in 1:n_chunks]
        potential_loss_chunks     = [T[] for _ in 1:n_chunks]
        charges_loss_chunks       = [T[] for _ in 1:n_chunks]
        vdw_loss_chunks           = [T[] for _ in 1:n_chunks]
        torsions_loss_chunks      = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for GEMS dataset
        enth_vap_loss_chunks     = [T[] for _ in 1:n_chunks]
        enth_mix_loss_chunks     = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for J couplings
        grads_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
        print_chunks = fill("", n_chunks)
        conf_data = read_conformation(conf_train, train_order, start_i, end_i)

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
                mol_id = conf_train[conf_i,:mol_name]
                
                # Index dataframe for features
                feat_df = occursin("maceoff", mol_id) ? FEATURE_DATAFRAMES[2] : FEATURE_DATAFRAMES[1]
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

                    forces_intra_loss_sum, forces_inter_loss_sum = zero(T), zero(T)
                    potential_loss_sum, charges_loss_sum, vdw_loss_sum, torsions_loss_sum, reg_loss_sum = zero(T), zero(T), zero(T), zero(T), zero(T)

                    # Forward pass and feat prediction
                    sys,
                    forces, potential_i, charges,
                    vdw_size, torsion_size,
                    elements, mol_inds,
                    forces_loss_inter, forces_loss_intra,
                    charges_loss, vdw_loss,
                    torsions_loss, reg_loss = fwd_and_loss(mol_id, feat_df, coords_i, forces_i, charges_i, has_charges_i, boundary_inf, models)

                    if MODEL_PARAMS["training"]["verbose"]
                        ignore_derivatives() do 
                            print_chunks[chunk_id] *="loss forces intra $forces_loss_intra forces inter $forces_loss_inter charge $charges_loss vdw params $vdw_loss torsion ks $torsions_loss regularisation $reg_loss\n"
                        end
                    end

                    loss_success,
                    forces_intra_loss_sum, forces_inter_loss_sum,
                    potential_loss_sum, charges_loss_sum,
                    vdw_loss_sum, torsions_loss_sum, reg_loss_sum = loss_update(
                        chunk_id,
                        forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                        charges_loss_chunks, vdw_loss_chunks, torsions_loss_chunks,
                        forces_intra_loss_sum, forces_inter_loss_sum, potential_loss_sum,
                        charges_loss_sum, vdw_loss_sum, torsions_loss_sum, reg_loss_sum,
                        forces_loss_intra, forces_loss_inter,
                        charges_loss, vdw_loss, torsions_loss, reg_loss,
                        false)

                    if !loss_success
                        println("NaNs found in losses!")
                        return zero(T)
                    end

                    if pair_present

                        # Forward pass and feat prediction
                        sys,
                        forces, potential_j, charges,
                        vdw_size, torsion_size,
                        elements, mol_inds,
                        forces_loss_inter, forces_loss_intra,
                        charges_loss, vdw_loss,
                        torsions_loss, reg_loss = fwd_and_loss(mol_id, feat_df, coords_j, forces_j, charges_j, has_charges_j, boundary_inf, models)
                        

                        pe_diff     = potential_j - potential_i
                        dft_pe_diff = energy_j - energy_i

                        loss_success,
                        forces_intra_loss_sum, forces_inter_loss_sum,
                        potential_loss_sum, charges_loss_sum,
                        vdw_loss_sum, torsions_loss_sum, reg_loss_sum = loss_update(
                            chunk_id,
                            forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                            charges_loss_chunks, vdw_loss_chunks, torsions_loss_chunks,
                            forces_intra_loss_sum, forces_inter_loss_sum, potential_loss_sum,
                            charges_loss_sum, vdw_loss_sum, torsions_loss_sum, reg_loss_sum,
                            forces_loss_intra, forces_loss_inter,
                            charges_loss, vdw_loss, torsions_loss, reg_loss,
                            true;
                            epoch_n = 1,
                            pe_diff = pe_diff,
                            dft_pe_diff = dft_pe_diff,
                            test_train = "train")

                        if !loss_success
                            println("NaNs found in losses! Mol j")
                            return zero(T)
                        end
                    end

                    return forces_intra_loss_sum * MODEL_PARAMS["training"]["train_on_forces_intra"] +
                           forces_inter_loss_sum * MODEL_PARAMS["training"]["train_on_forces_inter"] +
                           potential_loss_sum    * MODEL_PARAMS["training"]["train_on_pe"] +
                           charges_loss_sum      * MODEL_PARAMS["training"]["train_on_charges"] +
                           vdw_loss_sum + torsions_loss_sum + reg_loss_sum 
                end

                if check_no_nans(grads)
                    grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads)
                end
            end
        end

        MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)

        print_chunks = fill("", n_chunks)
        time_spice  += time() - time_group

        time_group = time()

        #=
        TODO: Add functionality to train on GEMS dataset
        =#

        # Now train on condensed data
        if training_sim_dir != "" && epoch_n >= MODEL_PARAMS["training"]["condensed_data_first_epoch"] && (MODEL_PARAMS["training"]["loss_weight_enth_vap"] > zero(T) || MODEL_PARAMS["training"]["loss_weight_enth_mixing"] > zero(T))
            cond_mol_indices = collect(batch_i:n_batches_train:length(COND_MOL_TRAIN))
            
            @timeit TO "Condensed" Threads.@threads for chunk_id in 1:n_chunks
                for cond_inds_i in chunk_id:n_chunks:length(cond_mol_indices)
                    mol_i = cond_mol_indices[cond_inds_i]
                    mol_id, temp, frame_i, repeat_i = COND_MOL_TRAIN[mol_i]

                    feat_df = FEATURE_DATAFRAMES[3]
                    feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]
                    
                    if startswith(mol_id, "vapourisation_")
                        train_on_weight = MODEL_PARAMS["training"]["train_on_enth_vap"]
                        label = "ΔHvap"
                        mol_id_gas = replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                        df_gas = FEATURE_DATAFRAMES[3]
                        df_gas = df_gas[df_gas.MOLECULE .== mol_id_gas, :]

                    else
                        train_on_weight = MODEL_PARAMS["training"]["train_on_enth_mix"]
                        label = "ΔHmix"
                        _, _, smiles_1, smiles_2 = split_grad_safe(mol_id, "_")
                        mol_id_1 = "mixing_single_$smiles_1"
                        mol_id_2 = "mixing_single_$smiles_2"

                        df_mix_1 = FEATURE_DATAFRAMES[3]
                        df_mix_1 = df_mix_1[df_mix_1.MOLECULE .== mol_id_1, :]
                        
                        df_mix_2 = FEATURE_DATAFRAMES[3]
                        df_mix_2 = df_mix_2[df_mix_2.MOLECULE .== mol_id_2, :]

                    end

                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "$mol_id training -"
                    end

                    grads = Zygote.gradient(models...) do models...

                        if label == "ΔHvap"

                            coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)

                            _,
                            _, potential, _,
                            vdw_size, torsion_size, 
                            _, mol_inds = mol_to_preds(mol_id, feat_df, coords, boundary, models...)

                            mean_U_gas = calc_mean_U_gas(mol_id_gas, df_gas, training_sim_dir, temp, models...)

                            cond_loss =  enth_vap_loss(potential, mean_U_gas, temp, frame_i, repeat_i, maximum(mol_inds), mol_id)

                        else

                            coords_1, boundary_1     = read_sim_data(mol_id_1, training_sim_dir, frame_i, temp)
                            coords_2, boundary_2     = read_sim_data(mol_id_2, training_sim_dir, frame_i, temp)
                            coords_com, boundary_com = read_sim_data(mol_id, training_sim_dir, frame_i, temp)

                            _,
                            _, potential_1, _,
                            vdw_size_1, torsion_size_1, 
                            _, mol_inds_1 = mol_to_preds(mol_id_1, df_mix_1, coords_1, boundary_1, models...)

                            _,
                            _, potential_2, _,
                            vdw_size_2, torsion_size_2, 
                            _, mol_inds_2 = mol_to_preds(mol_id_2, df_mix_2, coords_2, boundary_2, models...)

                            _,
                            _, potential_com, _,
                            vdw_size, torsion_size, 
                            _, mol_inds_com = mol_to_preds(mol_id, feat_df, coords_com, boundary_com, models...)

                            cond_loss = enth_mixing_loss(potential_com, potential_1, potential_2,
                                                         boundary_com, boundary_1, boundary_2, 
                                                         maximum(mol_inds_com), maximum(mol_inds_1), maximum(mol_inds_2),
                                                         mol_id, frame_i, repeat_i)
                        end

                        vdw_loss      = vdw_params_loss(vdw_size)
                        torsions_loss = torsion_ks_loss(torsion_size)
                        reg_loss      = param_regularisation((models...,))

                        if MODEL_PARAMS["training"]["verbose"]
                            ignore_derivatives() do 
                                print_chunks[chunk_id] *= "loss $label $cond_loss\n"
                            end
                        end

                        if isnan(cond_loss) || isnan(vdw_loss) || isnan(torsions_loss) || isnan(reg_loss)
                            return zero(T)
                        else
                            ignore_derivatives() do
                                if startswith(mol_id, "vapourisation_")
                                    push!(enth_vap_loss_chunks[chunk_id], cond_loss)
                                else
                                    push!(enth_mix_loss_chunks[chunk_id], cond_loss)
                                end
                                push!(vdw_loss_chunks[chunk_id], vdw_loss)
                                push!(torsions_loss_chunks[chunk_id], torsions_loss)
                            end
                        end
                        return cond_loss * train_on_weight + 
                                vdw_loss + torsions_loss + reg_loss
                    end
                    if check_no_nans(grads)
                        grads_chunks[chunk_id] =  accum_grads.(grads_chunks[chunk_id], grads)
                    end
                end
            end
            MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)
        end

        time_cond += time() - time_group

        time_group = time()

        #TODO: Add functionality for J-coupling data for proteins

        @timeit TO "Gradients" begin
            grads_minibatch = convert(Vector{Any}, fill(nothing, length(models)))
            for chunk_id in 1:n_chunks
                grads_minibatch = accum_grads.(grads_minibatch, grads_chunks[chunk_id])
            end
            grad_vals, restructure = Flux.destructure(grads_minibatch)
            grads_clamp = restructure(clamp.(grad_vals, -MODEL_PARAMS["training"]["grad_clamp_val"], MODEL_PARAMS["training"]["grad_clamp_val"]))
            for model_i in eachindex(models)
                Flux.update!(optims[model_i], models[model_i], grads_clamp[model_i])
            end
        end

        @timeit TO "Logging" begin
            
            log_forces_intra_loss = vcat(forces_intra_loss_chunks...)
            log_forces_inter_loss = vcat(forces_inter_loss_chunks...)
            log_potential_loss    = vcat(potential_loss_chunks...)
            log_charges_loss      = vcat(charges_loss_chunks...)
            log_vdw_loss          = vcat(vdw_loss_chunks...)
            log_torsions_loss     = vcat(torsions_loss_chunks...)
            log_enth_vap_loss     = vcat(enth_vap_loss_chunks...)
            log_enth_mix_loss     = vcat(enth_mix_loss_chunks...)

            if length(log_forces_intra_loss) > 0

                forces_intra_sum_loss_train += sum(log_forces_intra_loss)
                forces_inter_sum_loss_train += sum(log_forces_inter_loss)
                potential_sum_loss_train    += sum(log_potential_loss)
                charges_sum_loss_train      += sum(log_charges_loss)
                vdw_sum_loss_train          += sum(log_vdw_loss)
                torsions_sum_loss_train     += sum(log_torsions_loss)
                enth_vap_sum_loss_train     += sum(log_enth_vap_loss)
                enth_mix_sum_loss_train     += sum(log_enth_mix_loss)

                count_train              += length(log_forces_intra_loss)
                count_forces_intra_train += count(!iszero, log_forces_intra_loss)
                count_forces_inter_train += count(!iszero, log_forces_inter_loss)
                count_potential_train    += length(log_potential_loss)
                count_charges_train      += count(!iszero, log_charges_loss)
                count_torsions_train     += length(log_torsions_loss)
                count_vap_train          += length(log_enth_vap_loss)
                count_vap_train          += length(log_enth_mix_loss)
            end
        end
    end

    Flux.testmode!(models)
    for batch_i in 1:n_batches_val
        
        # TODO: This following piece of code is the same as for train mode, put into function
        # Getting the indices for the first and last conformations of batch
        start_i = (batch_i - 1) * MODEL_PARAMS["training"]["n_minibatch"] + 1
        end_i   = min(start_i + MODEL_PARAMS["training"]["n_minibatch"] - 1, n_conf_pairs_val)

        # Initialize vectors to store the losses of each chunk in parallel
        forces_intra_loss_chunks  = [T[] for _ in 1:n_chunks]
        forces_inter_loss_chunks  = [T[] for _ in 1:n_chunks]
        potential_loss_chunks     = [T[] for _ in 1:n_chunks]
        charges_loss_chunks       = [T[] for _ in 1:n_chunks]
        vdw_loss_chunks           = [T[] for _ in 1:n_chunks]
        torsions_loss_chunks      = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for GEMS dataset
        enth_vap_loss_chunks     = [T[] for _ in 1:n_chunks]
        enth_mix_loss_chunks     = [T[] for _ in 1:n_chunks]

        print_chunks = fill("", n_chunks)
        conf_data = read_conformation(conf_val, val_order, start_i, end_i)

        Threads.@threads for chunk_id in 1:n_chunks
            for i in (start_i-1 + chunk_id):n_chunks:end_i
                # TODO: Again, this is the same as for the train data, refactor in a method
                # Read Conformation indices
                conf_i, conf_j, repeat_i = val_order[i]
                mol_id = conf_val[conf_i,:mol_name]
                
                # Index dataframe for features
                feat_df = occursin("maceoff", mol_id) ? FEATURE_DATAFRAMES[2] : FEATURE_DATAFRAMES[1]
                feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]

                # Get the relevant conformation data
                coords_i, forces_i, energy_i, 
                charges_i, has_charges_i,
                coords_j, forces_j, energy_j,
                charges_j, has_charges_j,
                exceeds_force, pair_present = conf_data[i - start_i + 1]

                #TODO: Maybe it is a good idea to make logging modular
                if MODEL_PARAMS["training"]["verbose"]
                    print_chunks[chunk_id] *= "$mol_id conf $conf_i validation - "
                end
                if exceeds_force
                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "max force exceeded! \n"
                    end
                    continue # We break out if the structure shows too much force
                end

                # Forward pass and feat prediction
                sys,
                forces, potential_i, charges,
                vdw_size, torsion_size,
                elements, mol_inds,
                forces_loss_inter, forces_loss_intra,
                charges_loss, vdw_loss,
                torsions_loss, reg_loss = fwd_and_loss(mol_id, feat_df, coords_i, forces_i, charges_i, has_charges_i, boundary_inf, models)

                if MODEL_PARAMS["training"]["verbose"]
                    print_chunks[chunk_id] *="loss forces intra $forces_loss_intra forces inter $forces_loss_inter charge $charges_loss vdw params $vdw_loss torsion ks $torsions_loss regularisation $reg_loss\n"
                end

                loss_success,
                _, _,
                _, _,
                _, _, _ = loss_update(
                    chunk_id,
                    forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                    charges_loss_chunks, vdw_loss_chunks, torsions_loss_chunks,
                    zero(T), zero(T), zero(T),
                    zero(T), zero(T), zero(T), zero(T),
                    forces_loss_intra, forces_loss_inter,
                    charges_loss, vdw_loss, torsions_loss, reg_loss,
                    false)
                
                if pair_present

                    # Forward pass and feat prediction
                    sys,
                    forces, potential_j, charges,
                    vdw_size, torsion_size,
                    elements, mol_inds,
                    forces_loss_inter, forces_loss_intra,
                    charges_loss, vdw_loss,
                    torsions_loss, reg_loss = fwd_and_loss(mol_id, feat_df, coords_j, forces_j, charges_j, has_charges_j, boundary_inf, models)

                    pe_diff     = potential_j - potential_i
                    dft_pe_diff = energy_j - energy_i

                    loss_success,
                    _, _,
                    _, _,
                    _, _, _ = loss_update(
                        chunk_id,
                        forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                        charges_loss_chunks, vdw_loss_chunks, torsions_loss_chunks,
                        zero(T), zero(T), zero(T),
                        zero(T), zero(T), zero(T), zero(T),
                        forces_loss_intra, forces_loss_inter,
                        charges_loss, vdw_loss, torsions_loss, reg_loss,
                        true;
                        epoch_n = 1,
                        pe_diff = pe_diff,
                        dft_pe_diff = dft_pe_diff,
                        test_train = "train")
                end
            end
        end

        MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)
        print_chunks = fill("", n_chunks)

        # TODO: Add functionality for GEMS 

        if training_sim_dir != "" && epoch_n >= MODEL_PARAMS["training"]["condensed_data_first_epoch"] && (MODEL_PARAMS["training"]["loss_weight_enth_vap"] > zero(T) || MODEL_PARAMS["training"]["loss_weight_enth_mixing"] > zero(T))
            cond_mol_indices = collect(batch_i:n_batches_val:length(COND_MOL_VAL))
            #= Threads.@threads =# for chunk_id in 1:n_chunks
                for cond_inds_i in chunk_id:n_chunks:length(cond_mol_indices)
                    # TODO: This is the same as for the train loop, refactor into method
                    mol_i = cond_mol_indices[cond_inds_i]
                    mol_id, temp, frame_i, repeat_i = COND_MOL_TRAIN[mol_i]
                    feat_df = FEATURE_DATAFRAMES[3]
                    feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]

                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "$mol_id validation -"
                    end

                    if startswith(mol_id, "vapourisation_")
                        train_on_weight = MODEL_PARAMS["training"]["train_on_enth_vap"]
                        label = "ΔHvap"
                        mol_id_gas = replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                        df_gas = FEATURE_DATAFRAMES[3]
                        df_gas = df_gas[df_gas.MOLECULE .== mol_id_gas, :]

                        coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)

                        # Now this can be instantly done here as we do not need Zygote to calculate the gradients
                        _,
                        _, potential, _,
                        vdw_size, torsion_size, 
                        _, mol_inds = mol_to_preds(mol_id, feat_df, coords, boundary, models...)

                        mean_U_gas = calc_mean_U_gas(mol_id_gas, df_gas, training_sim_dir, temp, models...)

                        cond_loss =  enth_vap_loss(potential, mean_U_gas, temp, frame_i, repeat_i, maximum(mol_inds), mol_id)

                    else
                        train_on_weight = MODEL_PARAMS["training"]["train_on_enth_mix"]
                        label = "ΔHmix"
                        _, _, smiles_1, smiles_2 = split_grad_safe(mol_id, "_")
                        mol_id_1 = "mixing_single_$smiles_1"
                        mol_id_2 = "mixing_single_$smiles_2"
                        df_mix_1 = FEATURE_DATAFRAMES[3]
                        df_mix_1 = df_mix_1[df_mix_1.MOLECULE .== mol_id_1, :]

                        df_mix_2 = FEATURE_DATAFRAMES[3]
                        df_mix_2 = df_mix_2[df_mix_2.MOLECULE .== mol_id_2, :]

                        # Same here, we can get the losses instantly as we do not need the gradients from Zygote

                        coords_1, boundary_1     = read_sim_data(mol_id_1, training_sim_dir, frame_i, temp)
                        coords_2, boundary_2     = read_sim_data(mol_id_2, training_sim_dir, frame_i, temp)
                        coords_com, boundary_com = read_sim_data(mol_id, training_sim_dir, frame_i, temp)

                        _,
                        _, potential_1, _,
                        vdw_size_1, torsion_size_1, 
                        _, mol_inds_1 = mol_to_preds(mol_id_1, df_mix_1, coords_1, boundary_1, models...)

                        _,
                        _, potential_2, _,
                        vdw_size_2, torsion_size_2, 
                        _, mol_inds_2 = mol_to_preds(mol_id_2, df_mix_2, coords_2, boundary_2, models...)

                        _,
                        _, potential_com, _,
                        vdw_size, torsion_size, 
                        _, mol_inds_com = mol_to_preds(mol_id, feat_df, coords_com, boundary_com, models...)

                        cond_loss = enth_mixing_loss(potential_com, potential_1, potential_2,
                                                        boundary_com, boundary_1, boundary_2, 
                                                        maximum(mol_inds_com), maximum(mol_inds_1), maximum(mol_inds_2),
                                                        mol_id, frame_i, repeat_i)

                    end

                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "loss $label $cond_loss\n"
                    end

                    if !isnan(cond_loss)
                        if label == "ΔHvap"
                            push!(enth_vap_loss_chunks[chunk_id], cond_loss)
                        else
                            push!(enth_mix_loss_chunks[chunk_id], cond_loss)
                        end
                    end
                end
            end
            MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)
        end

        log_forces_intra_loss = vcat(forces_intra_loss_chunks...)
        log_forces_inter_loss = vcat(forces_inter_loss_chunks...)
        log_potential_loss    = vcat(potential_loss_chunks...)
        log_charges_loss      = vcat(charges_loss_chunks...)
        log_vdw_loss          = vcat(vdw_loss_chunks...)
        log_torsions_loss     = vcat(torsions_loss_chunks...)
        log_enth_vap_loss     = vcat(enth_vap_loss_chunks...)
        log_enth_mix_loss     = vcat(enth_mix_loss_chunks...)

        if length(log_forces_intra_loss) > 0

            forces_intra_sum_loss_val += sum(log_forces_intra_loss)
            forces_inter_sum_loss_val += sum(log_forces_inter_loss)
            potential_sum_loss_val    += sum(log_potential_loss)
            charges_sum_loss_val      += sum(log_charges_loss)
            vdw_sum_loss_val          += sum(log_vdw_loss)
            torsions_sum_loss_val     += sum(log_torsions_loss)
            enth_vap_sum_loss_val     += sum(log_enth_vap_loss)
            enth_mix_sum_loss_val     += sum(log_enth_mix_loss)

            count_val              += length(log_forces_intra_loss)
            count_forces_intra_val += count(!iszero, log_forces_intra_loss)
            count_forces_inter_val += count(!iszero, log_forces_inter_loss)
            count_potential_val    += length(log_potential_loss)
            count_charges_val      += count(!iszero, log_charges_loss)
            count_torsions_val     += length(log_torsions_loss)
            count_vap_val          += length(log_enth_vap_loss)
            count_vap_val          += length(log_enth_mix_loss)
        end
    end

    forces_intra_mean_loss_train = forces_intra_sum_loss_train / count_forces_intra_train
    forces_inter_mean_loss_train = forces_inter_sum_loss_train / count_forces_inter_train
    potential_mean_loss_train    = potential_sum_loss_train / count_potential_train
    charges_mean_loss_train      = charges_sum_loss_train / count_charges_train
    vdw_mean_loss_train          = vdw_sum_loss_train / count_vdw_train
    torsions_mean_loss_train     = torsions_sum_loss_train / count_torsions_train
    enth_vap_mean_loss_train     = enth_vap_sum_loss_train / count_vap_train
    enth_mix_mean_loss_train     = enth_mix_sum_loss_train / count_mix_train

    forces_intra_mean_loss_val = forces_intra_sum_loss_val / count_forces_intra_val
    forces_inter_mean_loss_val = forces_inter_sum_loss_val / count_forces_inter_val
    potential_mean_loss_val    = potential_sum_loss_val / count_potential_val
    charges_mean_loss_val      = charges_sum_loss_val / count_charges_val
    vdw_mean_loss_val          = vdw_sum_loss_val / count_vdw_val
    torsions_mean_loss_val     = torsions_sum_loss_val / count_torsions_val
    enth_vap_mean_loss_val     = enth_vap_sum_loss_val / count_vap_val
    enth_mix_mean_loss_val     = enth_mix_sum_loss_val / count_mix_val

    push!(epochs_mean_fs_intra_train     , forces_intra_mean_loss_train)
    push!(epochs_mean_fs_intra_val       , forces_intra_mean_loss_val)
    push!(epochs_mean_fs_inter_train     , forces_inter_mean_loss_train)
    push!(epochs_mean_fs_inter_val       , forces_inter_mean_loss_val)
    push!(epochs_mean_pe_train           , potential_mean_loss_train)
    push!(epochs_mean_pe_val             , potential_mean_loss_val)
    push!(epochs_mean_charges_train      , charges_mean_loss_train)
    push!(epochs_mean_charges_val        , charges_mean_loss_val)
    push!(epochs_mean_vdw_params_train   , vdw_mean_loss_train)
    push!(epochs_mean_vdw_params_val     , vdw_mean_loss_val)
    push!(epochs_mean_torsion_ks_train   , torsions_mean_loss_train)
    push!(epochs_mean_torsion_ks_val     , torsions_mean_loss_val)
    #= push!(epochs_mean_fs_intra_train_gems, loss_mean_fs_intra_train_gems)
    push!(epochs_mean_fs_intra_val_gems  , loss_mean_fs_intra_val_gems  )
    push!(epochs_mean_fs_inter_train_gems, loss_mean_fs_inter_train_gems)
    push!(epochs_mean_fs_inter_val_gems  , loss_mean_fs_inter_val_gems  ) =#
    push!(epochs_mean_enth_vap_train     , enth_vap_mean_loss_train)
    push!(epochs_mean_enth_vap_val       , enth_vap_mean_loss_val)
    push!(epochs_mean_enth_mixing_train  , enth_mix_mean_loss_train)
    push!(epochs_mean_enth_mixing_val    , enth_mix_mean_loss_val)
    #= push!(epochs_mean_J_coupling_train   , loss_mean_J_coupling_train   )
    push!(epochs_mean_J_coupling_val     , loss_mean_J_coupling_val     )
    push!(epochs_mean_chem_shift_train   , loss_mean_chem_shift_train   )
    push!(epochs_mean_chem_shift_val     , loss_mean_chem_shift_val     ) =#

    loss_regularisation = param_regularisation(models)
    push!(epochs_loss_regularisation, loss_regularisation)

    progress_str = ""
    if !isnothing(out_dir)
        for (store_id, default_str) in (
                ("val-val", "?"),
                ("ΔHvap"  , "ΔHvap water -, exp -, loss -"),
                ("ΔHmix"  , "ΔHmix CCCCO_OC1=NCCC1 - (- - -), exp -, loss -"),
            )
            store_path = joinpath(out_dir, "store_$store_id.txt")
            if ispath(store_path)
                progress_str *=  " - " * only(readlines(store_path))
            else
                progress_str *=  " - " * default_str
            end
        end
        plot_training(
            joinpath(out_dir, "training.pdf"), models,
            epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
            epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
            epochs_mean_pe_train, epochs_mean_pe_val,
            epochs_mean_charges_train, epochs_mean_charges_val,
            epochs_mean_vdw_params_train, epochs_mean_vdw_params_val,
            epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
            #= epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems,
            epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems, =#
            epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
            epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
            #= epochs_mean_J_coupling_train, epochs_mean_J_coupling_val,
            epochs_mean_chem_shift_train, epochs_mean_chem_shift_val, =#
            epochs_loss_regularisation, 
        )

        out_fp_models = joinpath(out_dir, "model.bson")
        out_fp_optims = joinpath(out_dir, "optim.bson")
        BSON.@save out_fp_models models
        BSON.@save out_fp_optims optims
        if save_every_epoch
            out_fp_models_epoch = joinpath(out_dir, "models", "model_ep_$epoch_n.bson")
            out_fp_optims_epoch = joinpath(out_dir, "optims", "optim_ep_$epoch_n.bson")
            BSON.@save out_fp_models_epoch models
            BSON.@save out_fp_optims_epoch optims
        end
    end

    time_epoch = now() - time_start
    time_wait_sims_perc = Int(round(100 * Dates.Second(round(time_wait_sims)) / time_epoch; digits=0))
    time_spice_perc     = Int(round(100 * Dates.Second(round(time_spice    )) / time_epoch; digits=0))
    #time_gems_perc      = Int(round(100 * Dates.Second(round(time_gems     )) / time_epoch; digits=0))
    time_cond_perc      = Int(round(100 * Dates.Second(round(time_cond     )) / time_epoch; digits=0))
    #time_protein_perc   = Int(round(100 * Dates.Second(round(time_protein  )) / time_epoch; digits=0))
    time_epoch_str = round(time_epoch, Minute)

    forces

    report("Epoch $epoch_n - mean training loss forces intra $forces_intra_mean_loss_train force inter $forces_inter_mean_loss_train pe $potential_mean_loss_train charge $charges_mean_loss_train vdw params $vdw_mean_loss_train torsion ks $torsions_mean_loss_train ΔHvap $enth_vap_mean_loss_train ΔHmix $enth_mix_mean_loss_train - mean validation loss forces intra $forces_intra_mean_loss_val force inter $forces_inter_mean_loss_val pe $potential_mean_loss_val charge $charges_mean_loss_val vdw params $vdw_mean_loss_val torsion ks $torsions_mean_loss_val ΔHvap $enth_vap_mean_loss_val ΔHmix $enth_vap_mean_loss_val $progress_str - $simulation_str - $time_spice_perc% SPICE, $time_cond_perc% condensed, $time_wait_sims_perc% sim waiting - took $time_epoch_str\n")

    GC.gc()

    return models, optims
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
    #epochs_mean_fs_intra_train_gems, epochs_mean_fs_intra_val_gems = T[], T[]
    #epochs_mean_fs_inter_train_gems, epochs_mean_fs_inter_val_gems = T[], T[]
    epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val      = T[], T[]
    epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val   = T[], T[]
    #epochs_mean_J_coupling_train   , epochs_mean_J_coupling_val    = T[], T[]
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

    conf_val  = conf_val_test[val_ids,:]
    conf_test = conf_val_test[test_ids,:]

    out_dir = MODEL_PARAMS["paths"]["out_dir"]

    if !isnothing(out_dir) && isfile(joinpath(out_dir, "training.log"))
        for line in readlines(joinpath(out_dir, "training.log"))
            if startswith(line, "Epoch")
                # TODO: This needs a good refactoring
                cols = split(line)
                push!(epochs_mean_fs_intra_train     , parse(T, cols[9 ]))
                push!(epochs_mean_fs_inter_train     , parse(T, cols[12]))
                push!(epochs_mean_pe_train           , parse(T, cols[14]))
                push!(epochs_mean_charges_train      , parse(T, cols[16]))
                push!(epochs_mean_vdw_params_train   , parse(T, cols[19]))
                push!(epochs_mean_torsion_ks_train   , parse(T, cols[22]))
                #push!(epochs_mean_fs_intra_train_gems, parse(T, cols[10]))
                #push!(epochs_mean_fs_inter_train_gems, parse(T, cols[14]))
                push!(epochs_mean_enth_vap_train     , parse(T, cols[24]))
                push!(epochs_mean_enth_mixing_train  , parse(T, cols[26]))
                #push!(epochs_mean_J_coupling_train   , parse(T, cols[30]))
                #push!(epochs_mean_chem_shift_train   , parse(T, cols[33]))
                #push!(epochs_loss_regularisation     , parse(T, cols[28]))
                push!(epochs_mean_fs_intra_val       , parse(T, cols[33]))
                push!(epochs_mean_fs_inter_val       , parse(T, cols[36]))
                push!(epochs_mean_pe_val             , parse(T, cols[38]))
                push!(epochs_mean_charges_val        , parse(T, cols[40]))
                push!(epochs_mean_vdw_params_val     , parse(T, cols[43]))
                push!(epochs_mean_torsion_ks_val     , parse(T, cols[46]))
                #push!(epochs_mean_fs_intra_val_gems  , parse(T, cols[43]))
                #push!(epochs_mean_fs_inter_val_gems  , parse(T, cols[47]))
                push!(epochs_mean_enth_vap_val       , parse(T, cols[48]))
                push!(epochs_mean_enth_mixing_val    , parse(T, cols[50]))
                #push!(epochs_mean_J_coupling_val     , parse(T, cols[63]))
                #push!(epochs_mean_chem_shift_val     , parse(T, cols[66]))
            end
        end
        starting_epoch = length(epochs_mean_fs_intra_train) + 1
        trained_model = joinpath(out_dir, "model.bson")
        trained_optim = joinpath(out_dir, "optim.bson")
        BSON.@load trained_model models
        BSON.@load trained_optim optims
        report("Restarting training from epoch ", starting_epoch, " on ",
               Threads.nthreads(), " thread(s)\n")
    else
        starting_epoch = 1
        report("Starting training on ", Threads.nthreads(), " thread(s)\n")
    end

    for epoch_n in starting_epoch:MODEL_PARAMS["training"]["n_epochs"]
        models, optims = train_epoch!(models, optims, epoch_n, 
                                      conf_train, conf_val, conf_test,
                                      epochs_mean_fs_inter_train, epochs_mean_fs_intra_val,
                                      epochs_mean_fs_intra_train, epochs_mean_fs_inter_val,
                                      epochs_mean_pe_train, epochs_mean_pe_val,
                                      epochs_mean_charges_train, epochs_mean_charges_val,
                                      epochs_mean_vdw_params_train, epochs_mean_vdw_params_val,
                                      epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
                                      epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
                                      epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
                                      epochs_loss_regularisation)
    end
    return models, optims
end
