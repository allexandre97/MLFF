const TO = TimerOutput()

const kB = T(8.314462618e-3)
const beta = T(1.0f0/(kB*305.0f0))

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

sigmoid_switch(epoch_n::Int, epoch_switch::Int, tension::T) = T(1.0f0 / (1.0f0 + exp(-tension * (T(epoch_n)- T(epoch_switch)))))

function build_rw_states(mol_id::String, epoch_n::Int, lag_epochs::Int, use_own_sims::Bool, temps::NTuple)

    exp_type, sim_type, smiles = split(mol_id, "_"; limit = 3)

    states = ThermoState[]
    trajs  = TrajSystem[]

    ffs_path    = "/lmb/home/alexandrebg/Documents/OpenMM_FF/amber"
    initial_dir = joinpath(DATASETS_PATH, "condensed_data", "starting_structures")

    if !use_own_sims # Then FF is always GAFF

        training_sim_dir = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")

        ff = MolecularForceField(T,
            joinpath.(ffs_path, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
            units = false)
        sys = System(joinpath(initial_dir, "vapourisation_liquid_O_new.pdb"),
                    ff;
                    units               = false,
                    array_type          = Array,
                    rename_terminal_res = false,
                    nonbonded_method    = "cutoff",)
        
        for t in temps
            
            kBT = ustrip(u"kJ/mol", Unitful.R * t*u"K")
            β   = T(1/kBT)

            trj_path = joinpath(training_sim_dir, "$(exp_type)_liquid", "$(smiles)_$(Int(t))K.dcd")

            tstate = ThermoState("sys_gaff_$(t)", β, T(1), deepcopy(sys))
            push!(states, tstate)
            push!(trajs,  TrajSystem(tstate.system, trj_path))

        end

    else # Then we use our own forcefields

        first_ep = epoch_n - lag_epochs - 1
        last_ep  = epoch_n - 2

        for ep in first_ep:last_ep

            ff_path    = joinpath(MODEL_PARAMS["paths"]["out_dir"], "ff_xml", "model_epoch_$(ep).xml")
            epoch_path = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "model_epoch_$(ep)")
            
            ff = MolecularForceField(T, ff_path, units = false)
            sys = System(joinpath(initial_dir, "vapourisation_liquid_O_new.pdb"),
                    ff;
                    units               = false,
                    array_type          = Array,
                    rename_terminal_res = false,
                    nonbonded_method    = "cutoff",)
            
            for t in temps

                kBT = ustrip(u"kJ/mol", Unitful.R * t*u"K")
                β   = T(1/kBT)

                trj_path = joinpath(epoch_path, "$(exp_type)_liquid", "$(smiles)=$(Int(t))K.dcd")

                tstate = ThermoState("sys_$(ep)_$(t)", β, T(1), deepcopy(sys))
                push!(states, tstate)
                push!(trajs,  TrajSystem(tstate.system, trj_path))
                        
            end
        end
    end

    return states, trajs

end

function sample_trajs(mol_id, feats, trajs, states; grads::Bool = true)


    nstates = length(states)

    coords = Vector{<:Any}(undef, nstates)
    bounds = Vector{<:Any}(undef, nstates)
    gradsU = Vector{<:Any}(undef, nstates)

    Ldens = Vector{<:Any}(undef, nstates)
    Lcomp = Vector{<:Any}(undef, nstates)

    kB = ustrip(u"kJ/K", Unitful.k)
    Na = ustrip(Unitful.Na)

    for state_n in 1:nstates

        tstate = states[state_n]
        trj    = trajs[state_n]

        β  = tstate.β

        temp = T((1/(kB*β))/Na)
        
        n_frames = Int(length(trj.trajectory))
        u = Vector{<:Any}(undef, n_frames)
        g = Vector{<:Any}(undef, n_frames)
        c = Vector{<:Any}(undef, n_frames)
        b = Vector{<:Any}(undef, n_frames)
        v = Vector{<:Any}(undef, n_frames)
        d = Vector{<:Any}(undef, n_frames)
        
        for n in 1:n_frames # This can not be multithreaded as Chemfile complains with concurrent reads from file
            current = read_frame!(trj, n)
            currc   = Molly.from_device(current.coords)
            bound   = current.boundary
            vol     = Molly.volume(bound) # nm^3

            dens  = (current.total_mass / vol)*u"g * mol^-1 * nm^-3"
            dens /= Unitful.Na
            dens  = ustrip(u"g/L", dens) 

            u[n] = potential_energy(current)
            c[n] = currc
            b[n] = bound
            v[n] = vol
            d[n] = dens
        end
        comp = calc_compressibility(calc_RT_nomol(temp), v; win_size = 10, step_size = 1)

        g, stride, = Molly.statistical_inefficiency(u; maxlag=n_frames-1)
        sub_coords = Molly.subsample(c, stride; first = 1)
        sub_bounds = Molly.subsample(b, stride; first = 1)
        sub_dens   = Molly.subsample(d, stride; first = 1)
        sub_comp   = Molly.subsample(comp, stride; first = 1)

        if grads
            gu = Vector{<:Any}(undef, length(sub_coords))
            Threads.@threads for sub_idx in eachindex(sub_coords)
                co = sub_coords[sub_idx]
                bo = sub_bounds[sub_idx]
                _, ∂θUθ = Zygote.withgradient(pe_from_snapshot, 1, mol_id, feats, co, bo, models...)
                ∂θUθ = ∂θUθ[6:end]
                gu[sub_idx] = ∂θUθ 
            end
        else
            gu = nothing
        end

        coords[state_n] = ustrip_vec.(sub_coords)
        bounds[state_n] = ustrip.(sub_bounds)
        gradsU[state_n] = gu
        Ldens[state_n]  = (sub_dens .- WATER_DENSITY[temp] ) * MODEL_PARAMS["training"]["loss_weight_density"]
        Lcomp[state_n]  = (sub_comp .- WATER_COMPRESS[temp]) * MODEL_PARAMS["training"]["loss_weight_compress"]

    end

    return coords, bounds, gradsU, Ldens, Lcomp

end

function grads_reweight(loss_flat, β, w_target, w_sum, gradsU_w, C, n_models)

    #loss_flat = collect(Iterators.flatten(loss))
    idx = length(loss_flat) # Needed for properties that are calculated with moving averages
    loss_w     = loss_flat .* view(w_target, 1:idx)
    loss_w_sum = sum(loss_w)
    loss_gradsU_w = multiply_grads.(view(gradsU_w, 1:idx), loss_flat)
    loss_gradsU_w_sum = convert(Vector{Any}, fill(nothing, n_models))
    for lgUw in loss_gradsU_w
        loss_gradsU_w_sum = accum_grads.(loss_gradsU_w_sum, lgUw)
    end

    A = multiply_grads(loss_gradsU_w_sum, 1/w_sum)
    B = loss_w_sum / w_sum

    BC = multiply_grads.(C, -1*B)

    covar = convert(Vector{Any}, fill(nothing, n_models))
    covar = accum_grads(covar, A)
    covar = accum_grads(covar, BC)

    grads = multiply_grads(covar, -1*β)

    return grads
end

function fwd_and_loss(
    epoch_n,
    weight_Ω,
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
    forces_intra, forces_inter,
    potential, charges,
    func_probs, weights_vdw,
    torsion_size, 
    elements, mol_inds = mol_to_preds(epoch_n, mol_id, feat_df, coords, boundary_inf, models...)

    # Split the forces in inter and intramolecular contributions
    #pred_force_intra, pred_force_inter = split_forces(forces, coords, mol_inds, elements)
    dft_force_intra, dft_force_inter   = split_forces(dft_forces, coords, mol_inds, elements)

    # Calculate the losses

    forces_loss_intra::T = force_loss(forces_intra, dft_force_intra)
    forces_loss_inter::T = T(MODEL_PARAMS["training"]["loss_weight_force_inter"]) * force_loss(forces_inter, dft_force_inter)
    vdw_params_reg::T    = vdw_params_regularisation(sys.atoms, sys.pairwise_inters[1].inters, vdw_fnc_idx) * MODEL_PARAMS["training"]["loss_weight_vdw_params"] * λ_reg
    charges_loss::T      = (has_charges ? charge_loss(charges, dft_charges) : zero(T))
    torsions_loss::T     = torsion_ks_loss(torsion_size)
    reg_loss::T          = param_regularisation((models...,))

    return (
        sys,
        forces_intra, forces_inter,
        potential,
        charges,
        weights_vdw, torsion_size,
        elements, mol_inds,
        forces_loss_inter, forces_loss_intra,
        charges_loss, 
        vdw_params_reg,
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
    vdw_params_reg_chunks::Vector{Vector{T}},
    torsions_loss_chunks::Vector{Vector{T}},

    # Running totals
    forces_intra_loss_sum::T,
    forces_inter_loss_sum::T,
    potential_loss_sum::T,
    charges_loss_sum::T,
    vdw_params_reg_sum::T,
    torsions_loss_sum::T,
    reg_loss_sum::T,

    # Current loss values
    forces_loss_intra::T,
    forces_loss_inter::T,
    charges_loss::T,
    vdw_params_reg::T,
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
    if any(isnan, (forces_loss_intra, forces_loss_inter, charges_loss, torsions_loss, reg_loss))
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
        push!(vdw_params_reg_chunks[chunk_id], vdw_params_reg)
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
        vdw_params_reg_sum    += vdw_params_reg
        torsions_loss_sum     += torsions_loss
        reg_loss_sum          += reg_loss
        if pair_present
            potential_loss_sum += loss_pe
        end
    end

    return true,
           forces_intra_loss_sum, forces_inter_loss_sum,
           potential_loss_sum, charges_loss_sum, 
           vdw_params_reg_sum,
           torsions_loss_sum, reg_loss_sum
end

function train_epoch!(models, optims, epoch_n, weight_Ω, conf_train, conf_val, conf_test,
    epochs_mean_fs_intra_train, epochs_mean_fs_intra_val,
    epochs_mean_fs_inter_train, epochs_mean_fs_inter_val,
    epochs_mean_pe_train, epochs_mean_pe_val,
    epochs_mean_charges_train, epochs_mean_charges_val,
    epochs_mean_vdw_params_train, epochs_mean_vdw_params_val,
    epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
    epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
    epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
    epochs_mean_dens_train, epochs_mean_dens_val,
    epochs_mean_comp_train, epochs_mean_comp_val,
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
    n_batches_val   = cld(n_conf_pairs_val, MODEL_PARAMS["training"]["n_minibatch"])

    train_order_cond, val_order_cond = shuffle(COND_MOL_TRAIN), shuffle(COND_MOL_VAL)
    time_wait_sims, time_spice, time_cond, time_reweight = zero(T), zero(T), zero(T), zero(T)

    # --- Configurable timing knobs ---
    lag_epochs_cond = 2
    lag_epochs_rewt = get(MODEL_PARAMS["training"], "reweight_lag", 2) - 1   # read sims from model (me - lag_epochs_rewt) for reweighting
    
    # --- Local epoch & thresholds (unchanged logic for indiv vs joint) ---
    if vdw_fnc_idx != zero(Int) && haskey(vdw_start_epochs, vdw_fnc_idx)
        epoch_local = epoch_n - vdw_start_epochs[vdw_fnc_idx] + 1
        first_condensed_epoch = MODEL_PARAMS["training"]["condensed_indiv_first_epoch"]
    else
        epoch_local = epoch_n - vdw_start_epochs["global"] + 1
        first_condensed_epoch = MODEL_PARAMS["training"]["condensed_joint_first_epoch"]
    end

    min_epoch_valid = MODEL_PARAMS["training"]["min_epoch_valid_sims"] + 1

    first_own_epoch = min_epoch_valid + min(lag_epochs_cond, lag_epochs_rewt)
    first_own_epoch = first_own_epoch ≥ first_condensed_epoch ? first_own_epoch : Inf

    # When we *will* use own sims
    condensed_training_active = epoch_local ≥ first_condensed_epoch
    use_own_sims_now          = (epoch_local - lag_epochs_cond ≥ min_epoch_valid) && condensed_training_active
    use_own_sims_rw           = (epoch_local - lag_epochs_rewt ≥ min_epoch_valid) && condensed_training_active
    
    maxlag = max(lag_epochs_cond, lag_epochs_rewt)
    @assert first_own_epoch - (maxlag + 1) ≥ 1 "First simulations requested at epoch $(first_own_epoch), using previous $(maxlag + 1) epochs, which is not possible"

    # Start submitting BEFORE we need them, regardless of whether condensed is active yet
    should_submit_now = epoch_local ≥ (first_own_epoch - maxlag - 1) && epoch_local ≥ min_epoch_valid

    # --- Paths & status ---
    training_sim_dir = ""
    simulation_str = "did not use simulations"

    # Model epoch in effect at the START of epoch_n
    me = epoch_n - 1

    # 1) SUBMIT (decoupled from condensed activation)
    if should_submit_now
        submit_me   = me
        submit_dir  = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "model_epoch_$submit_me")
        log_path    = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "model_epoch_$submit_me.log")
        ff_xml_path = joinpath(MODEL_PARAMS["paths"]["out_dir"], "ff_xml",        "model_epoch_$submit_me.xml")

        unique_mols = unique(val[1] for val in COND_MOL_TRAIN)

        #mkpath(submit_dir)
        for mol in unique_mols
            _ = features_to_xml(ff_xml_path, me, mol, 141, 295, FEATURE_DATAFRAMES[3], models...)
            #run(`sbatch --partition=agpu --gres=gpu:1 --time=4:0:0 --output=$log_path --job-name=sim$epoch_n --wrap="/lmb/home/alexandrebg/miniconda3/envs/rdkit/bin/python sim_training.py $submit_dir $ff_xml_path"`)
            run(pipeline(`/lmb/home/alexandrebg/miniconda3/envs/rdkit/bin/python sim_training.py $submit_dir $ff_xml_path`, stdout=devnull, stderr=devnull), wait=false)
        end

        # tidy: drop very old model-epoch sims
        old_me = me - maxlag - 1
        if old_me ≥ 0
            old_dir = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "model_epoch_$old_me")
            if isdir(old_dir); rm(old_dir; recursive=true); end
        end
    end

    #= @show first_own_epoch
    @show condensed_training_active
    @show should_submit_now
    @show use_own_sims_now
    @show use_own_sims_rw
    if use_own_sims_rw
        first_ep = epoch_n - lag_epochs_rewt - 1
        last_ep  = epoch_n - 2
        @show first_ep
        @show last_ep
    end =#

    # 2) READ/USE (only when configured to use own sims)
    if use_own_sims_now
        use_me = me - lag_epochs_cond + 1 # e.g., at epoch 3 with lag=1 → me=2, use_me=1
        if use_me ≥ 0
            training_sim_dir = joinpath(MODEL_PARAMS["paths"]["out_dir"], "training_sims", "model_epoch_$use_me")
            done_file  = joinpath(training_sim_dir, "done.txt")
            error_file = joinpath(training_sim_dir, "error.txt")

            # block until done or error
            t0 = time()
            while !(isfile(done_file) || isfile(error_file))
                sleep(10.0)
            end

            if isfile(done_file)
                time_wait_sims += time() - t0
                simulation_str = "used simulations from model epoch $use_me"
            else
                if get(MODEL_PARAMS["training"], "use_gaff_simulations", false)
                    gaff_dir = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")
                    gaff_dcd = joinpath(gaff_dir, "vapourisation_liquid", "O_295K.dcd")
                    if isfile(gaff_dcd)
                        training_sim_dir = gaff_dir
                        simulation_str   = "fallback to GAFF/TIP3P simulations (own sims errored)"
                    else
                        training_sim_dir = ""
                        simulation_str   = "own sims errored; GAFF fallback requested but not found"
                    end
                else
                    training_sim_dir = ""
                    simulation_str   = "own sims errored; no fallback"
                end
            end
        else
            # too early to have produced use_me; optional GAFF
            if get(MODEL_PARAMS["training"], "use_gaff_simulations", false)
                training_sim_dir = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")
                simulation_str   = "used GAFF/TIP3P simulations (too early for own sims)"
            end
        end
    else
        # not yet using own sims → optional GAFF
        if get(MODEL_PARAMS["training"], "use_gaff_simulations", false)
            training_sim_dir = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")
            simulation_str   = "used GAFF/TIP3P simulations (pre-own-sim epoch)"
        end
    end

    #return models, optims

    #=
    The commented-out lines correspond to datasets I am still not using
    =#

    forces_intra_sum_loss_train, forces_inter_sum_loss_train = zero(T), zero(T)
    forces_intra_sum_loss_val,   forces_inter_sum_loss_val   = zero(T), zero(T)
    
    potential_sum_loss_train = zero(T)
    potential_sum_loss_val   = zero(T)
    
    charges_sum_loss_train   = zero(T)
    charges_sum_loss_val     = zero(T)

    vdw_params_reg_sum_train   = zero(T)
    vdw_params_reg_sum_val     = zero(T)

    torsions_sum_loss_train  = zero(T)
    torsions_sum_loss_val    = zero(T)

    enth_vap_sum_loss_train  = zero(T)
    enth_vap_sum_loss_val    = zero(T)

    enth_mix_sum_loss_train  = zero(T)
    enth_mix_sum_loss_val    = zero(T)

    dens_sum_loss_train      = zero(T)
    dens_sum_loss_val        = zero(T)
    
    comp_sum_loss_train      = zero(T)
    comp_sum_loss_val        = zero(T)
    
    count_train = 0
    count_val   = 0

    count_forces_intra_train, count_forces_inter_train = 0, 0
    count_forces_intra_val,   count_forces_inter_val   = 0, 0

    count_potential_train = 0
    count_potential_val   = 0

    count_charges_train        = 0
    count_charges_val          = 0

    count_vdw_params_reg_train = 0
    count_vdw_params_reg_val   = 0

    count_torsions_train  = 0
    count_torsions_val    = 0

    count_vap_train       = 0
    count_vap_val         = 0

    count_mix_train       = 0
    count_mix_val         = 0

    count_dens_train = 0
    count_dens_val   = 0

    count_comp_train = 0
    count_comp_val   = 0

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
        vdw_params_reg_chunks     = [T[] for _ in 1:n_chunks]
        torsions_loss_chunks      = [T[] for _ in 1:n_chunks]
        #TODO: Add loss for GEMS dataset
        enth_vap_loss_chunks     = [T[] for _ in 1:n_chunks]
        enth_mix_loss_chunks     = [T[] for _ in 1:n_chunks]

        vdw_weights_chunks = [zeros(T, 5) for _ in 1:n_chunks]
        nvdw_chunks        = zeros(Int, n_chunks)

        # Reweighting Losses
        dens_loss_chunks = [T[] for _ in 1:n_chunks]
        comp_loss_chunks = [T[] for _ in 1:n_chunks]

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
                feat_qm = occursin("maceoff", mol_id) ? FEATURE_DATAFRAMES[2] : FEATURE_DATAFRAMES[1]
                feat_qm = feat_qm[feat_qm.MOLECULE .== mol_id, :]

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

                vdw_weight_arr = zeros(T, 5)
                nvdw           = 0

                grads = Zygote.gradient(models...) do models...

                    forces_intra_loss_sum, forces_inter_loss_sum = zero(T), zero(T)
                    potential_loss_sum, charges_loss_sum, torsions_loss_sum, reg_loss_sum = zero(T), zero(T), zero(T), zero(T)
                    vdw_params_reg_sum = zero(T)

                    # Forward pass and feat prediction
                    sys,
                    forces_intra, forces_inter, potential_i, charges,
                    weights_vdw, torsion_size,
                    elements, mol_inds,
                    forces_loss_inter, forces_loss_intra,
                    charges_loss, 
                    vdw_params_reg,
                    torsions_loss, reg_loss = fwd_and_loss(epoch_n, weight_Ω, mol_id, feat_qm, coords_i, forces_i, charges_i, has_charges_i, boundary_inf, models)

                    ignore_derivatives() do 
                        vdw_weight_arr += weights_vdw
                        nvdw += 1
                    end

                    if MODEL_PARAMS["training"]["verbose"]
                        ignore_derivatives() do 
                            print_chunks[chunk_id] *="loss forces intra $forces_loss_intra forces inter $forces_loss_inter charge $charges_loss torsion ks $torsions_loss regularisation $reg_loss\n"
                        end
                    end

                    loss_success,
                    forces_intra_loss_sum, forces_inter_loss_sum,
                    potential_loss_sum, charges_loss_sum, 
                    vdw_params_reg_sum,
                    torsions_loss_sum, reg_loss_sum = loss_update(
                        chunk_id,
                        forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                        charges_loss_chunks, vdw_params_reg_chunks, torsions_loss_chunks,
                        forces_intra_loss_sum, forces_inter_loss_sum, potential_loss_sum,
                        charges_loss_sum, vdw_params_reg_sum, torsions_loss_sum, reg_loss_sum,
                        forces_loss_intra, forces_loss_inter,
                        charges_loss, vdw_params_reg, torsions_loss, reg_loss,
                        false)

                    if !loss_success
                        println("NaNs found in losses!")
                        return zero(T)
                    end

                    if pair_present

                        # Forward pass and feat prediction
                        sys,
                        forces_intra, forces_inter, potential_j, charges,
                        weights_vdw, torsion_size,
                        elements, mol_inds,
                        forces_loss_inter, forces_loss_intra,
                        charges_loss,
                        vdw_params_reg,
                        torsions_loss, reg_loss = fwd_and_loss(epoch_n, weight_Ω, mol_id, feat_qm, coords_j, forces_j, charges_j, has_charges_j, boundary_inf, models)

                        ignore_derivatives() do 
                            vdw_weight_arr += weights_vdw
                            nvdw += 1
                        end

                        pe_diff     = potential_j - potential_i
                        dft_pe_diff = energy_j - energy_i

                        loss_success,
                        forces_intra_loss_sum, forces_inter_loss_sum,
                        potential_loss_sum, charges_loss_sum, 
                        vdw_params_reg_sum,
                        torsions_loss_sum, reg_loss_sum = loss_update(
                            chunk_id,
                            forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                            charges_loss_chunks, vdw_params_reg_chunks, torsions_loss_chunks,
                            forces_intra_loss_sum, forces_inter_loss_sum, potential_loss_sum,
                            charges_loss_sum, vdw_params_reg_sum, torsions_loss_sum, reg_loss_sum,
                            forces_loss_intra, forces_loss_inter,
                            charges_loss, vdw_params_reg, torsions_loss, reg_loss,
                            true;
                            epoch_n = epoch_n,
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
                           reg_loss_sum          * MODEL_PARAMS["training"]["loss_weight_regularisation"] + 
                           torsions_loss_sum + 
                           vdw_params_reg_sum
                end

                if check_no_nans(grads)
                    grads_chunks[chunk_id] = accum_grads.(grads_chunks[chunk_id], grads)
                end

                vdw_weights_chunks[chunk_id] += vdw_weight_arr
                nvdw_chunks[chunk_id]        += nvdw

            end
        end

        total_vdw_weights = sum(vdw_weights_chunks)
        total_nvdw = sum(nvdw_chunks)

        @assert total_nvdw > 0 "No vdW interactions processed this epoch"
        avg_func_probs = total_vdw_weights / total_nvdw

        @show avg_func_probs

        MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)

        print_chunks = fill("", n_chunks)
        time_spice  += time() - time_group

        time_group = time()

        # Now train on condensed data
        if training_sim_dir != "" && condensed_training_active && (MODEL_PARAMS["training"]["loss_weight_enth_vap"] > zero(T) || MODEL_PARAMS["training"]["loss_weight_enth_mixing"] > zero(T))
            println("CONDENSED!")
            cond_mol_indices = collect(batch_i:n_batches_train:length(train_order_cond))

            @timeit TO "Condensed" Threads.@threads for chunk_id in 1:n_chunks
                
                for cond_inds_i in chunk_id:n_chunks:length(cond_mol_indices)

                    mol_idx = cond_mol_indices[cond_inds_i]
                    mol_id, temp, frame_idx, repeats = train_order_cond[mol_idx]
                    feat_cd = FEATURE_DATAFRAMES[3]
                    feat_cd = feat_cd[feat_cd.MOLECULE .== mol_id, :]

                    coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_idx, temp)
                    
                    train_on_weight = MODEL_PARAMS["training"]["train_on_enth_vap"]
                    mol_id_gas  = replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                    feat_cd_gas = FEATURE_DATAFRAMES[3]
                    feat_cd_gas = feat_cd_gas[feat_cd_gas.MOLECULE .== mol_id_gas, :]


                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "$mol_id training -"
                    end

                    grads = Zygote.gradient(models...) do models...

                        sys_cond, 
                        _, _, potential_cond, 
                        charges_cond, 
                        func_probs_cond, weights_vdw_cond, 
                        ks_cond, 
                        elements_cond, mol_inds_cond = mol_to_preds(epoch_n, mol_id, feat_cd, coords, boundary, models...)

                        mean_U_gas = 0.0f0#calc_mean_U_gas(epoch_n, mol_id_gas, feat_cd_gas, training_sim_dir, temp, models...)
                        cond_loss  = enth_vap_loss(potential_cond, mean_U_gas, temp, frame_idx, repeats, maximum(mol_inds_cond), mol_id) 
                        
                        vdw_params_reg = vdw_params_regularisation(sys_cond.atoms, sys_cond.pairwise_inters[1].inters, vdw_fnc_idx) * MODEL_PARAMS["training"]["loss_weight_vdw_params"] * λ_reg

                        torsions_loss = zero(T)#torsion_ks_loss(torsion_size)
                        reg_loss      = param_regularisation((models...,))

                        if MODEL_PARAMS["training"]["verbose"]
                            ignore_derivatives() do 
                                print_chunks[chunk_id] *= "loss $label $cond_loss\n"
                            end
                        end

                        if isnan(cond_loss) || isnan(torsions_loss) || isnan(reg_loss)
                            return zero(T)
                        else
                            ignore_derivatives() do

                                push!(enth_vap_loss_chunks[chunk_id], cond_loss)
                                push!(torsions_loss_chunks[chunk_id], torsions_loss)
                                push!(vdw_params_reg_chunks[chunk_id], vdw_params_reg)

                            end
                        end

                        return cond_loss * train_on_weight + 
                               torsions_loss + 
                               reg_loss * MODEL_PARAMS["training"]["loss_weight_regularisation"] +
                               vdw_params_reg
                    
                    end
                    
                    if check_no_nans(grads)
                        grads_chunks[chunk_id] =  accum_grads.(grads_chunks[chunk_id], grads)
                    end

                end

                MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)

            end
            
        end

        time_cond += time() - time_group

        time_group = time() 

        reweight_n_batches = 2

        train_density  = T(MODEL_PARAMS["training"]["train_on_dens_rw"])
        train_compress = T(MODEL_PARAMS["training"]["train_on_comp_rw"])

        if training_sim_dir != "" && condensed_training_active && (train_density  != T(0) || train_compress != T(0))

            println("REWEIGHTING!")

            #mol_id =  "vapourisation_liquid_O"
            
            feat_rw = FEATURE_DATAFRAMES[3]
            feat_rw = feat_rw[feat_rw.MOLECULE .== "vapourisation_liquid_O", :]
            
            @timeit TO "Reweighting" if (batch_i - 1) % reweight_n_batches == 0

                temps = (285, 295, 305, 315, 325)
                n_temp = length(temps)

                states, trajs = build_rw_states("vapourisation_liquid_O", epoch_n, lag_epochs_rewt, use_own_sims_rw, temps)

                coords, bounds, gradsU, Ldens, Lcomp = sample_trajs("vapourisation_liquid_O", feat_rw, trajs, states)

                Ldens      = collect(Iterators.flatten(Ldens))
                Lcomp      = collect(Iterators.flatten(Lcomp))
                Ldens_mean = mean(Ldens)
                Lcomp_mean = mean(Lcomp)

                if !isnan(Ldens_mean) && !isnan(Lcomp_mean)

                    push!(dens_loss_chunks[1], Ldens_mean)
                    push!(comp_loss_chunks[1], Lcomp_mean)

                    if MODEL_PARAMS["training"]["verbose"]
                        report("loss ρ $(Ldens_mean) loss κ $(Lcomp_mean)")
                    end

                    sys_ML, = mol_to_system(1, "vapourisation_liquid_O", feat_rw, coords[1][1], ustrip(trajs[1].system.boundary), models...)

                    for temp in temps

                        kBT = ustrip(u"kJ/mol", Unitful.R * temp*u"K")
                        β   = T(1/kBT)

                        target   = ThermoState("MLFF_$(temp)", β, T(1), deepcopy(sys_ML))
                        gen_mbar = assemble_mbar_inputs(coords, bounds, states; target_state = target, energy_units = sys_ML.energy_units)

                        _, w_target, _ = mbar_weights(gen_mbar)

                        w_sum = sum(w_target)
                        gradsU_flat  = collect(Iterators.flatten(gradsU))
                        gradsU_w     = multiply_grads.(gradsU_flat, w_target)
                        gradsU_w_sum = convert(Vector{Any}, fill(nothing, length(models)))
                        for gUw in gradsU_w 
                            gradsU_w_sum = accum_grads.(gradsU_w_sum, gUw)
                        end

                        C = multiply_grads.(gradsU_w_sum, 1/w_sum)
                        grads_dens = grads_reweight(Ldens, β, w_target, w_sum, gradsU_w, C, length(models))
                        grads_comp = grads_reweight(Lcomp, β, w_target, w_sum, gradsU_w, C, length(models))

                        if check_no_nans(grads_dens)
                            grads_chunks[1] = accum_grads.(grads_chunks[1], grads_dens)
                        end
                        if check_no_nans(grads_comp)
                            grads_chunks[1] = accum_grads.(grads_chunks[1], grads_comp)
                        end
                    end
                else
                    println("NaNs in density or compressibility losses $(Ldens_mean), $(Lcomp_mean)")
                end
            end
        end

        time_reweight += time() - time_group

        time_group = time()

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
            log_vdw_params_reg    = vcat(vdw_params_reg_chunks...)
            log_torsions_loss     = vcat(torsions_loss_chunks...)
            log_enth_vap_loss     = vcat(enth_vap_loss_chunks...)
            log_enth_mix_loss     = vcat(enth_mix_loss_chunks...)
            log_dens_loss         = vcat(dens_loss_chunks...)
            log_comp_loss         = vcat(comp_loss_chunks...)

            if length(log_forces_intra_loss) > 0

                forces_intra_sum_loss_train += sum(log_forces_intra_loss)
                forces_inter_sum_loss_train += sum(log_forces_inter_loss)
                potential_sum_loss_train    += sum(log_potential_loss)
                charges_sum_loss_train      += sum(log_charges_loss)
                vdw_params_reg_sum_train    += sum(log_vdw_params_reg)
                torsions_sum_loss_train     += sum(log_torsions_loss)
                enth_vap_sum_loss_train     += sum(log_enth_vap_loss)
                enth_mix_sum_loss_train     += sum(log_enth_mix_loss)
                dens_sum_loss_train         += sum(log_dens_loss)
                comp_sum_loss_train         += sum(log_comp_loss)

                count_train                += length(log_forces_intra_loss)
                count_forces_intra_train   += count(!iszero, log_forces_intra_loss)
                count_forces_inter_train   += count(!iszero, log_forces_inter_loss)
                count_potential_train      += length(log_potential_loss)
                count_charges_train        += count(!iszero, log_charges_loss)
                count_vdw_params_reg_train += length(log_vdw_params_reg)
                count_torsions_train       += length(log_torsions_loss)
                count_vap_train            += length(log_enth_vap_loss)
                count_mix_train            += length(log_enth_mix_loss)
                count_dens_train           += length(log_dens_loss)
                count_comp_train           += length(log_comp_loss)

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
        vdw_params_reg_chunks     = [T[] for _ in 1:n_chunks]
        torsions_loss_chunks      = [T[] for _ in 1:n_chunks]

        enth_vap_loss_chunks     = [T[] for _ in 1:n_chunks]
        enth_mix_loss_chunks     = [T[] for _ in 1:n_chunks]

        # Reweighting Losses
        dens_loss_chunks = [T[] for _ in 1:n_chunks]
        comp_loss_chunks = [T[] for _ in 1:n_chunks]

        print_chunks = fill("", n_chunks)
        conf_data = read_conformation(conf_val, val_order, start_i, end_i)

        Threads.@threads for chunk_id in 1:n_chunks
            for i in (start_i-1 + chunk_id):n_chunks:end_i
                # TODO: Again, this is the same as for the train data, refactor in a method
                # Read Conformation indices
                conf_i, conf_j, repeat_i = val_order[i]
                mol_id = conf_val[conf_i,:mol_name]
                
                # Index dataframe for features
                feat_qm = occursin("maceoff", mol_id) ? FEATURE_DATAFRAMES[2] : FEATURE_DATAFRAMES[1]
                feat_qm = feat_qm[feat_qm.MOLECULE .== mol_id, :]

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
                forces_intra, forces_inter, potential_i, charges,
                weights_vdw, torsion_size,
                elements, mol_inds,
                forces_loss_inter, forces_loss_intra,
                charges_loss, vdw_params_reg,
                torsions_loss, reg_loss = fwd_and_loss(epoch_n, weight_Ω, mol_id, feat_qm, coords_i, forces_i, charges_i, has_charges_i, boundary_inf, models)

                if MODEL_PARAMS["training"]["verbose"]
                    print_chunks[chunk_id] *= "loss forces intra $forces_loss_intra forces inter $forces_loss_inter charge $charges_loss torsion ks $torsions_loss regularisation $reg_loss\n"
                end

                loss_success,
                _, _,
                _, _,
                _, _, _ = loss_update(
                    chunk_id,
                    forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                    charges_loss_chunks, vdw_params_reg_chunks, torsions_loss_chunks,
                    zero(T), zero(T), zero(T), zero(T),
                    zero(T), zero(T), zero(T),
                    forces_loss_intra, forces_loss_inter,
                    charges_loss, vdw_params_reg, torsions_loss, reg_loss,
                    false)
                
                if pair_present

                    # Forward pass and feat prediction
                    sys,
                    forces_intra, forces_inter, potential_j, charges,
                    weights_vdw, torsion_size,
                    elements, mol_inds,
                    forces_loss_inter, forces_loss_intra,
                    charges_loss, vdw_params_reg,
                    torsions_loss, reg_loss = fwd_and_loss(epoch_n, weight_Ω, mol_id, feat_qm, coords_j, forces_j, charges_j, has_charges_j, boundary_inf, models)

                    pe_diff     = potential_j - potential_i
                    dft_pe_diff = energy_j - energy_i

                    loss_success,
                    _, _,
                    _, _,
                    _, _, = loss_update(
                        chunk_id,
                        forces_intra_loss_chunks, forces_inter_loss_chunks, potential_loss_chunks,
                        charges_loss_chunks, vdw_params_reg_chunks, torsions_loss_chunks,
                        zero(T), zero(T), zero(T), zero(T),
                        zero(T), zero(T), zero(T),
                        forces_loss_intra, forces_loss_inter,
                        charges_loss, vdw_params_reg, torsions_loss, reg_loss,
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
        if training_sim_dir != "" && condensed_training_active && (MODEL_PARAMS["training"]["loss_weight_enth_vap"] > zero(T) || MODEL_PARAMS["training"]["loss_weight_enth_mixing"] > zero(T))
        
            cond_mol_indices = collect(batch_i:n_batches_train:length(val_order_cond))

            @timeit TO "Condensed" Threads.@threads for chunk_id in 1:n_chunks
            
                for cond_inds_i in chunk_id:n_chunks:length(cond_mol_indices)

                    mol_idx = cond_mol_indices[cond_inds_i]
                    mol_id, temp, frame_idx, repeats = val_order_cond[mol_idx]
                    feat_cd = FEATURE_DATAFRAMES[3]
                    feat_cd = feat_cd[feat_cd.MOLECULE .== mol_id, :]

                    coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_idx, temp)
                    
                    train_on_weight = MODEL_PARAMS["training"]["train_on_enth_vap"]
                    mol_id_gas  = replace(mol_id, "vapourisation_liquid_" => "vapourisation_gas_")
                    feat_cd_gas = FEATURE_DATAFRAMES[3]
                    feat_cd_gas = feat_cd_gas[feat_cd_gas.MOLECULE .== mol_id_gas, :]


                    if MODEL_PARAMS["training"]["verbose"]
                        print_chunks[chunk_id] *= "$mol_id training -"
                    end


                    sys_cond, 
                    _, _, potential_cond, 
                    charges_cond, 
                    func_probs_cond, weights_vdw_cond, 
                    ks_cond, 
                    elements_cond, mol_inds_cond = mol_to_preds(epoch_n, mol_id, feat_cd, coords, boundary, models...)

                    mean_U_gas = T(0)#calc_mean_U_gas(epoch_n, mol_id_gas, feat_cd_gas, training_sim_dir, temp, models...)
                    cond_loss  = enth_vap_loss(potential_cond, mean_U_gas, temp, frame_idx, repeats, maximum(mol_inds_cond), mol_id) 
                    
                    vdw_params_reg = vdw_params_regularisation(sys_cond.atoms, sys_cond.pairwise_inters[1].inters, vdw_fnc_idx) * MODEL_PARAMS["training"]["loss_weight_vdw_params"] * λ_reg

                    torsions_loss = zero(T)#torsion_ks_loss(torsion_size)
                    reg_loss      = param_regularisation((models...,))

                    if MODEL_PARAMS["training"]["verbose"]
                        ignore_derivatives() do 
                            print_chunks[chunk_id] *= "loss $label $cond_loss\n"
                        end
                    end

                    if isnan(cond_loss) || isnan(torsions_loss) || isnan(reg_loss)
                        return zero(T)
                    else
                        ignore_derivatives() do

                            push!(enth_vap_loss_chunks[chunk_id], cond_loss)
                            push!(torsions_loss_chunks[chunk_id], torsions_loss)
                            push!(vdw_params_reg_chunks[chunk_id], vdw_params_reg)

                        end
                    end

                end

                MODEL_PARAMS["training"]["verbose"] && foreach(report, print_chunks)

            end
            
        end

        reweight_n_batches = 2

        train_density  = T(MODEL_PARAMS["training"]["train_on_dens_rw"])
        train_compress = T(MODEL_PARAMS["training"]["train_on_comp_rw"])

        if training_sim_dir != "" && condensed_training_active && (train_density  != T(0) || 
                                                                   train_compress != T(0))

            mol_id = "vapourisation_liquid_O"
            feat_rw = FEATURE_DATAFRAMES[3]
            feat_rw = feat_rw[feat_rw.MOLECULE .== mol_id, :] 

            @timeit TO "Reweighting" if (batch_i - 1) % reweight_n_batches == 0

                if batch_i != 1
                    continue
                end

                temps = (285, 295, 305, 315, 325)

                states, trajs = build_rw_states(mol_id, epoch_n, lag_epochs_rewt, use_own_sims_now, temps)

                _, _, _, Ldens, Lcomp = sample_trajs(mol_id, feat_rw, trajs, states; grads = false)
                
                Ldens      = collect(Iterators.flatten(Ldens))
                Lcomp      = collect(Iterators.flatten(Lcomp))
                Ldens_mean = mean(Ldens)
                Lcomp_mean = mean(Lcomp)

                if !isnan(Ldens_mean) && !isnan(Lcomp_mean)

                    push!(dens_loss_chunks[1], Ldens_mean)
                    push!(comp_loss_chunks[1], Lcomp_mean)

                    if MODEL_PARAMS["training"]["verbose"]
                        report("loss ρ $(Ldens_mean) loss κ $(Lcomp_mean)")
                    end

                end

            end
        end

        log_forces_intra_loss = vcat(forces_intra_loss_chunks...)
        log_forces_inter_loss = vcat(forces_inter_loss_chunks...)
        log_potential_loss    = vcat(potential_loss_chunks...)
        log_charges_loss      = vcat(charges_loss_chunks...)
        log_vdw_params_reg    = vcat(vdw_params_reg_chunks...)
        log_torsions_loss     = vcat(torsions_loss_chunks...)
        log_enth_vap_loss     = vcat(enth_vap_loss_chunks...)
        log_enth_mix_loss     = vcat(enth_mix_loss_chunks...)
        log_dens_loss         = vcat(dens_loss_chunks...)
        log_comp_loss         = vcat(comp_loss_chunks...)

        if length(log_forces_intra_loss) > 0

            forces_intra_sum_loss_val += sum(log_forces_intra_loss)
            forces_inter_sum_loss_val += sum(log_forces_inter_loss)
            potential_sum_loss_val    += sum(log_potential_loss)
            charges_sum_loss_val      += sum(log_charges_loss)
            vdw_params_reg_sum_val    += sum(log_vdw_params_reg)
            torsions_sum_loss_val     += sum(log_torsions_loss)
            enth_vap_sum_loss_val     += sum(log_enth_vap_loss)
            enth_mix_sum_loss_val     += sum(log_enth_mix_loss)
            dens_sum_loss_val         += sum(log_dens_loss)
            comp_sum_loss_val         += sum(log_comp_loss)

            count_val                += length(log_forces_intra_loss)
            count_forces_intra_val   += count(!iszero, log_forces_intra_loss)
            count_forces_inter_val   += count(!iszero, log_forces_inter_loss)
            count_potential_val      += length(log_potential_loss)
            count_charges_val        += count(!iszero, log_charges_loss)
            count_vdw_params_reg_val += length(log_vdw_params_reg)
            count_torsions_val       += length(log_torsions_loss)
            count_vap_val            += length(log_enth_vap_loss)
            count_mix_val            += length(log_enth_mix_loss)
            count_dens_val           += length(log_dens_loss)
            count_comp_val           += length(log_comp_loss)

        end
    end

    forces_intra_mean_loss_train = forces_intra_sum_loss_train / count_forces_intra_train
    forces_inter_mean_loss_train = forces_inter_sum_loss_train / count_forces_inter_train
    potential_mean_loss_train    = potential_sum_loss_train / count_potential_train
    charges_mean_loss_train      = charges_sum_loss_train / count_charges_train
    vdw_params_mean_reg_train    = vdw_params_reg_sum_train / count_vdw_params_reg_train
    torsions_mean_loss_train     = torsions_sum_loss_train / count_torsions_train
    enth_vap_mean_loss_train     = enth_vap_sum_loss_train / count_vap_train
    enth_mix_mean_loss_train     = enth_mix_sum_loss_train / count_mix_train
    dens_mean_loss_train         = dens_sum_loss_train / count_dens_train
    comp_mean_loss_train         = comp_sum_loss_train / count_comp_train

    forces_intra_mean_loss_val = forces_intra_sum_loss_val / count_forces_intra_val
    forces_inter_mean_loss_val = forces_inter_sum_loss_val / count_forces_inter_val
    potential_mean_loss_val    = potential_sum_loss_val / count_potential_val
    charges_mean_loss_val      = charges_sum_loss_val / count_charges_val
    vdw_params_mean_reg_val    = vdw_params_reg_sum_val / count_vdw_params_reg_val
    torsions_mean_loss_val     = torsions_sum_loss_val / count_torsions_val
    enth_vap_mean_loss_val     = enth_vap_sum_loss_val / count_vap_val
    enth_mix_mean_loss_val     = enth_mix_sum_loss_val / count_mix_val
    dens_mean_loss_val         = dens_sum_loss_val / count_dens_val
    comp_mean_loss_val         = comp_sum_loss_val / count_comp_val

    push!(epochs_mean_fs_intra_train     , forces_intra_mean_loss_train)
    push!(epochs_mean_fs_intra_val       , forces_intra_mean_loss_val)
    push!(epochs_mean_fs_inter_train     , forces_inter_mean_loss_train)
    push!(epochs_mean_fs_inter_val       , forces_inter_mean_loss_val)
    push!(epochs_mean_pe_train           , potential_mean_loss_train)
    push!(epochs_mean_pe_val             , potential_mean_loss_val)
    push!(epochs_mean_charges_train      , charges_mean_loss_train)
    push!(epochs_mean_charges_val        , charges_mean_loss_val)
    push!(epochs_mean_vdw_params_train   , vdw_params_mean_reg_train)
    push!(epochs_mean_vdw_params_val     , vdw_params_mean_reg_val)
    push!(epochs_mean_torsion_ks_train   , torsions_mean_loss_train)
    push!(epochs_mean_torsion_ks_val     , torsions_mean_loss_val)
    push!(epochs_mean_enth_vap_train     , enth_vap_mean_loss_train)
    push!(epochs_mean_enth_vap_val       , enth_vap_mean_loss_val)
    push!(epochs_mean_enth_mixing_train  , enth_mix_mean_loss_train)
    push!(epochs_mean_enth_mixing_val    , enth_mix_mean_loss_val)
    push!(epochs_mean_dens_train         , dens_mean_loss_train)
    push!(epochs_mean_dens_val           , dens_mean_loss_val)
    push!(epochs_mean_comp_train         , comp_mean_loss_train)
    push!(epochs_mean_comp_val           , comp_mean_loss_val)

    loss_regularisation = zero(T)#param_regularisation(models)
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
    time_cond_perc      = Int(round(100 * Dates.Second(round(time_cond     )) / time_epoch; digits=0))
    time_rwght_perc     = Int(round(100 * Dates.Second(round(time_reweight )) / time_epoch; digits=0))
    time_epoch_str = round(time_epoch, Minute)


    report("Epoch $epoch_n - mean training loss forces intra $forces_intra_mean_loss_train force inter $forces_inter_mean_loss_train pe $potential_mean_loss_train charge $charges_mean_loss_train vdw_par_reg $vdw_params_mean_reg_train λ_vdw $λ_reg torsion ks $torsions_mean_loss_train ΔHvap $enth_vap_mean_loss_train ΔHmix $enth_mix_mean_loss_train density $dens_mean_loss_train compressibility $comp_mean_loss_train - mean validation loss forces intra $forces_intra_mean_loss_val force inter $forces_inter_mean_loss_val pe $potential_mean_loss_val charge $charges_mean_loss_val vdw_par_reg $vdw_params_mean_reg_train λ_vdw $λ_reg torsion ks $torsions_mean_loss_val ΔHvap $enth_vap_mean_loss_val ΔHmix $enth_vap_mean_loss_val density $dens_mean_loss_val compressibility $comp_mean_loss_val $progress_str - $simulation_str - $time_spice_perc% SPICE, $time_cond_perc% condensed, $time_rwght_perc% reweighting, $time_wait_sims_perc% sim waiting - took $time_epoch_str\n")

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
    epochs_mean_enth_vap_train     , epochs_mean_enth_vap_val      = T[], T[]
    epochs_mean_enth_mixing_train  , epochs_mean_enth_mixing_val   = T[], T[]
    epochs_mean_dens_train         , epochs_mean_dens_val          = T[], T[]
    epochs_mean_comp_train         , epochs_mean_comp_val          = T[], T[]
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

                push!(epochs_mean_fs_intra_train        , parse(T, cols[8]))
                push!(epochs_mean_fs_inter_train        , parse(T, cols[11]))
                push!(epochs_mean_pe_train              , parse(T, cols[13]))
                push!(epochs_mean_charges_train         , parse(T, cols[15]))
                push!(epochs_mean_vdw_param_reg_train   , parse(T, cols[22]))
                push!(epochs_mean_torsion_ks_train      , parse(T, cols[26]))
                push!(epochs_mean_enth_vap_train        , parse(T, cols[28]))
                push!(epochs_mean_enth_mixing_train     , parse(T, cols[30]))

                push!(epochs_mean_fs_intra_val          , parse(T, cols[35]))
                push!(epochs_mean_fs_inter_val          , parse(T, cols[38]))
                push!(epochs_mean_pe_val                , parse(T, cols[40]))
                push!(epochs_mean_charges_val           , parse(T, cols[42]))
                push!(epochs_mean_vdw_param_reg_val     , parse(T, cols[49]))
                push!(epochs_mean_torsion_ks_val        , parse(T, cols[53]))
                push!(epochs_mean_enth_vap_val          , parse(T, cols[55]))
                push!(epochs_mean_enth_mixing_val       , parse(T, cols[57]))

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

    global vdw_fnc_idx = 1
    global λ_reg = T(1e-6)
    vdw_fn_epoch_switch = MODEL_PARAMS["training"]["vdw_fn_epoch_switch"]
    tension = T(1000.0)
    global vdw_start_epochs = Dict{Any, Int}(5=>1)

    # Initialize the first FF training start epoch
    vdw_start_epochs[1] = starting_epoch

    for epoch_n in starting_epoch:MODEL_PARAMS["training"]["n_epochs"]

        @show epoch_n

        # Annealing schedule for entropy weight
        if epoch_n < anneal_first_epoch
            weight_Ω = Ω_0
        else
            weight_Ω = annealing_schedule(epoch_n, Ω_0, Ω_min, decay_rate_Ω)
        end

        # Update vdw_fnc_idx
        if epoch_n < 5 * vdw_fn_epoch_switch + 1
            if (epoch_n - 1) % vdw_fn_epoch_switch == 0 && epoch_n != 1
                vdw_fnc_idx += 1
                vdw_start_epochs[vdw_fnc_idx] = epoch_n
            end
        else
            vdw_fnc_idx = zero(Int)
            if !haskey(vdw_start_epochs, "global")
                vdw_start_epochs["global"] = epoch_n
            end
        end

        # Compute λ_reg
        if vdw_fnc_idx != zero(Int)
            start_epoch = vdw_start_epochs[vdw_fnc_idx]
            epoch_local = epoch_n - start_epoch
            λ_reg = sigmoid_switch(epoch_local, MODEL_PARAMS["training"]["vdw_reg_epoch"], tension) + T(1e-6)
        else
            epoch_local = epoch_n - 25
            λ_reg = T(10)#sigmoid_switch(epoch_local, MODEL_PARAMS["training"]["vdw_reg_epoch"], tension) + T(1e-6)  # Small constant during joint training (or use separate logic)
        end

        models, optims = train_epoch!(models, optims, epoch_n, weight_Ω,
                                      conf_train, conf_val, conf_test,
                                      epochs_mean_fs_inter_train, epochs_mean_fs_intra_val,
                                      epochs_mean_fs_intra_train, epochs_mean_fs_inter_val,
                                      epochs_mean_pe_train, epochs_mean_pe_val,
                                      epochs_mean_charges_train, epochs_mean_charges_val,
                                      epochs_mean_vdw_params_train, epochs_mean_vdw_params_val,
                                      epochs_mean_torsion_ks_train, epochs_mean_torsion_ks_val,
                                      epochs_mean_enth_vap_train, epochs_mean_enth_vap_val,
                                      epochs_mean_enth_mixing_train, epochs_mean_enth_mixing_val,
                                      epochs_mean_dens_train, epochs_mean_dens_val,
                                      epochs_mean_comp_train, epochs_mean_comp_val,
                                      epochs_loss_regularisation)
    end
    return models, optims
end
