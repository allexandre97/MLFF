# used for Gumbel-Softmax Annealing
anneal_first_epoch = MODEL_PARAMS["training"]["anneal_first_epoch"]

τ_min              = T(MODEL_PARAMS["training"]["tau_min"])
τ_0                = T(MODEL_PARAMS["training"]["tau_0"])
τ_min_epoch        = T(MODEL_PARAMS["training"]["tau_min_epoch"])
tau_mult_first     = MODEL_PARAMS["training"]["tau_mult_first"]

β_min = T(MODEL_PARAMS["training"]["beta_min"])
γ_anneal     = T(MODEL_PARAMS["training"]["gamma"])

decay_rate_τ = T(log(τ_0 / τ_min) / τ_min_epoch)

function build_adj_list(mol_row::DataFrame)::Array
    
    n_atoms::Int16    = size(split(mol_row[!, :ATOMIC_MASS][1], ","))[1]
    bonds_list::Array = split(mol_row[!, :BONDS][1], ",")

    adj_list::Array = [[i] for i in 1:n_atoms]

    for bond in bonds_list
        i, j = parse.(Int16, split(bond, "/"))
        push!(adj_list[i], j)
        push!(adj_list[j], i)
    end

    return adj_list
end

function build_adj_list(g)
    adj_mol  = [Int[] for _ in 1:nv(g)]
    for e in edges(g)
        u, v = src(e), dst(e)
        push!(adj_mol[u], v)
        push!(adj_mol[v], u)
    end
    return adj_mol
end

Flux.@non_differentiable build_adj_list(mol_row::DataFrame)
Flux.@non_differentiable build_adj_list(args...)
Flux.@non_differentiable has_isomorph(args...)

function mol_to_preds(
    epoch_n::Int,
    mol_id::String,
    args...
)

    sys, partial_charges, func_probs, weights_vdw, torsion_size, elements, mol_inds = mol_to_system(epoch_n, mol_id, args...)

    neighbors = ignore_derivatives() do
        return find_neighbors(sys; n_threads = 1)
    end

    # Get interaction lists separate depending on the number of atoms involves
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    if any(startswith.(mol_id, ("vapourisation_", "mixing_", "protein_")))
        forces_intra = nothing
        forces_inter = nothing
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            (), (), (), neighbors)
    else

        forces_intra = specific_forces_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sils_2_atoms, sils_3_atoms, sils_4_atoms)
        forces_inter = pairwise_forces_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, to_pack(sys.pairwise_inters), neighbors)
    
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    end

    return sys, forces_intra, forces_inter, potential, partial_charges, func_probs, weights_vdw, torsion_size, elements, mol_inds

end

function setup_torsions(features_pad, periodicities, phases, proper)
    return broadcast(1:size(features_pad, 2)) do i
        PeriodicTorsion{6, T, T}(
            periodicities,                      # periodicities
            phases,                             # phases
            ntuple(j -> features_pad[j, i], 6), # ks
            proper,                             # proper
        )
    end
end

function generate_neighbors(
    n_atoms,
    bond_is, bond_js,
    angle_is, angle_ks,
    proper_is, proper_ls,
    dist_nb_cutoff,
)

    eligible = trues(n_atoms, n_atoms)
    for (i, j) in zip(bond_is, bond_js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(angle_is, angle_ks)
        eligible[i, k] = false
        eligible[k, i] = false
    end

    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(proper_is, proper_ls)
        special[i, l] = true
        special[l, i] = true
    end

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        special=special,
        dist_cutoff=(dist_nb_cutoff + T(0.001)),
    )

    return neighbor_finder
end

@non_differentiable generate_neighbors(args...)

# Can this function be non-differentiable?? --> It cannot, it takes as input (args) things that the NNet is predicting!
function build_sys(
    mol_id,
    masses,
    atom_types,
    atom_names,
    mol_inds,
    coords,
    boundary,
    partial_charges,
    
    func_probs::Matrix{T},
    σ_lj::Vector{T},
    ϵ_lj::Vector{T},
    σ_lj69::Vector{T},
    ϵ_lj69::Vector{T},
    σ_dexp::Vector{T},
    ϵ_dexp::Vector{T},
    σ_buff ::Vector{T},
    ϵ_buff ::Vector{T},
    A::Vector{T},
    B::Vector{T},
    C::Vector{T},
    α::T,
    β::T,
    δ::T,
    γ::T,

    weight_lj::T,
    weight_coul::T,

    bond_functional_form::String,
    bond_k::Vector{Float32},
    bond_r0::Vector{Float32},
    bond_a::Union{Vector{Float32}, Nothing},

    angle_functional_form::String,
    angle_k::Union{Vector{Float32}, Nothing},
    angle_θ0::Union{Vector{Float32}, Nothing},
    angle_kj::Union{Vector{Float32}, Nothing},
    angle_θ0j::Union{Vector{Float32}, Nothing},

    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    proper_feats, improper_feats,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,

)
    n_atoms = length(partial_charges)
    dist_nb_cutoff = T(MODEL_PARAMS["physics"]["dist_nb_cutoff"])

    #= @show σ_lj[1]
    @show ϵ_lj[1]
    @show σ_lj69[1]
    @show ϵ_lj69[1]
    @show α
    @show β
    @show σ_dexp[1]
    @show ϵ_dexp[1]
    @show δ
    @show γ
    @show σ_buff[1]
    @show ϵ_buff[1]
    @show A[1]
    @show B[1]
    @show C[1] =#

    atoms      = [GeneralAtom(i, one(Int),
                              T(masses[i]), T(partial_charges[i]),
                              T(σ_lj[i]),   T(ϵ_lj[i]),
                              T(σ_lj69[i]), T(ϵ_lj69[i]),
                              T(σ_dexp[i]), T(ϵ_dexp[i]),
                              T(σ_buff[i]), T(ϵ_buff[i]),
                              T(A[i]),      T(B[i]),      T(C[i]))
                  for i in 1:n_atoms]

    masses = [a.mass for a in atoms]

    if vdw_fnc_idx == zero(Int)
        weights_vdw = vec(mean(func_probs; dims=2))
    else
        weights_vdw = [i == vdw_fnc_idx ? 1.0f0 : 0.0 for i in 1:5]
    end

    global choice      = argmax(weights_vdw)

    global vdw_functional_form = CHOICE_TO_VDW[choice]

    vdw_inters = (
        inters = (LennardJones(DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, weight_lj),
         Mie(6, 9, DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, one(T), 1),
         DoubleExponential(α, β, σ_mixing, ϵ_mixing, one(T), dist_nb_cutoff),
         Buffered147(δ, γ, σ_mixing, ϵ_mixing, one(T), dist_nb_cutoff),
         Buckingham(one(T), dist_nb_cutoff)),

        weights = SVector{5, T}(weights_vdw)
    )

    ########## Coulomb interactions section ##########
    if vdw_functional_form == "nn"
        # Placeholder
    else

        if MODEL_PARAMS["physics"]["use_reaction_field"] &&
            any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_", "protein_")))
            inter_coulomb = CoulombReactionField(dist_nb_cutoff, T(Molly.crf_solvent_dielectric),
                                                 true, weight_coul, T(ustrip(Molly.coulomb_const)))
        else
            inter_coulomb = Coulomb(DistanceCutoff(dist_nb_cutoff),
                                    true, weight_coul, T(ustrip(Molly.coulomb_const)))
        end

        pairwise_inter = (vdw_inters, inter_coulomb)
    end

    ########## Bond Interactions section ##########
    if bond_functional_form == "harmonic"
        bond_inter = HarmonicBond.(T.(bond_k), T.(bond_r0))
    elseif bond_functional_form == "morse"
        bond_inter = MorseBond.(T.(bond_k), T.(bond_a), T.(bond_r0))
    end
    bonds = InteractionList2Atoms(bonds_i, bonds_j, bond_inter)

    ########## Angle Interactions section ##########
    if angle_functional_form == "harmonic"
        angle_inter = HarmonicAngle.(T.(angle_k), T.(angle_θ0))
    elseif angle_functional_form == "ub"
        angle_inter = UreyBradley.(T.(angle_k), T.(angle_θ0), T.(angle_kj), T.(angle_θ0j))
    end
    angles = InteractionList3Atoms(angles_i, angles_j, angles_k, angle_inter)

    ######### Torsion Interactions section ##########
    proper_inter = setup_torsions(proper_feats, torsion_periodicities, torsion_phases, true)
    propers = InteractionList4Atoms(propers_i, propers_j, propers_k, propers_l, proper_inter)

    improper_inter = setup_torsions(improper_feats, torsion_periodicities, torsion_phases, false)
    impropers = InteractionList4Atoms(impropers_j, impropers_k, impropers_i, impropers_l, improper_inter)

    if length(propers_i) > 0 && length(impropers_i) > 0
        specific_inter_lists = (bonds, angles, propers, impropers)
    elseif length(propers_i) > 0
        specific_inter_lists = (bonds, angles, propers)
    elseif length(impropers_i) > 0
        specific_inter_lists = (bonds, angles, impropers)
    elseif length(angles_i) > 0
        specific_inter_lists = (bonds, angles)
    elseif length(bonds_i) > 0
        specific_inter_lists = (bonds,)
    else
        specific_inter_lists = ()
    end

    neighbor_finder = generate_neighbors(n_atoms,
                                         bonds_i, bonds_j,
                                         angles_i, angles_k,
                                         propers_i, propers_l,
                                         dist_nb_cutoff)

    velocities = zero(coords)

    topo = ignore_derivatives() do
        MolecularTopology(bonds_i, bonds_j, n_atoms)
    end

    atoms_data = [AtomData(atom_types[i], atom_names[i], mol_inds[i], split_grad_safe(atom_types[i], "_")[1], "A", "?", true) for i in 1:n_atoms]

    sys = System{3, Array, T, typeof(atoms), typeof(coords), typeof(boundary), typeof(velocities), typeof(atoms_data),
                 typeof(topo), typeof(pairwise_inter), typeof(specific_inter_lists), typeof(()), typeof(()),
                 typeof(neighbor_finder), typeof(()), typeof(NoUnits),
                 typeof(NoUnits), T, Vector{T}, T, Nothing}(
        atoms, coords, boundary, velocities, atoms_data, topo, pairwise_inter, specific_inter_lists,
        (), (), neighbor_finder, (), 1, NoUnits, NoUnits, one(T), masses, sum(masses), nothing)

    return sys, weights_vdw
end

function atom_names_from_elements(el_list::Vector{Int},
                                   name_map::Vector{String})
    counts = Dict{String,Int}() 
    names  = String[]
    for el in el_list
        sym = name_map[el]
        n   = get(counts, sym, 0) + 1
        counts[sym] = n
        push!(names, "$(sym)$(n)")
    end
    return names
end

Flux.@non_differentiable atom_names_from_elements(args...)

function get_molecule_names(mol_id::String)::Vector{String}

    if any(startswith.(mol_id, ("vapourisation_", "mixing_")))
        if startswith(mol_id, "vapourisation")
            name = split(mol_id, "_")[end]
            if name == "O"
                mol_names = ["water"]
            else
                mol_names = [name]
            end
        else
            _, _, smiles = split(mol_id, "_"; limit = 3)
            names = split(smiles, "_")
            mol_names = [name != "water" ? name : "water" for name in names]
        end

    else
        if occursin("water", mol_id)
            mol_names = ["water"]
	else
	    mol_names = ["CH3Cl", "water"]
        end
    end
end

Flux.@non_differentiable get_molecule_names(args...)

function mol_to_system(
    epoch_n::Int,
    mol_id::String,
    feat_df::DataFrame,
    coords,
    boundary::CubicBoundary,
    
    atom_embedding_model::GNNChain,
    bond_pooling_model::Chain,
    angle_pooling_model::Chain,
    proper_pooling_model::Chain,
    improper_pooling_model::Chain,
    
    nonbonded_selection_model::Chain,
    
    charge_features_model::Chain,
    lj_features_model::Chain,
    lj69_features_model::Chain,
    dexp_features_model::Chain,
    buff_features_model::Chain,
    buck_features_model::Chain,

    bond_features_model::Chain,
    angle_features_model::Chain,
    proper_features_model::Chain,
    improper_features_model::Chain,
    global_params::GlobalParams{T}
)

    atom_features_models = (charge_features_model,
                            lj_features_model,
                            lj69_features_model,
                            dexp_features_model,
                            buff_features_model,
                            buck_features_model)

    elements, formal_charges,
    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,
    mol_inds, _, n_atoms, atom_features = decode_feats(feat_df)

    masses = [NAME_TO_MASS[ELEMENT_TO_NAME[e]] for e in elements]

    atom_types = fill("", length(elements))
    atom_names = fill("", length(elements))

    global_graph = build_global_graph(length(elements), zip(bonds_i, bonds_j))
    all_graphs, all_indices = extract_all_subgraphs(global_graph)
    unique_graphs, unique_indices, graph_to_unique = filter_unique(all_graphs, all_indices)

    # Prediction arrays
    #Charges
    #partial_charges = zeros(T, n_atoms)

    # Let's try to predict the partial charges for the whole system instead for just each mol template
    charges_k1 = zeros(T, n_atoms)
    charges_k2 = zeros(T, n_atoms)

    #vdW Parameters
    func_probs = zeros(T, 5, n_atoms)

    vdw_σ_lj   = zeros(T, n_atoms)
    vdw_ϵ_lj   = zeros(T, n_atoms)
    vdw_σ_lj69 = zeros(T, n_atoms)
    vdw_ϵ_lj69 = zeros(T, n_atoms)
    vdw_σ_dexp = zeros(T, n_atoms)
    vdw_ϵ_dexp = zeros(T, n_atoms)
    vdw_σ_buff = zeros(T, n_atoms)
    vdw_ϵ_buff = zeros(T, n_atoms)
    vdw_A      = zeros(T, n_atoms)
    vdw_B      = zeros(T, n_atoms)
    vdw_C      = zeros(T, n_atoms)

    weight_lj   = sigmoid(global_params.params[1])
    weight_coul = sigmoid(global_params.params[2])
    
    α = transform_dexp_α(global_params.params[3])
    β = transform_dexp_β(global_params.params[4])
    δ = transform_buff_δ(global_params.params[5])
    γ = transform_buff_γ(global_params.params[6])
    
    #Bond Parameters
    bond_functional_form = MODEL_PARAMS["physics"]["bond_functional_form"]
    n_bonds = length(bonds_i)
    bonds_k  = nothing
    bonds_r0 = nothing
    bonds_a  = nothing
    if bond_functional_form == "harmonic"
        bonds_k  = zeros(T, n_bonds)
        bonds_r0 = zeros(T, n_bonds)
    elseif bond_functional_form == "morse"
        bonds_k  = zeros(T, n_bonds)
        bonds_r0 = zeros(T, n_bonds)
        bonds_a  = zeros(T, n_bonds)
    end

    #Angle parameters
    angle_functional_form = MODEL_PARAMS["physics"]["angle_functional_form"]
    n_angles = length(angles_i)
    angles_ki  = nothing
    angles_θ0i = nothing
    angles_kj  = nothing
    angles_θ0j = nothing
    if angle_functional_form == "harmonic"
        angles_ki  = zeros(T, n_angles)
        angles_θ0i = zeros(T, n_angles)
    elseif angle_functional_form == "ub"
        angles_ki   = zeros(T, n_angles)
        angles_θ0i  = zeros(T, n_angles)
        angles_kj  = zeros(T, n_angles)
        angles_θ0j = zeros(T, n_angles)
    end

    # Proper and improper torsions
    proper_feats   = zeros(T, (n_proper_terms, length(propers_i)))
    improper_feats = zeros(T, (n_improper_terms, length(impropers_i)))

    # Get name of molecules present in system
    mol_names = get_molecule_names(mol_id)

    # Loop over all unique molecule types
    for (t, (g, vs_template)) in enumerate(zip(unique_graphs, unique_indices))

        equivs = find_atom_equivalences(g, vs_template, elements)
        labels = ["$(mol_names[t])_" * l for l in label_molecule(vs_template, equivs, elements)]
        names  = atom_names_from_elements(elements[vs_template], ELEMENT_TO_NAME)

        feat_mol = atom_features[:, vs_template]

        adj_mol = build_adj_list(g)

        ### Atom pooling and feature prediction ###
        embeds_mol = calc_embeddings(adj_mol, feat_mol, atom_embedding_model)
        logits     = nonbonded_selection_model(embeds_mol)  # (5, n_atoms)


        if epoch_n < anneal_first_epoch
            τ = T(tau_mult_first * τ_0)
            β_noise_min = τ_min
            β_noise = T(β_noise_min)
        else
            relative_epoch = epoch_n - anneal_first_epoch
            β_noise_min = τ_min
            τ = annealing_schedule(relative_epoch, τ_0, τ_min, decay_rate_τ)
            β_noise = annealing_schedule_β(relative_epoch, β_noise_min, τ_0, γ_anneal)
        end

        func_probs_mol = gumbel_softmax_symmetric(logits, labels, τ, β_noise)                # (5, n_atoms)

        feats_mol  = predict_atom_features(labels, embeds_mol, atom_features_models...; vdw_fnc_idx)

        ### Bonds pooling and feature prediction ###
        bond_feats_mol, bond_to_local_idx = predict_bond_features(g, labels, embeds_mol, bond_pooling_model, bond_features_model)

        ### Angle pooling and feature prediction ###
        angle_feats_mol, angle_triples, angle_to_local_idx = predict_angle_features(angles_i, angles_j, angles_k, vs_template, labels, embeds_mol, angle_pooling_model, angle_features_model)

        proper_feats_mol, improper_feats_mol, 
        torsion_to_key_proper, torsion_to_key_improper,
        unique_proper_keys, unique_improper_keys = predict_torsion_features(propers_i, propers_j, propers_k, propers_l,
                                                                            impropers_i, impropers_j, impropers_k, impropers_l,
                                                                            vs_template, labels, embeds_mol,
                                                                            proper_pooling_model, proper_features_model,
                                                                            improper_pooling_model, improper_features_model)


        ### Predict charges from atom features ###
        #charges_mol = atom_feats_to_charges(feats_mol, formal_charges[vs_template])

        ### Predict vdw params ###
        #vdw_mol = atom_feats_to_vdW(feats_mol)
        vdw_params_all = feats_mol[3:end, :]            # remove charge predictions
        vdw_mol = combine_vdw_params_gumbel(vdw_params_all)

        ### Predict bonds params ###
        bonds_mol = feats_to_bonds(bond_feats_mol)

        ### Predict angle feats ###
        angles_mol = feats_to_angles(angle_feats_mol)

        for (idx, vs_instance) in enumerate(all_indices)

            if graph_to_unique[idx] == t

                global_to_local = Dict(g => i for (i, g) in enumerate(vs_instance))

                ignore_derivatives() do 
                    for global_i in eachindex(elements)
                        local_i = get(global_to_local, global_i, nothing)
                        if !isnothing(local_i)
                            atom_types[global_i] *= labels[local_i]
                            atom_names[global_i] *= names[local_i]
                        end
                    end
                end

                charges_k1, charges_k2,
                func_probs,
                vdw_σ_lj,   vdw_ϵ_lj,
                vdw_σ_lj69, vdw_ϵ_lj69,
                vdw_σ_dexp, vdw_ϵ_dexp,
                vdw_σ_buff, vdw_ϵ_buff,
                vdw_A, vdw_B, vdw_C  = broadcast_atom_data!(charges_k1, feats_mol[1,:],
                                                            charges_k2, feats_mol[2,:],
                                                            func_probs, func_probs_mol,
                                                            vdw_σ_lj,   vdw_mol[1],
                                                            vdw_ϵ_lj,   vdw_mol[2],
                                                            vdw_σ_lj69, vdw_mol[3],
                                                            vdw_ϵ_lj69, vdw_mol[4],
                                                            vdw_σ_dexp, vdw_mol[5],
                                                            vdw_ϵ_dexp, vdw_mol[6],
                                                            vdw_σ_buff, vdw_mol[7],
                                                            vdw_ϵ_buff, vdw_mol[8],
                                                            vdw_A,      vdw_mol[9],
                                                            vdw_B,      vdw_mol[10],
                                                            vdw_C,      vdw_mol[11],
                                                            global_to_local)

                mapping = Dict(i => vs_instance[i] for i in eachindex(vs_instance))
                bond_global_to_local = Dict{Tuple{Int,Int}, Int}()
                for e in edges(g)
                    i, j = mapping[src(e)], mapping[dst(e)]
                    global_pair = (min(i, j), max(i, j))
                    local_pair = (min(src(e), dst(e)), max(src(e), dst(e)))
                    bond_global_to_local[global_pair] = bond_to_local_idx[local_pair]
                end

                bonds_k, bonds_r0, bonds_a = broadcast_bond_data!(bonds_k, bonds_r0, bonds_a, bonds_mol[1], bonds_mol[2], bonds_mol[3], bond_functional_form, bonds_i, bonds_j, bond_global_to_local)

                angle_global_to_local = Dict{Tuple{Int,Int,Int}, Int}()
                for (i, j, k) in angle_triples
                    gi, gj, gk = mapping[i], mapping[j], mapping[k]
                    angle_global_to_local[(gi, gj, gk)] = angle_to_local_idx[(i, j, k)]
                end

                angles_ki, angles_θ0i, angles_kj, angles_θ0j = broadcast_angle_data!(angles_ki, angles_θ0i, angles_kj, angles_θ0j, angles_mol[1], angles_mol[2], angles_mol[3], angles_mol[4], angle_functional_form, angles_i, angles_j, angles_k, angle_global_to_local)

                # Broadcast proper torsion features
                proper_feats = broadcast_proper_torsion_feats!(proper_feats, proper_feats_mol, propers_i, propers_j, propers_k, propers_l, vs_instance, mapping, torsion_to_key_proper, unique_proper_keys)

                # Broadcast improper torsion features
                improper_feats = broadcast_improper_torsion_feats!(improper_feats, improper_feats_mol, impropers_i, impropers_j, impropers_k, impropers_l, vs_instance, mapping, torsion_to_key_improper, unique_improper_keys)
            end
        end
    end

    torsion_ks_size = zero(T)
    if length(proper_feats) > 0
        torsion_ks_size += mean(abs, proper_feats)
    end
    if length(improper_feats) > 0
        torsion_ks_size += mean(abs, improper_feats)
    end

    # Why is this padding needed?
    proper_feats_pad   = cat(proper_feats, zeros(T, 6 - n_proper_terms, length(propers_i)); dims = 1)
    improper_feats_pad = cat(improper_feats, zeros(T, 6 - n_improper_terms, length(impropers_i)); dims = 1)

    partial_charges = atom_feats_to_charges(charges_k1, charges_k2, formal_charges)

    molly_sys, weights_vdw = build_sys(mol_id, 
    masses, atom_types, atom_names, mol_inds, coords, boundary, partial_charges,
    func_probs, 
    vdw_σ_lj,   vdw_ϵ_lj,
    vdw_σ_lj69, vdw_ϵ_lj69,
    vdw_σ_dexp, vdw_ϵ_dexp,
    vdw_σ_buff, vdw_ϵ_buff,
    vdw_A, vdw_B, vdw_C,
    α, β, δ, γ, 
    weight_lj, weight_coul,
    bond_functional_form, bonds_k, bonds_r0, bonds_a, angle_functional_form,
    angles_ki, angles_θ0i, angles_kj, angles_θ0j, bonds_i, bonds_j, angles_i, angles_j, angles_k, proper_feats_pad,
    improper_feats_pad, propers_i, propers_j, propers_k, propers_l, impropers_i, impropers_j, impropers_k,
    impropers_l)
    
    return (
        molly_sys,
        partial_charges,
        func_probs,
        weights_vdw,
        torsion_ks_size,
        elements,
        mol_inds
    )

end
