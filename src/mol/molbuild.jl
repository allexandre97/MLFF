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
    mol_id::String,
    args...
)

    sys, partial_charges, vdw_size, torsion_size, elements, mol_inds = mol_to_system(mol_id, args...)
    neighbors = find_neighbors(sys; n_threads = 1)
    # Get interaction lists separate depending on the number of atoms involves
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
    if any(startswith.(mol_id, ("vapourisation_", "mixing_", "protein_")))
        forces = nothing
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    else
        forces = forces_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                             sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    end

    return sys, forces, potential, partial_charges, vdw_size, torsion_size, elements, mol_inds

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
    vdW_dict,
    bonds_dict,
    angles_dict,
    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    proper_feats, improper_feats,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l
)
    n_atoms  = length(partial_charges)

    dist_nb_cutoff = T(MODEL_PARAMS["physics"]["dist_nb_cutoff"])

    vdw    = vdW_dict["functional"]
    weight = vdW_dict["weight_vdw"]
    bond   = bonds_dict["functional"]
    angle  = angles_dict["functional"]

    ########## van der Waals section ##########
    if vdw in ("lj", "lj69", "dexp", "buff")
        σ = vdW_dict["σ"]
        ϵ = vdW_dict["ϵ"]
        atoms = [Atom(i, one(T), masses[i], partial_charges[i], σ[i], ϵ[i])
                 for i in 1:n_atoms]
        if vdw == "lj"
            inter_vdw = LennardJones(DistanceCutoff(dist_nb_cutoff),
                                     true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, weight)
        elseif vdw == "lj69"
            inter_vdw = Mie(6, 9, DistanceCutoff(dist_nb_cutoff),
                            true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, weight, 1)
        elseif vdw == "dexp"
            α = vdW_dict["α"]
            β = vdW_dict["β"]
            inter_vdw = DoubleExponential(α, β, σ_mixing, ϵ_mixing, weight, dist_nb_cutoff)
        elseif vdw == "buff"
            δ = vdW_dict["δ"]
            γ = vdW_dict["γ"]
            inter_vdw = Buffered147(δ, γ, σ_mixing, ϵ_mixing, weight, dist_nb_cutoff)
        end
    
    elseif vdw == "buck"

        A = vdW_dict["A"]
        B = vdW_dict["B"]
        C = vdW_dict["C"]

        atoms = [BuckinghamAtom(i, one(T), masses[i], partial_charges[i], A[i], B[i], C[i])
                 for i in 1:n_atoms]
        
        inter_vdw = Buckingham(weight, dist_nb_cutoff)

    elseif vdw == "nn"
        # TODO: Add functionality for NNet vdW interactions. See comment in 
        # relevant method of transformers.jl
    
    end

    ########## Coulomb interactions section ##########

    if vdw == "nn"
        # See previous TODO
    else
        weight_14_coul = sigmoid(global_params[2])
        
        if MODEL_PARAMS["physics"]["use_reaction_field"] &&
            any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_", "protein_")))
            
            inter_coulomb = CoulombReactionField(dist_nb_cutoff, T(Molly.crf_solvent_dielectric),
            true, weight_14_coul, T(ustrip(Molly.coulomb_const)))
        
        else

            inter_coulomb = Coulomb(DistanceCutoff(dist_nb_cutoff),
            true, weight_14_coul, T(ustrip(Molly.coulomb_const)))
        
        end

        pairwise_inter = (inter_vdw, inter_coulomb)

    end

    ########## Bond Interactions section ##########

    if bond == "harmonic"
        bond_inter = HarmonicBond.(T.(bonds_dict["k"]), T.(bonds_dict["r0"]))
    elseif bond == "morse"
        bond_inter = MorseBond.(T.(bonds_dict["k"]), T.(bonds_dict["a"]), T.(bonds_dict["r0"]))
    end
    bonds = InteractionList2Atoms(bonds_i, bonds_j, bond_inter)

    ########## Angle Interactions section ##########

    if angle == "harmonic"
        angle_inter = HarmonicAngle.(T.(angles_dict["k"]), T.(angles_dict["θ0"]))
    elseif angle == "ub"
        # TODO: I think this is not yet defined. Check with JG and train.jl script
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
                 typeof(NoUnits), T, Vector{T}, Nothing}(
        atoms, coords, boundary, velocities, atoms_data, topo, pairwise_inter, specific_inter_lists,
        (), (), neighbor_finder, (), 1, NoUnits, NoUnits, one(T), zeros(T, n_atoms), nothing)

    return sys

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

function mol_to_system(
    mol_id::String,
    feat_df::DataFrame,
    coords,
    boundary::CubicBoundary{Float32},
    atom_embedding_model::GNNChain,
    bond_pooling_model::Chain,
    angle_pooling_model::Chain,
    proper_pooling_model::Chain,
    improper_pooling_model::Chain,
    atom_features_model::Chain,
    bond_features_model::Chain,
    angle_features_model::Chain,
    proper_features_model::Chain,
    improper_features_model::Chain
)

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
    unique_graphs, unique_indices, counts = filter_unique(all_graphs, all_indices)

    # Prediction arrays
    partial_charges = zeros(T, n_atoms)
    
    vdw_functional_form = MODEL_PARAMS["physics"]["vdw_functional_form"]
    if vdw_functional_form == "lj"
        vdw_dict = Dict(
            "functional" => vdw_functional_form,
            "params_size" => zero(T),
            "weight_vdw" => zero(T),
            "σ" => zeros(T, n_atoms),
            "ϵ" => zeros(T, n_atoms)
        )
    elseif vdw_functional_form == "lj69"
        vdw_dict = Dict(
            "functional" => vdw_functional_form,
            "params_size" => zero(T),
            "weight_vdw" => zero(T),
            "σ" => zeros(T, n_atoms),
            "ϵ" => zeros(T, n_atoms)
        )
    elseif vdw_functional_form == "dexp"
        vdw_dict = Dict(
            "functional" => vdw_functional_form,
            "params_size" => zero(T),
            "weight_vdw" => zero(T),
            "σ" => Vector{T}(undef, n_atoms),
            "ϵ" => Vector{T}(undef, n_atoms),
            "α" => zeros(T, n_atoms),
            "β" => zeros(T, n_atoms)
        )
    elseif vdw_functional_form == "buff"
        vdw_dict = Dict(
            "functional" => vdw_functional_form,
            "params_size" => zero(T),
            "weight_vdw" => zero(T),
            "σ" => Vector{T}(undef, n_atoms),
            "ϵ" => Vector{T}(undef, n_atoms),
            "δ" => zeros(T, n_atoms),
            "γ" => zeros(T, n_atoms)
        )
    elseif vdw_functional_form == "buck"
        vdw_dict = Dict(
            "functional" => vdw_functional_form,
            "params_size" => zero(T),
            "weight_vdw" => zero(T),
            "A" => zeros(T, n_atoms),
            "B" => zeros(T, n_atoms),
            "C" => zeros(T, n_atoms)
        )
    end

    bond_functional_form = MODEL_PARAMS["physics"]["bond_functional_form"]
    if bond_functional_form == "harmonic"
        bonds_dict = Dict(
            "functional" => bond_functional_form,
            "k"  => zeros(T, length(bonds_i)),
            "r0" => zeros(T, length(bonds_i))
        )
    elseif bond_functional_form == "morse"
        bonds_dict = Dict(
            "functional" => bond_functional_form,
            "k"  => zeros(T, length(bonds_i)),
            "r0" => zeros(T, length(bonds_i)),
            "a"  => zeros(T, length(bonds_i))
        )
    end

    angle_functional_form = MODEL_PARAMS["physics"]["angle_functional_form"]
    if angle_functional_form == "harmonic"
        angles_dict = Dict(
            "functional" => angle_functional_form,
            "k" => zeros(T, length(angles_i)),
            "θ0" => zeros(T, length(angles_i))
        )
    elseif angle_functional_form == "ub"
        angles_dict = Dict(
            "functional" => angle_functional_form,
            "ki" => zeros(T, length(angles_i)),
            "θ0i" => zeros(T, length(angles_i)),
            "kj" => zeros(T, length(angles_i)),
            "θ0j" => zeros(T, length(angles_i))
        )
    end

    proper_feats   = zeros(T, (n_proper_terms, length(propers_i)))
    improper_feats = zeros(T, (n_improper_terms, length(impropers_i)))

    if any(startswith.(mol_id, ("vapourisation_", "mixing_")))

        ignore_derivatives() do
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
        end

    else

        if occursin("water", mol_id)
            mol_names = ["water"]
        end

    end

    for (t, (g, vs_template)) in enumerate(zip(unique_graphs, unique_indices))

        equivs = find_atom_equivalences(g, vs_template, elements)
        labels = ["$(mol_names[t])_" * l for l in label_molecule(vs_template, equivs, elements)]
        names  = atom_names_from_elements(elements[vs_template], ELEMENT_TO_NAME)

        feat_mol = atom_features[:, vs_template]

        adj_mol = build_adj_list(g)

        ### Atom pooling and feature prediction ###

        _, embeds_mol = calc_embeddings(adj_mol, feat_mol, atom_embedding_model, atom_features_model)

        label_to_index = Dict{String, Int}()
        for (i, label) in enumerate(labels)
            if !haskey(label_to_index, label)
                label_to_index[label] = i
            end
        end
        unique_label_indices = ignore_derivatives() do
            return collect(values(label_to_index))
        end
        unique_embeds = embeds_mol[:, unique_label_indices]
        unique_feats  = atom_features_model(unique_embeds)

        feats_mol = map(labels) do label
            return unique_feats[:, label_to_index[label]]
        end
        feats_mol = hcat(feats_mol...)

        ### Bonds pooling and feature prediction ###

        # First we create a dict that converts bonds represented as indices as bonds represented by molecule type
        bond_to_key = Dict{Tuple{Int,Int}, Tuple{String,String}}()
        bond_to_local_idx = Dict{Tuple{Int,Int}, Int}()
        
        edges_list = [e for e in edges(g)]
        bond_key = map(1:length(edges_list)) do k
            e = edges_list[k]
            u, v = src(e), dst(e)
            bond_to_local_idx[(min(u,v), max(u,v))] = k
            lu, lv = labels[u], labels[v]
            key = lu < lv ? (lu, lv) : (lv, lu)
            bond_to_key[(min(u,v), max(u,v))] = key
            return key
        end

        # Then we get the unique bonds represented by atom type
        unique_keys = Dict{Tuple{String,String}, Int}()
        unique_bond_keys = Tuple{String,String}[]
        ignore_derivatives() do
            for key in bond_key
                if !haskey(unique_keys, key)
                    push!(unique_bond_keys, key)
                    unique_keys[key] = length(unique_keys) + 1
                end
            end
        end

        # We pass only the unique bonds to the pooling model
        emb_i = embeds_mol[:, [findfirst(==(l), labels) for (l, _) in unique_bond_keys]]
        emb_j = embeds_mol[:, [findfirst(==(l), labels) for (_, l) in unique_bond_keys]]
        bond_pool_1 = bond_pooling_model(cat(emb_i, emb_j; dims=1))
        bond_pool_2 = bond_pooling_model(cat(emb_j, emb_i; dims=1))
        bond_pool = bond_pool_1 .+ bond_pool_2 # Bond symmetry preserved

        # Predict features 
        unique_bond_feats = bond_features_model(bond_pool)
        bond_feats_mol = map(1:length(edges(g))) do k
            e = [_ for _  in edges(g)][k]
            u, v = src(e), dst(e)
            key = bond_to_key[(min(u,v), max(u,v))]
            idx = unique_keys[key]
            return unique_bond_feats[:,idx]
        end
        bond_feats_mol = hcat(bond_feats_mol...)

        ### Angle Feature Pooling ###
        angle_to_key = Dict{NTuple{3,Int}, NTuple{3,String}}()
        angle_triples = [(i,j,k) for (i,j,k) in zip(angles_i, angles_j, angles_k) if i in vs_template && j in vs_template && k in vs_template]
        
        
        # Map triplets from whole system indexing to local molecule indexing
        local_map = Dict(glo => loc for (loc, glo) in enumerate(vs_template))
        angle_triples = [(local_map[i], local_map[j], local_map[k]) for (i,j,k) in angle_triples]
        
        # From index to molecule type
        angle_to_local_idx = Dict{Tuple{Int,Int,Int}, Int}()
        angle_key = Tuple{String,String,String}[]
        ignore_derivatives() do
            for (idx, (i, j, k)) in enumerate(angle_triples)
                angle_to_local_idx[(i,j,k)] = idx
                li, lj, lk = labels[i], labels[j], labels[k]
                key = (li, lj, lk) < (lk, lj, li) ? (li, lj, lk) : (lk, lj, li)
                push!(angle_key, key)
                angle_to_key[(i,j,k)] = key
            end
        end

        # Get unique representation by molecule type
        unique_angle_keys = Dict{NTuple{3,String}, Int}()
        angle_key_order = NTuple{3,String}[]
        ignore_derivatives() do 
            for key in angle_key
                if !haskey(unique_angle_keys, key)
                    push!(angle_key_order, key)
                    unique_angle_keys[key] = length(unique_angle_keys) + 1
                end
            end
        end

        # Get features for just the unique angles
        angle_emb_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _) in angle_key_order]]
        angle_emb_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _) in angle_key_order]]
        angle_emb_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk) in angle_key_order]]

        # Symmetry preserving pooling
        angle_com_emb_1 = cat(angle_emb_i, angle_emb_j, angle_emb_k; dims=1)
        angle_com_emb_2 = cat(angle_emb_k, angle_emb_j, angle_emb_i; dims=1)
        angle_pool_1 = angle_pooling_model(angle_com_emb_1)
        angle_pool_2 = angle_pooling_model(angle_com_emb_2)

        # Get features
        angle_pool = angle_pool_1 .+ angle_pool_2
        unique_angle_feats = angle_features_model(angle_pool)

        # Broadcast from unique bonds to whole molecule
        angle_feats_mol = map(1:length(angle_triples)) do idx
            ijk = angle_triples[idx]
            key = angle_to_key[ijk]
            key_idx = unique_angle_keys[key]
            return unique_angle_feats[:, key_idx]
        end
        angle_feats_mol = hcat(angle_feats_mol...)

        ### Torsion Feature Pooling ###
        torsion_to_key_proper = Dict{NTuple{4,Int}, NTuple{4,String}}()
        torsion_to_key_improper = Dict{NTuple{4,Int}, NTuple{4,String}}()

        # Get global indices that appear in molecular template indices
        torsion_proper_quads = [(i,j,k,l) for (i,j,k,l) in zip(propers_i, propers_j, propers_k, propers_l) if i in vs_template && j in vs_template && k in vs_template && l in vs_template]
        torsion_improper_quads = [(i,j,k,l) for (i,j,k,l) in zip(impropers_i, impropers_j, impropers_k, impropers_l) if i in vs_template && j in vs_template && k in vs_template && l in vs_template]

        # Map indices from global to local indexing
        local_map = Dict(glo => loc for (loc, glo) in enumerate(vs_template))
        torsion_proper_quads = [(local_map[i], local_map[j], local_map[k], local_map[l]) for (i,j,k,l) in torsion_proper_quads]
        torsion_improper_quads = [(local_map[i], local_map[j], local_map[k], local_map[l]) for (i,j,k,l) in torsion_improper_quads]

        # From indices to atom types
        torsion_key_proper = map(torsion_proper_quads) do quad
            i,j,k,l = quad
            li, lj, lk, ll = labels[i], labels[j], labels[k], labels[l]
            key = (li, lj, lk, ll) < (ll, lk, lj, li) ? (li, lj, lk, ll) : (ll, lk, lj, li)
            torsion_to_key_proper[(i,j,k,l)] = key
            return key
        end

        torsion_key_improper = map(torsion_improper_quads) do quad
            i,j,k,l = quad
            li, lj, lk, ll = labels[i], labels[j], labels[k], labels[l]
            key = (li, lj, lk, ll)
            torsion_to_key_improper[(i,j,k,l)] = key
            return key
        end

        # We get the unique torsions depending on atom type
        unique_proper_keys = Dict{NTuple{4,String}, Int}()
        unique_improper_keys = Dict{NTuple{4,String}, Int}()
        proper_key_order = NTuple{4,String}[]
        improper_key_order = NTuple{4,String}[]
        ignore_derivatives() do 
            for key in torsion_key_proper
                if !haskey(unique_proper_keys, key)
                    push!(proper_key_order, key)
                    unique_proper_keys[key] = length(unique_proper_keys) + 1
                end
            end

            for key in torsion_key_improper
                if !haskey(unique_improper_keys, key)
                    push!(improper_key_order, key)
                    unique_improper_keys[key] = length(unique_improper_keys) + 1
                end
            end
        end

        # Symmetry preserving pooling

        prop_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _, _) in proper_key_order]]
        prop_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _, _) in proper_key_order]]
        prop_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk, _) in proper_key_order]]
        prop_l = embeds_mol[:, [findfirst(==(ll), labels) for (_, _, _, ll) in proper_key_order]]

        prop_1 = cat(prop_i, prop_j, prop_k, prop_l; dims=1)
        prop_2 = cat(prop_l, prop_k, prop_j, prop_i; dims=1)
        proper_pool = proper_pooling_model(prop_1) .+ proper_pooling_model(prop_2)
        unique_proper_feats = proper_features_model(proper_pool)

        imp_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _, _) in improper_key_order]]
        imp_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _, _) in improper_key_order]]
        imp_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk, _) in improper_key_order]]
        imp_l = embeds_mol[:, [findfirst(==(ll), labels) for (_, _, _, ll) in improper_key_order]]

        imp_1 = cat(imp_i, imp_j, imp_k, imp_l; dims=1)
        imp_2 = cat(imp_i, imp_k, imp_j, imp_l; dims=1)
        imp_3 = cat(imp_i, imp_l, imp_j, imp_k; dims=1)
        improper_pool = improper_pooling_model(imp_1) .+ improper_pooling_model(imp_2) .+ improper_pooling_model(imp_3)
        unique_improper_feats = improper_features_model(improper_pool)

        # Broadcast from unique torsions to whole molecule
        proper_feats_mol = map(1:length(torsion_proper_quads)) do idx
            quad    = torsion_proper_quads[idx]
            key     = torsion_to_key_proper[quad]
            key_idx = unique_proper_keys[key]
            return unique_proper_feats[:, key_idx]
        end
        proper_feats_mol = hcat(proper_feats_mol...)

        improper_feats_mol = map(1:length(torsion_improper_quads)) do idx
            quad    = torsion_improper_quads[idx]
            key     = torsion_to_key_improper[quad]
            key_idx = unique_improper_keys[key]
            return unique_improper_feats[:, key_idx]
        end
        improper_feats_mol = hcat(improper_feats_mol...)

        ### Predict charges from atom features ###
        charges_mol = atom_feats_to_charges(feats_mol, formal_charges[vs_template])

        ### Predict vdw params ###
        vdw_mol = atom_feats_to_vdW(feats_mol)
        
        ### Predict bonds params ###
        bonds_mol = feats_to_bonds(bond_feats_mol)

        ### Predict angle feats ###
        angles_mol = feats_to_angles(angle_feats_mol)

        for (g2, vs_instance) in zip(all_graphs, all_indices)
            
            if has_isomorph(g, g2, VF2())
                
                atom_data = map(1:n_atoms) do global_i
                    local_i = findfirst(x -> x == global_i, vs_instance)
                    charge_return = isnothing(local_i) ? zero(T) : charges_mol[local_i]

                    name_return = isnothing(local_i) ? "" : names[local_i]
                    type_return = isnothing(local_i) ? "" : labels[local_i]

                    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
                        vdw_return_1 = isnothing(local_i) ? zero(T) : vdw_mol["σ"][local_i]
                        vdw_return_2 = isnothing(local_i) ? zero(T) : vdw_mol["ϵ"][local_i]
                        vdw_returns = (vdw_return_1, vdw_return_2)
                    else
                        vdw_return_1 = isnothing(local_i) ? zero(T) : vdw_mol["A"][local_i]
                        vdw_return_2 = isnothing(local_i) ? zero(T) : vdw_mol["B"][local_i]
                        vdw_return_3 = isnothing(local_i) ? zero(T) : vdw_mol["C"][local_i]
                        vdw_returns = (vdw_return_1, vdw_return_2, vdw_return_3)
                    end

                    return charge_return, name_return, type_return, vdw_returns...
                end

                partial_charges = partial_charges .+ [d[1] for d in atom_data]
                atom_names      = atom_names      .* [d[2] for d in atom_data]
                atom_types      = atom_types      .* [d[3] for d in atom_data]
                
                if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
                    vdw_dict["σ"] = vdw_dict["σ"] .+ [d[4] for d in atom_data]
                    vdw_dict["ϵ"] = vdw_dict["ϵ"] .+ [d[5] for d in atom_data]
                    if vdw_functional_form == "dexp"
                        vdw_dict["α"] = vdw_mol["α"]
                        vdw_dict["β"] = vdw_mol["β"]
                    elseif vdw_functional_form == "buff"
                        vdw_dict["δ"] = vdw_mol["δ"]
                        vdw_dict["γ"] = vdw_mol["γ"]
                    end
                else
                    vdw_dict["A"] = vdw_dict["A"] .+ [d[4] for d in atom_data]
                    vdw_dict["B"] = vdw_dict["B"] .+ [d[5] for d in atom_data]
                    vdw_dict["C"] = vdw_dict["C"] .+ [d[6] for d in atom_data]
                end

                mapping = Dict(i => vs_instance[i] for i in 1:length(vs_instance))
                bond_global_to_local = Dict{Tuple{Int,Int}, Int}()
                for e in edges(g)
                    i, j = mapping[src(e)], mapping[dst(e)]
                    global_pair = (min(i, j), max(i, j))
                    local_pair = (min(src(e), dst(e)), max(src(e), dst(e)))
                    bond_global_to_local[global_pair] = bond_to_local_idx[local_pair]
                end

                bond_data = map(1:length(bonds_i)) do idx
                    bond = (bonds_i[idx], bonds_j[idx])
                    if !haskey(bond_global_to_local, bond)
                        if bond_functional_form == "harmonic"
                            return zero(T), zero(T)
                        elseif bond_functional_form == "morse"
                            return zero(T), zero(T), zero(T)
                        end
                    else
                        local_i = bond_global_to_local[bond]
                        if bond_functional_form == "harmonic"
                            return_k = bonds_mol["k"][local_i]
                            return_r = bonds_mol["r0"][local_i]
                            return_params = (return_k, return_r)
                        elseif bond_functional_form == "morse"
                            return_k = bonds_mol["k"][local_i]
                            return_r = bonds_mol["r0"][local_i]
                            return_a = bonds_mol["a"][local_i]
                            return_params = (return_k, return_r, return_a)
                        end
                        return return_params
                    end
                end
                if bond_functional_form == "harmonic"
                    bonds_dict["k"]  = bonds_dict["k"]  .+ [d[1] for d in bond_data]
                    bonds_dict["r0"] = bonds_dict["r0"] .+ [d[2] for d in bond_data]
                elseif bond_functional_form == "morse"
                    bonds_dict["k"]  = bonds_dict["k"]  .+ [d[1] for d in bond_data]
                    bonds_dict["r0"] = bonds_dict["r0"] .+ [d[2] for d in bond_data]
                    bonds_dict["a"] = bonds_dict["a"] .+ [d[3] for d in bond_data]
                end

                angle_global_to_local = Dict{Tuple{Int,Int,Int}, Int}()
                for (i, j, k) in angle_triples
                    gi, gj, gk = mapping[i], mapping[j], mapping[k]
                    angle_global_to_local[(gi, gj, gk)] = angle_to_local_idx[(i, j, k)]
                end

                angle_data = map(1:length(angles_i)) do idx
                    angle = (angles_i[idx], angles_j[idx], angles_k[idx])
                    if !haskey(angle_global_to_local, angle)
                        if angle_functional_form == "harmonic"
                            return zero(T), zero(T)
                        elseif angle_functional_form == "ub"
                            return zero(T), zero(T), zero(T), zero(T)
                        end
                    else
                        local_i = angle_global_to_local[angle]
                        if angle_functional_form == "harmonic"
                            return_k = angles_mol["k"][local_i]
                            return_θ = angles_mol["θ0"][local_i]
                            return_params = (return_k, return_θ)
                        elseif angle_functional_form == "ub"
                            return_ki = angles_mol["ki"][local_i]
                            return_θi = angles_mol["θ0i"][local_i]
                            return_kj = angles_mol["kj"][local_i]
                            return_θj = angles_mol["θ0j"][local_i]
                            return_params = (return_ki, return_θi, return_kj, return_θj)
                        end
                        return return_params
                    end
                end
                if angle_functional_form == "harmonic"
                    angles_dict["k"]  = angles_dict["k"]  .+ [d[1] for d in angle_data]
                    angles_dict["θ0"] = angles_dict["θ0"] .+ [d[2] for d in angle_data]
                elseif angle_functional_form == "ub"
                    angles_dict["ki"]  = angles_dict["ki"]  .+ [d[1] for d in angle_data]
                    angles_dict["θ0i"] = angles_dict["θ0i"] .+ [d[2] for d in angle_data]
                    angles_dict["kj"]  = angles_dict["kj"]  .+ [d[3] for d in angle_data]
                    angles_dict["θ0j"] = angles_dict["θ0j"] .+ [d[4] for d in angle_data]
                end

                # Broadcast proper torsion features
                proper_data = map(1:length(propers_i)) do idx
                    global_quad = (propers_i[idx], propers_j[idx], propers_k[idx], propers_l[idx])
                    feat = zeros(T, size(proper_feats_mol, 1))
                    if all(x -> haskey(mapping, x), global_quad)
                        local_quad = (findfirst(==(global_quad[1]), vs_instance),
                                    findfirst(==(global_quad[2]), vs_instance),
                                    findfirst(==(global_quad[3]), vs_instance),
                                    findfirst(==(global_quad[4]), vs_instance))
                        if all(!isnothing, local_quad) && haskey(torsion_to_key_proper, local_quad)
                            key = torsion_to_key_proper[local_quad]
                            if haskey(unique_proper_keys, key)
                                idx_feat = unique_proper_keys[key]
                                feat = proper_feats_mol[:, idx_feat]
                            end
                        end
                    end
                    return feat
                end
                if !isempty(proper_data) 
                    proper_feats += hcat(proper_data...)
                end

                # Broadcast improper torsion features
                improper_data = map(1:length(impropers_i)) do idx
                    global_quad = (impropers_i[idx], impropers_j[idx], impropers_k[idx], impropers_l[idx])
                    feat = zeros(T, size(improper_feats_mol, 1))
                    if all(x -> haskey(mapping, x), global_quad)
                        local_quad = (findfirst(==(global_quad[1]), vs_instance),
                                    findfirst(==(global_quad[2]), vs_instance),
                                    findfirst(==(global_quad[3]), vs_instance),
                                    findfirst(==(global_quad[4]), vs_instance))
                        if all(!isnothing, local_quad) && haskey(torsion_to_key_improper, local_quad)
                            key = torsion_to_key_improper[local_quad]
                            if haskey(unique_improper_keys, key)
                                idx_feat = unique_improper_keys[key]
                                feat = improper_feats_mol[:, idx_feat]
                            end
                        end
                    end
                    return feat
                end
                if !isempty(improper_data) 
                    improper_feats += hcat(improper_data...)
                end

            end
        end
    end

    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
        vdw_dict["params_size"] = T(0.5*(mean(vdw_dict["σ"]) + mean(vdw_dict["ϵ"])))
    else
        vdw_dict["params_size"] = zero(T)
    end
    vdw_dict["weight_vdw"] = (vdw_functional_form == "lj" ? sigmoid(global_params[1]) : one(T))

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

    molly_sys = build_sys(mol_id,
                          masses,
                          atom_types,
                          atom_names,
                          mol_inds,
                          coords,
                          boundary,
                          partial_charges,
                          vdw_dict,
                          bonds_dict, 
                          angles_dict, 
                          bonds_i, bonds_j,
                          angles_i, angles_j, angles_k,
                          proper_feats_pad, improper_feats_pad,
                          propers_i, propers_j, propers_k, propers_l,
                          impropers_i, impropers_j, impropers_k, impropers_l)

    return (
        molly_sys,
        partial_charges, 
        vdw_dict["params_size"],
        torsion_ks_size,
        elements,
        mol_inds
    )

end