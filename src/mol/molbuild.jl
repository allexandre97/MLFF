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

Flux.@non_differentiable build_adj_list(mol_row::DataFrame)

#= function connected_components(adjacency)
    visited = falses(length(adjacency))
    components = []

    for i in 1:length(adjacency)
        if !visited[i]
            queue = [i]
            component = Int[]
            while !isempty(queue)
                node = popfirst!(queue)
                if !visited[node]
                    visited[node] = true
                    push!(component, node)
                    append!(queue, setdiff(adjacency[node], component))
                end
            end
            push!(components, sort(component))
        end
    end
    return components
end =#

#= Flux.@non_differentiable connected_components(args...) =#

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

    # Read the relevant features and store them in vectors
    
    elements, formal_charges,
    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,
    mol_inds, adj_list, n_atoms, atom_features = decode_feats(feat_df)

    masses = [NAME_TO_MASS[ELEMENT_TO_NAME[e]] for e in elements]

    # The total number of molecules in a conformation
    n_mols    = maximum(mol_inds)
    n_repeats = (startswith(mol_id, "mixing_combined_") ? (n_mols ÷ 2) : n_mols)

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

        n_elements_rep  = length(elements   ) ÷ n_repeats
        n_bonds_rep     = length(bonds_i    ) ÷ n_repeats
        n_angles_rep    = length(angles_i   ) ÷ n_repeats
        n_propers_rep   = length(propers_i  ) ÷ n_repeats
        n_impropers_rep = length(impropers_i) ÷ n_repeats
    else

        if occursin("water", mol_id)
            mol_names = ["water"]
        end

        n_elements_rep  = length(elements   )
        n_bonds_rep     = length(bonds_i    )
        n_angles_rep    = length(angles_i   )
        n_propers_rep   = length(propers_i  )
        n_impropers_rep = length(impropers_i)
    end

    
    atom_types = String[]
    atom_names = String[]
    ignore_derivatives() do
        # We build a graph for the whole system
        global_graph = build_global_graph(length(elements), [(Int(i),Int(j)) for (i, j) in zip(bonds_i, bonds_j)])
        # We split it in subgraphs where each one represents a given molecule
        all_graphs, all_indices = extract_all_subgraphs(global_graph)
        # We find unique molecule types
        unique_graphs, unique_indices, counts = filter_unique(all_graphs, all_indices)
        # for each type: find equivs, then label
        for (t, (g, vs)) in enumerate(zip(unique_graphs, unique_indices))
            equivs = find_atom_equivalences(g, vs, elements)
            labels = label_molecule(vs, equivs, elements)
            names  = atom_names_from_elements(elements[vs], ELEMENT_TO_NAME)
            labels = repeat(["$(mol_names[t])_"*label for label in labels], counts[t])
            names  = repeat(names, counts[t])
            push!(atom_types, labels...)
            push!(atom_names, names...)
        end
    end

    atom_feats, atom_embeds = calc_embeddings(mol_id, adj_list, atom_features,
                                              atom_embedding_model, atom_features_model, n_atoms, n_repeats)
    
    bond_pool, angle_pool, proper_pool, improper_pool = embed_to_pool(atom_embeds,
                                                                      bonds_i[1:n_bonds_rep], bonds_j[1:n_bonds_rep],
                                                                      angles_i[1:n_angles_rep], angles_j[1:n_angles_rep], angles_k[1:n_angles_rep],
                                                                      propers_i[1:n_propers_rep], propers_j[1:n_propers_rep], propers_k[1:n_propers_rep], propers_l[1:n_propers_rep],
                                                                      impropers_i[1:n_impropers_rep], impropers_j[1:n_impropers_rep], impropers_k[1:n_impropers_rep], impropers_l[1:n_impropers_rep],
                                                                      bond_pooling_model, 
                                                                      angle_pooling_model, 
                                                                      proper_pooling_model, 
                                                                      improper_pooling_model)

    bond_feats, angle_feats, proper_feats, improper_feats = pool_to_feats(mol_id,
                                                                          n_repeats,
                                                                          bond_pool,
                                                                          angle_pool,
                                                                          proper_pool,
                                                                          improper_pool,
                                                                          bond_features_model,
                                                                          angle_features_model,
                                                                          proper_features_model,
                                                                          improper_features_model)

    partial_charges = atom_feats_to_charges(mol_id, n_atoms, n_mols, atom_feats, formal_charges, mol_inds)

    vdw_dict    = atom_feats_to_vdW(atom_feats)
    bonds_dict  = feats_to_bonds(bond_feats)
    angles_dict = feats_to_angles(angle_feats)

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