include("../nets/models.jl")
include("../physics/transformer.jl")

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

function mol_to_preds(
    mol_id::String,
    args...
)

    mol_to_system(mol_id, args...)

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

    # The total number of molecules in a conformation
    n_mols    = maximum(mol_inds)
    n_repeats = (startswith(mol_id, "mixing_combined_") ? (n_mols ÷ 2) : n_mols)

    atom_feats, atom_embeds = calc_embeddings(mol_id, adj_list, atom_features,
                                              atom_embedding_model, atom_features_model)
    
    bond_pool, angle_pool, proper_pool, improper_pool = embed_to_pool(atom_embeds,
                                                                      bonds_i, bonds_j,
                                                                      angles_i, angles_j, angles_k,
                                                                      propers_i, propers_j, propers_k, propers_l,
                                                                      impropers_i, impropers_j, impropers_k, impropers_l,
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

    

end