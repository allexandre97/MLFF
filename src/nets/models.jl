using Statistics
using Flux
using GraphNeuralNetworks

include("../physics/definitions.jl")

NET_PARAMS = MODEL_PARAMS["networks"]

Flux.@non_differentiable GraphNeuralNetworks.GNNGraph(args...)

########## SOME PARSING LOGIC ##########

# Define the activation and initialization functions
if NET_PARAMS["activation_str"] == "relu"
    const activation_gnn, activation_dense = relu, relu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif NET_PARAMS["activation_str"] == "leakyrelu"
    const activation_gnn, activation_dense = leakyrelu, leakyrelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif NET_PARAMS["activation_str"] == "gelu"
    const activation_gnn, activation_dense = gelu, gelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif NET_PARAMS["activation_str"] == "swish"
    const activation_gnn, activation_dense = swish, swish
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
else
    error("unknown activation $activation_str")
end

# Define the aggregation strategy
if NET_PARAMS["aggr_str"] == "mean"
    const aggr_gnn = mean
elseif NET_PARAMS["aggr_str"] == "sum"
    const aggr_gnn = +
elseif NET_PARAMS["aggr_str"] == "max"
    const aggr_gnn = max
else
    error("unknown aggregation function $aggr_str")
end

########## HELPER METHODS ##########

function gcn_conv(in_out_dims, activation=identity, init=Flux.glorot_uniform)
    #=
    Defines the convolutional layer of the GNN.
    Args:
        in_out_dims Key{Int}=>Val{Int} --> Input dimensions => output dimensions
        activation --> The activation function to use
        init --> The initialisation function to use
    Returns:
        GNN --> Grap Neural Network
    =#
    if NET_PARAMS["gcn_conv_layer"] == "GCNConv"
        return GCNConv(in_out_dims, activation; init=init)
    elseif NET_PARAMS["gcn_conv_layer"] == "GATv2Conv"
        return GATv2Conv(in_out_dims, activation; init=init)
    elseif NET_PARAMS["gcn_conv_layer"] == "SAGEConv"
        return SAGEConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif NET_PARAMS["gcn_conv_layer"] == "GATConv"
        return GATConv(in_out_dims, activation; init=init)
    elseif NET_PARAMS["gcn_conv_layer"] == "GraphConv"
        return GraphConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif NET_PARAMS["gcn_conv_layer"] == "ChebConv5" # Gives LAPACKException
        return ChebConv(in_out_dims, 5; init=init)
    else
        error("unknown graph convolutional layer $gcn_conv_layer")
    end
end

function generate_gnn_layers(n_layers)
    #=
    Generates a number of hidden layers for the GNN
    Args:
        n_layers --> The number of hidden layers to generate
    Returns:
        layers::Array{GNN} --> An array of hidden layers
    =#
    layers = []
    if dropout_gnn > 0
        push!(layers, Dropout(dropout_gnn))
    end
    for _ in 1:n_layers
        push!(layers, gcn_conv(dim_hidden_gnn => dim_hidden_gnn, activation_gnn, init_gnn))
        if dropout_gnn > 0
            push!(layers, Dropout(dropout_gnn))
        end
    end
    return layers
end

function generate_gnn_layers(n_layers)
    dropout_gnn = NET_PARAMS["dropout_dense"]
    layers = []
    if dropout_gnn > 0
        push!(layers, Dropout(dropout_gnn))
    end
    for _ in 1:n_layers
        push!(layers, gcn_conv(NET_PARAMS["dim_hidden_gnn"] => NET_PARAMS["dim_hidden_gnn"],
              activation_gnn, init_gnn))
        if dropout_gnn > 0
            push!(layers, Dropout(dropout_gnn))
        end
    end
    return layers
end

function generate_dense_layers(n_layers)
    dropout_dense = NET_PARAMS["dropout_dense"]
    layers = []
    if dropout_dense > 0
        push!(layers, Dropout(dropout_dense))
    end
    for _ in 1:n_layers
        push!(layers, Dense(NET_PARAMS["dim_hidden_dense"] => NET_PARAMS["dim_hidden_dense"],
                            activation_dense; init=init_dense))
        if dropout_dense > 0
            push!(layers, Dropout(dropout_dense))
        end
    end
    return layers
end

########## MODELS ##########

function build_models()
    # Pooling step
    atom_embedding_model = GNNChain(
        gcn_conv(NET_PARAMS["n_atom_features_in"] => NET_PARAMS["dim_hidden_gnn"],
                 activation_gnn, init_gnn),
        generate_gnn_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        gcn_conv(NET_PARAMS["dim_hidden_gnn"] => NET_PARAMS["dim_embed_atom"]),
    )

    bond_pooling_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] * 2 => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => NET_PARAMS["dim_embed_inter"]),
    )

    angle_pooling_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] * 3 => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => NET_PARAMS["dim_embed_inter"]),
    )

    proper_pooling_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] * 4 => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => NET_PARAMS["dim_embed_inter"]),
    )

    improper_pooling_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] * 4 => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => NET_PARAMS["dim_embed_inter"]),
    )

    #

    # Feature prediction step
    atom_features_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init = init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => 2 + n_vdw_atom_params)
    )

    bond_features_model = Chain(
        Dense(NET_PARAMS["dim_embed_inter"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => n_bonded_params),
    )

    angle_features_model = Chain(
        Dense(NET_PARAMS["dim_embed_inter"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => n_angle_params),
    )

    proper_features_model = Chain(
        Dense(NET_PARAMS["dim_embed_inter"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => n_proper_terms),
    )

    improper_features_model = Chain(
        Dense(NET_PARAMS["dim_embed_inter"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=init_dense),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => n_improper_terms),
    )

    models = [atom_embedding_model, bond_pooling_model, angle_pooling_model, proper_pooling_model, improper_pooling_model,
              atom_features_model, bond_features_model, angle_features_model, proper_features_model, improper_features_model]

    optims = [Flux.setup(Adam(NET_PARAMS["learning_rate"]), m) for m in models]

    return models, optims

end

function calc_embeddings(
    mol_id::String,
    adj_list::Vector{Vector{Int64}},
    atom_feats,
    atom_embedding_model::GNNChain,
    atom_features_model::Chain
)

    if any(startswith.(mol_id, ("vapourisation_", "mixing_"))) # Condensed phase

    elseif startswith(mol_id, "protein") # Unused for now
    
    else

        mol_graph       = GNNGraph(adj_list)
        atom_embeddings = atom_embedding_model(mol_graph, atom_feats)
        atom_features   = atom_features_model(atom_embeddings)

    end

    return atom_features, atom_embeddings

end

function embed_to_pool(
    atom_embeddings,
    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,
    bond_pooling_model::Chain,
    angle_pooling_model::Chain,
    proper_pooling_model::Chain,
    improper_pooling_model::Chain
)

    bond_emb_i,
    bond_emb_j = atom_embeddings[:, bonds_i],
                 atom_embeddings[:, bonds_j]

    angle_emb_i,
    angle_emb_j,
    angle_emb_k = atom_embeddings[:, angles_i, :],
                  atom_embeddings[:, angles_j, :],
                  atom_embeddings[:, angles_k, :]

    proper_emb_i,
    proper_emb_j,
    proper_emb_k,
    proper_emb_l = atom_embeddings[:, propers_i],
                   atom_embeddings[:, propers_j],
                   atom_embeddings[:, propers_k],
                   atom_embeddings[:, propers_l]

    improper_emb_i,
    improper_emb_j,
    improper_emb_k,
    improper_emb_l = atom_embeddings[:, impropers_i],
                     atom_embeddings[:, impropers_j],
                     atom_embeddings[:, impropers_k],
                     atom_embeddings[:, impropers_l]

    bond_com_emb_1 = cat(bond_emb_i, bond_emb_j; dims=1)
    bond_com_emb_2 = cat(bond_emb_j, bond_emb_i; dims=1)
    bond_pool_1    = bond_pooling_model(bond_com_emb_1)
    bond_pool_2    = bond_pooling_model(bond_com_emb_2)
    bond_pool = bond_pool_1 .+ bond_pool_2

    angle_com_emb_1 = cat(angle_emb_i, angle_emb_j, angle_emb_k; dims=1)
    angle_com_emb_2 = cat(angle_emb_k, angle_emb_j, angle_emb_i; dims=1)
    angle_pool_1    = angle_pooling_model(angle_com_emb_1)
    angle_pool_2    = angle_pooling_model(angle_com_emb_2)
    angle_pool = angle_pool_1 .+ angle_pool_2

    proper_com_emb_1 = cat(proper_emb_i, proper_emb_j, proper_emb_k, proper_emb_l; dims=1)
    proper_com_emb_2 = cat(proper_emb_l, proper_emb_k, proper_emb_j, proper_emb_i; dims=1)
    proper_pool_1    = proper_pooling_model(proper_com_emb_1)
    proper_pool_2    = proper_pooling_model(proper_com_emb_2)
    proper_pool = proper_pool_1 .+ proper_pool_2

    improper_com_emb_1 = cat(improper_emb_i, improper_emb_j, improper_emb_k, improper_emb_l; dims=1)
    improper_com_emb_2 = cat(improper_emb_i, improper_emb_k, improper_emb_j, improper_emb_l; dims=1)
    improper_com_emb_3 = cat(improper_emb_i, improper_emb_l, improper_emb_j, improper_emb_k; dims=1)
    improper_pool_1    = improper_pooling_model(improper_com_emb_1)
    improper_pool_2    = improper_pooling_model(improper_com_emb_2)
    improper_pool_3    = improper_pooling_model(improper_com_emb_3)
    improper_pool = improper_pool_1 .+ improper_pool_2 .+ improper_pool_3

    return (
        bond_pool,
        angle_pool,
        proper_pool,
        improper_pool
    )

end

function pool_to_feats(
    mol_id::String,
    n_repeats::Int,
    bond_pool,
    angle_pool,
    proper_pool,
    improper_pool,
    bond_features_model::Chain,
    angle_features_model::Chain,
    proper_features_model::Chain,
    improper_features_model::Chain
)
    if any(startswith.(mol_id, ("vapourisation_", "mixing_")))
        bond_feats     = repeat(bond_features_model(bond_pool), 1, n_repeats)
        angle_feats    = repeat(angle_features_model(angle_pool), 1, n_repeats)
        proper_feats   = repeat(proper_features_model(proper_pool), 1, n_repeats)
        improper_feats = repeat(improper_features_model(improper_pool), 1, n_repeats)
    
    elseif startswith(mol_id, "protein")
        #TODO: Add this magic
    else
        bond_feats     = bond_features_model(bond_pool)
        angle_feats    = angle_features_model(angle_pool)
        proper_feats   = proper_features_model(proper_pool)
        improper_feats = improper_features_model(improper_pool)
    end

    return(
        bond_feats,
        angle_feats,
        proper_feats,
        improper_feats
    )

end