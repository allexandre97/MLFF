using Statistics
using Flux
using GraphNeuralNetworks

include("../physics/definitions.jl")

const NET_PARAMS = MODEL_PARAMS["networks"]

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

    return [atom_embedding_model, bond_pooling_model, angle_pooling_model,
            atom_features_model, bond_features_model, angle_features_model]

end