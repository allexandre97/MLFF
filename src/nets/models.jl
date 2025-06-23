using Flux
using GraphNeuralNetworks

if MODEL_PARAMS["activation_str"] == "relu"
    const activation_gnn, activation_dense = relu, relu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif MODEL_PARAMS["activation_str"] == "leakyrelu"
    const activation_gnn, activation_dense = leakyrelu, leakyrelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif MODEL_PARAMS["activation_str"] == "gelu"
    const activation_gnn, activation_dense = gelu, gelu
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
elseif MODEL_PARAMS["activation_str"] == "swish"
    const activation_gnn, activation_dense = swish, swish
    const init_gnn, init_dense = Flux.kaiming_uniform, Flux.kaiming_uniform
else
    error("unknown activation $activation_str")
end

function gcn_conv(in_out_dims, activation=identity, init=Flux.glorot_uniform)
    if gcn_conv_layer == "GCNConv"
        return GCNConv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "GATv2Conv"
        return GATv2Conv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "SAGEConv"
        return SAGEConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif gcn_conv_layer == "GATConv"
        return GATConv(in_out_dims, activation; init=init)
    elseif gcn_conv_layer == "GraphConv"
        return GraphConv(in_out_dims, activation; init=init, aggr=aggr_gnn)
    elseif gcn_conv_layer == "ChebConv5" # Gives LAPACKException
        return ChebConv(in_out_dims, 5; init=init)
    else
        error("unknown graph convolutional layer $gcn_conv_layer")
    end
end

atom_embedding_model = GNNChain(
    gcn_conv(MODEL_PARAMS["n_atom_features_in"] => MODEL_PARAMS["dim_hidden_gnn"],
             activation_gnn, init_gnn)
)