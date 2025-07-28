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

# Not sharing
PARAM_LAYOUT = Dict(
    :lj   => (1:2,  [:σ, :ϵ]),
    :lj69 => (3:4,  [:σ, :ϵ]),
    :dexp => (5:6,  [:σ, :ϵ]),
    :buff => (7:8,  [:σ, :ϵ]),
    :buck => (9:11, [:A, :B, :C]),
)

CHOICE_TO_VDW = Dict(
    1 => "lj",
    2 => "lj69",
    3 => "dexp",
    4 => "buff",
    5 => "buck"
)


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
    nonbonded_selection_model = Chain(
        Dense(NET_PARAMS["dim_embed_atom"] => NET_PARAMS["dim_hidden_dense"],
              activation_dense; init=Flux.ones32),
        generate_dense_layers(NET_PARAMS["n_layers_nn"] - 2)...,
        Dense(NET_PARAMS["dim_hidden_dense"] => 5; init = Flux.ones32)
    )

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
              nonbonded_selection_model,
              atom_features_model, bond_features_model, angle_features_model, proper_features_model, improper_features_model,
              model_global_params]

    optims = [Flux.setup(Adam(NET_PARAMS["learning_rate"]), m) for m in models]

    return models, optims

end

annealing_schedule(relative_epoch, τ_0, τ_min, decay_rate) = T(max(τ_min, (τ_0 - τ_min) * exp(-decay_rate * relative_epoch) + τ_min))
annealing_schedule_β(relative_epoch, β_min, τ, γ) = T(τ * (1.0 - exp(-γ * relative_epoch) + β_min))

function gumbel_softmax_symmetric(
    logits::Matrix{T},
    labels::Vector{String}, 
    τ::T = T(1e-1),
    β::T = T(1.0))
    n_forms, n_atoms = size(logits)
    
    noise = ignore_derivatives() do
        noise = zeros(T, n_forms, n_atoms)
        # Group atoms by label
        label_to_inds = Dict{String, Vector{Int}}()
        for (i, label) in enumerate(labels)
            push!(get!(label_to_inds, label, Int[]), i)
        end

        # Assign same noise to all atoms in a label group
        for (_, inds) in label_to_inds
            shared_noise = -β * log.(-log.(rand(T, n_forms)))
            for j in inds
                noise[:, j] = shared_noise
            end
        end

        noise
    end

    y = softmax((logits + noise) / τ; dims=1)
    return y
end


function calc_embeddings(
    adj_list::Vector{Vector{Int}},
    atom_features::Matrix{T},
    atom_embedding_model::GNNChain
)
    gnn_input = atom_features
    graph = GNNGraph(adj_list)
    atom_embeddings = atom_embedding_model(graph, gnn_input)
 
    return atom_embeddings
end

function predict_atom_features(
    labels,
    embeds_mol,
    atom_features_model
)

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
    return feats_mol

end

function predict_bond_features(
    g,
    labels,
    embeds_mol,
    bond_pooling_model,
    bond_features_model
)

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
    return bond_feats_mol, bond_to_local_idx
end

function predict_angle_features(
    angles_i, angles_j, angles_k,
    vs_template,
    labels,
    embeds_mol,
    angle_pooling_model, angle_features_model
)
    ### Angle Feature Pooling ###
    angle_to_key = Dict{NTuple{3,Int}, NTuple{3,String}}()
    angle_triples = [(i,j,k) for (i,j,k) in zip(angles_i, angles_j, angles_k) if i in vs_template && j in vs_template && k in vs_template]
    
    # Map triplets from whole system indexing to local molecule indexing
    local_map = Dict(glo => loc for (loc, glo) in enumerate(vs_template))
    angle_triples = [(local_map[i], local_map[j], local_map[k]) for (i,j,k) in angle_triples]
    
    # From index to molecule type
    angle_to_local_idx = Dict{Tuple{Int,Int,Int}, Int}()
    angle_key = Tuple{String,String,String}[]
    angle_key = map(eachindex(angle_triples)) do idx
        i, j, k = angle_triples[idx]
        angle_to_local_idx[(i,j,k)] = idx
        li, lj, lk = labels[i], labels[j], labels[k]
        key = (li, lj, lk) < (lk, lj, li) ? (li, lj, lk) : (lk, lj, li)
        angle_to_key[(i,j,k)] = key
        return key
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
    return angle_feats_mol, angle_triples, angle_to_local_idx
end

function predict_torsion_features(
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,
    vs_template,
    labels,
    embeds_mol,
    proper_pooling_model, proper_features_model,
    improper_pooling_model, improper_features_model
)
    
    torsion_to_key_proper   = Dict{NTuple{4,Int}, NTuple{4,String}}()
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
    if !isempty(proper_feats_mol)
        proper_feats_mol = hcat(proper_feats_mol...)
    else
        proper_feats_mol = zeros(T, n_proper_terms, 0)
    end

    improper_feats_mol = map(1:length(torsion_improper_quads)) do idx
        quad    = torsion_improper_quads[idx]
        key     = torsion_to_key_improper[quad]
        key_idx = unique_improper_keys[key]
        return unique_improper_feats[:, key_idx]
    end
    if !isempty(proper_feats_mol)
        improper_feats_mol = hcat(improper_feats_mol...)
    else
        improper_feats_mol = zeros(T, n_improper_terms, 0)
    end
    return proper_feats_mol, improper_feats_mol, torsion_to_key_proper, torsion_to_key_improper, unique_proper_keys, unique_improper_keys
end