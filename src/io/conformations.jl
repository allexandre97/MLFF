
function gen_conf_pairs(df::DataFrame; rng=Random.GLOBAL_RNG)
    # 1) group rows by (source, n_atoms)
    groups = Dict{Tuple{eltype(df.source), eltype(df.n_atoms)}, Vector{Int}}()
    for (i, row) in enumerate(eachrow(df))
        key = (row.source, row.n_atoms)
        push!(get!(groups, key, Int[]), i)
    end

    # 2) shuffle each group, emit pairs and singletons
    out = Vector{Tuple{Int,Int,Int}}()
    for idxs in values(groups)
        
        shuffle!(rng, idxs)
        
        # pair off in twos
        for k in 1:2:length(idxs)-1
            i, j = idxs[k], idxs[k+1]
            source = df[i,:source]
            repeats = SUBSET_N_REPEATS[source]
            for repeat_i in 1:repeats
                push!(out, i < j ? (i, j, repeat_i) : (j, i, repeat_i))
            end
        end

        # leftover?
        if isodd(length(idxs))
            leftover = last(idxs)
            source = df[leftover,:source]
            repeats = SUBSET_N_REPEATS[source]
            for repeat_i in 1:repeats
                push!(out, (leftover, 0, repeats))
            end
        end
    end

    return out
end