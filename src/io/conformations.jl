using LinearAlgebra

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

function read_coords(df::DataFrameRow)
    
    px = parse.(T, split(df.px, ",")[1:end-1])
    py = parse.(T, split(df.py, ",")[1:end-1])
    pz = parse.(T, split(df.pz, ",")[1:end-1])
    
    return hcat(px,py,pz) # Returns Vector of shape (N_atoms, 3)

end

function read_forces(df::DataFrameRow)

    fx = parse.(T, split(df.fx, ",")[1:end-1])
    fy = parse.(T, split(df.fy, ",")[1:end-1])
    fz = parse.(T, split(df.fz, ",")[1:end-1])

    force = hcat(fx, fy, fz)
    exceeds = maximum(norm.(force)) > T(1e4)

    return force, exceeds

end

function read_energies(df::DataFrameRow)
    energy = T(df.energy)
    return energy
end

function read_charges(df::DataFrameRow)
    has_charge = df.charge == "-" ? false : true
    if !has_charge
        return T[], has_charge
    else
        charge = parse.(T, split(df.charge, ",")[1:end-1])
        return charge, has_charge
    end
end

function read_conformation(mol_order, start_mol, end_mol)

    map_result = map(start_mol:end_mol) do i 
        
        conf_i, conf_j, repeat = mol_order[i]
        df_i = CONF_DATAFRAME[conf_i, :]

        coords_i                 = read_coords(df_i)
        forces_i, exceeds_i      = read_forces(df_i)
        energy_i                 = read_energies(df_i)
        charges_i, has_charges_i = read_charges(df_i)
        pair_present = !iszero(conf_j)

        if pair_present
            df_j = CONF_DATAFRAME[conf_j, :]
            coords_j                 = read_coords(df_j)
            forces_j, exceeds_j      = read_forces(df_j)
            energy_j                 = read_energies(df_j)
            charges_j, has_charges_j = read_charges(df_j)
        else
            coords_j, 
            forces_j, exceeds_j,
            energy_j,
            charges_j, has_charges_j = coords_i,
                                       forces_i, exceeds_i,
                                       energy_i,
                                       charges_i, has_charges_i
        end

        exceeds_max_force = exceeds_i || exceeds_j

        return (coords_i,
                forces_i,
                energy_i,
                charges_i, has_charges_i,
                coords_j,
                forces_j,
                energy_j,
                charges_j, has_charges_j,
                exceeds_max_force, pair_present
        )

    end

    return map_result

end