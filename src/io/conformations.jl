function gen_conf_pairs(
    df::DataFrame; 
    rng=Random.GLOBAL_RNG
)::Vector{Tuple{Int, Int, Int}}
    #= 
    Generates a Vector of tuples in which each tuple represents
    a pair of conformations that can be compared. To know if two
    conformations can be compared we match them by their source 
    dataset AND by the number of atoms present, given that they
    represent the same molecule type.
    =#
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
                push!(out, (leftover, 0, repeat_i))
            end
        end
    end

    return out
end

function read_coords(
    df::DataFrameRow,
    n_atoms::Int
)::Vector{SVector{3,T}}
    #=
    Reads the coordinates for a given conformation stored in a 
    conformation dataframe. It returns the coordinates as a 
    SVector{3, Float32}, which is the data type expected by Molly. 
    =#
    
    px = parse.(T, split(df.px, ",")[1:end-1])
    py = parse.(T, split(df.py, ",")[1:end-1])
    pz = parse.(T, split(df.pz, ",")[1:end-1])

    raw = reshape(hcat(px,py,pz)', n_atoms * 3)

    # This returns the approprate shape and type for Molly
    return reinterpret(SVector{3, T}, raw) # Returns Vector of shape (3, N_atoms)

end

function read_forces(
    df::DataFrameRow,
    n_atoms::Int
)::Tuple{Vector{SVector{3,T}}, Bool}
    #=
    Analogous to the previous function, reads the dft forces from
    the conformations dataframe. Returns the forces as a SVector{3, T} 
    and also a boolean flag indicating if the forces are too big.
    =#

    fx = parse.(T, split(df.fx, ",")[1:end-1])
    fy = parse.(T, split(df.fy, ",")[1:end-1])
    fz = parse.(T, split(df.fz, ",")[1:end-1])

    force = reinterpret(SVector{3, T}, reshape(hcat(fx, fy, fz)', n_atoms * 3))
    exceeds = maximum(norm.(force)) > T(1e4)

    return force, exceeds

end

function read_energies(df::DataFrameRow)::T
    #=
    Reads the dft energy from the conformations dataframe. Returns it as Float32 
    =#
    energy = T(df.energy)
    return energy
end

function read_charges(df::DataFrameRow)::Tuple{Vector{T}, Bool}
    #=
    Reads the dft charges from the conformation dataframe. Returns 
    charges as Vector{T} and a boolean flag indicating if the conformation
    had charges present or not.
    =#
    has_charge = df.charge == "-" ? false : true
    if !has_charge
        return T[], has_charge
    else
        charge = parse.(T, split(df.charge, ",")[1:end-1])
        return charge, has_charge
    end
end

function read_conformation(
    df::DataFrame,
    mol_order::Vector{Tuple{Int, Int, Int}}, 
    start_mol::Int, 
    end_mol::Int
)
    #=
    Wrapper over all the previous functions. Returns a vector of tuples
    where each tuple contains the coordinates, forces, energy, charges of
    a given conformation. The vector is over all conformations to read.
    =#
    map_result = map(start_mol:end_mol) do i 
        
        conf_i, conf_j, repeat = mol_order[i]
        df_i = df[conf_i, :]
        n_atoms_i = df_i.n_atoms

        coords_i                 = read_coords(df_i, n_atoms_i)
        forces_i, exceeds_i      = read_forces(df_i, n_atoms_i)
        energy_i                 = read_energies(df_i)
        charges_i, has_charges_i = read_charges(df_i)
        pair_present = !iszero(conf_j)

        if pair_present
            df_j = df[conf_j, :]
            n_atoms_j = df_j.n_atoms
            coords_j                 = read_coords(df_j, n_atoms_j)
            forces_j, exceeds_j      = read_forces(df_j, n_atoms_j)
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