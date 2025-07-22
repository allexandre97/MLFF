struct XYZFile
    filepath::String
end

function read_file(file::HDF5.File, species::String = "water")
    #=
    Reads SPICE-format HDF5 for one species, returning:
      n_confs::Int
      atom_number::Vector{Int16}
      conformations::Vector{Vector{SVector{3, T}}}
      dft_energies::Vector{T}
      dft_forces::Vector{Vector{SVector{3, T}}}
      mbis_charges::Vector{Vector{T}}
      has_mbis_charges::Bool
    =#

    #— read basic arrays —#
    subset              = read(file["$species/subset"])[1]
    atom_number         = read(file["$species/atomic_numbers"])
    conformations_raw   = read(file["$species/conformations"])
    dft_energies        = read(file["$species/dft_total_energy"])
    dft_grads_raw       = read(file["$species/dft_total_gradient"])

    #— optional dispersion correction —#
    has_disp = haskey(file, "$species/dispersion_correction_gradient")
    disp_raw = has_disp ? read(file["$species/dispersion_correction_gradient"])::Array{T,3} : nothing
    is_rna   = startswith(species, "RNA")

    #— optional MBIS charges —#
    has_mbis_charges = haskey(file, "$species/mbis_charges")
    mbis_charges_raw = has_mbis_charges ? read(file["$species/mbis_charges"])::Array{T,3} : nothing

    # sizes
    n_atoms = size(conformations_raw, 2)
    n_confs = size(conformations_raw, 3)

    #— build conformations → SVector{3,T} —#
    conformations = Vector{Vector{SVector{3, T}}}(undef, n_confs)
    for ci in 1:n_confs
        conformations[ci] = [
            SVector{3, T}(conformations_raw[:, ai, ci] .* bohr_to_nm)
            for ai in 1:n_atoms
        ]
    end

    #— build DFT forces (negate grads + convert) —#
    dft_forces = Vector{Vector{SVector{3, T}}}(undef, n_confs)
    for ci in 1:n_confs
        grad = dft_grads_raw[:, :, ci]
        if is_rna && has_disp
            grad .+= disp_raw[:, :, ci]
        end
        dft_forces[ci] = SVector{3, T}.(eachcol(-grad .* force_conversion))
    end

    #— build MBIS charges per atom (if present) —#
    mbis_charges = Vector{Vector{T}}(undef, n_confs)
    if has_mbis_charges
        for ci in 1:n_confs
            mbis_charges[ci] = vec(mbis_charges_raw[1, :, ci])  # <- fixed indexing here
        end
    else
        fill!(mbis_charges, Vector{T}())
    end

    return (subset,
            n_confs,
            atom_number,
            conformations,
            dft_energies,
            dft_forces,
            mbis_charges,
            has_mbis_charges)
end


function read_file(file::XYZFile)
    #=
    Reads the SPICE dataset hdf5 file for a given species.
    
    ARGS:
        file::XYZFile --> The file to open
        
    RETURN:
        conformations::Array{Float32, 3}
        dft_energies::Array{Float32}
        dft_gradients::Array{Float32, 3}

    =#

    coords = map(readlines(file.filepath)[3:end]) do line
        SVector{3, T}(parse.(T, split(line)[2:4])) ./ 10
    end

    forces = map(readlines(file.filepath)[3:end]) do line
        SVector{3, T}(parse.(T, split(line)[5:7])) .* eVpÅ_to_kJpmolpnm
    end

    eline = readlines(file.filepath)[2]

    energy::T =  parse(T, split(split(eline, " ")[2], "=")[2])

    n_atoms::Int16 = parse(Int16, readlines(file.filepath)[1])
    
    return n_atoms, coords, forces, energy

end

function read_feat_file(path::String)::DataFrame
    #=
    Reads a .tsv file where features for a given set of molecules
    is encoded. 
    =#

    dataframe::DataFrame = CSV.read(joinpath(DATASETS_PATH, path), DataFrame; delim='\t', header = false)

    rename!(dataframe, FEATURE_COL_NAMES)

    return dataframe

end


function read_conf_data()


    hdf5_list = [h5open(joinpath(DATASETS_PATH, file), "r") for file in HDF5_FILES]
    xyz_list  = [XYZFile(joinpath(MACEOFF_PATH, "conf_$i.xyz")) for i in 1:1_681]

    row_buffer = Vector{Dict{Symbol, Any}}()

    # Process HDF5 files
    for hdf5 in hdf5_list
        #=
        TODO: Right now we are just getting the information for the water clusters
        (see method read_file for HDF5 files). In the future we have to get the information
        for all the molecules in the SPICE datasets and so on. A first idea on how to 
        approach this is to iterate over keys = keys(HDF5_file) and do the relevant extraction
        for each one of those.
        =#

        #species = "ccl o"
        species = "water"

        subset, n_confs, atom_numbers,
        conformations, energies, forces,
        charges, has_charges = read_file(hdf5, species)

        for i in 1:n_confs

            row = Dict{Symbol, Any}()
            row[:mol_name] = species
            row[:source] = subset
            row[:conf_id] = i
            row[:n_atoms] = length(atom_numbers)
            row[:energy] = energies[i]

            if !has_charges
                row[:charge] = "-"
            else
                charge_str = ""
                for j in 1:length(atom_numbers)
                    c = charges[i][j]
                    charge_str *= string(c) * ","
                end
                row[:charge] = charge_str
            end

            x_str = ""; y_str = ""; z_str = ""
            fx_str = ""; fy_str = ""; fz_str = ""

            for j in 1:length(atom_numbers)
                pos = conformations[i][j]
                grad = forces[i][j]
                x_str  *= string(pos[1]) * ","
                y_str  *= string(pos[2]) * ","
                z_str  *= string(pos[3]) * ","
                fx_str *= string(grad[1]) * ","
                fy_str *= string(grad[2]) * ","
                fz_str *= string(grad[3]) * ","
            end

            row[:px] = x_str; row[:py] = y_str; row[:pz] = z_str
            row[:fx] = fx_str; row[:fy] = fy_str; row[:fz] = fz_str

            push!(row_buffer, row)
        end
    end

    # Process XYZ files
    # I believe that the only datasets in .xyz format belong to the MACE-OFF water
    for (conf_i, xyz) in enumerate(xyz_list)
        n_atoms, coords, forces, energy = read_file(xyz)

        row = Dict{Symbol, Any}()
        row[:mol_name] = "maceoff_water_$conf_i"
        row[:source] = "MACE-OFF water"
        row[:conf_id] = conf_i
        row[:n_atoms] = n_atoms
        row[:energy] = energy

        row[:charge] = "-"

        x_str = ""; y_str = ""; z_str = ""
        fx_str = ""; fy_str = ""; fz_str = ""

        for j in 1:n_atoms
            pos = coords[j]
            grad = forces[j]
            x_str  *= string(pos[1]) * ","
            y_str  *= string(pos[2]) * ","
            z_str  *= string(pos[3]) * ","
            fx_str *= string(grad[1]) * "," # This is sub-optimal as it leaves a trailing comma that needs to be filtered out downstream
            fy_str *= string(grad[2]) * ","
            fz_str *= string(grad[3]) * ","
        end

        row[:px] = x_str; row[:py] = y_str; row[:pz] = z_str
        row[:fx] = fx_str; row[:fy] = fy_str; row[:fz] = fz_str

        push!(row_buffer, row)
    end

    df = DataFrame(row_buffer)
    desired_order = [:mol_name, :source, :conf_id, :n_atoms, :energy, :charge, :px, :py, :pz, :fx, :fy, :fz]
    df = df[:, desired_order]
    return df

end

function decode_feats(df::DataFrame)
    #=
    Takes an entry of a feature dataframe and decodes it as vectors
    that can then be fed to the NN model.
    =#

    elements::Vector{Int}       = parse.(Int, split(df.ATOMIC_MASS[1], ","))
    formal_charges::Vector{Int} = parse.(Int, split(df.FORMAL_CHARGE[1], ","))

    aromatics::Vector{Int} = parse.(Int, split(df.AROMATICITY[1], ","))

    n_bonds::Vector{Int} = parse.(T, split(df.N_BONDS[1], ","))

    bonds::Matrix{Int}     = reduce(vcat, [parse.(Int, split(pair, "/"))' for pair in split(df.BONDS[1], ",")])
    angles::Matrix{Int}    = reduce(vcat, [parse.(Int, split(trio, "/"))' for trio in split(df.ANGLES[1], ",")])
    propers::Matrix{Int}   = df.PROPER[1] == "-"   ? Matrix{Int}(undef,0,4) : reduce(vcat, [parse.(Int, split(quad, "/"))' for quad in split(df.PROPER[1], ",")])
    impropers::Matrix{Int} = df.IMPROPER[1] == "-" ? Matrix{Int}(undef,0,4) : reduce(vcat, [parse.(Int, split(quad, "/"))' for quad in split(df.IMPROPER[1], ",")])
    
    mol_inds::Vector{Int} = parse.(Int, split(df.MOL_ID[1], ","))
    
    adj_list::Vector{Vector{Int}} = build_adj_list(df)

    n_atoms::Int          = length(elements)
    atom_feats = zeros(T, MODEL_PARAMS["networks"]["n_atom_features_in"], n_atoms)

    for i in 1:n_atoms
        atom_feats[elements[i], i] = one(T)
        atom_feats[19 + formal_charges[i], i] = one(T)
        atom_feats[22, i] = aromatics[i]
        atom_feats[23 + min(n_bonds[i], 6), i] = one(T)
    end

    return (
        elements,
        formal_charges,
        bonds[:,1], bonds[:,2],
        angles[:,1], angles[:,2], angles[:,3],
        propers[:,1], propers[:,2], propers[:,3], propers[:,4],
        impropers[:,1], impropers[:,2], impropers[:,3], impropers[:,4],
        mol_inds,
        adj_list,
        n_atoms,
        atom_feats
    )
end

Flux.@non_differentiable decode_feats(df::DataFrame)
