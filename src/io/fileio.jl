using HDF5
using StaticArrays
using CSV, DataFrames

include("../physics/definitions.jl")

struct XYZFile
    filepath::String
end

function read_file(file::HDF5.File, species::String = "water")
    #=
    Reads a SPICE-format HDF5 file for a given species.

    ARGS:
        file::HDF5.File --> Open HDF5 file handle
        species::String --> Molecule name/group inside file

    RETURNS:
        n_confs::Int
        atom_number::Vector{Int16}
        conformations::Vector{Vector{SVector{3, T}}}
        dft_energies::Vector{T}
        dft_forces::Vector{Vector{SVector{3, T}}}
    =#

    subset::String                = read(file["$species/subset"])[1]
    atom_number::Vector{Int16}    = read(file["$species/atomic_numbers"])
    conformations_raw::Array{T,3} = read(file["$species/conformations"])
    dft_energies::Vector{T}       = read(file["$species/dft_total_energy"])
    dft_gradients_raw::Array{T,3} = read(file["$species/dft_total_gradient"])

    n_atoms = size(conformations_raw, 2)
    n_confs = size(conformations_raw, 3)

    # Optional dispersion correction for RNA
    has_disp_corr = haskey(file, "$species/dispersion_correction_gradient")
    disp_corr = has_disp_corr ? read(file["$species/dispersion_correction_gradient"]) : nothing
    is_rna = startswith(species, "RNA")

    # Convert conformations to Vector{Vector{SVector{3,T}}}
    conformations = [
        [SVector{3, T}(conformations_raw[:, atom_i, conf_i] .* bohr_to_nm)
         for atom_i in 1:n_atoms]
        for conf_i in 1:n_confs
    ]

    # Compute DFT forces (with optional dispersion correction)
    dft_forces = Vector{Vector{SVector{3, T}}}(undef, n_confs)

    for conf_i in 1:n_confs
        total_grad = dft_gradients_raw[:, :, conf_i]
        if is_rna && has_disp_corr
            total_grad .+= disp_corr[:, :, conf_i]
        end
        # Convert: negate and scale
        dft_forces[conf_i] = SVector{3, T}.(eachcol(-total_grad .* force_conversion))
    end

    return species, subset, n_confs, atom_number, conformations, dft_energies, dft_forces
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

        species = "water"

        species, subset, n_confs, atom_numbers,
        conformations, energies, gradients = read_file(hdf5, 
                                                       species)
        for i in 1:n_confs
            row = Dict{Symbol, Any}()
            row[:mol_name] = species
            row[:source] = subset
            row[:conf_id] = i
            row[:n_atoms] = length(atom_numbers)
            row[:energy] = energies[i]

            x_str = ""; y_str = ""; z_str = ""
            fx_str = ""; fy_str = ""; fz_str = ""

            for j in 1:length(atom_numbers)
                pos = conformations[i][j]
                grad = gradients[i][j]
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
    for (conf_i, xyz) in enumerate(xyz_list)
        n_atoms, coords, forces, energy = read_file(xyz)

        row = Dict{Symbol, Any}()
        row[:mol_name] = "water"
        row[:source] = "MACE-OFF water"
        row[:conf_id] = conf_i
        row[:n_atoms] = n_atoms
        row[:energy] = energy

        x_str = ""; y_str = ""; z_str = ""
        fx_str = ""; fy_str = ""; fz_str = ""

        for j in 1:n_atoms
            pos = coords[j]
            grad = forces[j]
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

    df = DataFrame(row_buffer)
    desired_order = [:mol_name, :source, :conf_id, :n_atoms, :energy, :px, :py, :pz, :fx, :fy, :fz]
    df = df[:, desired_order]
    return df

end

