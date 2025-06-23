using HDF5
using CSV, DataFrames

function read_hdf5(file::HDF5.File,
                   species::String = "water")
    #=
    Reads the SPICE dataset hdf5 file for a given species.
    
    ARGS:
        file::HDF5.File --> The file to open
        species::String --> The species to read. Defaults to water
    
    RETURN:
        conformations::Array{Float32, 3}
        dft_energies::Array{Float32}
        dft_gradients::Array{Float32, 3}

    =#

    atom_number::Array{Int16}          = read(file["$species/atomic_numbers"])
    conformations::Array{Float32, 3}   = read(file["$species/conformations"])
    dft_energies::Array{Float32}       = read(file["$species/dft_total_energy"])
    dft_gradients::Array{Float32, 3}   = read(file["$species/dft_total_gradient"])
    n_conf::Tuple{Int16, Int16, Int16} = size(conformations)

    println(n_conf)

    return n_conf[3], atom_number, conformations, dft_energies, dft_gradients
    
end
#= 
function read_feat_files(file_list::Tuple{String})

    full_paths = [joinpath(DATASETS_PATH, file) for file in file_list]

    line_feats = vcat(readlines.(full_paths)...)
    
    mol_feats = Dict(Pair(String.(split(line, "\t"; limit=2))...) for line in line_feats)

    return mol_feats

end
=#

function read_feat_file(path::String)::DataFrame

    dataframe::DataFrame = CSV.read(joinpath(DATASETS_PATH, path), DataFrame; delim='\t')

    rename!(dataframe, FEATURE_COL_NAMES)

    return dataframe

end

function read_conf_data(mol_order,
                        start_i,
                        end_i)

    #=
    Read conformations from the HDF5 files
    =#
 
    hdf5_list = [h5open(joinpath(DATASETS_PATH, file), "r") for file in HDF5_FILES]

    _,_,_,_ = read_hdf5(hdf5_list[1])

end
