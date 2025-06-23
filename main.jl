using ArgParse
using JSON

include("src/io/fileio.jl")    # Handles file input output
include("src/mol/molbuild.jl") # Helpers to build molecule connectivity
include("src/nets/models.jl")  # Methods and variables related to Neural Nets

function parse_commandline()::Dict{String, Any}

    s = ArgParseSettings()
    @add_arg_table s begin
        "--db"
            help    = "Path to where datasets are found"
            default = "/lmb/home/alexandrebg/Documents/QuarantineScripts/JG/typing"
            arg_type = String
    end

    return parse_args(s)

end


############### MAIN LOGIC ###############

parsed_args::Dict{String, Any} = parse_commandline() # Read args from cli
const global MODEL_PARAMS::Dict = JSON.parsefile("params.json") # Read model parameters from JSON file

const global DATASETS_PATH::String = parsed_args["db"]
const global HDF5_FILES = ("SPICE-2.0.1.hdf5",)#, Just use SPICE water for now
#                           "RNA-DIVERSE-OPENFF-DEFAULT.hdf5",
#                           "RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5",
#                           "RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5")

const global FEATURE_COL_NAMES = ["MOLECULE", "ATOMIC_MASS", "FORMAL_CHARGE", "AROMATICITY", "N_BONDS", "BONDS", "ANGLES", "PROPER", "IMPROPER", "MOL_ID"]
const global FEATURE_FILES = ("features.tsv",) # Just the SPICE features for now too. Generated with custom script!!! TODO: Take a look at said script, maybe integrate here?

read_conf_data(1,2,3)

const global mol_features = read_feat_file(FEATURE_FILES[1])

build_adj_list(mol_features[mol_features[!, :MOLECULE] .== "water", :])