using ArgParse
using JSON

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

const T = Float32

parsed_args::Dict{String, Any} = parse_commandline() # Read args from cli
const global MODEL_PARAMS::Dict = JSON.parsefile("params.json") # Read model parameters from JSON file

include("src/io/fileio.jl")    # Handles file input output
include("src/mol/molbuild.jl") # Helpers to build molecule connectivity
include("src/nets/models.jl")  # Methods and variables related to Neural Nets
include("src/nets/trainer.jl")  # Methods and variables related to Neural Nets

############### MAIN LOGIC ###############

const global DATASETS_PATH::String = parsed_args["db"]
const global MACEOFF_PATH::String  = "/lmb/home/alexandrebg/Documents/QuarantineScripts/JG/typing/data_kovacs2023/water"
const global HDF5_FILES = ("SPICE-2.0.1.hdf5",)#, Just use SPICE water for now
#                           "RNA-DIVERSE-OPENFF-DEFAULT.hdf5",
#                           "RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5",
#                           "RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5")

const global FEATURE_COL_NAMES = ["MOLECULE", "ATOMIC_MASS", "FORMAL_CHARGE", "AROMATICITY", "N_BONDS", "BONDS", "ANGLES", "PROPER", "IMPROPER", "MOL_ID"]
const global FEATURE_FILES = ("features.tsv",
                              "features_maceoff.tsv",
                              "features_cond.tsv",) # Just the SPICE features for now too. Generated with custom script!!! TODO: Take a look at said script, maybe integrate here?

const global CONF_DATAFRAME     = read_conf_data()
const global FEATURE_DATAFRAMES = [read_feat_file(file) for file in FEATURE_FILES]

models, optims     = build_models()

train!(models, optims)