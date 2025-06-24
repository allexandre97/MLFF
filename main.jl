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
const global HDF5_FILES = ("SPICE-2.0.1.hdf5",)

const global SUBSET_N_REPEATS = Dict(
    # SPICE
    "SPICE Solvated Amino Acids Single Points Dataset v1.1" => 100, # 1,300   -> 130,000
    "SPICE Ion Pairs Single Points Dataset v1.2"            => 10 , # 1,426   -> 14,260
    "SPICE Dipeptides Single Points Dataset v1.3"           => 10 , # 33,850  -> 338,500
    "SPICE Amino Acid Ligand v1.0"                          => 2  , # 194,174 -> 388,348
    "SPICE Solvated PubChem Set 1 v1.0"                     => 20 , # 13,934  -> 278,680
    "SPICE Water Clusters v1.0"                             => 100, # 1,000   -> 100,000
    # Takaba2024 Espaloma
    "RNA Single Point Dataset v1.0"                         => 10 , # 8,560   -> 85,600
    "RNA Nucleoside Single Point Dataset v1.0"              => 10 , # 120     -> 1,200
    "RNA Trinucleotide Single Point Dataset v1.0"           => 10 , # 6,080   -> 60,800
    # Kovacs2023 MACE-OFF23
    "MACE-OFF water"                                        => 100, # 1,681   -> 168,100
    # GEMS
    "GEMS crambin"                                          => 100, # 5,140   -> 514,000
    # Condensed ΔHvap
    "vapourisation"                                         => 2  , # 2,250   -> 4,500
    # Condensed ΔHmix
    "mixing"                                                => 4  , # 600     -> 2,400
)

const global FEATURE_COL_NAMES = ["MOLECULE", "ATOMIC_MASS", "FORMAL_CHARGE", "AROMATICITY", "N_BONDS", "BONDS", "ANGLES", "PROPER", "IMPROPER", "MOL_ID"]
const global FEATURE_FILES = ("features.tsv",
                              "features_maceoff.tsv",
                              "features_cond.tsv",) # Just the SPICE features for now too. Generated with custom script!!! TODO: Take a look at said script, maybe integrate here?

const global CONF_DATAFRAME     = read_conf_data()
const global FEATURE_DATAFRAMES = [read_feat_file(file) for file in FEATURE_FILES]

models, optims     = build_models()

conf_pairs = train!(models, optims)

#= for pair in conf_pairs
    a, b = pair

    if b == 0
        continue
    end

    source_a, n_a, id_a = CONF_DATAFRAME[a,:source], CONF_DATAFRAME[a,:n_atoms], CONF_DATAFRAME[a,:conf_id]
    source_b, n_b, id_b = CONF_DATAFRAME[b,:source], CONF_DATAFRAME[b,:n_atoms], CONF_DATAFRAME[b,:conf_id]
    
    println("$source_a : $n_a : $id_a, $source_b : $n_b : $id_b")
end =#