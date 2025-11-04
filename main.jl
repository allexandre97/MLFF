using ArgParse
using JSON
using EzXML

using Flux
using GraphNeuralNetworks
using Zygote
using Enzyme
#Enzyme.Compiler.VERBOSE_ERRORS[] = true
using ChainRulesCore
using BSON

using Random
using Statistics
using Polynomials
using LinearAlgebra
using StaticArrays

using DataStructures
using OrderedCollections
using CSV
using HDF5
using DataFrames
using Dates
using TimerOutputs

using Graphs
using Graphs.Experimental

import Chemfiles
using Molly

using CairoMakie

import Base: +
import Base: *


function parse_commandline()::Dict{String, Any}

    s = ArgParseSettings()
    @add_arg_table s begin
        "--db"
            help    = "Path to where datasets are found"
            default = "/lmb/home/alexandrebg/Documents/QuarantineScripts/JG/typing"
            arg_type = String
        "--json"
            help    = "Path to where the hyperparam json file is found"
            default = "./params.json"
            arg_type = String
    end

    return parse_args(s)

end

const T = Float32

parsed_args::Dict{String, Any} = parse_commandline() # Read args from cli
global MODEL_PARAMS::Dict = JSON.parsefile(parsed_args["json"]) # Read model parameters from JSON file

include("./src/physics/definitions.jl")

include("./src/physics/molly_ext/general_atom.jl")
include("./src/physics/molly_ext/functionals_nonbonded.jl")
include("./src/physics/molly_ext/functionals_bonded.jl")

include("./src/physics/forces.jl")
include("./src/physics/condensed_phase.jl")
include("./src/physics/potentials.jl")
include("./src/physics/transformer.jl")

include("./src/io/conformations.jl")
include("./src/io/fileio.jl")
include("./src/io/logging.jl")
include("./src/io/simread.jl")
include("./src/io/graphics.jl")
include("./src/io/forcefield.jl")

include("./src/mol/broadcast.jl")
include("./src/mol/molbuild.jl")
include("./src/mol/graphs.jl")
include("./src/mol/newtypes.jl")

include("./src/nets/losses.jl")
include("./src/nets/models.jl")
include("./src/nets/trainer.jl")

############### MAIN LOGIC ###############

const global DATASETS_PATH::String = parsed_args["db"]
const global MACEOFF_PATH::String  = joinpath(DATASETS_PATH, "data_kovacs2023/water")
const global EXP_DATA_DIR::String  = joinpath(DATASETS_PATH, "condensed_data/exp_data")
const global HDF5_FILES = ("SPICE-2.0.1.hdf5",)

global OUTDIR = MODEL_PARAMS["paths"]["out_dir"]
global GAFF_DIR = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")

const global SUBSET_N_REPEATS = Dict(
    # SPICE
    "SPICE Solvated Amino Acids Single Points Dataset v1.1" => 100, # 1,300   -> 130,000
    "SPICE Ion Pairs Single Points Dataset v1.2"            => 10 , # 1,426   -> 14,260
    "SPICE Dipeptides Single Points Dataset v1.3"           => 10 , # 33,850  -> 338,500
    "SPICE Amino Acid Ligand v1.0"                          => 2  , # 194,174 -> 388,348
    "SPICE Solvated PubChem Set 1 v1.0"                     => 20 , # 13,934  -> 278,680
    "SPICE Water Clusters v1.0"                             => 1, #100, # 1,000   -> 100,000
    "SPICE DES370K Single Points Dataset v1.0"              => 1,
    # Takaba2024 Espaloma
    "RNA Single Point Dataset v1.0"                         => 10 , # 8,560   -> 85,600
    "RNA Nucleoside Single Point Dataset v1.0"              => 10 , # 120     -> 1,200
    "RNA Trinucleotide Single Point Dataset v1.0"           => 10 , # 6,080   -> 60,800
    # Kovacs2023 MACE-OFF23
    "MACE-OFF water"                                        => 1, #100, # 1,681   -> 168,100
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
                              "features_cond_bigbox.tsv",) # Just the SPICE features for now too. Generated with custom script!!! TODO: Take a look at said script, maybe integrate here?

const global CONF_DATAFRAME     = read_conf_data()
const global FEATURE_DATAFRAMES = ([read_feat_file(file) for file in FEATURE_FILES]...,)

const global CONDENSED_TEST_SYSTEMS = (
    "vapourisation_liquid_CC(=O)C", # Acetone
    "mixing_combined_CNCCO_O",
    "mixing_combined_CCC(C)=O_Nc1ccccc1",
)
const global COND_SIM_FRAMES = 101:250 
const global COND_MOLECULES  = Vector{Tuple{String, T, Int, Int}}()

for mol_id in FEATURE_DATAFRAMES[3].MOLECULE
    mol_id in CONDENSED_TEST_SYSTEMS && continue # Skip if in test systems
    if startswith(mol_id, "vapourisation_liquid") && endswith(mol_id, "_O") # Force to get only the data for water, remember to change this when moving to complex stuff
        repeats = SUBSET_N_REPEATS["vapourisation"]
        for temp in T.(285:10:325)
            for frame_i in COND_SIM_FRAMES
                for repeat_i in 1:repeats
                    push!(COND_MOLECULES, (mol_id, temp, frame_i, repeat_i))
                end
            end
        end

    #=
    TODO: This part of the logic is only needed when training on something else
    than just pure water, as there is no enthalpy of mixing for only single 
    components. 
    =#

    #= elseif startswith(mol_id, "mixing_combined_")
        repeats = SUBSET_N_REPEATS["mixing"]
        for frame_i in COND_SIM_FRAMES
            for repeat_i in 1:repeatsatoms
                push!(COND_MOLECULES, (mol_id, T(298.15), frame_i, repeat_i))
            end
        end
    =#
    end 
end
shuffle!(COND_MOLECULES)

const ENTH_VAP_EXP_DATA = Dict{String, Polynomial{Float64, :x}}()
for mol in COND_MOLECULES
    mol_id = mol[1]
    if startswith(mol_id, "vapourisation_liquid_")
        smiles = split(mol_id, "_")[end]
        enth_vap_data_fp = joinpath(EXP_DATA_DIR, "enth_vap", "$smiles.txt")
        # Float32 gave bad fit
        enth_vap_exp_xs = parse.(Float64, getindex.(split.(readlines(enth_vap_data_fp)), 1))
        enth_vap_exp_ys = parse.(Float64, getindex.(split.(readlines(enth_vap_data_fp)), 2))
        ENTH_VAP_EXP_DATA[mol_id] = fit(enth_vap_exp_xs, enth_vap_exp_ys, 3)
    end
end

const global COND_MOL_VAL   = COND_MOLECULES[1:MODEL_PARAMS["training"]["n_frames_val_cond"]]
const global COND_MOL_TRAIN = COND_MOLECULES[(MODEL_PARAMS["training"]["n_frames_val_cond"]+1):end]

# Molly Constants. TODO: How can I pack these in a json?
const global boundary_inf = CubicBoundary(T(Inf))

#models, optims     = build_models()

#= BSON.@save "init_models.bson" models
BSON.@save "init_optims.bson" optims =# 

BSON.@load "./init_models.bson" models
BSON.@load "./init_optims.bson" optims

@non_differentiable Molly.find_neighbors(args...)

out_dir = MODEL_PARAMS["paths"]["out_dir"]

const global save_every_epoch = false

if !isnothing(out_dir) && !isdir(out_dir)
    mkpath(out_dir)
    mkdir(joinpath(out_dir, "ff_xml"))
    mkdir(joinpath(out_dir, "training_sims"))
    if save_every_epoch
        mkdir(joinpath(out_dir, "models"))
        mkdir(joinpath(out_dir, "optims"))
    end
end


#= df = FEATURE_DATAFRAMES[1]
df = df[df.MOLECULE .== "water", :]

conf_data = read_conformation(CONF_DATAFRAME, [(1,2,1)], 1, 1)

coords_i, forces_i, energy_i, 
charges_i, has_charges_i,
coords_j, forces_j, energy_j,
charges_j, has_charges_j,
exceeds_force, pair_present = conf_data[1]

vdw_fnc_idx = 1

sys, _ = mol_to_system(1, "water", df, coords_i, boundary_inf, models...) =#

models, optims = train!(models, optims)
println()

#= df = FEATURE_DATAFRAMES[3]
df = df[df.MOLECULE .== "big_waterbox", :]

temps = (285, 295, 305, 315, 325)

vdw_fnc_idx = 1

ff_path    = joinpath(MODEL_PARAMS["paths"]["out_dir"], "ff_xml", "model_epoch_38.xml")

ff = MolecularForceField(T, ff_path, units = false)

sys_ML = features_to_xml(ff_path, 1, "vapourisation_liquid_O", 141, 295, df, models...)

states, trajs = build_rw_states("vapourisation_liquid_O", 40, 1, true, temps)
coords, bounds, gradsU, Ldens, Lcomp = sample_trajs("vapourisation_liquid_O", df, trajs, states) =#

#=Ldens      = collect(Iterators.flatten(Ldens))
Lcomp      = collect(Iterators.flatten(Lcomp))
Ldens_mean = mean(Ldens)
Lcomp_mean = mean(Lcomp)

ff_path    = joinpath(MODEL_PARAMS["paths"]["out_dir"], "ff_xml", "model_epoch_38.xml")

ff = MolecularForceField(T, ff_path, units = false)
initial_dir = joinpath(DATASETS_PATH, "condensed_data", "starting_structures")
sys = System(joinpath(initial_dir, "big_waterbox.pdb"),
        ff;
        units               = false,
        array_type          = Array,
        rename_terminal_res = false,
        nonbonded_method    = "cutoff",
        dist_cutoff         = T(0.8))

kBT = ustrip(u"kJ/mol", Unitful.R * 310*u"K")
β   = T(1/kBT)
target = ThermoState("target", β, T(1), deepcopy(sys))

gen_mbar = assemble_mbar_inputs(coords, bounds, states; target_state = target)

_, w_target, _ = mbar_weights(gen_mbar)

gradsU_flat  = collect(Iterators.flatten(gradsU))
gradsU_w     = multiply_grads.(gradsU_flat, w_target)
gradsU_w_sum = convert(Vector{Any}, fill(nothing, length(models)))
for gUw in gradsU_w 
    gradsU_w_sum = accum_grads.(gradsU_w_sum, gUw)
end

grads_dens = grads_reweight(Ldens, β, w_target, gradsU_w, gradsU_w_sum, length(models))
grads_comp = grads_reweight(Lcomp, β, w_target, gradsU_w, gradsU_w_sum, length(models)) =#