vdw_functional_form   = MODEL_PARAMS["physics"]["vdw_functional_form"]
mixing_function       = MODEL_PARAMS["physics"]["mixing_function"]
bond_functional_form  = MODEL_PARAMS["physics"]["bond_functional_form"]
angle_functional_form = MODEL_PARAMS["physics"]["angle_functional_form"]

bohr_to_nm = T(5.29177210903e-2)
hartree_to_kJpmol = T(4.3597447222071 * 6.02214076e2)
force_conversion = hartree_to_kJpmol / bohr_to_nm
eVpÅ_to_kJpmolpnm = T(964.8533212331)

inverse_sigmoid(x) = log(x / (1 - x))
starting_weight14_vdw  = inverse_sigmoid(T(0.5))
starting_weight14_coul = inverse_sigmoid(T(0.8333))

const ELEMENT_TO_NAME = ["H", "Li", "B" , "C", "N" , "O" , "F", "Na", "Mg", "Si",
                         "P", "S" , "Cl", "K", "Ca", "Br", "I"]

const ELEMENT_TO_Z = Dict(
    "H"=>1,  "He"=>2,  "Li"=>3,  "Be"=>4,  "B"=>5,   "C"=>6,   "N"=>7,   "O"=>8,   "F"=>9,   "Ne"=>10,
    "Na"=>11,"Mg"=>12, "Al"=>13, "Si"=>14, "P"=>15,  "S"=>16,  "Cl"=>17, "Ar"=>18, "K"=>19,  "Ca"=>20,
    "Sc"=>21,"Ti"=>22, "V"=>23,  "Cr"=>24, "Mn"=>25, "Fe"=>26, "Co"=>27, "Ni"=>28, "Cu"=>29, "Zn"=>30,
    "Ga"=>31,"Ge"=>32, "As"=>33, "Se"=>34, "Br"=>35, "Kr"=>36, "Rb"=>37, "Sr"=>38, "Y"=>39,  "Zr"=>40,
    "Nb"=>41,"Mo"=>42, "Tc"=>43, "Ru"=>44, "Rh"=>45, "Pd"=>46, "Ag"=>47, "Cd"=>48, "In"=>49, "Sn"=>50,
    "Sb"=>51,"Te"=>52, "I"=>53,  "Xe"=>54, "Cs"=>55, "Ba"=>56, "La"=>57, "Ce"=>58, "Pr"=>59, "Nd"=>60,
    "Pm"=>61,"Sm"=>62, "Eu"=>63, "Gd"=>64, "Tb"=>65, "Dy"=>66, "Ho"=>67, "Er"=>68, "Tm"=>69, "Yb"=>70,
    "Lu"=>71,"Hf"=>72, "Ta"=>73, "W"=>74,  "Re"=>75, "Os"=>76, "Ir"=>77, "Pt"=>78, "Au"=>79, "Hg"=>80,
    "Tl"=>81,"Pb"=>82, "Bi"=>83, "Po"=>84, "At"=>85, "Rn"=>86, "Fr"=>87, "Ra"=>88, "Ac"=>89, "Th"=>90,
    "Pa"=>91,"U"=>92,  "Np"=>93, "Pu"=>94, "Am"=>95, "Cm"=>96, "Bk"=>97, "Cf"=>98, "Es"=>99, "Fm"=>100
)


const ATOMIC_MASSES = [
    1.008 , 6.94        , 10.81, 12.011, 14.007 , 15.999, 18.998403163, 22.98976928, 24.305,
    28.085, 30.973761998, 32.06, 35.45 , 39.0983, 40.078, 79.904      , 126.90447  ,
]

global NAME_TO_MASS = Dict(Pair(e, m) for (e,m) in zip(ELEMENT_TO_NAME, ATOMIC_MASSES))

# Initialize some constants for non bonded interactions. Now we let the model choose the best functional form for vdw
const  global n_vdw_atom_params = 11

struct GlobalParams{T}
    params::Vector{T}
end
(model::GlobalParams)() = model.params

init_global_params = [inverse_sigmoid(T(0.5)),
                      inverse_sigmoid(T(0.833)),
                      zero(T), zero(T), zero(T), zero(T)]

global model_global_params = GlobalParams(init_global_params)


if mixing_function == "lb"
    const σ_mixing = Molly.lorentz_σ_mixing
    const ϵ_mixing = Molly.geometric_ϵ_mixing
elseif mixing_function == "geom"
    const σ_mixing = Molly.geometric_σ_mixing
    const ϵ_mixing = Molly.geometric_ϵ_mixing
elseif mixing_function == "wh"
    const σ_mixing = Molly.waldman_hagler_σ_mixing
    const ϵ_mixing = Molly.waldman_hagler_ϵ_mixing
else
    error("unknown mixing function $mixing_function")
end

# Initialize some constants based on the functioal form for bonds description
if bond_functional_form == "harmonic"
    const n_bonded_params = 2
elseif bond_functional_form == "morse"
    const n_bonded_params = 3
else
    error("unknown bond functional form $bond_functional_fo19rm")
end

if angle_functional_form == "harmonic"
    const n_angle_params = 2
elseif angle_functional_form == "ub" # UreyBradley
    const n_angle_params = 4
else
    error("unknown angle functional form $angle_functional_form")
end

const n_proper_terms   = MODEL_PARAMS["physics"]["n_proper_terms"]
const n_improper_terms = MODEL_PARAMS["physics"]["n_improper_terms"]

const torsion_periodicities = ntuple(i -> i, 6)
const torsion_phases = ntuple(i -> i % 2 == 0 ? T(π) : zero(T), 6)

# Some magic hackery. TODO: READ RELEVANT PAPERS
transform_lj_σ(x) = sigmoid(x) * T(0.42) + T(0.08) # 0.08 nm -> 0.5 nm
transform_lj_ϵ(x) = sigmoid(x) * T(0.95) + T(0.05) # 0.05 kJ/mol -> 1.0 kJ/mol

transform_dexp_α(x) = sigmoid(x) * T(8.0) + T(12.766) # 12.766 -> 20.766
transform_dexp_β(x) = sigmoid(x) * T(4.0) + T(2.427) # 2.427 -> 6.427

transform_buff_δ(x) = sigmoid(x) * T(0.08) + T(0.03) # 0.03 -> 0.11
transform_buff_γ(x) = sigmoid(x) * T(0.16) + T(0.04) # 0.04 -> 0.2

transform_buck_A(x) = sigmoid(x) * T(800_000.0) + T(100_000.0) # 100_000 kJ/mol -> 900_000 kJ/mol
transform_buck_B(x) = sigmoid(x) * T(40.0) + T(10.0) # 10 nm^-1 -> 50 nm^-1
transform_buck_C(x) = sigmoid(x) * T(0.75) + T(0.0) # 0 kJ/mol nm -> 0.75 kJ/mol nm this now reflects the change of how the potential is computed

transform_bond_k(  k1, k2) = max(k1 + k2, zero(T))
transform_angle_k( k1, k2) = max(k1 + k2, zero(T))
bond_r1, bond_r2   = T(MODEL_PARAMS["physics"]["bond_r1"]), T(MODEL_PARAMS["physics"]["bond_r2"])
angle_r1, angle_r2 = T(MODEL_PARAMS["physics"]["angle_r1"]), T(MODEL_PARAMS["physics"]["angle_r2"])
transform_bond_r0( k1, k2) = max(T((k1 * bond_r1  + k2 * bond_r2 ) / (k1 + k2)), zero(T))
transform_angle_θ0(k1, k2) = max(T((k1 * deg2rad(angle_r1) + k2 * deg2rad(angle_r2)) / (k1 + k2)), zero(T))


transform_morse_a(a) = max(a, zero(T))