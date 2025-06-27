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

const ATOMIC_MASSES = [
    1.008 , 6.94        , 10.81, 12.011, 14.007 , 15.999, 18.998403163, 22.98976928, 24.305,
    28.085, 30.973761998, 32.06, 35.45 , 39.0983, 40.078, 79.904      , 126.90447  ,
]

# Initialize some constants depending on the functional form of vdW interactions
if vdw_functional_form in ("lj", "lj69")
    const n_vdw_atom_params = 2
    const global_params = [starting_weight14_vdw, starting_weight14_coul]
elseif vdw_functional_form in ("dexp", "buff")
    const n_vdw_atom_params = 2
    const global_params = [starting_weight14_vdw, starting_weight14_coul, zero(T), zero(T)]
elseif vdw_functional_form == "buck"
    const n_vdw_atom_params = 3
    const global_params = [starting_weight14_vdw, starting_weight14_coul]
elseif vdw_functional_form == "nn"
    training_sims_first_epoch == 0 || error("cannot run training simulations with vdw functional form nn")
    const n_vdw_atom_params = nn_dim_atom
    const n_params_pairwise = (2 * nn_dim_atom + 3 + 1 + 1) * dim_hidden_pairwise
    const global_params = vcat(starting_weight14_vdw, T.(Flux.kaiming_uniform(n_params_pairwise)))
else
    error("unknown vdw functional form $vdw_functional_form")
end

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
    error("unknown bond functional form $bond_functional_form")
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
transform_buck_C(x) = sigmoid(x) * T(0.0095) + T(0.0005) # 0.0005 kJ/mol nm^6 -> 0.01 kJ/mol nm^6

transform_bond_k(  k1, k2) = max(k1 + k2, zero(T))
transform_angle_k( k1, k2) = max(k1 + k2, zero(T))
bond_r1, bond_r2 = MODEL_PARAMS["physics"]["bond_r1"], MODEL_PARAMS["physics"]["bond_r2"]
angle_r1, angle_r2 = MODEL_PARAMS["physics"]["angle_r1"], MODEL_PARAMS["physics"]["angle_r2"]
transform_bond_r0( k1, k2) = max((k1 * bond_r1  + k2 * bond_r2 ) / (k1 + k2), zero(T))
transform_angle_θ0(k1, k2) = max((k1 * deg2rad(angle_r1) + k2 * deg2rad(angle_r2)) / (k1 + k2), zero(T))

transform_morse_a(a) = max(a, zero(T))