function multi_mol_charge_factors!(mol_charge_factors, charge_e_inv_s, charge_inv_s,
                                   formal_charges, molecule_inds, n_molecules)
    for mi in 1:n_molecules
        mol_charge, mol_charge_e_inv_s, mol_charge_inv_s = zero(T), zero(T), zero(T)
        for (ai, ami) in enumerate(molecule_inds)
            if ami == mi
                mol_charge += formal_charges[ai]
                mol_charge_e_inv_s += charge_e_inv_s[ai]
                mol_charge_inv_s += charge_inv_s[ai]
            end
        end
        mol_charge_factors[mi] = (mol_charge + mol_charge_e_inv_s) / mol_charge_inv_s
    end
    return mol_charge_factors
end



function multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                  molecule_inds, n_molecules)
    mol_charge_factors = zeros(T, n_molecules)
    multi_mol_charge_factors!(mol_charge_factors, charge_e_inv_s, charge_inv_s, formal_charges,
                              molecule_inds, n_molecules)
    return mol_charge_factors
end

function ChainRulesCore.rrule(::typeof(multi_mol_charge_factors), charge_e_inv_s, charge_inv_s,
                              formal_charges, molecule_inds, n_molecules)
    Y = multi_mol_charge_factors(charge_e_inv_s, charge_inv_s, formal_charges,
                                 molecule_inds, n_molecules)
    function multi_mol_charge_factors_pullback(d_mol_charge_factors)
        mol_charge_factors = zeros(T, n_molecules)
        d_charge_e_inv_s = zero(charge_e_inv_s)
        d_charge_inv_s = zero(charge_inv_s)
        Enzyme.autodiff(
            Enzyme.Reverse,
            multi_mol_charge_factors!,
            Enzyme.Const,
            Enzyme.Duplicated(mol_charge_factors, d_mol_charge_factors),
            Enzyme.Duplicated(charge_e_inv_s, d_charge_e_inv_s),
            Enzyme.Duplicated(charge_inv_s, d_charge_inv_s),
            Enzyme.Const(formal_charges),
            Enzyme.Const(molecule_inds),
            Enzyme.Const(n_molecules),
        )
        return NoTangent(), d_charge_e_inv_s, d_charge_inv_s, NoTangent(),
               NoTangent(), NoTangent()
    end
    return Y, multi_mol_charge_factors_pullback
end

function atom_feats_to_charges(charges_k1::Vector{T}, charges_k2::Vector{T}, formal_charges::Vector{Int})
    e       = charges_k1
    inv_s   = inv.(charges_k2)
    e_over_s = e .* inv_s

    charge_factor = (sum(formal_charges) + sum(e_over_s)) / sum(inv_s)
    return -one(T)*(-e_over_s .+ inv_s .* charge_factor)
end

function extract_vdw_params(feats::Matrix{T}, layout::Dict{Symbol,Tuple{UnitRange{Int}}}) where {T}
    return Dict(k => feats[i, :] for (k, (i,)) in layout)
end

function combine_vdw_params_gumbel!(
    σs::Vector{T},
    ϵs::Vector{T},
    A::Vector{T},
    B::Vector{T},
    C::Vector{T},
    feats::Matrix{T},
    func_probs::Matrix{T}
)

    for (f_idx, f) in enumerate(keys(PARAM_LAYOUT))
        inds = PARAM_LAYOUT[f][1]
        w = func_probs[f_idx, :]
        p = feats[inds, :]

        if f in (:lj, :lj69, :dexp, :buff)
            σs .+= transform_lj_σ.(w .* p[1, :])
            ϵs .+= transform_lj_ϵ.(w .* p[2, :])
        end
        if f == :buck
            A .+= transform_buck_A.(w .* p[1, :])
            B .+= transform_buck_B.(w .* p[2, :])
            C .+= transform_buck_C.(w .* p[3, :])
        end
    end
end

function combine_vdw_params_gumbel(
    feats::Matrix{T},
    func_probs::Matrix{T}
)

    n_atoms = size(feats, 2)
    σs, ϵs, A, B, C = zeros(T, n_atoms), zeros(T, n_atoms), zeros(T, n_atoms), zeros(T, n_atoms), zeros(T, n_atoms)

    combine_vdw_params_gumbel!(σs, ϵs, A, B, C, feats, func_probs)

    return σs, ϵs, A, B, C
end


function ChainRulesCore.rrule(
    ::typeof(combine_vdw_params_gumbel),
    feats::Matrix{T},
    func_probs::Matrix{T}
)

    Y = combine_vdw_params_gumbel(feats, func_probs)

    function pullback(ŷ)
        n_atoms = size(feats, 2)

        σs = zeros(T, n_atoms)
        ϵs = zeros(T, n_atoms)
        A  = zeros(T, n_atoms)
        B  = zeros(T, n_atoms)
        C  = zeros(T, n_atoms)

        d_σs = iszero(ŷ[1]) ? zero(σs) : ŷ[1]
        d_ϵs = iszero(ŷ[2]) ? zero(ϵs) : ŷ[2]
        d_A  = iszero(ŷ[3]) ? zero(A)  : ŷ[3]
        d_B  = iszero(ŷ[4]) ? zero(B)  : ŷ[4]
        d_C  = iszero(ŷ[5]) ? zero(C)  : ŷ[5]

        d_feats = zeros(T, size(feats))
        d_func_probs = zeros(T, size(func_probs))

        Enzyme.autodiff(
            Enzyme.Reverse,
            combine_vdw_params_gumbel!,
            Enzyme.Const,  # function itself
            Enzyme.Duplicated(σs, d_σs),
            Enzyme.Duplicated(ϵs, d_ϵs),
            Enzyme.Duplicated(A,  d_A),
            Enzyme.Duplicated(B,  d_B),
            Enzyme.Duplicated(C,  d_C),
            Enzyme.Duplicated(feats, d_feats),
            Enzyme.Duplicated(func_probs, d_func_probs)
        )

        return NoTangent(), d_feats, d_func_probs
    end

    return Y, pullback
end


function atom_feats_to_vdW(
    atom_features
)
    #=
    Working towards not building atom lists in this method. I want to Keep
    all he Molly steps in a separate module.
    This method could return a Dict with all the needed params. 
    =#
    vdw_functional_form = MODEL_PARAMS["physics"]["vdw_functional_form"]
    weight_vdw = (vdw_functional_form == "lj" ? sigmoid(global_params[1]) : one(T))

    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")

        σs = transform_lj_σ.(atom_features[3, :])
        ϵs = transform_lj_ϵ.(atom_features[4, :])
        vdw_params_size = mean(σs) + mean(ϵs)/2.0
    
        if vdw_functional_form in ("lj", "lj69")
            return σs, ϵs, nothing, nothing

        elseif vdw_functional_form == "dexp"
            α = transform_dexp_α(global_params[3])
            β = transform_dexp_β(global_params[4])
            return σs, ϵs, α, β

        elseif vdw_functional_form == "buff"
            δ = transform_buff_δ(global_params[3])
            γ = transform_buff_γ(global_params[4])
            return σs, ϵs, δ, γ
        end

    elseif vdw_functional_form == "buck"
        As = transform_buck_A.(atom_features[3, :])
        Bs = transform_buck_B.(atom_features[4, :])
        Cs = transform_buck_C.(atom_features[5, :])
        return As, Bs, Cs, nothing
    end

    #=
    TODO: Maybe add functionality to implement Neural Network vdW Forces?
    I woud rather have all based off analytical forms. 
    =#

end

function feats_to_bonds(
    bond_feats
)
    bond_functional_form = MODEL_PARAMS["physics"]["bond_functional_form"]

    if bond_functional_form == "harmonic"
        k1 = T(100000.0)*softplus(bond_feats[1, :])
        k2 = T(100000.0)*softplus(bond_feats[2, :])
        k  = transform_bond_k.(k1, k2)
        r0 = transform_bond_r0.(k1, k2)
        return k, r0, nothing

    elseif bond_functional_form == "morse"
        k  = transform_bond_k.(bond_feats[1, :], bond_feats[2, :])
        r0 = transform_morse_a.(bond_feats[3, :])
        a  = transform_bond_r0.(bond_feats[1, :], bond_feats[2, :])
        return k, r0, a
    end

end

function feats_to_angles(
    angle_feats
)

    angle_functional_form = MODEL_PARAMS["physics"]["angle_functional_form"]

    if angle_functional_form == "harmonic"
        k1 = T(1000.0)*softplus(angle_feats[1, :])
        k2 = T(1000.0)*softplus(angle_feats[2, :])
        k  = transform_angle_k.(k1, k2)
        θ0 = transform_angle_θ0.(k1, k2)
        return k, θ0, nothing, nothing

    elseif angle_functional_form == "ub"
        ki  = transform_angle_k.(angle_feats[1, :], angle_feats[2, :])
        θ0i = transform_angle_θ0.(angle_feats[1, :], angle_feats[2, :])
        kj  = transform_angle_k.(angle_feats[3, :], angle_feats[4, :])
        θ0j = transform_angle_θ0.(angle_feats[3, :], angle_feats[4, :])
        return ki, θ0i, kj, θ0j
    end
    
end