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
    
        if vdw_functional_form == "lj"
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
        Bs = transform_buck_A.(atom_features[4, :])
        Cs = transform_buck_A.(atom_features[5, :])
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
        k1, k2 = softplus(bond_feats[1, :]), softplus(bond_feats[2, :])
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
        k1 = softplus(angle_feats[1, :])
        k2 = softplus(angle_feats[2, :])
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