using ChainRulesCore
using Enzyme

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

function atom_feats_to_charges(
    mol_id::String,
    n_atoms::Int,
    n_molecules::Int,
    atom_features,
    formal_charges,
    molecule_inds
)

    charge_e = atom_features[1, :]
    charge_inv_s = inv.(atom_features[2, :])
    charge_e_inv_s = charge_e .* charge_inv_s

    if any(startswith.(mol_id, ("vapourisation_", "mixing_")))
        n_atoms_mol = n_atoms ÷ n_molecules
        charge_factor = (sum(formal_charges[1:n_atoms_mol]) + sum(charge_e_inv_s[1:n_atoms_mol])) /
                        sum(charge_inv_s[1:n_atoms_mol])
        charge_factors = fill(charge_factor, n_atoms)
    elseif n_molecules == 1
        charge_factor = (sum(formal_charges) + sum(charge_e_inv_s)) / sum(charge_inv_s)
        charge_factors = fill(charge_factor, n_atoms)
    else
        mol_charge_factor = multi_mol_charge_factors(charge_e_inv_s, charge_inv_s,
                                                 formal_charges, molecule_inds,
                                                 n_molecules)
        charge_factors = [mol_charge_factor[mi] for mi in molecule_inds]
    end

    return -charge_e_inv_s .+ charge_inv_s .* charge_factors

end