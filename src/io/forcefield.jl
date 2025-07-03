function generate_forcefield_xml(
    atoms, mol_id,
    bond_names_by_moltype, bond_params_by_moltype,
    angle_names_by_moltype, angle_params_by_moltype,
    proper_names_by_moltype, proper_params_by_moltype;
    filename = "generated_forcefield.xml"
)
    doc = XMLDocument()
    root = ElementNode("ForceField")
    setroot!(doc, root)

    # --- AtomTypes ---
    atomtypes = ElementNode("AtomTypes")
    for (name, mass, _) in unique(atoms)
        elem = replace(name, r"\d+" => "")
        node = ElementNode("Type")
        link!(node, AttributeNode("name", name))
        link!(node, AttributeNode("class", name))
        link!(node, AttributeNode("element", elem))
        link!(node, AttributeNode("mass", string(mass)))
        link!(atomtypes, node)
    end
    link!(root, atomtypes)

    # --- Residues ---
    residues = ElementNode("Residues")
    mol_groups = Dict{Int, Vector{Int}}()
    for (i, mid) in enumerate(mol_id)
        push!(get!(mol_groups, mid, Int[]), i)
    end

    for (mol_i, indices) in mol_groups
        res = ElementNode("Residue")
        link!(res, AttributeNode("name", "MOL$mol_i"))
        for idx in indices
            atomname, _, charge = atoms[idx]
            at = ElementNode("Atom")
            link!(at, AttributeNode("name", atomname))
            link!(at, AttributeNode("type", atomname))
            link!(at, AttributeNode("charge", string(charge)))
            link!(res, at)
        end
        link!(residues, res)
    end
    link!(root, residues)

    # --- HarmonicBondForce ---
    bond_force = ElementNode("HarmonicBondForce")
    for (moltype_i, bond_list) in enumerate(bond_names_by_moltype)
        for (j, (a1, a2)) in enumerate(bond_list)
            k, r0 = bond_params_by_moltype[moltype_i][j]
            b = ElementNode("Bond")
            link!(b, AttributeNode("class1", a1))
            link!(b, AttributeNode("class2", a2))
            link!(b, AttributeNode("length", string(r0)))
            link!(b, AttributeNode("k", string(k)))
            link!(bond_force, b)
        end
    end
    link!(root, bond_force)

    # --- HarmonicAngleForce ---
    angle_force = ElementNode("HarmonicAngleForce")
    for (moltype_i, angle_list) in enumerate(angle_names_by_moltype)
        for (j, (a1, a2, a3)) in enumerate(angle_list)
            k, theta0 = angle_params_by_moltype[moltype_i][j]
            a = ElementNode("Angle")
            link!(a, AttributeNode("class1", a1))
            link!(a, AttributeNode("class2", a2))
            link!(a, AttributeNode("class3", a3))
            link!(a, AttributeNode("angle", string(theta0)))
            link!(a, AttributeNode("k", string(k)))
            link!(angle_force, a)
        end
    end
    link!(root, angle_force)

    # --- PeriodicTorsionForce ---
    torsion_force = ElementNode("PeriodicTorsionForce")
    for (moltype_i, torsion_list) in enumerate(proper_names_by_moltype)
        for (j, (a1, a2, a3, a4)) in enumerate(torsion_list)
            k, phase, n = proper_params_by_moltype[moltype_i][j]
            t = ElementNode("Proper")
            link!(t, AttributeNode("class1", a1))
            link!(t, AttributeNode("class2", a2))
            link!(t, AttributeNode("class3", a3))
            link!(t, AttributeNode("class4", a4))
            link!(t, AttributeNode("periodicity", string(n)))
            link!(t, AttributeNode("phase", string(phase)))
            link!(t, AttributeNode("k", string(k)))
            link!(torsion_force, t)
        end
    end
    link!(root, torsion_force)

    # --- Write XML ---
    open(filename, "w") do f
        prettyprint(f, doc)
    end
end

function features_to_ff_xml(out_fp::AbstractString, args...)
    open(out_fp, "w") do of
        features_to_ff_xml(of, args...)
    end
end

function features_to_xml(io, 
    mol_id::String,
    training_sim_dir,
    frame_i,
    temp,
    feat_df::DataFrame,
    models...)

    coords, boundary = read_sim_data(mol_id, training_sim_dir, frame_i, temp)
    feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]

    sys, partial_charges, vdw_size, torsion_size, elements, mol_inds = mol_to_system(mol_id, feat_df, coords, boundary, models...)

    return sys

end
