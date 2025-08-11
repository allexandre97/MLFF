
"""
Writes an OpenMM-compatible XML for the given system.
"""
function write_openmm_xml(sys::Molly.System, filename::String)

    all_atoms = collect_atoms(sys)
    canonical_molecules = find_canonical_molecules(all_atoms, sys.topology.atom_molecule_inds)
    atomtypes_list = extract_unique_types(all_atoms)

    doc = XMLDocument()
    root = ElementNode("ForceField")
    setroot!(doc, root)

    build_atomtypes_node(root, atomtypes_list)
    build_residues_node(root, all_atoms, sys)
    build_bond_forces(root, all_atoms, canonical_molecules, sys)
    build_harmonic_angle_force(root, all_atoms, canonical_molecules, sys)
    build_torsion_forces(root, all_atoms, canonical_molecules, sys)
    build_improper_forces(root, all_atoms, canonical_molecules, sys)
    build_nonbonded_force(root, atomtypes_list, unique_all_atoms(all_atoms))

    open(filename, "w") do f
        prettyprint(f, doc)
    end
end

# --- Subfunctions ---

function collect_atoms(sys::Molly.System)
    data = BundledAtomData[]

    α = vdw_functional_form == "dexp" ? sys.pairwise_inters[1].inters[3].α : nothing
    β = vdw_functional_form == "dexp" ? sys.pairwise_inters[1].inters[3].β : nothing
    δ = vdw_functional_form == "buff" ? sys.pairwise_inters[1].inters[4].δ : nothing
    γ = vdw_functional_form == "buff" ? sys.pairwise_inters[1].inters[4].γ : nothing

    for (gatom, atom_data) in zip(sys.atoms, sys.atoms_data)

        if choice == 1
            σ = gatom.σ_lj
            ϵ = gatom.ϵ_lj
            A = nothing
            B = nothing
            C = nothing
        elseif choice == 2
            σ = gatom.σ_lj69
            ϵ = gatom.ϵ_lj69
            A = nothing
            B = nothing
            C = nothing
        elseif choice == 3
            σ = gatom.σ_dexp
            ϵ = gatom.ϵ_dexp
            A = nothing
            B = nothing
            C = nothing
        elseif choice == 4
            σ = gatom.σ_buff
            ϵ = gatom.ϵ_buff
            A = nothing
            B = nothing
            C = nothing
        elseif choice == 5
            σ = nothing
            ϵ = nothing
            A = gatom.A_buck
            B = gatom.B_buck
            C = gatom.C_buck
        end

        push!(data, BundledAtomData(
            atom_data.atom_name,
            atom_data.atom_type,
            atom_data.res_name,
            gatom.mass,
            gatom.charge,
            α,
            β,
            δ,
            γ,
            σ,
            ϵ,
            A,
            B,
            C
        ))
    end
    return data
end

function find_canonical_molecules(all_atoms, mol_inds)
    canonical = Vector{Vector{BundledAtomData}}()
    seen = Set{Vector{BundledAtomData}}()
    for mol_i in unique(mol_inds)
        inds = findall(==(mol_i), mol_inds)
        group = all_atoms[inds]
        if !(group in seen)
            push!(canonical, group)
            push!(seen, group)
        end
    end
    return canonical
end

function extract_unique_types(all_atoms)
    seen = Set{String}()
    types = Vector{Tuple{String,String,Float32}}()
    for atom in all_atoms
        if !(atom.type in seen)
            push!(seen, atom.type)
            elem = replace(atom.name, r"\d+" => "")
            push!(types, (atom.type, elem, atom.mass))
        end
    end
    return types
end

function build_atomtypes_node(root, atomtypes_list)
    atomtypes = ElementNode("AtomTypes")
    for (typ, elem, mass) in atomtypes_list
        node = ElementNode("Type")
        link!(node, AttributeNode("element", elem))
        link!(node, AttributeNode("name", typ))
        link!(node, AttributeNode("class", typ))
        link!(node, AttributeNode("mass", string(mass)))
        link!(atomtypes, node)
    end
    link!(root, atomtypes)
end

function build_residues_node(root, all_atoms, sys)
    residues = ElementNode("Residues")
    bonds = sys.specific_inter_lists[1]
    bond_pairs = zip(bonds.is, bonds.js)

    atom_name_map = Dict(i => atom.name for (i, atom) in enumerate(all_atoms))
    mol_inds = sys.topology.atom_molecule_inds
    seen = Set{String}()

    for mol_i in unique(mol_inds)
        inds = findall(==(mol_i), mol_inds)
        group = all_atoms[inds]
        resname = group[1].resname
        if resname in seen continue end
        push!(seen, resname)

        res = ElementNode("Residue")
        link!(res, AttributeNode("name", resname))

        idx_to_name = Dict{Int,String}(i => all_atoms[i].name for i in inds)
        for i in inds
            atom = all_atoms[i]
            at = ElementNode("Atom")
            link!(at, AttributeNode("name", atom.name))
            link!(at, AttributeNode("type", atom.type))
            link!(at, AttributeNode("charge", string(atom.charge)))
            link!(res, at)
        end
        for (i,j) in bond_pairs
            if i in inds && j in inds
                bond = ElementNode("Bond")
                link!(bond, AttributeNode("atomName1", idx_to_name[i]))
                link!(bond, AttributeNode("atomName2", idx_to_name[j]))
                link!(res, bond)
            end
        end
        link!(residues, res)
    end
    link!(root, residues)
end

function build_bond_forces(root, all_atoms, canon_mols, sys)
    bonds = sys.specific_inter_lists[1]
    pairs = zip(bonds.is, bonds.js, bonds.inters)

    harm   = ElementNode("HarmonicBondForce")
    custom = ElementNode("CustomBondForce")
    has_harm, has_morse = false, false
    seen = Set{Tuple{String,String}}()

    for group in canon_mols
        inds = findall(x->x in group, all_atoms)
        type_map = Dict(i=>all_atoms[i].type for i in inds)

        for (i,j,inter) in pairs
            if i in inds && j in inds
                t1, t2 = type_map[i], type_map[j]
                key = (t1,t2)
                if key in seen || (t2,t1) in seen continue end
                push!(seen, key)
                if inter isa HarmonicBond
                    has_harm = true
                    bond = ElementNode("Bond")
                    link!(bond, AttributeNode("type1", t1)); link!(bond, AttributeNode("type2", t2))
                    link!(bond, AttributeNode("length", string(inter.r0)))
                    link!(bond, AttributeNode("k", string(inter.k)))
                    link!(harm, bond)
                elseif inter isa MorseBond
                    has_morse = true
                    bond = ElementNode("Bond")
                    link!(bond, AttributeNode("type1", t1)); link!(bond, AttributeNode("type2", t2))
                    link!(bond, AttributeNode("D", string(inter.D)))
                    link!(bond, AttributeNode("a", string(inter.a)))
                    link!(bond, AttributeNode("r0", string(inter.r0)))
                    link!(custom, bond)
                end
            end
        end
    end
    if has_harm link!(root, harm) end
    if has_morse
        link!(custom, AttributeNode("energy", "D*(1-exp(-a*(r-r0)))^2"))
        for name in ("D","a","r0")
            link!(custom, link!(ElementNode("PerBondParameter"), AttributeNode("name", name)))
        end
        link!(root, custom)
    end
end

function build_harmonic_angle_force(root, all_atoms, canon_mols, sys)
    angles = sys.specific_inter_lists[2]
    harm = ElementNode("HarmonicAngleForce")
    seen = Set{Tuple{String,String,String}}()

    for group in canon_mols
        inds = findall(x->x in group, all_atoms)
        type_map = Dict(i=>all_atoms[i].type for i in inds)
        for (i,j,k,inter) in zip(angles.is, angles.js, angles.ks, angles.inters)
            if i in inds && j in inds && k in inds && inter isa HarmonicAngle
                t1, t2, t3 = type_map[i], type_map[j], type_map[k]
                key = (t1,t2,t3)
                if key in seen || reverse(key) in seen continue end
                push!(seen, key)
                angle = ElementNode("Angle")
                link!(angle, AttributeNode("type1", t1)); link!(angle, AttributeNode("type2", t2))
                link!(angle, AttributeNode("type3", t3)); link!(angle, AttributeNode("angle", string(inter.θ0)))
                link!(angle, AttributeNode("k", string(inter.k)))
                link!(harm, angle)
            end
        end
    end
    link!(root, harm)
end

function build_torsion_forces(root, all_atoms, canon_mols, sys)
    if length(sys.specific_inter_lists) < 3 return end
    torsions = sys.specific_inter_lists[3]
    force = ElementNode("PeriodicTorsionForce")
    seen = Set{NTuple{4,String}}()
    for group in canon_mols
        inds = findall(x->x in group, all_atoms)
        type_map = Dict(i=>all_atoms[i].type for i in inds)
        for (i,j,k,l,inter) in zip(torsions.is, torsions.js, torsions.ks, torsions.ls, torsions.inters)
            if all(x->x in inds, (i,j,k,l))
                key = (type_map[i],type_map[j],type_map[k],type_map[l])
                if key in seen || reverse(key) in seen continue end
                push!(seen, key)
                node = ElementNode("Proper")
                for (attr,val) in zip(("type1","type2","type3","type4"), key)
                    link!(node, AttributeNode(attr, val))
                end
                link!(node, AttributeNode("periodicity1", string(inter.periodicity)))
                link!(node, AttributeNode("phase1", string(inter.phase)))
                link!(node, AttributeNode("k1", string(inter.k)))
                link!(force, node)
            end
        end
    end
    link!(root, force)
end

function build_improper_forces(root, all_atoms, canon_mols, sys)
    if length(sys.specific_inter_lists) < 4 return end
    impropers = sys.specific_inter_lists[4]
    force = ElementNode("PeriodicTorsionForce")
    seen = Set{NTuple{4,String}}()
    for group in canon_mols
        inds = findall(x->x in group, all_atoms)
        type_map = Dict(i=>all_atoms[i].type for i in inds)
        for (i,j,k,l,inter) in zip(impropers.is, impropers.js, impropers.ks, impropers.ls, impropers.inters)
            if all(x->x in inds, (i,j,k,l))
                key = (type_map[i],type_map[j],type_map[k],type_map[l])
                if key in seen || reverse(key) in seen continue end
                push!(seen, key)
                node = ElementNode("Improper")
                for (attr,val) in zip(("type1","type2","type3","type4"), key)
                    link!(node, AttributeNode(attr, val))
                end
                link!(node, AttributeNode("periodicity1", string(inter.periodicity)))
                link!(node, AttributeNode("phase1", string(inter.phase)))
                link!(node, AttributeNode("k1", string(inter.k)))
                link!(force, node)
            end
        end
    end
    link!(root, force)
end

function build_nonbonded_force(root, atomtypes_list, unique_atoms)

    if vdw_functional_form == "lj"

        nb = ElementNode("NonbondedForce")
        link!(nb, AttributeNode("coulomb14scale", "$(model_global_params.params[2])"))
        link!(nb, AttributeNode("lj14scale", "0.5"))

        # Properly create and attach the UseAttributeFromResidue node with its attribute
        use_charge = ElementNode("UseAttributeFromResidue")
        link!(use_charge, AttributeNode("name", "charge"))
        link!(nb, use_charge)

        for (typ, _, _) in atomtypes_list
            atom = unique_atoms[typ]
            nb_atom = ElementNode("Atom")
            link!(nb_atom, AttributeNode("type", typ))
            link!(nb_atom, AttributeNode("sigma", string(atom.σ)))
            link!(nb_atom, AttributeNode("epsilon", string(atom.ϵ)))
            link!(nb, nb_atom)
        end

        link!(root, nb)
    
    else
        nb = ElementNode("NonbondedForce")
        link!(nb, AttributeNode("coulomb14scale", "$(model_global_params.params[2])"))
        link!(nb, AttributeNode("lj14scale", "0.5"))
        use_charge = ElementNode("UseAttributeFromResidue")
        link!(use_charge, AttributeNode("name", "charge"))
        link!(nb, use_charge)
        for (typ, _, _) in atomtypes_list
            atom = unique_atoms[typ]
            nb_atom = ElementNode("Atom")
            link!(nb_atom, AttributeNode("type", typ))
            link!(nb_atom, AttributeNode("sigma", "1"))
            link!(nb_atom, AttributeNode("epsilon", "0"))
            link!(nb, nb_atom)
        end
        link!(root, nb)

        if vdw_functional_form == "lj69"

            nb = ElementNode("CustomNonbondedForce")
            link!(nb, AttributeNode("energy", "sqrt(epsilon1*epsilon2)*((((sigma1 + sigma2) / 2) / r)^9 - (((sigma1 + sigma2) / 2) / r)^6)"))
            link!(nb, AttributeNode("bondCutoff", "3"))

            # Properly create and attach the UseAttributeFromResidue node with its attribute
            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "sigma"))
            link!(nb, perpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "epsilon"))
            link!(nb, perpar)

            for (typ, _, _) in atomtypes_list
                atom = unique_atoms[typ]
                nb_atom = ElementNode("Atom")
                link!(nb_atom, AttributeNode("type", typ))
                link!(nb_atom, AttributeNode("sigma", string(atom.σ)))
                link!(nb_atom, AttributeNode("epsilon", string(atom.ϵ)))
                link!(nb, nb_atom)
            end

            link!(root, nb)

        elseif vdw_functional_form == "dexp"

            α = unique_atoms[atomtypes_list[1][1]].α
            β = unique_atoms[atomtypes_list[1][1]].α

            nb = ElementNode("CustomNonbondedForce")
            link!(nb, AttributeNode("energy", "sqrt(epsilon1*epsilon2)*(((β*exp(α))/(α-β))*exp(-α*(r/((2^(1/6))*(sigma1+sigma2)/2)))-((α*exp(β))/(α-β))*exp(-β*(r/((2^(1/6))*(sigma1+sigma2)/2))))"))
            link!(nb, AttributeNode("bondCutoff", "3"))

            # Properly create and attach the UseAttributeFromResidue node with its attribute
            globpar = ElementNode("GlobalParameter")
            link!(globpar, AttributeNode("name", "α"))
            link!(globpar, AttributeNode("defaultValue", string(α)))
            link!(nb, globpar)

            globpar = ElementNode("GlobalParameter")
            link!(globpar, AttributeNode("name", "β"))
            link!(globpar, AttributeNode("defaultValue", string(β)))
            link!(nb, globpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "sigma"))
            link!(nb, perpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "epsilon"))
            link!(nb, perpar)

            for (typ, _, _) in atomtypes_list
                atom = unique_atoms[typ]
                nb_atom = ElementNode("Atom")
                link!(nb_atom, AttributeNode("type", typ))
                link!(nb_atom, AttributeNode("sigma", string(atom.σ)))
                link!(nb_atom, AttributeNode("epsilon", string(atom.ϵ)))
                link!(nb, nb_atom)
            end

            link!(root, nb)

        elseif vdw_functional_form == "buff"

            δ = unique_atoms[atomtypes_list[1][1]].δ
            γ = unique_atoms[atomtypes_list[1][1]].γ

            nb = ElementNode("CustomNonbondedForce")
            link!(nb, AttributeNode("energy", "sqrt(epsilon1*epsilon2)*(((1+δ)/((r/(((sigma1+sigma2)/2)*(2^(1/6))))+δ))^7)*(((1+γ)/(((r/(((sigma1+sigma2)/2)*(2^(1/6))))^7)+γ))-2)"))
            link!(nb, AttributeNode("bondCutoff", "3"))

            # Properly create and attach the UseAttributeFromResidue node with its attribute
            globpar = ElementNode("GlobalParameter")
            link!(globpar, AttributeNode("name", "δ"))
            link!(globpar, AttributeNode("defaultValue", string(δ)))
            link!(nb, globpar)

            globpar = ElementNode("GlobalParameter")
            link!(globpar, AttributeNode("name", "γ"))
            link!(globpar, AttributeNode("defaultValue", string(γ)))
            link!(nb, globpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "sigma"))
            link!(nb, perpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "epsilon"))
            link!(nb, perpar)

            for (typ, _, _) in atomtypes_list
                atom = unique_atoms[typ]
                nb_atom = ElementNode("Atom")
                link!(nb_atom, AttributeNode("type", typ))
                link!(nb_atom, AttributeNode("sigma", string(atom.σ)))
                link!(nb_atom, AttributeNode("epsilon", string(atom.ϵ)))
                link!(nb, nb_atom)
            end

            link!(root, nb)

        elseif vdw_functional_form == "buck"

            nb = ElementNode("CustomNonbondedForce")
            link!(nb, AttributeNode("energy", "W1*W2*(((A1+A2)/2)*exp(((B1+B2)/2)*(-r)) - (((C1+C2)/2)/r)^6)"))
            link!(nb, AttributeNode("bondCutoff", "3"))

            # Properly create and attach the UseAttributeFromResidue node with its attribute
            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "A"))
            link!(nb, perpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "B"))
            link!(nb, perpar)

            perpar = ElementNode("PerParticleParameter")
            link!(perpar, AttributeNode("name", "C"))
            link!(nb, perpar)

            for (typ, _, _) in atomtypes_list
                atom = unique_atoms[typ]
                nb_atom = ElementNode("Atom")
                link!(nb_atom, AttributeNode("type", typ))
                link!(nb_atom, AttributeNode("A", string(atom.A)))
                link!(nb_atom, AttributeNode("B", string(atom.B)))
                link!(nb_atom, AttributeNode("C", string(atom.C)))
                link!(nb, nb_atom)
            end

            link!(root, nb)

        end

    end

end

# Helper to map type=>BundledAtomData
function unique_all_atoms(all_atoms)
    dict = Dict{String,BundledAtomData}()
    for atom in all_atoms
        dict[atom.type] = atom
    end
    return dict
end

function features_to_ff_xml(out_fp::AbstractString, args...)
    open(out_fp, "w") do of
        features_to_ff_xml(of, args...)
    end
end

function features_to_xml(io,
    epoch_n::Int,
    mol_id::String,
    frame_i,
    temp,
    feat_df::DataFrame,
    models...)


    cond_sim_dir = joinpath(DATASETS_PATH, "condensed_data", "trajs_gaff")

    coords, boundary = read_sim_data(mol_id, cond_sim_dir, frame_i, temp)
    feat_df = feat_df[feat_df.MOLECULE .== mol_id, :]

    sys, _ = mol_to_system(epoch_n, mol_id, feat_df, coords, boundary, models...)

    write_openmm_xml(sys, io)

end
