function read_sim_data(mol_id, training_sim_dir, frame_i, temp=nothing)
    
    exp_type, sim_type, smiles = split(mol_id, "_"; limit=3)
    if exp_type == "vapourisation"
        traj_fp = "$(smiles)_$(Int(temp))K.dcd"
    else
        traj_fp = "$smiles.dcd"
    end
    traj = Chemfiles.Trajectory(joinpath(training_sim_dir, "$(exp_type)_$sim_type", traj_fp))
    frame = Chemfiles.read_step(traj, frame_i - 1) # Zero-based indexing
    pos = Chemfiles.positions(frame)
    coords_unordered = SVector{3, T}.(eachcol(pos)) ./ 10 # Convert to nm
    if exp_type == "mixing" && sim_type == "combined"
        # PDB files have molecules in order 1,1,2,2 so we reorder to 1,2,1,2
        molecule_inds_str = split(mol_features_cond[mol_id], "\t")[end]
        molecule_inds = parse.(Int, split(molecule_inds_str, ","))
        n_molecules_each = maximum(molecule_inds) ÷ 2
        n_atoms_1 = findlast(isequal(1), molecule_inds)
        n_atoms_2 = findlast(isequal(2), molecule_inds) - n_atoms_1
        n_atoms_com = n_atoms_1 + n_atoms_2
        n_atoms_1_all = n_atoms_1 * n_molecules_each
        coords = zero(coords_unordered)
        for mi in 1:n_molecules_each
            for ai in 1:n_atoms_1
                coords[(mi-1) * n_atoms_com + ai] = coords_unordered[(mi-1) * n_atoms_1 + ai]
            end
            for ai in 1:n_atoms_2
                coords[(mi-1) * n_atoms_com + n_atoms_1 + ai] = coords_unordered[n_atoms_1_all + (mi-1) * n_atoms_2 + ai]
            end
        end
    else
        coords = coords_unordered
    end
    if sim_type == "gas"
        boundary = boundary_inf
    else
        box_sides = T.(Chemfiles.lengths(Chemfiles.UnitCell(frame))) ./ 10 # Convert to nm
        boundary = CubicBoundary(box_sides...)
        coords .= wrap_coords.(coords, (boundary,))
    end
    return coords, boundary
end

Flux.@non_differentiable read_sim_data(args...)