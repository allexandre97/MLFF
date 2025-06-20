using HDF5
using AutomaticDocstrings

@autodoc
function read_conf_data(mol_order, start_i, end_i)
    #=
     
    =#
    hdf5_list = [h5open(joinpath(data_dir, hdf5_file), "r") for hdf5_file in hdf5_files]
    map_res = map(start_i:end_i) do i
        mol_id, conf_i, conf_i_p1, repeat_i = mol_order[i]
        mol_hdf5_or_xyz, _, _ = extract_hdf5_or_xyz(mol_id, hdf5_list)
        coords = read_coordinates(mol_hdf5_or_xyz, conf_i)
        dft_fs, exceeds_max = read_dft_forces(mol_hdf5_or_xyz, conf_i)
        dft_pe = read_dft_pe(mol_hdf5_or_xyz, conf_i)
        dft_charges, has_charges = read_dft_charges(mol_hdf5_or_xyz, conf_i)
        pair_present = !iszero(conf_i_p1)
        if pair_present
            coords_p1 = read_coordinates(mol_hdf5_or_xyz, conf_i_p1)
            dft_fs_p1, exceeds_max_p1 = read_dft_forces(mol_hdf5_or_xyz, conf_i_p1)
            dft_pe_p1 = read_dft_pe(mol_hdf5_or_xyz, conf_i_p1)
            dft_charges_p1, has_charges_p1 = read_dft_charges(mol_hdf5_or_xyz, conf_i_p1)
        else
            coords_p1, dft_fs_p1, exceeds_max_p1, dft_pe_p1, dft_charges_p1, has_charges_p1 = coords,
                                dft_fs, exceeds_max, dft_pe, dft_charges, has_charges
        end
        exceeds_max_either = exceeds_max || exceeds_max_p1
        return coords, dft_fs, dft_pe, dft_charges, has_charges, coords_p1, dft_fs_p1,
                dft_pe_p1, dft_charges_p1, has_charges_p1, exceeds_max_either, pair_present
    end
    close.(hdf5_list)
    return map_res
end

