function build_adj_list(mol_row::DataFrame)::Array
    
    n_atoms::Int16    = size(split(mol_row[!, :ATOMIC_MASS][1], ","))[1]
    bonds_list::Array = split(mol_row[!, :BONDS][1], ",")

    adj_list::Array = [[i] for i in 1:n_atoms]

    for bond in bonds_list
        i, j = parse.(Int16, split(bond, "/"))
        push!(adj_list[i], j)
        push!(adj_list[j], i)
    end

    return adj_list
end

function build_adj_list(g)
    adj_mol  = [Int[] for _ in 1:nv(g)]
    for e in edges(g)
        u, v = src(e), dst(e)
        push!(adj_mol[u], v)
        push!(adj_mol[v], u)
    end
    return adj_mol
end

Flux.@non_differentiable build_adj_list(mol_row::DataFrame)
Flux.@non_differentiable build_adj_list(args...)
Flux.@non_differentiable has_isomorph(args...)


function mol_to_preds(
    mol_id::String,
    args...
)

    sys, partial_charges, vdw_size, torsion_size, elements, mol_inds = mol_to_system(mol_id, args...)
    neighbors = find_neighbors(sys; n_threads = 1)
    # Get interaction lists separate depending on the number of atoms involves
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
    if any(startswith.(mol_id, ("vapourisation_", "mixing_", "protein_")))
        forces = nothing
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    else
        forces = forces_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                             sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
        potential = pe_wrap(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.pairwise_inters,
                            sils_2_atoms, sils_3_atoms, sils_4_atoms, neighbors)
    end

    return sys, forces, potential, partial_charges, vdw_size, torsion_size, elements, mol_inds

end

function setup_torsions(features_pad, periodicities, phases, proper)
    return broadcast(1:size(features_pad, 2)) do i
        PeriodicTorsion{6, T, T}(
            periodicities,                      # periodicities
            phases,                             # phases
            ntuple(j -> features_pad[j, i], 6), # ks
            proper,                             # proper
        )
    end
end

function generate_neighbors(
    n_atoms,
    bond_is, bond_js,
    angle_is, angle_ks,
    proper_is, proper_ls,
    dist_nb_cutoff,
)

    eligible = trues(n_atoms, n_atoms)
    for (i, j) in zip(bond_is, bond_js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(angle_is, angle_ks)
        eligible[i, k] = false
        eligible[k, i] = false
    end

    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(proper_is, proper_ls)
        special[i, l] = true
        special[l, i] = true
    end

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        special=special,
        dist_cutoff=(dist_nb_cutoff + T(0.001)),
    )

    return neighbor_finder
end

@non_differentiable generate_neighbors(args...)

# Can this function be non-differentiable?? --> It cannot, it takes as input (args) things that the NNet is predicting!
function build_sys(
    mol_id,
    masses,
    atom_types,
    atom_names,
    mol_inds,
    coords,
    boundary,
    partial_charges,
    vdw_functional_form::String,
    weight_vdw::Float32,
    
    σ::Union{Vector{Float32}, Nothing},
    ε::Union{Vector{Float32}, Nothing},
    A::Union{Vector{Float32}, Nothing},
    B::Union{Vector{Float32}, Nothing},
    C::Union{Vector{Float32}, Nothing},
    α::Union{Float32, Nothing},
    β::Union{Float32, Nothing},
    δ::Union{Float32, Nothing},
    γ::Union{Float32, Nothing},

    bond_functional_form::String,
    bond_k::Vector{Float32},
    bond_r0::Vector{Float32},
    bond_a::Union{Vector{Float32}, Nothing},

    angle_functional_form::String,
    angle_k::Union{Vector{Float32}, Nothing},
    angle_θ0::Union{Vector{Float32}, Nothing},
    angle_kj::Union{Vector{Float32}, Nothing},
    angle_θ0j::Union{Vector{Float32}, Nothing},

    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    proper_feats, improper_feats,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l
)
    n_atoms = length(partial_charges)
    dist_nb_cutoff = T(MODEL_PARAMS["physics"]["dist_nb_cutoff"])

    ########## van der Waals section ##########
    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
        atoms = [Atom(i, one(T), masses[i], partial_charges[i], σ[i], ε[i]) for i in 1:n_atoms]
        if vdw_functional_form == "lj"
            inter_vdw = LennardJones(DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, weight_vdw)
        elseif vdw_functional_form == "lj69"
            inter_vdw = Mie(6, 9, DistanceCutoff(dist_nb_cutoff), true, Molly.lj_zero_shortcut, σ_mixing, ϵ_mixing, weight_vdw, 1)
        elseif vdw_functional_form == "dexp"
            inter_vdw = DoubleExponential(α, β, σ_mixing, ϵ_mixing, weight_vdw, dist_nb_cutoff)
        elseif vdw_functional_form == "buff"
            inter_vdw = Buffered147(δ, γ, σ_mixing, ϵ_mixing, weight_vdw, dist_nb_cutoff)
        end

    elseif vdw_functional_form == "buck"
        atoms = [BuckinghamAtom(i, one(T), masses[i], partial_charges[i], A[i], B[i], C[i]) for i in 1:n_atoms]
        inter_vdw = Buckingham(weight_vdw, dist_nb_cutoff)

    elseif vdw_functional_form == "nn"
        # TODO: Add functionality for NNet vdW interactions
    end

    ########## Coulomb interactions section ##########
    if vdw_functional_form == "nn"
        # Placeholder
    else
        weight_14_coul = sigmoid(global_params[2])

        if MODEL_PARAMS["physics"]["use_reaction_field"] &&
            any(startswith.(mol_id, ("vapourisation_liquid_", "mixing_", "protein_")))
            inter_coulomb = CoulombReactionField(dist_nb_cutoff, T(Molly.crf_solvent_dielectric),
                                                 true, weight_14_coul, T(ustrip(Molly.coulomb_const)))
        else
            inter_coulomb = Coulomb(DistanceCutoff(dist_nb_cutoff),
                                    true, weight_14_coul, T(ustrip(Molly.coulomb_const)))
        end

        pairwise_inter = (inter_vdw, inter_coulomb)
    end

    ########## Bond Interactions section ##########
    if bond_functional_form == "harmonic"
        bond_inter = HarmonicBond.(T.(bond_k), T.(bond_r0))
    elseif bond_functional_form == "morse"
        bond_inter = MorseBond.(T.(bond_k), T.(bond_a), T.(bond_r0))
    end
    bonds = InteractionList2Atoms(bonds_i, bonds_j, bond_inter)

    ########## Angle Interactions section ##########
    if angle_functional_form == "harmonic"
        angle_inter = HarmonicAngle.(T.(angle_k), T.(angle_θ0))
    end
    angles = InteractionList3Atoms(angles_i, angles_j, angles_k, angle_inter)

    ######### Torsion Interactions section ##########
    proper_inter = setup_torsions(proper_feats, torsion_periodicities, torsion_phases, true)
    propers = InteractionList4Atoms(propers_i, propers_j, propers_k, propers_l, proper_inter)

    improper_inter = setup_torsions(improper_feats, torsion_periodicities, torsion_phases, false)
    impropers = InteractionList4Atoms(impropers_j, impropers_k, impropers_i, impropers_l, improper_inter)

    if length(propers_i) > 0 && length(impropers_i) > 0
        specific_inter_lists = (bonds, angles, propers, impropers)
    elseif length(propers_i) > 0
        specific_inter_lists = (bonds, angles, propers)
    elseif length(impropers_i) > 0
        specific_inter_lists = (bonds, angles, impropers)
    elseif length(angles_i) > 0
        specific_inter_lists = (bonds, angles)
    elseif length(bonds_i) > 0
        specific_inter_lists = (bonds,)
    else
        specific_inter_lists = ()
    end

    neighbor_finder = generate_neighbors(n_atoms,
                                         bonds_i, bonds_j,
                                         angles_i, angles_k,
                                         propers_i, propers_l,
                                         dist_nb_cutoff)

    velocities = zero(coords)

    topo = ignore_derivatives() do
        MolecularTopology(bonds_i, bonds_j, n_atoms)
    end

    atoms_data = [AtomData(atom_types[i], atom_names[i], mol_inds[i], split_grad_safe(atom_types[i], "_")[1], "A", "?", true) for i in 1:n_atoms]

    sys = System{3, Array, T, typeof(atoms), typeof(coords), typeof(boundary), typeof(velocities), typeof(atoms_data),
                 typeof(topo), typeof(pairwise_inter), typeof(specific_inter_lists), typeof(()), typeof(()),
                 typeof(neighbor_finder), typeof(()), typeof(NoUnits),
                 typeof(NoUnits), T, Vector{T}, Nothing}(
        atoms, coords, boundary, velocities, atoms_data, topo, pairwise_inter, specific_inter_lists,
        (), (), neighbor_finder, (), 1, NoUnits, NoUnits, one(T), zeros(T, n_atoms), nothing)

    return sys
end

function atom_names_from_elements(el_list::Vector{Int},
                                   name_map::Vector{String})
    counts = Dict{String,Int}() 
    names  = String[]
    for el in el_list
        sym = name_map[el]
        n   = get(counts, sym, 0) + 1
        counts[sym] = n
        push!(names, "$(sym)$(n)")
    end
    return names
end

Flux.@non_differentiable atom_names_from_elements(args...)

# For LJ, LJ69
function broadcast_atom_data!(
    charges_sys::Vector{T}, 
    charges_mol::Vector{T},
    vdw_σ::Vector{T}, vdw_σ_mol::Vector{T},
    vdw_ϵ::Vector{T}, vdw_ϵ_mol::Vector{T},
    global_to_local::Dict{Int, Int}
)
    for global_i in 1:length(charges_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_sys[global_i] = charges_mol[local_i]
            vdw_σ[global_i] = vdw_σ_mol[local_i]
            vdw_ϵ[global_i] = vdw_ϵ_mol[local_i]
        end
    end
end


function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_sys::Vector{Float32},
               charges_mol::Vector{Float32},
               vdw_σ::Vector{Float32}, vdw_σ_mol::Vector{Float32},
               vdw_ϵ::Vector{Float32}, vdw_ϵ_mol::Vector{Float32},
               global_to_local::Dict{Int, Int})

    broadcast_atom_data!(charges_sys, charges_mol, vdw_σ, vdw_σ_mol, vdw_ϵ, vdw_ϵ_mol, global_to_local)

    function pullback(ȳ)
        d_charges_sys, d_vdw_σ, d_vdw_ϵ = ȳ

        d_charges_mol = zeros(T, length(charges_mol))
        d_vdw_σ_mol   = zeros(T, length(vdw_σ_mol))
        d_vdw_ϵ_mol   = zeros(T, length(vdw_ϵ_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Duplicated(charges_sys, d_charges_sys),
            Enzyme.Duplicated(charges_mol, d_charges_mol),
            Enzyme.Duplicated(vdw_σ, d_vdw_σ),
            Enzyme.Duplicated(vdw_σ_mol, d_vdw_σ_mol),
            Enzyme.Duplicated(vdw_ϵ, d_vdw_ϵ),
            Enzyme.Duplicated(vdw_ϵ_mol, d_vdw_ϵ_mol),
            Enzyme.Const(global_to_local)
        )

        return NoTangent(), NoTangent(), d_charges_mol,
               NoTangent(), d_vdw_σ_mol, NoTangent(), d_vdw_ϵ_mol, NoTangent()
    end

    return nothing, pullback
end

# For BUCK
function broadcast_atom_data!(
    charges_sys::Vector{T}, 
    charges_mol::Vector{T},
    vdw_A::Vector{T}, vdw_A_mol::Vector{T},
    vdw_B::Vector{T}, vdw_B_mol::Vector{T},
    vdw_C::Vector{T}, vdw_C_mol::Vector{T},
    global_to_local::Dict{Int, Int}
)
    for global_i in 1:length(charges_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_sys[global_i] = charges_mol[local_i]
            vdw_A[global_i] = vdw_A_mol[local_i]
            vdw_B[global_i] = vdw_B_mol[local_i]
            vdw_C[global_i] = vdw_C_mol[local_i]
        end
    end
end

function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_sys::Vector{Float32},
               charges_mol::Vector{Float32},
               vdw_A::Vector{Float32}, vdw_A_mol::Vector{Float32},
               vdw_B::Vector{Float32}, vdw_B_mol::Vector{Float32},
               vdw_C::Vector{Float32}, vdw_C_mol::Vector{Float32},
               global_to_local::Dict{Int, Int})

    broadcast_atom_data!(charges_sys, charges_mol, vdw_A, vdw_A_mol, vdw_B, vdw_B_mol, vdw_C, vdw_C_mol, global_to_local)

    function pullback(ȳ)
        d_charges_sys, d_vdw_A, d_vdw_B, d_vdw_C = ȳ

        d_charges_mol = zeros(T, length(charges_mol))
        d_vdw_A_mol   = zeros(T, length(vdw_A_mol))
        d_vdw_B_mol   = zeros(T, length(vdw_B_mol))
        d_vdw_C_mol   = zeros(T, length(vdw_C_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Duplicated(charges_sys, d_charges_sys),
            Enzyme.Duplicated(charges_mol, d_charges_mol),
            Enzyme.Duplicated(vdw_A, d_vdw_A),
            Enzyme.Duplicated(vdw_A_mol, d_vdw_A_mol),
            Enzyme.Duplicated(vdw_B, d_vdw_B),
            Enzyme.Duplicated(vdw_B_mol, d_vdw_B_mol),
            Enzyme.Duplicated(vdw_C, d_vdw_C),
            Enzyme.Duplicated(vdw_C_mol, d_vdw_C_mol),
            Enzyme.Const(global_to_local)
        )

        return NoTangent(), NoTangent(), d_charges_mol,
               NoTangent(), d_vdw_A_mol, NoTangent(), d_vdw_B_mol, NoTangent(), d_vdw_C_mol, NoTangent()
    end

    return nothing, pullback
end

# For DEXP and BUFF
function broadcast_atom_data!(
    charges_sys::Vector{Float32}, 
    charges_mol::Vector{Float32},
    vdw_σ::Vector{Float32}, vdw_σ_mol::Vector{Float32},
    vdw_ϵ::Vector{Float32}, vdw_ϵ_mol::Vector{Float32},
    vdw_α::Base.RefValue{Float32}, vdw_α_mol::Base.RefValue{Float32},
    vdw_β::Base.RefValue{Float32}, vdw_β_mol::Base.RefValue{Float32},
    global_to_local::Dict{Int, Int}
)
    for global_i in 1:length(charges_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_sys[global_i] = charges_mol[local_i]
            vdw_σ[global_i] = vdw_σ_mol[local_i]
            vdw_ϵ[global_i] = vdw_ϵ_mol[local_i]
        end
    end

    vdw_α[] = vdw_α_mol[]
    vdw_β[] = vdw_β_mol[]
end

function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_sys::Vector{Float32},
               charges_mol::Vector{Float32},
               vdw_σ::Vector{Float32}, vdw_σ_mol::Vector{Float32},
               vdw_ϵ::Vector{Float32}, vdw_ϵ_mol::Vector{Float32},
               vdw_α::Base.RefValue{Float32}, vdw_α_mol::Base.RefValue{Float32},
               vdw_β::Base.RefValue{Float32}, vdw_β_mol::Base.RefValue{Float32},
               global_to_local::Dict{Int, Int})

    broadcast_atom_data!(charges_sys, charges_mol, vdw_σ, vdw_σ_mol, vdw_ϵ, vdw_ϵ_mol, vdw_α, vdw_α_mol, vdw_β, vdw_β_mol, global_to_local)

    function pullback(ȳ)
        d_charges_sys, d_vdw_σ, d_vdw_ϵ, d_vdw_α, d_vdw_β = ȳ

        d_charges_mol = zeros(T, length(charges_mol))
        d_vdw_σ_mol   = zeros(T, length(vdw_σ_mol))
        d_vdw_ϵ_mol   = zeros(T, length(vdw_ϵ_mol))
        d_vdw_α_mol   = Ref(zero(T))
        d_vdw_β_mol   = Ref(zero(T))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Duplicated(charges_sys, d_charges_sys),
            Enzyme.Duplicated(charges_mol, d_charges_mol),
            Enzyme.Duplicated(vdw_σ, d_vdw_σ),
            Enzyme.Duplicated(vdw_σ_mol, d_vdw_σ_mol),
            Enzyme.Duplicated(vdw_ϵ, d_vdw_ϵ),
            Enzyme.Duplicated(vdw_ϵ_mol, d_vdw_ϵ_mol),
            Enzyme.Duplicated(vdw_α, d_vdw_α),
            Enzyme.Duplicated(vdw_α_mol, d_vdw_α_mol),
            Enzyme.Duplicated(vdw_β, d_vdw_β),
            Enzyme.Duplicated(vdw_β_mol, d_vdw_β_mol),
            Enzyme.Const(global_to_local)
        )

        return NoTangent(), NoTangent(), d_charges_mol,
               NoTangent(), d_vdw_σ_mol, NoTangent(), d_vdw_ϵ_mol,
               NoTangent(), d_vdw_α_mol, NoTangent(), d_vdw_β_mol, NoTangent()
    end

    return nothing, pullback
end

function broadcast_bond_data!(
    bonds_k::Union{Vector{T}, Nothing},
    bonds_r0::Union{Vector{T}, Nothing},
    bonds_a::Union{Vector{T}, Nothing},
    bonds_k_mol::Union{Vector{T}, Nothing},
    bonds_r0_mol::Union{Vector{T}, Nothing},
    bonds_a_mol::Union{Vector{T}, Nothing},
    bond_functional_form::String,
    bonds_i::Vector{Int}, 
    bonds_j::Vector{Int},
    bond_global_to_local::Dict{Tuple{Int, Int}, Int}
)
    n_bonds = length(bonds_i)

    for idx in 1:n_bonds
        bond = (bonds_i[idx], bonds_j[idx])
        if haskey(bond_global_to_local, bond)
            local_i = bond_global_to_local[bond]
            if bond_functional_form == "harmonic"
                @assert bonds_k !== nothing && bonds_r0 !== nothing
                @assert bonds_k_mol !== nothing && bonds_r0_mol !== nothing
                bonds_k[idx] = bonds_k_mol[local_i]
                bonds_r0[idx] = bonds_r0_mol[local_i]
            elseif bond_functional_form == "morse"
                @assert bonds_k !== nothing && bonds_r0 !== nothing && bonds_a !== nothing
                @assert bonds_k_mol !== nothing && bonds_r0_mol !== nothing && bonds_a_mol !== nothing
                bonds_k[idx]  = bonds_k_mol[local_i]
                bonds_r0[idx] = bonds_r0_mol[local_i]
                bonds_a[idx]  = bonds_a_mol[local_i]
            end
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(broadcast_bond_data!),
    bonds_k::Union{Vector{T}, Nothing},
    bonds_r0::Union{Vector{T}, Nothing},
    bonds_a::Union{Vector{T}, Nothing},
    bonds_k_mol::Union{Vector{T}, Nothing},
    bonds_r0_mol::Union{Vector{T}, Nothing},
    bonds_a_mol::Union{Vector{T}, Nothing},
    bond_functional_form::String,
    bonds_i::Vector{Int},
    bonds_j::Vector{Int},
    bond_global_to_local::Dict{Tuple{Int, Int}, Int}
)

    broadcast_bond_data!(
        bonds_k, bonds_r0, bonds_a,
        bonds_k_mol, bonds_r0_mol, bonds_a_mol,
        bond_functional_form, bonds_i, bonds_j, bond_global_to_local
    )

    function pullback((ȳ_k, ȳ_r0, ȳ_a))
        d_bonds_k_mol  = bonds_k_mol  === nothing ? nothing : zeros(T, length(bonds_k_mol))
        d_bonds_r0_mol = bonds_r0_mol === nothing ? nothing : zeros(T, length(bonds_r0_mol))
        d_bonds_a_mol  = bonds_a_mol  === nothing ? nothing : zeros(T, length(bonds_a_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_bond_data!,
            bonds_k  === nothing ? Enzyme.Const(bonds_k)  : Enzyme.Duplicated(bonds_k, ȳ_k),
            bonds_r0 === nothing ? Enzyme.Const(bonds_r0) : Enzyme.Duplicated(bonds_r0, ȳ_r0),
            bonds_a  === nothing ? Enzyme.Const(bonds_a)  : Enzyme.Duplicated(bonds_a, ȳ_a),
            bonds_k_mol  === nothing ? Enzyme.Const(bonds_k_mol)  : Enzyme.Duplicated(bonds_k_mol, d_bonds_k_mol),
            bonds_r0_mol === nothing ? Enzyme.Const(bonds_r0_mol) : Enzyme.Duplicated(bonds_r0_mol, d_bonds_r0_mol),
            bonds_a_mol  === nothing ? Enzyme.Const(bonds_a_mol)  : Enzyme.Duplicated(bonds_a_mol, d_bonds_a_mol),
            Enzyme.Const(bond_functional_form),
            Enzyme.Const(bonds_i),
            Enzyme.Const(bonds_j),
            Enzyme.Const(bond_global_to_local)
        )

        return NoTangent(), NoTangent(), NoTangent(),
               d_bonds_k_mol, d_bonds_r0_mol, d_bonds_a_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return nothing, pullback
end

function broadcast_angle_data!(
    angles_ki::Union{Vector{T}, Nothing},
    angles_θ0i::Union{Vector{T}, Nothing},
    angles_kj::Union{Vector{T}, Nothing},
    angles_θ0j::Union{Vector{T}, Nothing},
    angles_ki_mol::Union{Vector{T}, Nothing},
    angles_θ0i_mol::Union{Vector{T}, Nothing},
    angles_kj_mol::Union{Vector{T}, Nothing},
    angles_θ0j_mol::Union{Vector{T}, Nothing},
    angle_functional_form::String,
    angles_i::Vector{Int}, 
    angles_j::Vector{Int},
    angles_k::Vector{Int},
    angle_global_to_local::Dict{Tuple{Int, Int, Int}, Int}
)
    n_angles = length(angles_i)

    for idx in 1:n_angles
        angle = (angles_i[idx], angles_j[idx], angles_k[idx])
        if haskey(angle_global_to_local, angle)
            local_i = angle_global_to_local[angle]
            if angle_functional_form == "harmonic"
                @assert angles_ki     !== nothing && angles_θ0i     !== nothing
                @assert angles_ki_mol !== nothing && angles_θ0i_mol !== nothing
                angles_ki[idx]  = angles_ki_mol[local_i]
                angles_θ0i[idx] = angles_θ0i_mol[local_i]
            elseif angle_functional_form == "ub"
                @assert angles_ki     !== nothing && angles_θ0i     !== nothing && angles_kj     !== nothing && angles_θ0j     !== nothing
                @assert angles_ki_mol !== nothing && angles_θ0i_mol !== nothing && angles_kj_mol !== nothing && angles_θ0j_mol !== nothing
                angles_ki[idx]  = angles_ki_mol[local_i]
                angles_θ0i[idx] = angles_θ0i_mol[local_i]
                angles_kj[idx]  = angles_kj_mol[local_i]
                angles_θ0j[idx] = angles_θ0j_mol[local_i]
            end
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(broadcast_angle_data!),
    angles_ki::Union{Vector{T}, Nothing},
    angles_θ0i::Union{Vector{T}, Nothing},
    angles_kj::Union{Vector{T}, Nothing},
    angles_θ0j::Union{Vector{T}, Nothing},
    angles_ki_mol::Union{Vector{T}, Nothing},
    angles_θ0i_mol::Union{Vector{T}, Nothing},
    angles_kj_mol::Union{Vector{T}, Nothing},
    angles_θ0j_mol::Union{Vector{T}, Nothing},
    angle_functional_form::String,
    angles_i::Vector{Int}, 
    angles_j::Vector{Int},
    angles_k::Vector{Int},
    angle_global_to_local::Dict{Tuple{Int, Int, Int}, Int}
)
    broadcast_angle_data!(angles_ki, angles_θ0i, angles_kj, angles_θ0j, angles_ki_mol, angles_θ0i_mol, angles_kj_mol, angles_θ0j_mol, angle_functional_form, angles_i, angles_j, angles_k, angle_global_to_local)

    function pullback((ȳ_ki, ȳ_θ0i, ȳ_kj, ȳ_θ0j))

        d_angles_ki_mol  = angles_ki_mol  === nothing ? nothing : zeros(T, length(angles_ki_mol))
        d_angles_θ0i_mol = angles_θ0i_mol === nothing ? nothing : zeros(T, length(angles_θ0i_mol))
        d_angles_kj_mol  = angles_kj_mol  === nothing ? nothing : zeros(T, length(angles_kj_mol))
        d_angles_θ0j_mol = angles_θ0j_mol === nothing ? nothing : zeros(T, length(angles_θ0j_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_angle_data!,
            angles_ki  === nothing ? Enzyme.Const(angles_ki)  : Enzyme.Duplicated(angles_ki,  ȳ_ki),
            angles_θ0i === nothing ? Enzyme.Const(angles_θ0i) : Enzyme.Duplicated(angles_θ0i, ȳ_θ0i),
            angles_kj  === nothing ? Enzyme.Const(angles_kj)  : Enzyme.Duplicated(angles_kj,  ȳ_kj),
            angles_θ0j === nothing ? Enzyme.Const(angles_θ0j) : Enzyme.Duplicated(angles_θ0j, ȳ_θ0j),
            angles_ki_mol  === nothing ? Enzyme.Const(angles_ki_mol)  : Enzyme.Duplicated(angles_ki_mol,  d_angles_ki_mol),
            angles_θ0i_mol === nothing ? Enzyme.Const(angles_θ0i_mol) : Enzyme.Duplicated(angles_θ0i_mol, d_angles_θ0i_mol),
            angles_kj_mol  === nothing ? Enzyme.Const(angles_kj_mol)  : Enzyme.Duplicated(angles_kj_mol,  d_angles_kj_mol),
            angles_θ0j_mol === nothing ? Enzyme.Const(angles_θ0j_mol) : Enzyme.Duplicated(angles_θ0j_mol, d_angles_θ0j_mol),
            Enzyme.Const(angle_functional_form),
            Enzyme.Const(angles_i),
            Enzyme.Const(angles_j),
            Enzyme.Const(angles_k),
            Enzyme.Const(angle_global_to_local)
        )
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               d_angles_ki_mol, d_angles_θ0i_mol, d_angles_kj_mol, d_angles_θ0j_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return nothing, pullback
end

function broadcast_proper_torsion_feats!(
    proper_feats::Matrix{T},
    proper_feats_mol::Matrix{T},
    propers_i::Vector{Int},
    propers_j::Vector{Int},
    propers_k::Vector{Int},
    propers_l::Vector{Int},
    vs_instance::Vector{Int},
    mapping::Dict{Int, Int},
    torsion_to_key_proper::Dict{NTuple{4, Int}, NTuple{4, String}},
    unique_proper_keys::Dict{NTuple{4, String}, Int}
)
    for idx in 1:length(propers_i)
        global_quad = (propers_i[idx], propers_j[idx], propers_k[idx], propers_l[idx])
        if all(x -> haskey(mapping, x), global_quad)
            local_quad = (
                findfirst(==(global_quad[1]), vs_instance),
                findfirst(==(global_quad[2]), vs_instance),
                findfirst(==(global_quad[3]), vs_instance),
                findfirst(==(global_quad[4]), vs_instance)
            )
            if all(!isnothing, local_quad) && haskey(torsion_to_key_proper, local_quad)
                key = torsion_to_key_proper[local_quad]
                if haskey(unique_proper_keys, key)
                    idx_feat = unique_proper_keys[key]
                    proper_feats[:, idx] .= proper_feats_mol[:, idx_feat]
                end
            end
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(broadcast_proper_torsion_feats!),
    proper_feats::Matrix{T},
    proper_feats_mol::Matrix{T},
    propers_i::Vector{Int},
    propers_j::Vector{Int},
    propers_k::Vector{Int},
    propers_l::Vector{Int},
    vs_instance::Vector{Int},
    mapping::Dict{Int, Int},
    torsion_to_key_proper::Dict{NTuple{4, Int}, NTuple{4, String}},
    unique_proper_keys::Dict{NTuple{4, String}, Int}
)
    broadcast_proper_torsion_feats!(proper_feats, proper_feats_mol, propers_i, propers_j, propers_k, propers_l,
                                    vs_instance, mapping, torsion_to_key_proper, unique_proper_keys)

    function pullback(ȳ)
        d_proper_feats_mol = zeros(size(proper_feats_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_proper_torsion_feats!,
            Enzyme.Duplicated(proper_feats, ȳ),
            Enzyme.Duplicated(proper_feats_mol, d_proper_feats_mol),
            Enzyme.Const(propers_i),
            Enzyme.Const(propers_j),
            Enzyme.Const(propers_k),
            Enzyme.Const(propers_l),
            Enzyme.Const(vs_instance),
            Enzyme.Const(mapping),
            Enzyme.Const(torsion_to_key_proper),
            Enzyme.Const(unique_proper_keys)
        )

        return (
            NoTangent(), 
            NoTangent(), d_proper_feats_mol,
            NoTangent(), NoTangent(), NoTangent(), NoTangent(),
            NoTangent(), NoTangent(), NoTangent()
        )
    end

    return nothing, pullback
end

function broadcast_improper_torsion_feats!(
    improper_feats::Matrix{T},
    improper_feats_mol::Matrix{T},
    impropers_i::Vector{Int},
    impropers_j::Vector{Int},
    impropers_k::Vector{Int},
    impropers_l::Vector{Int},
    vs_instance::Vector{Int},
    mapping::Dict{Int, Int},
    torsion_to_key_improper::Dict{NTuple{4, Int}, NTuple{4, String}},
    unique_improper_keys::Dict{NTuple{4, String}, Int}
)
    for idx in 1:length(impropers_i)
        global_quad = (impropers_i[idx], impropers_j[idx], impropers_k[idx], impropers_l[idx])
        if all(x -> haskey(mapping, x), global_quad)
            local_quad = (
                findfirst(==(global_quad[1]), vs_instance),
                findfirst(==(global_quad[2]), vs_instance),
                findfirst(==(global_quad[3]), vs_instance),
                findfirst(==(global_quad[4]), vs_instance)
            )
            if all(!isnothing, local_quad) && haskey(torsion_to_key_improper, local_quad)
                key = torsion_to_key_improper[local_quad]
                if haskey(unique_improper_keys, key)
                    idx_feat = unique_improper_keys[key]
                    improper_feats[:, idx] .= improper_feats_mol[:, idx_feat]
                end
            end
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(broadcast_improper_torsion_feats!),
    improper_feats::Matrix{T},
    improper_feats_mol::Matrix{T},
    impropers_i::Vector{Int},
    impropers_j::Vector{Int},
    impropers_k::Vector{Int},
    impropers_l::Vector{Int},
    vs_instance::Vector{Int},
    mapping::Dict{Int, Int},
    torsion_to_key_improper::Dict{NTuple{4, Int}, NTuple{4, String}},
    unique_improper_keys::Dict{NTuple{4, String}, Int}
)
    broadcast_improper_torsion_feats!(improper_feats, improper_feats_mol, impropers_i, impropers_j, impropers_k, impropers_l,
                                      vs_instance, mapping, torsion_to_key_improper, unique_improper_keys)

    function pullback(ȳ)
        d_improper_feats_mol = zeros(size(improper_feats_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_improper_torsion_feats!,
            Enzyme.Duplicated(improper_feats, ȳ),
            Enzyme.Duplicated(improper_feats_mol, d_improper_feats_mol),
            Enzyme.Const(impropers_i),
            Enzyme.Const(impropers_j),
            Enzyme.Const(impropers_k),
            Enzyme.Const(impropers_l),
            Enzyme.Const(vs_instance),
            Enzyme.Const(mapping),
            Enzyme.Const(torsion_to_key_improper),
            Enzyme.Const(unique_improper_keys)
        )

        return (
            NoTangent(),
            NoTangent(), d_improper_feats_mol,
            NoTangent(), NoTangent(), NoTangent(), NoTangent(),
            NoTangent(), NoTangent(), NoTangent()
        )
    end

    return nothing, pullback
end


function mol_to_system(
    mol_id::String,
    feat_df::DataFrame,
    coords,
    boundary::CubicBoundary{Float32},
    atom_embedding_model::GNNChain,
    bond_pooling_model::Chain,
    angle_pooling_model::Chain,
    proper_pooling_model::Chain,
    improper_pooling_model::Chain,
    atom_features_model::Chain,
    bond_features_model::Chain,
    angle_features_model::Chain,
    proper_features_model::Chain,
    improper_features_model::Chain
)

    elements, formal_charges,
    bonds_i, bonds_j,
    angles_i, angles_j, angles_k,
    propers_i, propers_j, propers_k, propers_l,
    impropers_i, impropers_j, impropers_k, impropers_l,
    mol_inds, _, n_atoms, atom_features = decode_feats(feat_df)

    masses = [NAME_TO_MASS[ELEMENT_TO_NAME[e]] for e in elements]

    atom_types = fill("", length(elements))
    atom_names = fill("", length(elements))

    global_graph = build_global_graph(length(elements), zip(bonds_i, bonds_j))
    all_graphs, all_indices = extract_all_subgraphs(global_graph)
    unique_graphs, unique_indices, graph_to_unique = filter_unique(all_graphs, all_indices)

    # Prediction arrays
    partial_charges = zeros(T, n_atoms)

    vdw_functional_form = MODEL_PARAMS["physics"]["vdw_functional_form"]
    
    
    vdw_size     = zero(T)
    weight_vdw   = zero(T)

    vdw_σ = nothing
    vdw_ϵ = nothing
    vdw_A = nothing
    vdw_B = nothing
    vdw_C = nothing
    vdw_α = nothing
    vdw_β = nothing
    vdw_δ = nothing
    vdw_γ = nothing
    if vdw_functional_form == "lj"
        vdw_σ = zeros(T, n_atoms)
        vdw_ϵ = zeros(T, n_atoms)

    elseif vdw_functional_form == "lj69"
        vdw_σ = zeros(T, n_atoms)
        vdw_ϵ = zeros(T, n_atoms)

    elseif vdw_functional_form == "dexp"
        vdw_σ = zeros(T, n_atoms)
        vdw_ϵ = zeros(T, n_atoms)
        vdw_α = zero(T)
        vdw_β = zero(T)

    elseif vdw_functional_form == "buff"
        vdw_σ = zeros(T, n_atoms)
        vdw_ϵ = zeros(T, n_atoms)
        vdw_δ = zero(T)
        vdw_γ = zero(T)

    elseif vdw_functional_form == "buck"
        vdw_A = zeros(T, n_atoms)
        vdw_B = zeros(T, n_atoms)
        vdw_C = zeros(T, n_atoms)
    end

    bond_functional_form = MODEL_PARAMS["physics"]["bond_functional_form"]
    n_bonds = length(bonds_i)

    bonds_k  = nothing
    bonds_r0 = nothing
    bonds_a  = nothing

    if bond_functional_form == "harmonic"
        bonds_k  = zeros(T, n_bonds)
        bonds_r0 = zeros(T, n_bonds)
    elseif bond_functional_form == "morse"
        bonds_k  = zeros(T, n_bonds)
        bonds_r0 = zeros(T, n_bonds)
        bonds_a  = zeros(T, n_bonds)
    end

    angle_functional_form = MODEL_PARAMS["physics"]["angle_functional_form"]
    n_angles = length(angles_i)
    angles_ki  = nothing
    angles_θ0i = nothing
    angles_kj  = nothing
    angles_θ0j = nothing
    if angle_functional_form == "harmonic"
        angles_ki  = zeros(T, n_angles)
        angles_θ0i = zeros(T, n_angles)

    elseif angle_functional_form == "ub"
        angles_ki   = zeros(T, n_angles)
        angles_θ0i  = zeros(T, n_angles)
        angles_kj  = zeros(T, n_angles)
        angles_θ0j = zeros(T, n_angles)

    end

    proper_feats   = zeros(T, (n_proper_terms, length(propers_i)))
    improper_feats = zeros(T, (n_improper_terms, length(impropers_i)))


    if any(startswith.(mol_id, ("vapourisation_", "mixing_")))

        ignore_derivatives() do
            if startswith(mol_id, "vapourisation")
                name = split(mol_id, "_")[end]
                if name == "O"
                    mol_names = ["water"]
                else
                    mol_names = [name]
                end
            else
                _, _, smiles = split(mol_id, "_"; limit = 3)
                names = split(smiles, "_")
                mol_names = [name != "water" ? name : "water" for name in names]
            end
        end

    else

        if occursin("water", mol_id)
            mol_names = ["water"]
        end

    end

    for (t, (g, vs_template)) in enumerate(zip(unique_graphs, unique_indices))

        equivs = find_atom_equivalences(g, vs_template, elements)
        labels = ["$(mol_names[t])_" * l for l in label_molecule(vs_template, equivs, elements)]
        names  = atom_names_from_elements(elements[vs_template], ELEMENT_TO_NAME)

        feat_mol = atom_features[:, vs_template]

        adj_mol = build_adj_list(g)

        ### Atom pooling and feature prediction ###

        embeds_mol = calc_embeddings(adj_mol, feat_mol, atom_embedding_model)

        label_to_index = Dict{String, Int}()
        for (i, label) in enumerate(labels)
            if !haskey(label_to_index, label)
                label_to_index[label] = i
            end
        end
        unique_label_indices = ignore_derivatives() do
            return collect(values(label_to_index))
        end
        unique_embeds = embeds_mol[:, unique_label_indices]
        unique_feats  = atom_features_model(unique_embeds)

        feats_mol = map(labels) do label
            return unique_feats[:, label_to_index[label]]
        end
        feats_mol = hcat(feats_mol...)

        ### Bonds pooling and feature prediction ###

        # First we create a dict that converts bonds represented as indices as bonds represented by molecule type
        bond_to_key = Dict{Tuple{Int,Int}, Tuple{String,String}}()
        bond_to_local_idx = Dict{Tuple{Int,Int}, Int}()
        
        edges_list = [e for e in edges(g)]
        bond_key = map(1:length(edges_list)) do k
            e = edges_list[k]
            u, v = src(e), dst(e)
            bond_to_local_idx[(min(u,v), max(u,v))] = k
            lu, lv = labels[u], labels[v]
            key = lu < lv ? (lu, lv) : (lv, lu)
            bond_to_key[(min(u,v), max(u,v))] = key
            return key
        end

        # Then we get the unique bonds represented by atom type
        unique_keys = Dict{Tuple{String,String}, Int}()
        unique_bond_keys = Tuple{String,String}[]
        ignore_derivatives() do
            for key in bond_key
                if !haskey(unique_keys, key)
                    push!(unique_bond_keys, key)
                    unique_keys[key] = length(unique_keys) + 1
                end
            end
        end

        # We pass only the unique bonds to the pooling model
        emb_i = embeds_mol[:, [findfirst(==(l), labels) for (l, _) in unique_bond_keys]]
        emb_j = embeds_mol[:, [findfirst(==(l), labels) for (_, l) in unique_bond_keys]]
        bond_pool_1 = bond_pooling_model(cat(emb_i, emb_j; dims=1))
        bond_pool_2 = bond_pooling_model(cat(emb_j, emb_i; dims=1))
        bond_pool = bond_pool_1 .+ bond_pool_2 # Bond symmetry preserved

        # Predict features 
        unique_bond_feats = bond_features_model(bond_pool)
        bond_feats_mol = map(1:length(edges(g))) do k
            e = [_ for _  in edges(g)][k]
            u, v = src(e), dst(e)
            key = bond_to_key[(min(u,v), max(u,v))]
            idx = unique_keys[key]
            return unique_bond_feats[:,idx]
        end
        bond_feats_mol = hcat(bond_feats_mol...)

        ### Angle Feature Pooling ###
        angle_to_key = Dict{NTuple{3,Int}, NTuple{3,String}}()
        angle_triples = [(i,j,k) for (i,j,k) in zip(angles_i, angles_j, angles_k) if i in vs_template && j in vs_template && k in vs_template]
        
        
        # Map triplets from whole system indexing to local molecule indexing
        local_map = Dict(glo => loc for (loc, glo) in enumerate(vs_template))
        angle_triples = [(local_map[i], local_map[j], local_map[k]) for (i,j,k) in angle_triples]
        
        # From index to molecule type
        angle_to_local_idx = Dict{Tuple{Int,Int,Int}, Int}()
        angle_key = Tuple{String,String,String}[]
        ignore_derivatives() do
            for (idx, (i, j, k)) in enumerate(angle_triples)
                angle_to_local_idx[(i,j,k)] = idx
                li, lj, lk = labels[i], labels[j], labels[k]
                key = (li, lj, lk) < (lk, lj, li) ? (li, lj, lk) : (lk, lj, li)
                push!(angle_key, key)
                angle_to_key[(i,j,k)] = key
            end
        end

        # Get unique representation by molecule type
        unique_angle_keys = Dict{NTuple{3,String}, Int}()
        angle_key_order = NTuple{3,String}[]
        ignore_derivatives() do 
            for key in angle_key
                if !haskey(unique_angle_keys, key)
                    push!(angle_key_order, key)
                    unique_angle_keys[key] = length(unique_angle_keys) + 1
                end
            end
        end

        # Get features for just the unique angles
        angle_emb_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _) in angle_key_order]]
        angle_emb_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _) in angle_key_order]]
        angle_emb_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk) in angle_key_order]]

        # Symmetry preserving pooling
        angle_com_emb_1 = cat(angle_emb_i, angle_emb_j, angle_emb_k; dims=1)
        angle_com_emb_2 = cat(angle_emb_k, angle_emb_j, angle_emb_i; dims=1)
        angle_pool_1 = angle_pooling_model(angle_com_emb_1)
        angle_pool_2 = angle_pooling_model(angle_com_emb_2)

        # Get features
        angle_pool = angle_pool_1 .+ angle_pool_2
        unique_angle_feats = angle_features_model(angle_pool)

        # Broadcast from unique bonds to whole molecule
        angle_feats_mol = map(1:length(angle_triples)) do idx
            ijk = angle_triples[idx]
            key = angle_to_key[ijk]
            key_idx = unique_angle_keys[key]
            return unique_angle_feats[:, key_idx]
        end
        angle_feats_mol = hcat(angle_feats_mol...)

        ### Torsion Feature Pooling ###
        torsion_to_key_proper = Dict{NTuple{4,Int}, NTuple{4,String}}()
        torsion_to_key_improper = Dict{NTuple{4,Int}, NTuple{4,String}}()

        # Get global indices that appear in molecular template indices
        torsion_proper_quads = [(i,j,k,l) for (i,j,k,l) in zip(propers_i, propers_j, propers_k, propers_l) if i in vs_template && j in vs_template && k in vs_template && l in vs_template]
        torsion_improper_quads = [(i,j,k,l) for (i,j,k,l) in zip(impropers_i, impropers_j, impropers_k, impropers_l) if i in vs_template && j in vs_template && k in vs_template && l in vs_template]

        # Map indices from global to local indexing
        local_map = Dict(glo => loc for (loc, glo) in enumerate(vs_template))
        torsion_proper_quads = [(local_map[i], local_map[j], local_map[k], local_map[l]) for (i,j,k,l) in torsion_proper_quads]
        torsion_improper_quads = [(local_map[i], local_map[j], local_map[k], local_map[l]) for (i,j,k,l) in torsion_improper_quads]

        # From indices to atom types
        torsion_key_proper = map(torsion_proper_quads) do quad
            i,j,k,l = quad
            li, lj, lk, ll = labels[i], labels[j], labels[k], labels[l]
            key = (li, lj, lk, ll) < (ll, lk, lj, li) ? (li, lj, lk, ll) : (ll, lk, lj, li)
            torsion_to_key_proper[(i,j,k,l)] = key
            return key
        end

        torsion_key_improper = map(torsion_improper_quads) do quad
            i,j,k,l = quad
            li, lj, lk, ll = labels[i], labels[j], labels[k], labels[l]
            key = (li, lj, lk, ll)
            torsion_to_key_improper[(i,j,k,l)] = key
            return key
        end

        # We get the unique torsions depending on atom type
        unique_proper_keys = Dict{NTuple{4,String}, Int}()
        unique_improper_keys = Dict{NTuple{4,String}, Int}()
        proper_key_order = NTuple{4,String}[]
        improper_key_order = NTuple{4,String}[]
        ignore_derivatives() do 
            for key in torsion_key_proper
                if !haskey(unique_proper_keys, key)
                    push!(proper_key_order, key)
                    unique_proper_keys[key] = length(unique_proper_keys) + 1
                end
            end

            for key in torsion_key_improper
                if !haskey(unique_improper_keys, key)
                    push!(improper_key_order, key)
                    unique_improper_keys[key] = length(unique_improper_keys) + 1
                end
            end
        end

        # Symmetry preserving pooling

        prop_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _, _) in proper_key_order]]
        prop_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _, _) in proper_key_order]]
        prop_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk, _) in proper_key_order]]
        prop_l = embeds_mol[:, [findfirst(==(ll), labels) for (_, _, _, ll) in proper_key_order]]

        prop_1 = cat(prop_i, prop_j, prop_k, prop_l; dims=1)
        prop_2 = cat(prop_l, prop_k, prop_j, prop_i; dims=1)
        proper_pool = proper_pooling_model(prop_1) .+ proper_pooling_model(prop_2)
        unique_proper_feats = proper_features_model(proper_pool)

        imp_i = embeds_mol[:, [findfirst(==(li), labels) for (li, _, _, _) in improper_key_order]]
        imp_j = embeds_mol[:, [findfirst(==(lj), labels) for (_, lj, _, _) in improper_key_order]]
        imp_k = embeds_mol[:, [findfirst(==(lk), labels) for (_, _, lk, _) in improper_key_order]]
        imp_l = embeds_mol[:, [findfirst(==(ll), labels) for (_, _, _, ll) in improper_key_order]]

        imp_1 = cat(imp_i, imp_j, imp_k, imp_l; dims=1)
        imp_2 = cat(imp_i, imp_k, imp_j, imp_l; dims=1)
        imp_3 = cat(imp_i, imp_l, imp_j, imp_k; dims=1)
        improper_pool = improper_pooling_model(imp_1) .+ improper_pooling_model(imp_2) .+ improper_pooling_model(imp_3)
        unique_improper_feats = improper_features_model(improper_pool)

        # Broadcast from unique torsions to whole molecule
        proper_feats_mol = map(1:length(torsion_proper_quads)) do idx
            quad    = torsion_proper_quads[idx]
            key     = torsion_to_key_proper[quad]
            key_idx = unique_proper_keys[key]
            return unique_proper_feats[:, key_idx]
        end
        if !isempty(proper_feats_mol)
            proper_feats_mol = hcat(proper_feats_mol...)
        else
            proper_feats_mol = zeros(T, n_proper_terms, 0)
        end

        improper_feats_mol = map(1:length(torsion_improper_quads)) do idx
            quad    = torsion_improper_quads[idx]
            key     = torsion_to_key_improper[quad]
            key_idx = unique_improper_keys[key]
            return unique_improper_feats[:, key_idx]
        end
        if !isempty(proper_feats_mol)
            improper_feats_mol = hcat(improper_feats_mol...)
        else
            improper_feats_mol = zeros(T, n_improper_terms, 0)
        end

        ### Predict charges from atom features ###
        charges_mol = atom_feats_to_charges(feats_mol, formal_charges[vs_template])

        ### Predict vdw params ###
        vdw_mol = atom_feats_to_vdW(feats_mol)
        
        ### Predict bonds params ###
        bonds_mol = feats_to_bonds(bond_feats_mol)

        ### Predict angle feats ###
        angles_mol = feats_to_angles(angle_feats_mol)

        for (idx, vs_instance) in enumerate(all_indices)

            if graph_to_unique[idx] == t

                global_to_local = Dict(g => i for (i, g) in enumerate(vs_instance))
                
                if vdw_functional_form in ("lj", "lj69")
                    broadcast_atom_data!(partial_charges, charges_mol, 
                                         vdw_σ, vdw_mol[1],
                                         vdw_ϵ, vdw_mol[2],
                                         global_to_local)
                elseif vdw_functional_form == "dexp"
                    broadcast_atom_data!(partial_charges, charges_mol, 
                                         vdw_σ, vdw_mol[1],
                                         vdw_ϵ, vdw_mol[2],
                                         vdw_α, vdw_mol[3],
                                         vdw_β, vdw_mol[4],
                                         global_to_local)
                elseif vdw_functional_form == "buff"
                    broadcast_atom_data!(partial_charges, charges_mol, 
                                         vdw_σ, vdw_mol[1],
                                         vdw_ϵ, vdw_mol[2],
                                         vdw_δ, vdw_mol[3],
                                         vdw_γ, vdw_mol[4],
                                         global_to_local)
                elseif vdw_functional_form == "buck"
                    broadcast_atom_data!(partial_charges, charges_mol, 
                                         vdw_A, vdw_mol[1],
                                         vdw_B, vdw_mol[2],
                                         vdw_C, vdw_mol[3],
                                         global_to_local)
                end

                mapping = Dict(i => vs_instance[i] for i in 1:length(vs_instance))
                bond_global_to_local = Dict{Tuple{Int,Int}, Int}()
                for e in edges(g)
                    i, j = mapping[src(e)], mapping[dst(e)]
                    global_pair = (min(i, j), max(i, j))
                    local_pair = (min(src(e), dst(e)), max(src(e), dst(e)))
                    bond_global_to_local[global_pair] = bond_to_local_idx[local_pair]
                end

                broadcast_bond_data!(bonds_k, bonds_r0, bonds_a, bonds_mol[1], bonds_mol[2], bonds_mol[3], bond_functional_form, bonds_i, bonds_j, bond_global_to_local)

                angle_global_to_local = Dict{Tuple{Int,Int,Int}, Int}()
                for (i, j, k) in angle_triples
                    gi, gj, gk = mapping[i], mapping[j], mapping[k]
                    angle_global_to_local[(gi, gj, gk)] = angle_to_local_idx[(i, j, k)]
                end

                broadcast_angle_data!(angles_ki, angles_θ0i, angles_kj, angles_θ0j, angles_mol[1], angles_mol[2], angles_mol[3], angles_mol[4], angle_functional_form, angles_i, angles_j, angles_k, angle_global_to_local)

                # Broadcast proper torsion features
                broadcast_proper_torsion_feats!(proper_feats, proper_feats_mol, propers_i, propers_j, propers_k, propers_l, vs_instance, mapping, torsion_to_key_proper, unique_proper_keys)

                # Broadcast improper torsion features
                broadcast_improper_torsion_feats!(improper_feats, improper_feats_mol, impropers_i, impropers_j, impropers_k, impropers_l, vs_instance, mapping, torsion_to_key_improper, unique_improper_keys)
            end
        end
    end

    if vdw_functional_form in ("lj", "lj69", "dexp", "buff")
        vdw_size = T(0.5*(mean(vdw_σ) + mean(vdw_ϵ)))
    else
        vdw_size = zero(T)
    end
    weight_vdw = (vdw_functional_form == "lj" ? sigmoid(global_params[1]) : one(T))

    torsion_ks_size = zero(T)
    if length(proper_feats) > 0
        torsion_ks_size += mean(abs, proper_feats)
    end
    if length(improper_feats) > 0
        torsion_ks_size += mean(abs, improper_feats)
    end

    # Why is this padding needed?
    proper_feats_pad   = cat(proper_feats, zeros(T, 6 - n_proper_terms, length(propers_i)); dims = 1)
    improper_feats_pad = cat(improper_feats, zeros(T, 6 - n_improper_terms, length(impropers_i)); dims = 1)

    molly_sys = build_sys(mol_id, 
    masses, atom_types, atom_names, mol_inds, coords, boundary_inf, partial_charges, 
    vdw_functional_form, weight_vdw, vdw_σ, vdw_ϵ, vdw_A, vdw_B, vdw_C, vdw_α, vdw_β,
    vdw_δ, vdw_γ, bond_functional_form, bonds_k, bonds_r0, bonds_a, angle_functional_form,
    angles_ki, angles_θ0i, angles_kj, angles_θ0j, bonds_i, bonds_j, angles_i, angles_j, angles_k, proper_feats_pad,
    improper_feats_pad, propers_i, propers_j, propers_k, propers_l, impropers_i, impropers_j, impropers_k,
    impropers_l)

    return (
        molly_sys,
        partial_charges, 
        vdw_size,
        torsion_ks_size,
        elements,
        mol_inds
    )

end