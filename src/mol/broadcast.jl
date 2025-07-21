# For LJ, LJ69, but using charge features instead of partial charges
function broadcast_atom_data!(
    charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
    charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
    vdw_σ::Vector{T}, vdw_σ_mol::Vector{T},
    vdw_ϵ::Vector{T}, vdw_ϵ_mol::Vector{T},
    global_to_local::Dict{Int, Int}
)

    for global_i in 1:length(charges_k1_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_k1_sys[global_i] = charges_k1_mol[local_i]
            charges_k2_sys[global_i] = charges_k2_mol[local_i]
            vdw_σ[global_i] = vdw_σ_mol[local_i]
            vdw_ϵ[global_i] = vdw_ϵ_mol[local_i]
        end
    end
    return charges_k1_sys, charges_k2_sys, vdw_σ, vdw_ϵ
end

function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
               charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
               vdw_σ::Vector{T}, vdw_σ_mol::Vector{T},
               vdw_ϵ::Vector{T}, vdw_ϵ_mol::Vector{T},
               global_to_local::Dict{Int, Int})

    Y = broadcast_atom_data!(charges_k1_sys, charges_k1_mol, charges_k2_sys, charges_k2_mol, vdw_σ, vdw_σ_mol, vdw_ϵ, vdw_ϵ_mol, global_to_local)

    function pullback(y_hat)

        d_charges_k1_sys = iszero(y_hat[1]) ? zero(charges_k1_sys) : y_hat[1]
        d_charges_k2_sys = iszero(y_hat[2]) ? zero(charges_k2_sys) : y_hat[2]
        d_vdw_σ          = iszero(y_hat[3]) ? zero(vdw_σ)          : y_hat[3]
        d_vdw_ϵ          = iszero(y_hat[4]) ? zero(vdw_ϵ)          : y_hat[4]

        d_charges_k1_mol = zeros(T, length(charges_k1_mol))
        d_charges_k2_mol = zeros(T, length(charges_k2_mol))
        d_vdw_σ_mol      = zeros(T, length(vdw_σ_mol))
        d_vdw_ϵ_mol      = zeros(T, length(vdw_ϵ_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Const,
            Enzyme.Duplicated(charges_k1_sys, d_charges_k1_sys),
            Enzyme.Duplicated(charges_k1_mol, d_charges_k1_mol),
            Enzyme.Duplicated(charges_k2_sys, d_charges_k2_sys),
            Enzyme.Duplicated(charges_k2_mol, d_charges_k2_mol),
            Enzyme.Duplicated(vdw_σ, d_vdw_σ),
            Enzyme.Duplicated(vdw_σ_mol, d_vdw_σ_mol),
            Enzyme.Duplicated(vdw_ϵ, d_vdw_ϵ),
            Enzyme.Duplicated(vdw_ϵ_mol, d_vdw_ϵ_mol),
            Enzyme.Const(global_to_local)
        )
        #println("ATOM PULLBACK")
        return NoTangent(), 
               NoTangent(), d_charges_k1_mol,
               NoTangent(), d_charges_k2_mol,
               NoTangent(), d_vdw_σ_mol,
               NoTangent(), d_vdw_ϵ_mol,
               NoTangent()
    end

    return Y, pullback
end

# For BUCK
function broadcast_atom_data!(
    charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
    charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
    vdw_A::Vector{T}, vdw_A_mol::Vector{T},
    vdw_B::Vector{T}, vdw_B_mol::Vector{T},
    vdw_C::Vector{T}, vdw_C_mol::Vector{T},
    global_to_local::Dict{Int, Int}
)
    for global_i in 1:length(charges_k1_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_k1_sys[global_i] = charges_k1_mol[local_i]
            charges_k2_sys[global_i] = charges_k2_mol[local_i]
            vdw_A[global_i] = vdw_A_mol[local_i]
            vdw_B[global_i] = vdw_B_mol[local_i]
            vdw_C[global_i] = vdw_C_mol[local_i]
        end
    end
    return charges_k1_sys, charges_k2_sys, vdw_A, vdw_B, vdw_C
end

function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
               charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
               vdw_A::Vector{T}, vdw_A_mol::Vector{T},
               vdw_B::Vector{T}, vdw_B_mol::Vector{T},
               vdw_C::Vector{T}, vdw_C_mol::Vector{T},
               global_to_local::Dict{Int, Int})

    Y = broadcast_atom_data!(charges_k1_sys, charges_k1_mol, charges_k2_sys, charges_k2_mol, vdw_A, vdw_A_mol, vdw_B, vdw_B_mol, vdw_C, vdw_C_mol, global_to_local)

    function pullback(y_hat)

        d_charges_k1_sys, d_charges_k2_sys, d_vdw_A, d_vdw_B, d_vdw_C = y_hat

        d_charges_k1_mol = zeros(T, length(charges_k1_mol))
        d_charges_k2_mol = zeros(T, length(charges_k2_mol))
        d_vdw_A_mol   = zeros(T, length(vdw_A_mol))
        d_vdw_B_mol   = zeros(T, length(vdw_B_mol))
        d_vdw_C_mol   = zeros(T, length(vdw_C_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Const,
            Enzyme.Duplicated(charges_k1_sys, d_charges_k1_sys),
            Enzyme.Duplicated(charges_k1_mol, d_charges_k1_mol),
            Enzyme.Duplicated(charges_k2_sys, d_charges_k2_sys),
            Enzyme.Duplicated(charges_k2_mol, d_charges_k2_mol),
            Enzyme.Duplicated(vdw_A, d_vdw_A),
            Enzyme.Duplicated(vdw_A_mol, d_vdw_A_mol),
            Enzyme.Duplicated(vdw_B, d_vdw_B),
            Enzyme.Duplicated(vdw_B_mol, d_vdw_B_mol),
            Enzyme.Duplicated(vdw_C, d_vdw_C),
            Enzyme.Duplicated(vdw_C_mol, d_vdw_C_mol),
            Enzyme.Const(global_to_local)
        )

        return NoTangent(),
               NoTangent(), d_charges_k1_mol,
               NoTangent(), d_charges_k2_mol,
               NoTangent(), d_vdw_A_mol,
               NoTangent(), d_vdw_B_mol,
               NoTangent(), d_vdw_C_mol,
               NoTangent()
    end

    return Y, pullback
end

# For DEXP and BUFF
function broadcast_atom_data!(
    charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
    charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
    vdw_σ::Vector{T}, vdw_σ_mol::Vector{T},
    vdw_ϵ::Vector{T}, vdw_ϵ_mol::Vector{T},
    vdw_α::Base.RefValue{T}, vdw_α_mol::Base.RefValue{T},
    vdw_β::Base.RefValue{T}, vdw_β_mol::Base.RefValue{T},
    global_to_local::Dict{Int, Int}
)
    for global_i in 1:length(charges_k1_sys)
        local_i = get(global_to_local, global_i, nothing)
        if !isnothing(local_i)
            charges_k1_sys[global_i] = charges_k1_mol[local_i]
            charges_k2_sys[global_i] = charges_k2_mol[local_i]
            vdw_σ[global_i] = vdw_σ_mol[local_i]
            vdw_ϵ[global_i] = vdw_ϵ_mol[local_i]
        end
    end
    vdw_α[] = vdw_α_mol[]
    vdw_β[] = vdw_β_mol[]
    return charges_k1_sys, charges_k2_sys, vdw_σ, vdw_ϵ, vdw_α, vdw_β
end

function ChainRulesCore.rrule(::typeof(broadcast_atom_data!),
               charges_k1_sys::Vector{T}, charges_k1_mol::Vector{T},
               charges_k2_sys::Vector{T}, charges_k2_mol::Vector{T},
               vdw_σ::Vector{T}, vdw_σ_mol::Vector{T},
               vdw_ϵ::Vector{T}, vdw_ϵ_mol::Vector{T},
               vdw_α::Base.RefValue{T}, vdw_α_mol::Base.RefValue{T},
               vdw_β::Base.RefValue{T}, vdw_β_mol::Base.RefValue{T},
               global_to_local::Dict{Int, Int})

    Y = broadcast_atom_data!(charges_k1_sys, charges_k1_mol, charges_k2_sys, charges_k2_mol, vdw_σ, vdw_σ_mol, vdw_ϵ, vdw_ϵ_mol, vdw_α, vdw_α_mol, vdw_β, vdw_β_mol, global_to_local)

    function pullback(y_hat)
        d_charges_k1_sys, d_charges_k2_sys, d_vdw_σ, d_vdw_ϵ, d_vdw_α, d_vdw_β = y_hat

        d_charges_k1_mol = zeros(T, length(charges_k1_mol))
        d_charges_k2_mol = zeros(T, length(charges_k2_mol))
        d_vdw_σ_mol   = zeros(T, length(vdw_σ_mol))
        d_vdw_ϵ_mol   = zeros(T, length(vdw_ϵ_mol))
        d_vdw_α_mol   = Ref(zero(T))
        d_vdw_β_mol   = Ref(zero(T))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_atom_data!,
            Enzyme.Const,
            Enzyme.Duplicated(charges_k1_sys, d_charges_k1_sys),
            Enzyme.Duplicated(charges_k1_mol, d_charges_k1_mol),
            Enzyme.Duplicated(charges_k2_sys, d_charges_k2_sys),
            Enzyme.Duplicated(charges_k2_mol, d_charges_k2_mol),
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

        return NoTangent(),
               NoTangent(), d_charges_k1_mol,
               NoTangent(), d_charges_k2_mol,
               NoTangent(), d_vdw_σ_mol,
               NoTangent(), d_vdw_ϵ_mol,
               NoTangent(), d_vdw_α_mol,
               NoTangent(), d_vdw_β_mol,
               NoTangent()
    end

    return Y, pullback
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
    return bonds_k, bonds_r0, bonds_a
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

    Y = broadcast_bond_data!(
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
            Enzyme.Const,
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

	return NoTangent(),
               NoTangent(), NoTangent(), NoTangent(),
               d_bonds_k_mol, d_bonds_r0_mol, d_bonds_a_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent()
               
    end

    return Y, pullback
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
    return angles_ki, angles_θ0i, angles_kj, angles_θ0j
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
    Y = broadcast_angle_data!(angles_ki, angles_θ0i, angles_kj, angles_θ0j,
                          angles_ki_mol, angles_θ0i_mol, angles_kj_mol, angles_θ0j_mol,
                          angle_functional_form, angles_i, angles_j, angles_k, angle_global_to_local)

    function pullback((ȳ_ki, ȳ_θ0i, ȳ_kj, ȳ_θ0j))

        d_angles_ki_mol  = angles_ki_mol  === nothing ? nothing : zeros(T, length(angles_ki_mol))
        d_angles_θ0i_mol = angles_θ0i_mol === nothing ? nothing : zeros(T, length(angles_θ0i_mol))
        d_angles_kj_mol  = angles_kj_mol  === nothing ? nothing : zeros(T, length(angles_kj_mol))
        d_angles_θ0j_mol = angles_θ0j_mol === nothing ? nothing : zeros(T, length(angles_θ0j_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_angle_data!,
            Enzyme.Const,
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

        return NoTangent(),
               NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               d_angles_ki_mol, d_angles_θ0i_mol, d_angles_kj_mol, d_angles_θ0j_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return Y, pullback
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
    return proper_feats
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
    Y = broadcast_proper_torsion_feats!(proper_feats, proper_feats_mol, propers_i, propers_j, propers_k, propers_l,
                                    vs_instance, mapping, torsion_to_key_proper, unique_proper_keys)

    function pullback(ȳ)
        d_proper_feats_mol = zeros(size(proper_feats_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_proper_torsion_feats!,
            Enzyme.Const,
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

        return NoTangent(),
               NoTangent(), d_proper_feats_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return Y, pullback
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
    return improper_feats
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
    Y = broadcast_improper_torsion_feats!(improper_feats, improper_feats_mol, impropers_i, impropers_j, impropers_k, impropers_l,
                                      vs_instance, mapping, torsion_to_key_improper, unique_improper_keys)

    function pullback(ȳ)
        d_improper_feats_mol = zeros(size(improper_feats_mol))

        Enzyme.autodiff(
            Enzyme.Reverse,
            broadcast_improper_torsion_feats!,
            Enzyme.Const,
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

        return NoTangent(),
               NoTangent(), d_improper_feats_mol,
               NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return Y, pullback
end
