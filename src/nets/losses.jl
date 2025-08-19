abs2_vec(x) = abs2.(x)
force_loss(fs, dft_fs)            = MODEL_PARAMS["training"]["loss_weight_force"] * mean(sqrt.(sum.(abs2_vec.(fs .- dft_fs))))
charge_loss(charges, dft_charges) = MODEL_PARAMS["training"]["loss_weight_charge"] * mean(abs2.(charges .- dft_charges))
vdw_params_loss(vdw_params_size)  = MODEL_PARAMS["training"]["loss_weight_vdw_params"] * -vdw_params_size
torsion_ks_loss(torsion_ks_size)  = MODEL_PARAMS["training"]["loss_weight_torsion_ks"] * torsion_ks_size
pe_loss(pe_diff, dft_pe_diff) = MODEL_PARAMS["training"]["loss_weight_energy"] * abs(pe_diff - dft_pe_diff)

ϵ_entropy = T(1e-8)

Ω_0          = T(MODEL_PARAMS["training"]["loss_weight_vdw_entropy_0"])
Ω_min        = T(MODEL_PARAMS["training"]["loss_weight_vdw_entropy_min"])
Ω_min_epoch  = T(MODEL_PARAMS["training"]["entropy_min_epoch"])
decay_rate_Ω = T(log(Ω_0 / Ω_min) / Ω_min_epoch)

entropy_loss(func_probs) = -mean(sum(func_probs .* log.(func_probs .+ ϵ_entropy); dims = 1))


using LambertW
using Base.MathConstants: e

# Largest real zero of A*exp(-B r) - (C/r)^6; 0 if none
@inline function buckingham_zero(A::T, B::T, C::T)::T where {T<:Real}
    z = -(B*C) / (T(6) * A^(one(T)/T(6)))
    if z >= zero(T) || z < -inv(T(e))             # need z ∈ [-1/e, 0)
        return zero(T)
    end
    y0  = -T(6) * T(lambertw(z))                  # W0
    y_1 = -T(6) * T(lambertw(z, -1))              # W-1
    r0, r1 = y0/B, y_1/B
    r = ifelse(r0 > r1, r0, r1)
    return r > zero(T) ? T(r) : zero(T)
end

# Precompute per-atom σ-like scale. This runs OUTSIDE Enzyme.
function precompute_rscale(atom, vdw_fnc_idx::Int)::typeof(atom.σ_lj)
    T = typeof(atom.σ_lj)
    buck_r0 = buckingham_zero(T(atom.A_buck), T(atom.B_buck), T(atom.C_buck))
    if vdw_fnc_idx == 0
        buck_r0 == zero(T) ?
            mean((atom.σ_lj, atom.σ_lj69, atom.σ_dexp, atom.σ_buff)) :
            mean((atom.σ_lj, atom.σ_lj69, atom.σ_dexp, atom.σ_buff, buck_r0))
    elseif vdw_fnc_idx == 1
        atom.σ_lj
    elseif vdw_fnc_idx == 2
        atom.σ_lj69
    elseif vdw_fnc_idx == 3
        atom.σ_dexp
    elseif vdw_fnc_idx == 4
        atom.σ_buff
    elseif vdw_fnc_idx == 5
        buck_r0 == zero(T) ? T(0.1) : buck_r0
    else
        T(0.17)
    end
end

# INTERNAL: Enzyme will differentiate this. No lambertw here.
function vdw_params_regularisation_core(
    atoms::Vector{GeneralAtom{T}},
    vdw_inters,
    vdw_fnc_idx::Int,
    rscale::Vector{T},
) where {T}
    loss = zero(T)
    N = 50
    vdw_lj, vdw_lj69, vdw_dexp, vdw_buff, vdw_buck = vdw_inters

    @inbounds for (ai, atom) in enumerate(atoms)
        ref_σ = rscale[ai]
        start = T(0.95) * ref_σ
        stop  = T(3.00) * ref_σ
        step  = (stop - start) / T(N - 1)

        r = Vector{T}(undef, N)
        @inbounds @simd for i in 1:N
            r[i] = start + (i-1) * step
        end

        pots = (
            vdw_potential(vdw_lj,   atom, r),
            vdw_potential(vdw_lj69, atom, r),
            vdw_potential(vdw_dexp, atom, r),
            vdw_potential(vdw_buff, atom, r),
            vdw_potential(vdw_buck, atom, r),
        )

        if vdw_fnc_idx == 0
            l = zero(T)
            @inbounds for i in 1:5, j in 1:5
                if i != j
                    pi, pj = pots[i], pots[j]
                    l += (length(pi) == length(pj)) ? mean(abs2.(pi .- pj)) : zero(T)
                end
            end
            loss += l / T(10)
        else
            ref_p = pots[vdw_fnc_idx]
            l = zero(T)
            @inbounds for i in 1:5
                if i != vdw_fnc_idx
                    pi = pots[i]
                    l += (length(pi) == length(ref_p)) ? mean(abs2.(pi .- ref_p)) : zero(T)
                end
            end
            loss += l / T(4)
        end
    end

    return loss / T(length(atoms))
end

function vdw_params_regularisation(
    atoms::Vector{GeneralAtom{T}},
    vdw_inters,
    vdw_fnc_idx::Int,
) where {T}
    rscale = similar([atom.σ_lj for atom in atoms])  # Vector{T}
    @inbounds for i in eachindex(atoms)
        rscale[i] = precompute_rscale(atoms[i], vdw_fnc_idx)
    end
    return vdw_params_regularisation_core(atoms, vdw_inters, vdw_fnc_idx, rscale)
end

function ChainRulesCore.rrule(::typeof(vdw_params_regularisation),
                              atoms::Vector{GeneralAtom{T}},
                              vdw_inters,
                              vdw_fnc_idx::Int) where {T}

    rscale = similar([atom.σ_lj for atom in atoms])
    @inbounds for i in eachindex(atoms)
        rscale[i] = precompute_rscale(atoms[i], vdw_fnc_idx)
    end

    y = vdw_params_regularisation_core(atoms, vdw_inters, vdw_fnc_idx, rscale)

    function pullback(ȳ)
        ȳT = T(ȳ)
        d_atoms      = zero.(atoms)
        d_vdw_inters = zero.(vdw_inters)

        Enzyme.autodiff(
            Enzyme.Reverse,
            vdw_params_regularisation_core,
            Enzyme.Active,
            Enzyme.Duplicated(atoms, d_atoms),
            Enzyme.Duplicated(vdw_inters, d_vdw_inters),
            Enzyme.Const(vdw_fnc_idx),
            Enzyme.Const(rscale),  # <- constant, not seen by Enzyme
        )

        return NoTangent(), ȳT .* d_atoms, ȳT .* d_vdw_inters, NoTangent()
    end

    return y, pullback
end

function param_regularisation(models)
    s = sum(abs2, Flux.destructure(models[1:(end-1)])[1])
    # Global parameters excluded from regularisation except for NNPairwise NN params
    if vdw_functional_form == "nn"
        s += sum(abs2, Flux.destructure(models[end])[1][2:end])
    end
    return MODEL_PARAMS["training"]["loss_weight_regularisation"] * s
end

function store_string(store_id, str)
    if !isnothing(out_dir)
        open(joinpath(out_dir, "store_$store_id.txt"), "w") do of
            println(of, str)
        end
    end
end

Flux.@non_differentiable store_string(args...)

function enth_vap_loss(snapshot_U_liquid, mean_U_gas, temp, frame_i, repeat_i, n_molecules, mol_id)
    ΔH_vap = enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    ΔH_vap_exp = T(ENTH_VAP_EXP_DATA[mol_id](temp))
    loss_ΔH_vap = MODEL_PARAMS["training"]["loss_weight_enth_vap"] * abs(ΔH_vap - ΔH_vap_exp)
    ignore_derivatives() do
        if mol_id == "vapourisation_liquid_O" && temp == T(295.0) && frame_i == 101 && repeat_i == 1
            store_string("ΔHvap", "ΔHvap water $ΔH_vap, exp $ΔH_vap_exp, loss $loss_ΔH_vap")
        end
    end
    return loss_ΔH_vap
end

const pressure_enth_mixing = ustrip(T, u"kJ * mol^-1", 1.0u"bar" * 1.0u"nm^3" * Unitful.Na)

function enth_mixing_loss(pe_com, pe_1, pe_2, bound_com, bound_1, bound_2,
                          n_mols_com, n_mols_1, n_mols_2, mol_id, frame_i, repeat_i)
    u_mol_com = (pe_com + pressure_enth_mixing * Molly.volume(bound_com)) / n_mols_com
    u_mol_1   = (pe_1   + pressure_enth_mixing * Molly.volume(bound_1)  ) / n_mols_1
    u_mol_2   = (pe_2   + pressure_enth_mixing * Molly.volume(bound_2)  ) / n_mols_2
    ΔH_mix = u_mol_com - (u_mol_1 + u_mol_2) / 2 # β cancels out
    ΔH_mix_exp = enth_mixing_exp_data[mol_id]
    loss_ΔH_mix = loss_weight_enth_mixing * abs(ΔH_mix - ΔH_mix_exp)
    ignore_derivatives() do
        if mol_id == "mixing_combined_CCCCO_OC1=NCCC1" && frame_i == 101 && repeat_i == 1
            store_string("ΔHmix", "ΔHmix CCCCO_OC1=NCCC1 $ΔH_mix ($u_mol_com $u_mol_1 $u_mol_2), exp $ΔH_mix_exp, loss $loss_ΔH_mix")
        end
    end
    return loss_ΔH_mix
end

function reweighting_loss_grads(mol_id, feat_df, temp, training_sim_dir, models...)
    
    TZ = Float64
    frames = 100:250

    # --- reference energies U_ref (kJ/mol) and coords/boundaries ---
    exp_type, sim_type, smiles = split(mol_id, "_"; limit=3)
    log_file_path = joinpath(training_sim_dir, "$(exp_type)_liquid", "$(smiles)_$(Int(temp))K.log")
    trj_file_path = joinpath(training_sim_dir, "$(exp_type)_liquid", "$(smiles)_$(Int(temp))K.dcd")
    log_data = CSV.read(log_file_path, DataFrame; delim = ",", header = 1)

    Uref = TZ.(log_data[frames, 2])                 # kJ/mol
    Tsim = TZ(mean(log_data[frames, 3]))            # K
    β = TZ(1) / calc_RT(Tsim)                       # 1/(kJ/mol)

    coords_list, boundary_list = read_sim_data(trj_file_path, frames)

    # --- per-frame density loss L_i ---
    total_mass = sum(ATOMIC_MASSES[parse.(Int, _ for _ in split(feat_df[!,:ATOMIC_MASS][1], ","))]) # g/mol
    volume = Molly.volume.(boundary_list)           # nm^3
    win_size = 10
    κ = calc_compressibility(TZ(calc_RT_nomol(Tsim)), TZ.(volume);
                            win_size = win_size, step_size = 1) # MPa^-1

    ρ = (total_mass ./ volume) / ustrip(u"mol^-1", TZ(Unitful.Na)) * 1e21  # g/L

    Lρ = abs.(ρ .- WATER_DENSITY[temp])  # scalar per frame
    Lκ = abs.(κ .- WATER_COMPRESS[temp]) # scalar per frame
    

    # --- chunk accumulators (log-sum-exp style) ---
    nmodels = length(models)
    n_chunks = Threads.nthreads()

    w_chunks      = zeros(TZ, n_chunks)                       # stores sum_i exp(a_i - a*_chunk)
    Lρw_chunks     = zeros(TZ, n_chunks)                       # stores sum_i L_i exp(a_i - a*_chunk)
    Lκw_chunks     = zeros(TZ, n_chunks)                       # stores sum_i L_i exp(a_i - a*_chunk)
    ∂θUθw_chunks  = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    Lρ∂θUθw_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    Lκ∂θUθw_chunks = [convert(Vector{Any}, fill(nothing, length(models))) for _ in 1:n_chunks]
    amax_chunks   = fill(-TZ(Inf), n_chunks)                   # a*_chunk = max_i a_i in the chunk

    Threads.@threads for chunk_id in 1:n_chunks
        amax = amax_chunks[chunk_id]
        for i in chunk_id:n_chunks:(length(frames)-win_size+1)
            frame_i, coords, boundary = frames[i], coords_list[i], boundary_list[i]
            Uθ, ∂θUθ = Zygote.withgradient(pe_from_snapshot, 1, mol_id, feat_df, coords, boundary, models...)
            ∂θUθ = ∂θUθ[6:end]

            ΔU = TZ(Uθ) - Uref[i]
            a  = -β * ΔU                               # a_i = -β ΔU_i

            if a > amax                                # new maximum -> rescale existing chunk sums
                s = exp(amax - a)                      # factor < 1
                w_chunks[chunk_id]     *= s
                Lρw_chunks[chunk_id]   *= s
                Lκw_chunks[chunk_id]   *= s
                ∂θUθw_chunks[chunk_id]  = multiply_grads.(∂θUθw_chunks[chunk_id], TZ(s))
                Lρ∂θUθw_chunks[chunk_id] = multiply_grads.(Lρ∂θUθw_chunks[chunk_id], TZ(s))
                Lκ∂θUθw_chunks[chunk_id] = multiply_grads.(Lκ∂θUθw_chunks[chunk_id], TZ(s))
                amax = a
            end

            t = exp(a - amax)                          # stabilized weight in [0,1]
            ∂θUθt  = multiply_grads(∂θUθ, TZ(t))
            
            Lρ∂θUθt = multiply_grads(∂θUθt, TZ(Lρ[i]))
            Lκ∂θUθt = multiply_grads(∂θUθt, TZ(Lκ[i]))
            
            Lρt     = TZ(Lρ[i]) * TZ(t)
            Lκt     = TZ(Lκ[i]) * TZ(t)

            w_chunks[chunk_id]      += TZ(t)
            Lρw_chunks[chunk_id]    += Lρt
            Lκw_chunks[chunk_id]    += Lκt
            ∂θUθw_chunks[chunk_id]   = accum_grads.(∂θUθw_chunks[chunk_id],   ∂θUθt)
            Lρ∂θUθw_chunks[chunk_id] = accum_grads.(Lρ∂θUθw_chunks[chunk_id], Lρ∂θUθt)
            Lκ∂θUθw_chunks[chunk_id] = accum_grads.(Lκ∂θUθw_chunks[chunk_id], Lκ∂θUθt)
        end
        amax_chunks[chunk_id] = amax
    end

    # --- reduce chunks with cross-chunk rescaling to the global a* ---
    amax_global = maximum(amax_chunks)
    scale = exp.(amax_chunks .- amax_global)                 # per-chunk factors

    w_sum   = sum(w_chunks   .* scale)
    Lρw_sum = sum(Lρw_chunks .* scale)
    Lκw_sum = sum(Lκw_chunks .* scale)

    ∂θUθw_sum   = convert(Vector{Any}, fill(nothing, length(models)))
    Lρ∂θUθw_sum = convert(Vector{Any}, fill(nothing, length(models)))
    Lκ∂θUθw_sum = convert(Vector{Any}, fill(nothing, length(models)))

    for chunk_id in 1:n_chunks
        s = TZ(scale[chunk_id])
        ∂θUθw_sum   = accum_grads.(∂θUθw_sum , multiply_grads.(∂θUθw_chunks[chunk_id],  s))
        Lρ∂θUθw_sum = accum_grads.(Lρ∂θUθw_sum, multiply_grads.(Lρ∂θUθw_chunks[chunk_id], s))
        Lκ∂θUθw_sum = accum_grads.(Lκ∂θUθw_sum, multiply_grads.(Lκ∂θUθw_chunks[chunk_id], s))
    end

    Lρ∂θUθw_w    = multiply_grads(Lρ∂θUθw_sum, TZ(1)/w_sum)
    Lκ∂θUθw_w    = multiply_grads(Lκ∂θUθw_sum, TZ(1)/w_sum)
    
    Lρw_w        = Lρw_sum / w_sum
    Lκw_w        = Lκw_sum / w_sum
    
    ∂θUθw_w     = multiply_grads(∂θUθw_sum, TZ(1)/w_sum)
    
    Lρw∂θUθw_ww  = multiply_grads(∂θUθw_w, Lρw_w)
    Lκw∂θUθw_ww  = multiply_grads(∂θUθw_w, Lκw_w)

    mLρw∂θUθw_ww = multiply_grads(Lρw∂θUθw_ww, TZ(-1))
    mLκw∂θUθw_ww = multiply_grads(Lκw∂θUθw_ww, TZ(-1))

    covar_ρ = accum_grads(Lρ∂θUθw_w, mLρw∂θUθw_ww)
    covar_κ = accum_grads(Lκ∂θUθw_w, mLκw∂θUθw_ww)

    ∂θLρ = multiply_grads(covar_ρ, TZ(-1)*β)
    ∂θLκ = multiply_grads(covar_κ, TZ(-1)*β)

    Lρ_mean = mean(@view Lρ[1:length(frames)-win_size+1])
    Lκ_mean = mean(Lκ)

    return Lρ_mean, Lκ_mean, ∂θLρ, ∂θLκ
end