calc_RT(temp) = ustrip(u"kJ/mol", T(Unitful.R) * temp * u"K")
calc_RT_nomol(temp) = ustrip(u"kJ", T(Unitful.k) * temp * u"K")
@non_differentiable calc_RT(args...)

function enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    # See https://docs.openforcefield.org/projects/evaluator/en/stable/properties/properties.html
    RT = calc_RT(temp)
    ΔH_vap = mean_U_gas - snapshot_U_liquid / n_molecules + RT
    return ΔH_vap
end

function calc_compressibility(
    kBT_kJ::T,
    volume::Vector{T};
    win_size::Int=5,
    step_size::Int=1) where {T}

    # volume → m^3
    s = T(1e-27)
    V = s .* volume

    κ = T[]  # MPa^-1
    kBT = kBT_kJ * T(1e3)  # J per molecule

    for i in 1:step_size:length(V)-win_size+1
        win = @view V[i:i+win_size-1]
        m = mean(win)            # m^3
        v = var(win)             # m^6
        push!(κ, (v / (kBT * m)) * T(1e6))  # Pa^-1 → MPa^-1
    end
    κ
end