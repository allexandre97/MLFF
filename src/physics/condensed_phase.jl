calc_RT(temp) = ustrip(u"kJ/mol", T(Unitful.R) * temp * u"K")
calc_RT_nomol(temp) = ustrip(u"kJ", T(Unitful.k) * temp * u"K")
@non_differentiable calc_RT(args...)

function enthalpy_vaporization(snapshot_U_liquid, mean_U_gas, temp, n_molecules)
    # See https://docs.openforcefield.org/projects/evaluator/en/stable/properties/properties.html
    RT = calc_RT(temp)
    ΔH_vap = mean_U_gas - snapshot_U_liquid / n_molecules + RT
    return ΔH_vap
end

function calc_permitivity(Ms, Vs, temp;
    win_size::Int=5,
    step_size::Int=1,
)
    ε0 = ustrip(u"F/m", Unitful.ϵ0)
    kB = ustrip(Unitful.k)

    nwin = 1 + (length(Vs) - win_size) ÷ step_size
    ϵ    = Vector{Float64}(undef, nwin)  # MPa^-1

    j = 1
    @views for i in 1:step_size:(length(Vs)-win_size+1)
        
        winV = Vs[i:i+win_size-1]
        winM = Ms[i:i+win_size-1]

        avgv  = mean(winV)
        avgm  = sum(winM) / length(winM)
        avgm2 = mean(dot.(winM, winM))
        m2    = dot(avgm, avgm)

        ϵ[j] = 1 + (avgm2 - m2) / (3 * ε0 * kB * temp * avgv)

        j += 1
    end

    return ϵ

end

function calc_compressibility(
    kBT_kJ::T,
    volume::AbstractVector;
    win_size::Int=5,
    step_size::Int=1,
) where {T<:Real}

    @assert win_size ≥ 2 "win_size must be ≥ 2"
    @assert step_size ≥ 1 "step_size must be ≥ 1"
    @assert length(volume) ≥ win_size "window larger than data"

    # Work in native units: volume in nm^3, kBT in kJ
    vol = Float64.(volume)
    kBT = Float64(kBT_kJ)

    # Unit conversion factor: (nm^3 / kJ) → MPa^-1
    conv = ustrip(uconvert(u"MPa^-1", 1u"nm^3"/1u"kJ"))

    nwin = 1 + (length(vol) - win_size) ÷ step_size
    κ = Vector{Float64}(undef, nwin)  # MPa^-1

    j = 1
    @views for i in 1:step_size:(length(vol)-win_size+1)
        win = vol[i:i+win_size-1]
        m = mean(win)         # nm^3
        v = var(win)          # nm^6  (sample variance)
        κ[j] = (v / (kBT * m)) * conv  # MPa^-1
        j += 1
    end
    return κ
end