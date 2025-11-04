function ChainRulesCore.rrule(TY::Type{<:InteractionList2Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList2Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList2Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList3Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList3Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList3Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:InteractionList4Atoms}, vs...)
    Y = TY(vs...)
    function InteractionList4Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters,
               NoTangent()
    end
    return Y, InteractionList4Atoms_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicBond}, vs...)
    Y = TY(vs...)
    function HarmonicBond_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.r0
    end
    return Y, HarmonicBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:MorseBond}, vs...)
    Y = TY(vs...)
    function MorseBond_pullback(Ȳ)
        return NoTangent(), Ȳ.D, Ȳ.a, Ȳ.r0
    end
    return Y, MorseBond_pullback
end

function ChainRulesCore.rrule(TY::Type{<:HarmonicAngle}, vs...)
    Y = TY(vs...)
    function HarmonicAngle_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.θ0
    end
    return Y, HarmonicAngle_pullback
end

function ChainRulesCore.rrule(TY::Type{<:UreyBradley}, vs...)
    Y = TY(vs...)
    function UreyBradley_pullback(Ȳ)
        return NoTangent(), Ȳ.kangle, Ȳ.θ0,  Ȳ.kbond, Ȳ.r0
    end
    return Y, UreyBradley_pullback
end

function ChainRulesCore.rrule(TY::Type{<:PeriodicTorsion}, vs...)
    Y = TY(vs...)
    function PeriodicTorsion_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.phases, Ȳ.ks, NoTangent()
    end
    return Y, PeriodicTorsion_pullback
end