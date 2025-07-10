# 1) Build the global graph
function build_global_graph(n_atoms, bonds)
    g = SimpleGraph{Int}(n_atoms)
    for (u,v) in bonds
        add_edge!(g, u, v)
    end
    return g
end

Flux.@non_differentiable build_global_graph(args...)

# 2) Extract every connected‐component subgraph + its global atom indices
function extract_all_subgraphs(g::SimpleGraph)
    comps = connected_components(g)            # Vector{Vector{Int}}
    subgs = SimpleGraph{Int}[]
    inds  = Vector{Vector{Int}}()
    for c in comps
        vs = Int.(c)
        sg, _ = induced_subgraph(g, vs)
        push!(subgs, sg)
        push!(inds, vs)
    end
    return subgs, inds
end

Flux.@non_differentiable extract_all_subgraphs(args...)

# 3) Filter to unique molecule types (by isomorphism),
#    and count how many of each type we have.
function filter_unique(subgs::Vector{SimpleGraph{Int}},
                       inds::Vector{Vector{Int}})
    uniq_s = SimpleGraph{Int}[]
    uniq_i = Vector{Vector{Int}}()
    counts = Int[]

    for (g, vs) in zip(subgs, inds)
        found = false
        # check against each already-collected unique graph
        for (k, ug) in enumerate(uniq_s)
            if has_isomorph(ug, g, VF2())
                counts[k] += 1
                found = true
                break
            end
        end

        # if it's truly new, record it and start its count at 1
        if !found
            push!(uniq_s, g)
            push!(uniq_i, vs)
            push!(counts, 1)
        end
    end

    return uniq_s, uniq_i, counts
end

Flux.@non_differentiable filter_unique(ars...)

# 4) Graph‐omission helper for atom‐equivalence test
function relabel_bonds(edges::Vector{Tuple{Int,Int}}, omit::Int)
    out = Tuple{Int,Int}[]
    for (u,v) in edges
        if omit in (u,v)
            continue
        end
        u2 = u > omit ? u-1 : u
        v2 = v > omit ? v-1 : v
        push!(out, (u2,v2))
    end
    return out
end

Flux.@non_differentiable relabel_bonds(args...)

# 5) Find equivalence pairs → Dict(global_index => [globals…])
function find_atom_equivalences(g::SimpleGraph, global_vs::Vector{Int}, elements::Vector{Int})
    n = nv(g)
    local_el   = [elements[v] for v in global_vs]
    edges_local = [(src(e), dst(e)) for e in edges(g)]
    equivs = Dict(v => Int[] for v in global_vs)

    for i in 1:n
        # build g_i
        gi = SimpleGraph(n-1)
        for (u,v) in relabel_bonds(edges_local, i)
            add_edge!(gi, u, v)
        end
        el_i = [local_el[k] for k in 1:n if k != i]

        for j in (i+1):n
            if local_el[j] != local_el[i]
                continue
            end
            gj = SimpleGraph(n-1)
            for (u,v) in relabel_bonds(edges_local, j)
                add_edge!(gj, u, v)
            end
            el_j = [local_el[k] for k in 1:n if k != j]

            vr = (u,v) -> el_i[u] == el_j[v]
            if has_isomorph(gi, gj, VF2(); vertex_relation=vr)
                gi_glob = global_vs[i]
                gj_glob = global_vs[j]
                push!(equivs[gi_glob], gj_glob)
                push!(equivs[gj_glob], gi_glob)
            end
        end
    end

    return equivs
end

Flux.@non_differentiable find_atom_equivalences(args...)

# 6) Turn equivalence‐dict + ELEMENT_TO_NAME into local labels
function label_molecule(global_vs::Vector{Int},
                        equivs::Dict{Int,Vector{Int}},
                        elements::Vector{Int})
    n = length(global_vs)
    # find connected components in the equivalence graph
    visited = falses(n)
    classes = Vector{Vector{Int}}()
    for i in 1:n
        if visited[i] continue end
        queue = [i]
        comp  = Int[]
        visited[i] = true
        while !isempty(queue)
            u = popfirst!(queue)
            push!(comp, u)
            for gj in equivs[ global_vs[u] ]
                # find local position
                pos = findfirst(==(gj), global_vs)
                if !visited[pos]
                    visited[pos] = true
                    push!(queue, pos)
                end
            end
        end
        push!(classes, sort(comp))
    end

    # assign suffix numbers per element type
    suffix = Dict{Int,Int}()  # class_idx -> suffix
    counts = Dict{Int,Int}()  # element -> next suffix
    for cls in sort(classes, by=first)
        # element at any member
        el = elements[ global_vs[ cls[1] ] ]
        k = counts[el] = get(counts, el, 0) + 1
        suffix[ cls[1] ] = k
    end

    # now build labels in order 1:n
    labels = String[]
    for i in 1:n
        el    = elements[ global_vs[i] ]
        name  = ELEMENT_TO_NAME[el]
        # find class that contains i
        cls1 = findfirst(c->i in c, classes)
        num   = suffix[ classes[cls1][1] ]
        push!(labels, "$(name)$(num)")
    end

    return labels
end

Flux.@non_differentiable label_molecule(args...)