function report(args...)
    print(args...)
    out_dir = MODEL_PARAMS["paths"]["out_dir"]
    if !isnothing(out_dir)
        open(joinpath(out_dir, "training.log"), "a") do of
            print(of, args...)
        end
    end
end