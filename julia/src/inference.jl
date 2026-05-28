# calls python utilies / inference scripts
using Pkg
Pkg.add("DotEnv")


using TOML
using Dates
using UUIDs
using SHA

using JSON3
using Base.Threads


using DataFrames
using CSV
using DotEnv
DotEnv.config()

# env keys
python = ENV["PYTHON_BIN"]
pyenv = ENV["PYENV_NAME"]
src = ENV["DATASET_PATH"]


repo = ENV["REPO_NAME"]
out_root = ENV["OUT_ROOT"]
summary_csv = ENV["CSV"]

function load_config(path::String)
    return TOML.parsefile(path)
end


function write_toml(path::String, cfg::Dict)
    open(path, "w") do io
        TOML.print(io, cfg)
    end
end

function hash_config(cfg::Dict)
    return bytes2hex(sha1(cfg))
end

function stable_run_id(cfg::Dict)
    s = JSON3.write(sort(collect(cfg)))
    l = bytes2hex(sha1(s))[1:12]
    return "run_" * l
end

function read_metrics(path:String)
    if !isfile(path)
        return Dict{String, Any}(
            "psnr" => NaN,
            "mse" => NaN,
            "aurc" => NaN,
            "finite" => 0,
        )
    end

    return Dict{String, Any}(JSON3.read(read(path, String)))
end


function flatten(prefix, d, out)
    for (k, v) in d
        key = isempty(prefix)? string(k) : "$(prefix).$(k)"

        if v isa Dict
            flatten(key, v, out)
        else
            out[key] = v
        end
    end
end

function flatten_dict(d::Dict)
    out = Dict{String, Any}()
    flatten("", d, out)
    return out
end

function resolve_config(cfg::Dict)

    flattened = flatten_dict(cfg)
    config = [{Dict{String, Any}}()]

    for (k, values) in sort(collect(flattened))

        values = values isa Vector ? values : [values]


        cfg = Vector{Dict{String, Any}}()
        
        for partial_cfg in cfg
            for v in values
                next_cfg = copy(partial_cfg)
                next_cfg[k] = v
                push!(cfg, next_cfg)
            end
        end

        config = cfg
    end

    return config
end

function expand_grid(dict::Dict{String, Vector})
    keys_ = collect(keys(dict))
    values_ = [dict[k] for k in keys]

    result = Vector{Dict{String, Any}}()

    
    function rec(i, cur)
        if i > length(keys_)
            push!(result, deepcopy(cur))
            return
        end

        k = keys_[i]
        for v in values_[i]
            cur[k] = v
            rec(i + 1, cur)
            
            println(keys[i], ':', v[i])
        end
    end

    rec(1, Dict{String, Any}())

    return result
end

function run_phase(cmd::Cmd; log_path::Union{Nothing, String}=nothing)
    println("run:", cmd)

    if log_path === nothing
        try
            run(cmd)
            return true
        catch
            return false
    end

    open(log_path, "w") do io
        try
            run(pipeline(cmd, stdout=io, stderr=io))
            return true
        catch
            return false
        end
    end
end

function build_sweeps(cfg)
    sweep = cfg["sweep"]

    flattened = flatten_dict(sweep)

    grid = Dict{String, Vector}()

    for (k, v) in flattened
        grid[k] = v isa Vector ? v : [v]
    end

    return expand_grid(grid)
end

function build_cmd(global_cfg, sweep_cfg, root_dir)
    repo = global_cfg["repo"]
    py = global_cfg["python"]
    
    pyenv = global_cfg["pyenv"]
    script = joinpath(repo, global_cfg["python_script"])

    args = String[
        script,
        "--src", global_cfg["src"],
        "--out", root_dir
    ]

    for (k, v) in sort(collect(sweep_cfg))
        cli_key = replace(k, "." => "_")
        push!(args, "--$cli_key")
        push!(args, string(v))
    end

    if !isempty(pyenv)
        return `conda run -n $(global["pyenv"]) python $(args...)`
    else
        return `$py $(args...)`
    end
end

function make_output_dir(base, cfg)
    hash = hash_config(cfg)
    dir = joinpath(base, hash[1:8])
    mkpath(dir)
    return dir
end

function save_config(output_dir, cfg)
    open(joinpath(output_dir, "config.json"), "w") do io
        stable_run_id(cfg)
        
    end
end

function run_sweep(cfg_path::String)
    cfg = load_config(cfg_path)

    global_cfg = resolve_config(get(raw, "global", Dict{String, Any}()))
    sweeps = build_sweeps(cfg)

    
    root_dir = make_output_dir(global_cfg["out_root"], global_cfg)
    output_dir = make_output_dir(global_cfg["summary_csv"], global_cfg)

    rows = DataFrame()

    try
        for (i, sweep_cfg) in enumerate(sweeps)

            run_id = stable_run_id(sweep_cfg)
            cfg = merge(globals, partial_cfg)

            cfg_path = joinpath(root_dir, "config.toml")
            metric_path = joinpath(root_dir, "metrics.json")
            log_path = joinpath(root_dir, "stdout.log")

            write_toml(cfg_path, cfg)

            save_config(output_dir, sweep_cfg)

            println("[$i/$length(sweeps)] ", run_id)

            if isfile(metric_path)
                println("skip existing:", run_id)
            else
                cmd = build_cmd(globals, sweep_cfg, root_dir)
                ok = run(cmd; log_path=log_path)

                if !ok
                    println("failed to run:" run_id)
                end
            end

            metrics = read_metrics(metric_path)

            row = marge(
                Dict{String, Any}("run_id" => run_id, "run_id" => "run_id" => run_id), sweep_cfg, metrics
            )

            push!(rows, row; cols=:union)

            CSV.write(summary_csv, rows)
        end
    
        println("summary saved:", summary_csv)

        println("[$i/$(length(sweeps))] running: ", cmd)
        
    catch e
        println("failed: ", e)
    end
end

function extract_name(path::String)
    return splitext(basename(path))[1]
end    
   
function main()
    # julia inference.jl configs/project.toml --repo /path/to/repo --pyenv $(name) --src ...
    args = Dict{String, String}()
    i = 1

    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            
            if i < length(ARGS) && !startswith(ARGS[i+1], "--")
                val = ARGS[i+1]
                i += 2
            
            else
                val = ""
                i += 1
            end

            args[key] = val

        else
            i += 1
        end
    end

    repo = get(args, "repo", pwd())
    h5 = get(args, "h5", "")
    src = get(args, "src", "")
    seed = get(args, "seed", "0")
    copy = get(args, "copy", "false")

    py = get(args, "python", "python")

    println("repo:", repo)
    println("src:", src)
    println("time", now())

    if isempty(h5)
        error('--h5 is required')
    end

    
    config_path = length(ARGS) >= 1 ? ARGS[1]: "configs/Project.toml"
    run_sweeps(config_path)
    
    return args

end

main()