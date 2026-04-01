# calls python utilies / inference scripts




using TOML
using Dates

function run(cmd::Cmd)
    println('run:', cmd)
    success = run(cmd)
    return success
end

function main()
    # julia inference.jl --repo /path/to/repo --pyenv bounded-dit --mode split --src ... --out ...
    args = Dict{String, String}()
    i = 1

    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            val = (i < length(ARGS)) ? ARGS[i+1] : ''
            args[key] = val

            i += 2

        else
            i += 1
        end
    end

    repo = get(args, 'repo', pwd())
    mode = get(args, 'mode', 'splith5')
    h5 = get(args, 'h5','')
    src = get(args, 'src', '')
    out = get(args, 'out', joinpath(repo, 'rough_pcam_h5'))
    seed = get(args, 'seed', '0')
    split = get(args, 'split', 'train')
    n = get(args, 'n', '64')
    copy = get(args, 'copy', 'false')

    py = get(args, 'python', 'python')
    diffusion_py = joinpath(repo, 'diffusion.py')

    println('repo:', repo)
    println('mode:', mode)
    println('time', now())

    if isempty(h5)
        error('--h5 is required')
    end

    if mode == 'split' # split_h5
        if isempty(src)
            error('--src is required for mode=split')
        end
        cmd = `$py $diffusion_py --make_split --src $src out $out --seed $seed`
        if lowercase(copy) == 'true'
            cmd = `$py $diffusion_py --make_split --src $src --out $out --seed $seed`

        end
        run(cmd)

    elseif mode == 'sample_demo'
        T = get(args, 'T', '50')
        schedule = get(args, 'schedule', 'cosine')
        device = get(args, 'device', 'cuda')
        B = get(args,'B', '2')
        C = get(args,'C', '3')
        L = get(args,'L', '96')
        W = get(args,'W', '96')
        cmd = `$py $diffusion_py --sample_demo --T $T --schedule $schedule`
        run(cmd)

    elseif mode == 'split_h5'
        cmd = `$py $diffusion_py --make_split_h5 --h5 $h5 --out $out --seed $seed`
        println('run', cmd)
        run(cmd)
    elseif mode == 'preview'
        cmd = `$py $diffusion_py --export_preview --h5 $h5 --out $out --seed $seed`
        println('run', cmd)
        run(cmd)
    else
        error(`unknown mode: $mode`)
    end

    println('done')
end

main()