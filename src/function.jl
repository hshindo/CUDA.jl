type CuFunction
    m::CuModule # avoid CuModule gc-ed
    ptr::Ptr{Void}
end

function CuFunction(m::CuModule, name::String)
    ref = CUfunction[0]
    cuModuleGetFunction(ref, m, name)
    CuFunction(m, ref[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr

macro nvrtc(key, expr)
    dict = Dict{Any,CuFunction}()
    quote
        local dict = $dict
        local key = $(esc(key))
        if haskey(dict, key)
            dict[key]
        else
            local code = $(esc(expr))
            local ptx = NVRTC.compile(code)
            f = load_ptx(ptx)
            dict[key] = f
            f
        end
    end
end

function load_ptx(code::String)
    ref = Ptr{Void}[0]
    cuModuleLoadData(ref, pointer(code))
    mod = CuModule(ref[1])

    for line in split(code,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        fname = Symbol(m[1])
        curr_dev = device()
        f = CuFunction(mod, string(fname))
        return f
    end
end

box(x) = pointer_from_objref(x)

function (f::CuFunction)(dx::Int, dy::Int, dz::Int, args...;
    blocksize::NTuple{3,Int}=(128,1,1), sharedmem=4, stream=C_NULL)

    argptrs = Ptr{Void}[box(a) for a in args]
    gx = ceil(dx / blocksize[1])
    gy = ceil(dy / blocksize[2])
    gz = ceil(dz / blocksize[3])
    cuLaunchKernel(f, gx, gy, gz, blocksize..., sharedmem, stream, argptrs, C_NULL)
end

#=
function compile(path=joinpath(Pkg.dir("CUDA"),"kernels"))
    for str in readdir(path)
        filename = joinpath(path, str)
        endswith(filename, ".cu") || continue
        cmd = `nvcc -ptx $(filename) -odir $(path)`
        println("Running...")
        println(cmd)
        run(cmd)
    end
end
=#
