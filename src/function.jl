export @nvrtc

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

export @nvrtc2
const cufuns = Dict{String,CuFunction}()

macro nvrtc2(src)
    src.head == :string || throw("expr is not string")
    dict = Dict()
    syms = Any[string(gensym())]
    for arg in src.args
        typeof(arg) != Symbol && continue
        haskey(dict, arg) && continue
        dict[arg] = arg
        push!(syms, "_", arg)
    end
    expr = Expr(:string, syms...)
    quote
        local key = $(esc(expr))
        if haskey(cufuns, key)
            cufuns[key]
        else
            local code = $(esc(src))
            local ptx = NVRTC.compile(code)
            f = load_ptx(ptx)
            cufuns[key] = f
            f
        end
    end
end

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
    bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)

    argptrs = Ptr{Void}[box(a) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, C_NULL)
end
