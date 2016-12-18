type CuFunction
    m::CuModule # avoid CuModule gc-ed
    ptr::Ptr{Void}
end

function CuFunction(m::CuModule, name::String)
    p = CUfunction[0]
    cuModuleGetFunction(p, m, name)
    CuFunction(m, p[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr

function compile(code::String)
    ptx = NVRTC.compile(code)
    p = Ptr{Void}[0]
    cuModuleLoadData(p, pointer(ptx))
    mod = CuModule(p[1])
    # TODO: multi-device
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        fname = Symbol(m[1])
        return CuFunction(mod, string(fname))
    end
end

box(x) = pointer_from_objref(x)

@generated function typesym{T}(::Type{T})
    syms = (Symbol(T),)
    :($syms[1])
end
@generated function typesym{T1,T2}(::Type{T1}, ::Type{T2})
    syms = (Symbol(T1, "_", T2),)
    :($syms[1])
end

macro compile(key, code)
    

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

function (f::CuFunction)(dx::Int, dy::Int, dz::Int, args...;
    bx=128, by=1, bz=1, sharedmem=0, stream=C_NULL)

    argptrs = Ptr{Void}[box(a) for a in args]
    gx = ceil(dx / bx)
    gy = ceil(dy / by)
    gz = ceil(dz / bz)
    cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedmem, stream, argptrs, C_NULL)
end
