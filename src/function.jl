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
        f = CuFunction(mod, string(fname))
        break
    end
    f
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
