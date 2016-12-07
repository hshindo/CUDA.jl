type CuModule
    ptr::Ptr{Void}

    function CuModule(ptr)
        m = new(ptr)
        finalizer(m, cuModuleUnload)
        m
    end
end

function CuModule(image::Vector{UInt8})
    ref = Ptr{Void}[0]
    cuModuleLoadData(ref, image)
    #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
    CuModule(ref[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, m::CuModule) = m.ptr
