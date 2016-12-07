# CUDA.jl
CUDA.jl is CUDA bindings for Julia.  
It supports runtime compiling provided by [NVRTC](http://docs.nvidia.com/cuda/nvrtc/index.html)
so that users can easily add custom kernels.

## Support
CUDA 7.0 or higher.

Check your compute capability from [here](https://developer.nvidia.com/cuda-gpus)

## Install
```julia
julia> Pkg.clone("https://github.com/hshindo/CUDA.jl.git")
```

## Usage
`CuArray{T,N}` is analogous to `Base.Array{T,N}` in Julia.

```julia
x = CuArray{Float32}(10,5)
xx = Array(x)
```

## Writing Custom Kernel
A custom kernel can be embedded into julia code.  

The following is an example of `fill` function:
```julia
function Base.fill!{T}(a::CuArray{T}, value)
    t = ctype(T)
    f = @cu t """
    __global__ void f($t *x, int length, $t value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) x[idx] = value;
    } """
    f(length(a), 1, 1, pointer(a), length(a), T(value))
    a
end
```
where `@cu` compiles the CUDA native code at runtime and returns a `CuFunction` object.  
The compiled output is automatically cached.  
