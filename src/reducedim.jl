import Base: sum, max
export reducedim_kernel, argmax

@generated function sum{T,N}(x::CuArray{T,N}, dim::Int)
    t = ctype(T)
    f = compile("""
    $array_h
    $reducedim_h
    __global__ void f(Array<$t,3> x, Array<$t,$N> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        if (idx_y >= x.dims[1]) return;

        $t val = x(idx_x, idx_y, idx_z);
        val = blockReduce<$t>(val);
        if (threadIdx.y == 0) y(blockIdx.x, 0, blockIdx.z) = val;
    } """)
    quote
        y = similar(x, ntuple(i -> i==dim ? 1 : size(x,i), N))
        x = reshape3d(x, dim)
        $f(size(x,1), size(x,2), size(x,3), x, reshape3d(y,dim), bx=1, by=128, sharedmem=sizeof(T)*32)
        y
    end
end

function argmax{T,N}(x::CuArray{T,N}, dim::Int)
    f = reducedim_kernel("v0 < v1 ? v0 : v1", "0")
    f(x, dim)
end

function reshape3d(x::CuArray, dim::Int)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x,i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x,i)
    end
    reshape(x, dim1, dim2, dim3)
end

type ReduceKernel
    code::String
end

function ReduceKernel(op::String, v0::String)
    t = "float"
    code = """
    $array_h

    template<typename T>
    __inline__ __device__ T warpReduce(T v0) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            T v1 = __shfl_down(v0, offset);
            v0 = $op;
        }
        return v0;
    }

    template<typename T>
    __inline__ __device__ T blockReduce(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.y % warpSize;
        int wid = threadIdx.y / warpSize;

        val = warpReduce<T>(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (lane < 4) ? shared[lane] : 0;
        //if (wid == 0) val = warpReduce<T>(val);
        val = warpReduce<T>(val);
        return val;
    }

    __global__ void f(Array<$t,3> x, Array<$t,3> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        if (idx_y >= x.dims[1]) return;

        $t val = x(idx_x, idx_y, idx_z);
        val = blockReduce<$t>(val);
        if (threadIdx.y == 0) y(blockIdx.x, 0, blockIdx.z) = val;
    } """
    ReduceKernel(code)
end

@generated function (kernel::ReduceKernel){T,N}(x::CuArray{T,N}, dim::Int, code::String)
    f = compile(:($code))
    quote
        y = similar(x, ntuple(i -> i==dim ? 1 : size(x,i), N))
        x = reshape3d(x, dim)
        $f(size(x,1), size(x,2), size(x,3), x, reshape3d(y,dim), bx=1, by=128, sharedmem=sizeof(T)*32)
        y
    end
end

function reducedim_kernel(op::String, v0::String)
    @eval begin
        @generated function reducedim{T,N}(x::CuArray{T,N}, dim::Int)
            t = ctype(T)
            f = compile("""
            $array_h

            template<typename T>
            __inline__ __device__ T warpReduce(T v0) {
                for (int offset = warpSize/2; offset > 0; offset /= 2) {
                    T v1 = __shfl_down(v0, offset);
                    v0 = v0 + v1;
                }
                return v0;
            }

            template<typename T>
            __inline__ __device__ T blockReduce(T val) {
                static __shared__ T shared[32];
                int lane = threadIdx.y % warpSize;
                int wid = threadIdx.y / warpSize;

                val = warpReduce<T>(val);
                if (lane == 0) shared[wid] = val;
                __syncthreads();

                val = (lane < 4) ? shared[lane] : 0;
                //if (wid == 0) val = warpReduce<T>(val);
                val = warpReduce<T>(val);
                return val;
            }

            __global__ void f(Array<$t,3> x, Array<$t,3> y) {
                int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
                int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
                int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
                if (idx_y >= x.dims[1]) return;

                $t val = x(idx_x, idx_y, idx_z);
                val = blockReduce<$t>(val);
                if (threadIdx.y == 0) y(blockIdx.x, 0, blockIdx.z) = val;
            } """)
            quote
                y = similar(x, ntuple(i -> i==dim ? 1 : size(x,i), N))
                x = reshape3d(x, dim)
                $f(size(x,1), size(x,2), size(x,3), x, reshape3d(y,dim), bx=1, by=128, sharedmem=sizeof(T)*32)
                y
            end
        end
    end
    reducedim
end

const reducedim_h = """
template<typename T>
__inline__ __device__ T warpReduce(T v0) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        T v1 = __shfl_down(v0, offset);
        v0 = $(reduce);
    }
    return v0;
}

template<typename T>
__inline__ __device__ T blockReduce(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.y % warpSize;
    int wid = threadIdx.y / warpSize;

    val = warpReduce<T>(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (lane < 4) ? shared[lane] : 0;
    //if (wid == 0) val = warpReduce<T>(val);
    val = warpReduce<T>(val);
    return val;
}
"""
