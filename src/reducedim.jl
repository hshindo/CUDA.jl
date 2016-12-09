import Base: sum, max

function sum{T,N}(x::CuArray{T,N}, dim::Int)
    size(x,dim) > 128 && throw("Not implemented yet.")
    (1 <= dim <= 2) || throw("Not implemented yet.")

    y = similar(x, ntuple(i -> i==dim ? 1 : size(x,i), N))
    t = ctype(T)
    f = @nvrtc CuArray{T,N} """
    template<typename T>
    __inline__ __device__ T warpReduce(T val) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += __shfl_down(val, offset);
        }
        return val;
    }

    template<typename T>
    __inline__ __device__ T blockReduce(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        val = warpReduce<T>(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        if (wid == 0) val = warpReduce<T>(val);
        return val;
    }

    $array_h
    __global__ void f(Array<$t,$N> x, int dim, Array<$t,$N> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (dim == 1) {
            if (idx_x >= x.dims[0]) return;
            $t val = x(idx_x, idx_y);
            val = blockReduce<$t>(val);
            if (threadIdx.x == 0) y(blockIdx.x, idx_y) = val;
        } else {
            if (idx_x >= x.dims[1]) return;
            $t val = x(idx_y, idx_x);
            val = blockReduce<$t>(val);
            if (threadIdx.x == 0) y(idx_y, blockIdx.x) = val;
        }
    }
    """
    dx = dim == 1 ? size(x,1) : size(x,2)
    dy = dim == 1 ? size(x,2) : size(x,1)
    f(dx, dy, 1, x, Cint(dim), y, sharedmem=sizeof(T)*32)
    y
end

function max{T,N}(x::CuArray{T,N}, dim::Int)
    size(x,dim) > 128 && throw("Not implemented yet.")
    (1 <= dim <= 2) || throw("Not implemented yet.")

    y = similar(x, ntuple(i -> i==dim ? 1 : size(x,i), N))
    t = ctype(T)
    f = @nvrtc CuArray{T,N} """
    template<typename T>
    __inline__ __device__ T warpReduce(T val) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            T comp = __shfl_down(val, offset);
            val = (comp > val) ? comp : val;
        }
        return val;
    }

    template<typename T>
    __inline__ __device__ T blockReduce(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        val = warpReduce<T>(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        if (wid == 0) val = warpReduce<T>(val);
        return val;
    }

    $array_h
    __global__ void f(Array<$t,$N> x, int dim, Array<$t,$N> y) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (dim == 1) {
            if (idx_x >= x.dims[0]) return;
            $t val = x(idx_x, idx_y);
            val = blockReduce<$t>(val);
            if (threadIdx.x == 0) y(blockIdx.x, idx_y) = val;
        } else {
            if (idx_x >= x.dims[1]) return;
            $t val = x(idx_y, idx_x);
            val = blockReduce<$t>(val);
            if (threadIdx.x == 0) y(idx_y, blockIdx.x) = val;
        }
    }
    """
    dx = dim == 1 ? size(x,1) : size(x,2)
    dy = dim == 1 ? size(x,2) : size(x,1)
    f(dx, dy, 1, x, Cint(dim), y, sharedmem=sizeof(T)*32)
    y
end
