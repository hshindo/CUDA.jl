__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void reduce(int *in, int *out, int N) {
    int sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    //sum = warpReduceSum(sum);
    //if (threadIdx.x & (warpSize - 1) == 0) atomicAdd(out, sum);
}
