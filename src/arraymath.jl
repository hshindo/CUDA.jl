import Base: .+, +, .-, -, .*, *, /, ./
import Base: exp, log

for op in (:exp,:log)
    @eval begin
        function $op{T,N}(x::AbstractCuArray{T,N})
            op = $op
            y = CuArray{T}(size(x))
            t = ctype(T)
            f = @nvrtc CuArray{T,N} """
            $array_h
            __global__ void f(Array<$t,$N> x, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    y(idx) = $op(x(idx));
                }
            }
            """
            f(length(y), 1, 1, x, y)
            y
        end
    end
end

for (fname,op) in ((:.+,:+),(:.-,:-),(:.*,:*),(:./,:/))
    @eval begin
        function $fname{T,N}(x1::AbstractCuArray{T,N}, x2::AbstractCuArray{T,N})
            size(x1) == size(x2) && return $op(x1, x2)
            op = $op
            y = CuArray{T}(ntuple(i -> max(size(x1,i),size(x2,i)), N))
            t = ctype(T)
            f = @nvrtc CuArray{T,N} """
            $array_h
            __global__ void f(Array<$t,$N> x1, Array<$t,$N> x2, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    int subs[$N];
                    y.idx2sub(idx, subs);
                    y(subs) = x1(subs) $op x2(subs);
                }
            }
            """
            f(length(y), 1, 1, x1, x2, y)
            y
        end
    end
end

for op in (:+,:-,:/)
    @eval begin
        function $op{T,N}(x1::AbstractCuArray{T,N}, x2::AbstractCuArray{T,N})
            size(x1) == size(x2) || throw(DimensionMismatch())
            op = $op
            y = similar(x1)
            t = ctype(T)
            f = @nvrtc CuArray{T,N} """
            $array_h
            __global__ void f(Array<$t,$N> x1, Array<$t,$N> x2, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    y(idx) = x1(idx) $op x2(idx);
                }
            }
            """
            f(length(y), 1, 1, x1, x2, y)
            y
        end
    end
end

*(x1::CuMatrix, x2::CuMatrix) = BLAS.gemm('N', 'N', x1, x2)
