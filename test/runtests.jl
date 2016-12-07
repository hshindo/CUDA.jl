using Base.Test
using CUDA

function checkdiff(x, y)

end

T = Float32
x = rand(T,10,8,5)
y = rand(T,10,8,5)
cux = CuArray(x)
cuy = CuArray(y)
copy!(cuy, cux)

# array


# arraymath
x = rand(T,10,5)
cux = CuArray(x)
Array(exp(cux)) - x
