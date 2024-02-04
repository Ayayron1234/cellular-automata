#include "temp.h"
#include <curand_kernel.h>

__global__
void test_kernel(GlobalReadWriteBuffer<int> buffer, Options options) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * 20 + col;

    buffer.Write(i, row * col);
}

void test(GlobalReadWriteBuffer<int> buffer, Options options) {
    dim3 gridSize(20, 20);
    dim3 blockSize(2, 2);
    test_kernel << < gridSize, blockSize >> > (buffer, options);
    buffer.AwaitDevice();
    buffer.SyncFromDevice();
}
