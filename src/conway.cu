#include <curand_kernel.h>
#include "global data/grid.h"
#include "IO.h"

__global__ 
void calcNextStateGameOfLife(Grid<int> grid) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = y * grid.Width() + x;

    int cell = grid.Get(x, y);
    int newCell = 0;
    int sum_neighborhood = grid.Get(x + 1, y) + grid.Get(x + 1, y + 1) + grid.Get(x, y + 1) + grid.Get(x - 1, y + 1)
        + grid.Get(x - 1, y) + grid.Get(x - 1, y - 1) + grid.Get(x, y - 1) + grid.Get(x + 1, y - 1);

    if (cell == 1) {
        if (sum_neighborhood < 2)
            newCell = 0;
        else if (sum_neighborhood < 4)
            newCell = 1;
        else
            newCell = 0;
    }
    else {
        if (sum_neighborhood == 3)
            newCell = 1;
    }

    grid.Set(x, y, newCell);
}

__global__
void calcNextStateFallingSand(Grid<int> grid, bool pipes = false) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = y * grid.Width() + x;

    if (grid.Get(x, y) == 0) {
        if (pipes && i % 100 == 1) grid.Set(x, y, 1);
        return;
    }

    if (grid.Get(x, y + 1, 1) == 0) {
        grid.Set(x, y + 1, 1);
        grid.Set(x, y, 0);
        return;
    }

    if (grid.Get(x + 1, y + 1, 1) == 0 && grid.Get(x - 1, y + 1) == 0) {
        grid.Set(x + (1 * (i % 2 == 1) ? -1 : 1), y + 1, 1);
        grid.Set(x, y, 0);
        return;
    }

    if (grid.Get(x + 1, y + 1, 1) == 0) {
        grid.Set(x + 1, y + 1, 1);
        grid.Set(x, y, 0);
        return;
    }

    if (grid.Get(x - 1, y + 1, 1) == 0) {
        grid.Set(x - 1, y + 1, 1);
        grid.Set(x, y, 0);
        return;
    }

    grid.Set(x, y, 1);
}

template<>
void Grid<int>::advanceState(Options options) {
    dim3 block_size(20, 20);
    dim3 grid_size(Width() / block_size.x, Height() / block_size.y);

    switch (options.automataType)
    {
    case AutomataType::GameOfLife: {
        calcNextStateGameOfLife << < grid_size, block_size >> > (*this);
    } break;
    case AutomataType::FallingSand: {
        calcNextStateFallingSand<<< grid_size, block_size >>>(*this, ::IO::IsButtonDown(2));
    } break;
    default:
        break;
    }

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}
