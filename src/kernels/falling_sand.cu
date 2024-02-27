#include "../kernels.h"
#include <curand_kernel.h>

__global__
void calcNextState(Grid<int> grid, bool pipes) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = y * grid.Width() + x;

    int type = grid.Get(x, y);
    if (type == 0) {
        if (pipes && i % 100 == 1) grid.Set(x, y, 1);
        return;
    }

    if (type == 1) {
        int nextPos = grid.Get(x, y + 1, 1);
        if (nextPos == 0 || nextPos == 2) {
            grid.Set(x, y + 1, 1);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x - 1, y + 1, 1);
        if (nextPos == 0 || nextPos == 2) {
            grid.Set(x - 1, y + 1, 1);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x + 1, y + 1, 1);
        if (nextPos == 0 || nextPos == 2) {
            grid.Set(x + 1, y + 1, 1);
            grid.Set(x, y, nextPos);
            return;
        }

        grid.Set(x, y, 1);
        return;
    }

    if (type == 2) {
        int nextPos = grid.Get(x, y + 1, 1);
        if (nextPos == 0) {
            grid.Set(x, y + 1, 2);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x - 1, y + 1, 1);
        if (nextPos == 0) {
            grid.Set(x - 1, y + 1, 2);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x + 1, y + 1, 1);
        if (nextPos == 0) {
            grid.Set(x + 1, y + 1, 2);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x - 1, y, 1);
        if (nextPos == 0) {
            grid.Set(x - 1, y, 2);
            grid.Set(x, y, nextPos);
            return;
        }

        nextPos = grid.Get(x + 1, y, 1);
        if (nextPos == 0) {
            grid.Set(x + 1, y, 2);
            grid.Set(x, y, nextPos);
            return;
        }

        grid.Set(x, y, 2);
        return;
    }
}

void fallingSand(Grid<int> grid, bool pipes) {
    dim3 block_size(32, 32);
    dim3 grid_size(grid.Width() / block_size.x, grid.Height() / block_size.y);

    grid.SyncFromHost();

    calcNextState << < grid_size, block_size >> > (grid, pipes);
}
