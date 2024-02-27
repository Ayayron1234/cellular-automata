#include "../kernels.h"
#include <curand_kernel.h>

__global__
void calcNextState(Grid<int> grid) {
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

void gameOfLife(Grid<int> grid) {
    dim3 block_size(32, 32);
    dim3 grid_size(grid.Width() / block_size.x, grid.Height() / block_size.y);

    grid.SyncFromHost();

    calcNextState << < grid_size, block_size >> > (grid);
}
