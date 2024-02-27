#include "../kernels.h"   
#include <curand_kernel.h>
#include "../IO.h"

__global__
void setPixel(GlobalBuffer<IO::RGB> pixelBuffer, Options options, Grid<int> grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)col / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)row / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    int x = floor(x0), y = floor(y0);

    int val = grid.Get(x, y);
    IO::RGB color;
    if (val == 1)
        color = { (char)191, (char)174, (char)111 };
    else if (val == 2)
        color = { (char)75, (char)147, (char)189 };

    pixelBuffer.Write(i, color);
}

void drawGrid(GlobalBuffer<IO::RGB> pixelBuffer, Grid<int> grid, Options options) {
    dim3 block_size(32, 32);
    dim3 grid_size(options.windowWidth / block_size.x, options.windowHeight / block_size.y);

    setPixel << < grid_size, block_size >> > (pixelBuffer, options, grid);
}
