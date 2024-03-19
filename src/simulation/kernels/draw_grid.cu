#include "../kernels.h"   
#include <curand_kernel.h>

//__device__ __host__
//IO::RGB getColor(Cell::Type type) {
//    if (type == Cell::Type::AIR) return IO::RGB(0, 0, 0);
//    if (type == Cell::Type::WATER) return IO::RGB(54, 124, 138);
//    if (type == Cell::Type::SAND) return IO::RGB(191, 174, 111);
//    if (type == Cell::Type::BEDROCK) return IO::RGB(66, 66, 66);
//    return IO::RGB(255, 255, 255);
//}

__device__
inline CellCoord cellCoordFromPixel(Options options, int x, int y) {
    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)x / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)y / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    return { floor(x0), floor(y0) };
}

__global__
void setPixel(GlobalBuffer<IO::RGB> pixelBuffer, World world, Options options, ColorPalette colorPalette) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    CellCoord coord = cellCoordFromPixel(options, col, row);

    Cell cell = world.getCell(coord);
    IO::RGB color;
    color = colorPalette.getColor(cell);

    if (options.showChunkBorders) {
        ChunkCoord chunkCoord = world.chunkCoordOf(coord);
        if (!world.getChunk(chunkCoord)->empty()) {
            bool isChunkEdge = world.chunkCoordOf(cellCoordFromPixel(options, col + 1, row)).x != chunkCoord.x
                            || world.chunkCoordOf(cellCoordFromPixel(options, col, row + 1)).y != chunkCoord.y
                            || world.chunkCoordOf(cellCoordFromPixel(options, col - 1, row + 1)).x != chunkCoord.x
                            || world.chunkCoordOf(cellCoordFromPixel(options, col, row - 1)).y != chunkCoord.y;

            if (isChunkEdge)
                color = IO::RGB::white();
        }
    }

    pixelBuffer[i] = color;
}

void drawWorld(GlobalBuffer<IO::RGB> pixelBuffer, World world, Options options, ColorPalette colorPalette) {
    dim3 block_size(32, 32);
    dim3 grid_size(options.windowWidth / block_size.x, options.windowHeight / block_size.y);

    setPixel << < grid_size, block_size >> > (pixelBuffer, world, options, colorPalette);
}
