#pragma once
#include "grid_display.h"
#include "../IO.h"

__global__
void setPixel(IO::RGB* pixelBuffer, Options options, Grid<int> grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)col / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)row / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    int x = floor(x0), y = floor(y0);

    char color = grid.Get(x, y) * 255;

    //// Store the RGB color values in the buffer, clamped to the range [0, 255]
    pixelBuffer[i].r = color;
    pixelBuffer[i].g = color;
    pixelBuffer[i].b = color;
}

// TODO: create class to handle most functionality of this method
void GridDisplay::DrawGrid(const Grid<int>& grid, Options options) {
    IO::RGB* devicePixelBuff = nullptr;
    cudaError_t cudaStatus;
    
    cudaStatus = cudaMalloc((void**)&devicePixelBuff, options.windowWidth * options.windowHeight * sizeof(IO::RGB));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(1);
    }

    dim3 block_size(20, 20);
    dim3 grid_size(options.windowWidth / block_size.x, options.windowHeight / block_size.y);

    setPixel<<< grid_size, block_size >>>(devicePixelBuff, options, grid);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy((unsigned char*)IO::GetOutputBuffer(), devicePixelBuff, options.windowWidth * options.windowHeight * sizeof(IO::RGB), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Cleanup;
    }

Cleanup:
    cudaFree(devicePixelBuff);

}
