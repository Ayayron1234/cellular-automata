#include "automata.h"

#include "utils/ExternalResource.h"
#include "IO.h"

#include <stdio.h>

__device__ Float maxf(Float a, Float b) {
    return (a > b) ? a : b;
}

__device__ Float minf(Float a, Float b) {
    return (a < b) ? a : b;
}

/**
 * CUDA kernel function: Computes the color of a pixel in the Mandelbrot or Julia set and stores it in the buffer.
 *
 * @param buffer - The output buffer storing RGB values of pixels.
 * @param options - The Mandelbrot set properties and camera configuration.
 * @param maxIterations - The maximum number of iterations for the fractal computation.
 */
__global__ void calcPixel(IO::RGB* buffer, Options options, const ConwayGrid grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)col / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)row / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    int x = floor(x0), y = floor(y0);

    char color = (grid.at_device(x, y) == 1) ? 0xff : 0x00;
        
    //color = grid.getBuffer()[0] == 1 ? 0xff : 0x01;
    //color = x + y;
    //color = grid.width();
    //color = x0 + y0;

    // Store the RGB color values in the buffer, clamped to the range [0, 255]
    buffer[i].r = minf(255.f, color);
    buffer[i].g = minf(255.f, color);
    buffer[i].b = minf(255.f, color);
}

template <typename T>
__global__ void calcNextState(Options options, const ConwayGrid grid) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = y * grid.width() + x;

    T cell = grid.at_device(x, y);
    T newCell = grid.outerValue;
    int sum_neighborhood = grid.at_device(x + 1, y) + grid.at_device(x + 1, y + 1) + grid.at_device(x, y + 1) + grid.at_device(x - 1, y + 1)
        + grid.at_device(x - 1, y) + grid.at_device(x - 1, y - 1) + grid.at_device(x, y - 1) + grid.at_device(x + 1, y - 1);
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

    grid.getBuffer()[i] = newCell;
}

cudaError_t conwayCuda(Options options, ConwayGrid* _grid, bool advanceState) {
    using cell_t = std::remove_pointer_t<decltype(_grid)>::value_t;

    IO::RGB* gpuBuffer = 0;
    cell_t* gridGpuBuffer = nullptr;
    cudaError_t cudaStatus;
    std::remove_pointer_t<decltype(_grid)> grid = *_grid;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&gridGpuBuffer, grid.width() * grid.height() * sizeof(cell_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }
    grid.setCudaBuffer(gridGpuBuffer);

    dim3 block_size(16, 16);

    dim3 grid_size(grid.width() / block_size.x, grid.height() / block_size.y);
    if (advanceState)
        calcNextState<cell_t> <<< grid_size, block_size >>> (options, grid);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    //// cudaDeviceSynchronize waits for the kernel to finish, and returns
    //// any errors encountered during the launch.
    //cudaStatus = cudaDeviceSynchronize();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //    goto Error;
    //}

    //// Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy((unsigned char*)grid->getBuffer(), gridGpuBuffer, grid->width() * grid->height() * sizeof(cell_t), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    grid_size = dim3(options.windowWidth / block_size.x, options.windowHeight / block_size.y);
    calcPixel <<< grid_size, block_size >>> (gpuBuffer, options, grid);

    grid.loadCudaBuffer();

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Cleanup;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy((unsigned char*)IO::GetOutputBuffer(), gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Cleanup;
    }

Cleanup:
    cudaFree(gpuBuffer);
    cudaFree(gridGpuBuffer);

    //std::cout << "done" << std::endl;

    return cudaStatus;
}
