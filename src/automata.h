#pragma once
#include "vec2.h"
#include "utils/Json.h"
#include "grid.h"

JSON_C(vec2, JSON_M(x), JSON_M(y))

// Structure representing a 2D camera with a position and zoom factor
struct Camera {
    vec2 position;
    Float zoom;
}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))

// Structure representing options for rendering fractals, including window dimensions,
// camera settings, fractal type, and iteration parameters
struct Options {
    int windowWidth = 800;                  // Width of the rendering window
    int windowHeight = 800;                 // Height of the rendering window
    Camera camera{ vec2(), 2 };             // Camera configuration with default values

    int stateTransitionTickDelay = 50;
}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera), JSON_M(stateTransitionTickDelay))

/**
 * Calculates the Mandelbrot set on the GPU using CUDA.
 *
 * @param options - The options specifying rendering parameters.
 * @param maxIterations - The maximum number of iterations for fractal computation.
 *
 * @return cudaError_t - Returns cudaSuccess on successful execution, or an error code otherwise.
 */
cudaError_t conwayCuda(Options options, ConwayGrid* grid, bool advanceState = false);
