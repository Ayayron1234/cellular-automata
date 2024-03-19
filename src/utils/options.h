#pragma once
#include "vec2.h"
#include "camera.h"
#include "ExternalResource.h"

JSON_C(vec2, JSON_M(x), JSON_M(y))

// Structure representing options for rendering fractals, including window dimensions,
// camera settings, fractal type, and iteration parameters
struct Options {
    int windowWidth = 800;                  // Width of the rendering window
    int windowHeight = 800;                 // Height of the rendering window
    Camera camera{ vec2(), 0.01 };             // Camera configuration with default values

    bool simulationEnabled = true;
    int stateTransitionTickDelay = 1;

    int brushSize = 12;
    bool showChunkBorders = false;
}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera), JSON_M(stateTransitionTickDelay), JSON_M(brushSize), JSON_M(simulationEnabled))
