#pragma once
#include "vec2.h"
#include "camera.h"
#include "ExternalResource.h"

JSON_C(vec2, JSON_M(x), JSON_M(y))

enum class AutomataType { GameOfLife = 0x00, FallingSand, _COUNT };

// Structure representing options for rendering fractals, including window dimensions,
// camera settings, fractal type, and iteration parameters
struct Options {
    int windowWidth = 800;                  // Width of the rendering window
    int windowHeight = 800;                 // Height of the rendering window
    Camera camera{ vec2(), 2 };             // Camera configuration with default values
    AutomataType automataType;

    int stateTransitionTickDelay = 50;
}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera), JSON_M(stateTransitionTickDelay), JSON_M(automataType))
