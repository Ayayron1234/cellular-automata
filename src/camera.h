#pragma once
#include "utils/Json.h"
#include "vec2.h"

// Structure representing a 2D camera with a position and zoom factor
struct Camera {
    vec2 position;
    Float zoom;
}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))
