#pragma once
#include "Json.h"
#include "vec2.h"

// Structure representing a 2D camera with a position and zoom factor
struct Camera {
    vec2 position;
    Float zoom;

    void getMinWorldPos(int windowWidth, int windowHeight, Float& minX, Float& minY) const {
        Float wph = (Float)windowWidth / (Float)windowHeight;

        minX = -0.5 / zoom - position.x;
        minY = (-0.5 / zoom - position.y) / wph;
    }

    void getMaxWorldPos(Float windowWidth, Float windowHeight, Float& minX, Float& minY) const {
        Float wph = windowWidth / windowHeight;

        minX = (Float)1.f / zoom - 0.5 / zoom - position.x;
        minY = ((Float)1.f / zoom - 0.5 / zoom - position.y) / wph;
    }

    vec2 screenToWorld(Float windowWidth, Float windowHeight, vec2 pos) {
        Float wph = windowWidth / windowHeight;

        return { 
            (pos.x / windowWidth) / zoom - (0.5 / zoom) - position.x,
            ((pos.y / windowHeight) / zoom - (0.5 / zoom) - position.y) / wph
        };
    }

}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))
