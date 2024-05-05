#pragma once
#include "Json.h"
#include "vec2.h"

// Structure representing a 2D camera with a position and zoom factor
struct Camera {
    vec2 position;
    Float zoom;

    void getMinWorldPos(int windowWidth, int windowHeight, Float& minX, Float& minY) const {
        vec2 min = screenToWorld(windowWidth, windowHeight, vec2(0, windowHeight));
        minX = min.x;
        minY = min.y;
    }

    void getMaxWorldPos(Float windowWidth, Float windowHeight, Float& maxX, Float& maxY) const {
        vec2 max = screenToWorld(windowWidth, windowHeight, vec2(windowWidth, 0));
        maxX = max.x;
        maxY = max.y;
    }

    vec2 screenToNds(Float windowWidth, Float windowHeight, vec2 pos) const {
        return {
            pos.x / (Float)windowWidth * 2.f - 1.f,
            //pos.y / (Float)windowHeight * 2.f - 1.f
            (-pos.y) / (Float)windowHeight * 2.f + 1.f
        };
    }

    vec2 screenToWorld(Float windowWidth, Float windowHeight, vec2 pos) const {
        Float aspectRatio = windowWidth / windowHeight;

        return screenToNds(windowWidth, windowHeight, pos) * vec2(aspectRatio, 1.f) / zoom - position;
    }

    vec2 worldToNds(Float windowWidth, Float windowHeight, vec2 pos) const {
        Float aspectRatio = windowWidth / windowHeight;

        return ((pos + position) * zoom) * vec2(1.f / aspectRatio, 1.f);
    }

    vec2 worldToScreen(Float windowWidth, Float windowHeight, vec2 pos) const {
        Float wph = windowWidth / windowHeight;

        return {
            (pos.x + position.x + (0.5 / zoom)) * zoom * windowWidth,
            (pos.y + position.y + (0.5 / zoom)) * zoom * windowHeight,
        };
    }

}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))
