#pragma once
#include "vec2.h"

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_sdl2.h"
#include "imgui/backends/imgui_impl_sdlrenderer2.h"
#include "imgui/imgui_stdlib.h"

#include <iostream>

namespace IO {

struct RGB {
	char r = 0;
	char g = 0;
	char b = 0;
	char a = 0;

	__host__ __device__
	RGB(int _r = 0, int _g = 0, int _b = 0, int _a = 1) : r(_r), g(_g), b(_b), a(_a) { }
}; 

void Render();

/**
 * \brief 
 * Handles various SDL events, updating the SDL_Instance state accordingly.
 * Also, processes ImGui events and resizes the output buffer if the window size changes.
 */
void HandleEvents();

/**
 * Quits and cleans up resources related to the application window.
 */
void Quit();

void OpenWindow(int width, int height);

bool Resized();

bool IsButtonDown(uint8_t button);

bool MouseClicked(uint8_t button);

bool MouseReleased(uint8_t button);

bool IsKeyDown(uint8_t key);

bool KeyPressed(uint8_t key);

bool KeyReleased(uint8_t key);

/**
 * Normalizes pixel coordinates to the range [0, 1].
 *
 * @param x - The x-coordinate of the pixel.
 * @param y - The y-coordinate of the pixel.
 * @return A vec2 containing normalized coordinates.
 */
vec2 NormalizePixel(int x, int y);

vec2 GetMousePos();

float GetMouseWheel();

long int GetTicks();

/**
 * Gets the output buffer for rendering graphics.
 *
 * @return A pointer to the RGB buffer.
 */
RGB* GetOutputBuffer();

int GetWindowWidth();

int GetWindowHeight();

/**
 * Checks if a file has been dropped onto the application window.
 *
 * @return True if a file has been dropped, false otherwise.
 */
bool FileDropped();

const std::wstring& GetDroppedFilePath();

void ResizeWindow(int width, int height);

} // namespace IO
