#pragma once
#include "vec2.h"

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_sdl2.h"
#include "imgui/backends/imgui_impl_sdlrenderer2.h"
#include "imgui/imgui_stdlib.h"

#include <iostream>

namespace IO {

struct RGB {
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;
	unsigned char a = 0;

	__host__ __device__
	RGB(int _r = 0, int _g = 0, int _b = 0, int _a = 1) : r(_r), g(_g), b(_b), a(_a) { }

	__host__ __device__
	static RGB red() { return RGB(255, 0, 0); }
	__host__ __device__
	static RGB black() { return RGB(0, 0, 0); }
	__host__ __device__
	static RGB white() { return RGB(255, 255, 255); }

	__host__ __device__
	RGB operator*(float scale) {
		return RGB(r * scale, g * scale, b * scale, a);
	}

	__host__ __device__
	RGB operator+(RGB color) {
		return RGB(r + color.r, g + color.g, b + color.b, a);
	}
}; 

enum class WindowMode {
	Windowed = 0x00, FullscreenWindowed, Fullscreen 
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

void SetWindowMode(WindowMode mode);

WindowMode GetWindowMode();

void ToggleFullscreen();

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
vec2 NormalizePixel(vec2 pos);

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

void DrawRect(int x, int y, int w, int h, RGB color);

} // namespace IO

inline std::ostream& operator<<(std::ostream& os, IO::RGB rgb) {
	os << "rgba(" << (unsigned)rgb.r << "," << (unsigned)rgb.g << "," << (unsigned)rgb.b << "," << (unsigned)rgb.a << ")";
	return os;
}
