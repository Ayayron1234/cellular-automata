#include "IO.h"
#include <fstream>

#include <SDL2/SDL.h>
#undef main

#define SDL SDL_Instance::instance()

const wchar_t* GetWC(const char* c) {
	const size_t cSize = strlen(c) + 1;
	wchar_t* wc = new wchar_t[cSize];
	mbstowcs(wc, c, cSize);

	return wc;
}

namespace IO {

struct OutputBuffer {
	RGB* buffer = nullptr;
	SDL_Texture* texture;

	void resize(int width, int height);
};

struct SDL_Instance {
	SDL_Renderer* renderer = nullptr;

	long int tickCount = 0;

	SDL_Window* window = nullptr;
	int windowWidth, windowHeight;
	bool didResize = false;

	OutputBuffer output;

	Uint8 mouseButtons = 0;
	Uint8 prevMouseButtons = 0;
	float mouseScrollAmount = 0;

	int keyboardStateArraySize = 0;
	const Uint8* keyboardState = nullptr;
	Uint8* prevKeyboardState = nullptr;

	std::wstring droppedFilePath;

	static SDL_Instance& instance();
};

void Render() {
	//// Update SDL texture with the output buffer data and 
	//// render the texture to the entire window
	//SDL_UpdateTexture(SDL.output.texture, nullptr, SDL.output.buffer, sizeof(IO::RGB) * SDL.windowWidth);
	//SDL_RenderCopy(SDL.renderer, SDL.output.texture, nullptr, nullptr);

	// Render ImGui draw data
	ImGui::Render();
	ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());

	// Present the rendered frame
	SDL_RenderPresent(SDL.renderer);

	// Start a new frame for ImGui rendering
	ImGui_ImplSDLRenderer2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);

	// Clear the renderer
	SDL_SetRenderDrawColor(SDL.renderer, 0, 0, 0, 255);
	SDL_RenderClear(SDL.renderer);
}

SDL_Renderer* GetRenderer() {
	return SDL.renderer;
}

void OutputBuffer::resize(int width, int height) {
	delete[] buffer;
	SDL_DestroyTexture(texture);

	buffer = new RGB[width * height];
	texture = SDL_CreateTexture(SDL.renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, width, height);
}

void IO::HandleEvents() {
	++SDL.tickCount;

	// Store previous mouse button state, reset scroll amount, and clear dropped file path
	SDL.prevMouseButtons = SDL.mouseButtons;
	SDL.mouseScrollAmount = 0;
	SDL.droppedFilePath.clear();

	// Check for window resize and update output buffer accordingly
	int newWidth, newHeight;
	SDL_GetWindowSize(SDL.window, &newWidth, &newHeight);
	SDL.didResize = (newWidth != SDL.windowWidth || newHeight != SDL.windowHeight);
	if (SDL.didResize) {
		SDL.output.resize(newWidth, newHeight);

		SDL.windowWidth = newWidth;
		SDL.windowHeight = newHeight;
	}

	// Update Keyboard state
	memcpy(SDL.prevKeyboardState, SDL.keyboardState, SDL.keyboardStateArraySize);

	// Process SDL events
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		ImGui_ImplSDL2_ProcessEvent(&event);

		switch (event.type)
		{
		case SDL_MOUSEBUTTONDOWN: {
			SDL.mouseButtons |= (1u << event.button.button);
		} break;
		case SDL_MOUSEBUTTONUP: {
			SDL.mouseButtons &= ~(1u << event.button.button);
		} break;
		case SDL_MOUSEWHEEL:
			SDL.mouseScrollAmount = event.wheel.preciseY;
			break;
		case (SDL_DROPFILE): {
			const wchar_t* path = GetWC(event.drop.file);
			SDL.droppedFilePath = path;
			delete[] path;
			break;
		}
		default:
			break;
		}
	}
}

void Quit() {
	SDL_DestroyRenderer(SDL.renderer);
	SDL_DestroyWindow(SDL.window);
	SDL_Quit();
}

void OpenWindow(int width, int height) {
	SDL.windowWidth = width;
	SDL.windowHeight = height;

	// Init SDL and open a window
	SDL_Init(SDL_INIT_EVERYTHING);

	SDL.window = SDL_CreateWindow("GPGPU", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SDL.windowWidth, SDL.windowHeight, SDL_WINDOW_RESIZABLE | SDL_RENDERER_ACCELERATED);
	SDL.renderer = SDL_CreateRenderer(SDL.window, -1, SDL_RENDERER_ACCELERATED);

	// Init keyboard state
	SDL.keyboardState = SDL_GetKeyboardState(&SDL.keyboardStateArraySize);
	SDL.prevKeyboardState = new Uint8[SDL.keyboardStateArraySize];

	// Init ImGUI
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui::StyleColorsClassic();

	ImGui_ImplSDL2_InitForSDLRenderer(SDL.window, SDL.renderer);
	ImGui_ImplSDLRenderer2_Init(SDL.renderer);

	// Init output buffer
	SDL.output.texture = SDL_CreateTexture(SDL.renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, SDL.windowWidth, SDL.windowHeight);
	SDL.output.buffer = new RGB[width * height];

	// Start first frame
	ImGui_ImplSDLRenderer2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);
}

bool Resized() {
	return SDL.didResize;
}

void SetWindowMode(WindowMode mode) {
	auto flags = SDL_GetWindowFlags(SDL.window);

	switch (mode)
	{
	case IO::WindowMode::Windowed:
		SDL_SetWindowFullscreen(SDL.window, ~SDL_WINDOW_FULLSCREEN & flags);
		break;
	case IO::WindowMode::FullscreenWindowed:
		SDL_SetWindowFullscreen(SDL.window, SDL_WINDOW_FULLSCREEN_DESKTOP | flags);
		break;
	case IO::WindowMode::Fullscreen:
		SDL_SetWindowFullscreen(SDL.window, SDL_WINDOW_FULLSCREEN | flags);
		break;
	default:
		break;
	}
}

WindowMode GetWindowMode() {
	auto flags = SDL_GetWindowFlags(SDL.window);

	if ((flags & SDL_WINDOW_FULLSCREEN) == 0u)
		return IO::WindowMode::Windowed;

	if ((flags & 0x00001000) == 0u)
		return IO::WindowMode::FullscreenWindowed;

	return IO::WindowMode::Fullscreen;
}

void ToggleFullscreen() {
	static WindowMode c_modeBeforeWindowed = WindowMode::FullscreenWindowed;

	if (GetWindowMode() == WindowMode::Windowed)
		SetWindowMode(c_modeBeforeWindowed);

	c_modeBeforeWindowed = GetWindowMode();
	SetWindowMode(WindowMode::Windowed);
}

bool IsButtonDown(uint8_t button) {
	button = (1u << button);
	return (SDL.mouseButtons & button) == button;
}

bool MouseClicked(uint8_t button) {
	button = (1u << button);
	return ((SDL.mouseButtons & button) == button) && !((SDL.prevMouseButtons & button) == button);
}

bool MouseReleased(uint8_t button) {
	button = (1u << button);
	return !((SDL.mouseButtons & button) == button) && ((SDL.prevMouseButtons & button) == button);
}

bool IsKeyDown(uint8_t key) {
	return SDL.keyboardState[key];
}

bool KeyPressed(uint8_t key) {
	return SDL.keyboardState[key] && !SDL.prevKeyboardState[key];
}

bool KeyReleased(uint8_t key) {
	return !SDL.keyboardState[key] && SDL.prevKeyboardState[key];
}

vec2 NormalizePixel(int x, int y) {
	return { (2.f * x) / (Float)SDL.windowWidth - 1.f, ((2.f * y) / (Float)SDL.windowHeight - 1.f) };
}

vec2 NormalizePixel(vec2 pos) {
	return { (2.f * pos.x) / (Float)SDL.windowWidth - 1.f, ((2.f * pos.y) / (Float)SDL.windowHeight - 1.f) };
}

vec2 GetMousePos() {
	int x, y;
	SDL_GetMouseState(&x, &y);
	return vec2(x, y);
}

float GetMouseWheel() {
	return SDL.mouseScrollAmount;
}

long int GetTicks() {
	return SDL.tickCount;
}

RGB* GetOutputBuffer() {
	return SDL.output.buffer;
}

int GetWindowWidth() {
	return SDL.windowWidth;
}

int GetWindowHeight() {
	return SDL.windowHeight;
}

bool FileDropped() {
	return SDL.droppedFilePath.length() > 0;
}

const std::wstring& GetDroppedFilePath() {
	return SDL.droppedFilePath;
}

void ResizeWindow(int width, int height) {
	SDL.output.resize(width, height);
	SDL_SetWindowSize(SDL.window, width, height);
}

void DrawRect(int x, int y, int w, int h, RGB color) {
	SDL_SetRenderDrawColor(SDL.renderer, color.r, color.g, color.b, color.a);
	SDL_Rect rect{ x, y, w, h };
	SDL_RenderDrawRect(SDL.renderer, &rect);
}

inline SDL_Instance& SDL {
	static SDL_Instance c_instance;
	return c_instance;
}

}
