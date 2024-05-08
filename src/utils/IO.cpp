#include "IO.h"
#include <fstream>

#include "../graphics_headers.h"

#define SDL SDL_Instance::instance()

const wchar_t* GetWC(const char* c) {
	const size_t cSize = strlen(c) + 1;
	wchar_t* wc = new wchar_t[cSize];
	mbstowcs(wc, c, cSize);

	return wc;
}

namespace IO {

struct SDL_Instance {
	SDL_GLContext glContext;

	long int tickCount = 0;

	SDL_Window* window = nullptr;
	int windowWidth, windowHeight;
	bool didResize = false;

	Uint8 mouseButtons = 0;
	Uint8 prevMouseButtons = 0;
	float mouseScrollAmount = 0;

	int keyboardStateArraySize = 0;
	const Uint8* keyboardState = nullptr;
	Uint8* prevKeyboardState = nullptr;

	std::wstring droppedFilePath;

	static SDL_Instance& instance();
};

void endFrame() {
	// Render ImGui draw data
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Swap buffers
	SDL_GL_SwapWindow(SDL.window);

	// Clear buffer
	glClearColor(104 / 256.f, 147 / 256.f, 156 / 256.f, 1);
	glClear(GL_COLOR_BUFFER_BIT);
}

void beginFrame() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);

	glViewport(0, 0, SDL.windowWidth, SDL.windowHeight);
}

void Render() {
	endFrame();
	beginFrame();
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
	SDL_DestroyWindow(SDL.window);
	SDL_Quit();
}

bool OpenWindow(int width, int height) {
	std::cout << "Opening window." << std::endl;

	SDL.windowWidth = width;
	SDL.windowHeight = height;

	// Init SDL and open a window
	SDL_Init(SDL_INIT_EVERYTHING);

	// Create window
	SDL.window = SDL_CreateWindow("Falling Sand Automata", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
								  SDL.windowWidth, SDL.windowHeight, SDL_WINDOW_RESIZABLE | SDL_RENDERER_ACCELERATED);

	// Use OpenGL 3.3
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	// Create opengl context
	SDL.glContext = SDL_GL_CreateContext(SDL.window);
	if (SDL.glContext == nullptr) {
		std::cerr << "OpenGL context could not be created! SDL Error: " << SDL_GetError() << std::endl;
		return false;
	}
	SDL_GL_MakeCurrent(SDL.window, SDL.glContext);

	// Initialize GLEW after creating OpenGL context
	glewExperimental = GL_TRUE;
	GLenum glewError = glewInit();
	if (glewError != GLEW_OK) {
		std::cerr << "Error initializing GLEW! " << glewGetErrorString(glewError) << std::endl;
		return false;
	}
	// Remove error caused by glewExperimental
	glGetError();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Use Vsync
	if (SDL_GL_SetSwapInterval(0) < 0) {
		std::cerr << "Warning: Unable to set VSync! SDL Error: " << SDL_GetError() << std::endl;
	}

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

	ImGui_ImplSDL2_InitForOpenGL(SDL.window, SDL.glContext);
	ImGui_ImplOpenGL3_Init("#version 330");


	beginFrame();

	return true;
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
	return { (2.f * x) / (Float)SDL.windowWidth - 1.f, ((- 2.f * y) / (Float)SDL.windowHeight + 1.f) };
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
	SDL_SetWindowSize(SDL.window, width, height);
}

inline SDL_Instance& SDL {
	static SDL_Instance c_instance;
	return c_instance;
}

}
