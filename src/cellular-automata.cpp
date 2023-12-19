#include "fractals.h"

#include "utils/ExternalResource.h"
#include "IO.h"

#include <SDL2/SDL.h>
#undef main

auto& g_options = external_resource<"options.json", Json::wrap<Options>>::value;

bool g_doShowPropertiesWindow = true;
bool g_propertiesWindowHovered = false;

/**
 * Shows the properties window, allowing users to modify Mandelbrot set options.
 * Updates global variables based on user interactions.
 */
void ShowPropertiesWindow() {
    if (!g_doShowPropertiesWindow) return;

    // Initialize window state and set its size
    bool isWindowOpen = true;
    ImGui::Begin("Properties", &isWindowOpen, ImGuiWindowFlags_NoResize);
    ImGui::SetWindowSize({ 256 , 216 });

    // Check if the properties window is hovered
    g_propertiesWindowHovered = ImGui::IsWindowHovered();

    // Display radio buttons to choose between Mandelbrot and Julia sets
    ImGui::SeparatorText("Type:");
    ImGui::RadioButton("Mandelbrot", (int*)&g_options.type, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Julia", (int*)&g_options.type, 1);

    // Display and allow modification of the maximum iterations
    ImGui::SeparatorText("Max iterations:");
    ImGui::Text("Maximum iterations at zoom: 1.0");
    ImGui::DragInt("##maxIterations", &g_options.baseIterations, 1, 25, 400);

    // Display and allow modification of the iteration increase falloff
    ImGui::SeparatorText("Iteration increase falloff:");
    ImGui::DragFloat("##iterationIncreaseFalloff", &g_options.iterationIncreaseFallOff, 0.02f, 2.f, 20.f);

    ImGui::End();
}

/**
 * Handles a dropped file, attempting to load and apply options from a JSON file.
 *
 * @param path - The path to the dropped file in wide-string format.
 */
void HandleDroppedFile(const std::wstring& path) {
    // Check if the file has a valid JSON extension
    if (path.length() < 5 || path.substr(path.length() - 5).compare(L".json") != 0) {
        std::cerr << "Invalid format. " << std::endl;
        return;
    }

    // Attempt to open the file
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open()) {
        // Display an error message if the file opening fails
        size_t len = wcslen(path.c_str()) + 1;
        char* charPath = new char[len];
        wcstombs(charPath, path.c_str(), len);

        std::cerr << "Failed to open file: " << charPath << std::endl;
        delete[] charPath;

        return;
    }

    // Load options from the JSON file
    Options newOptions;
    Json newOptionsJson;
    try {
        ifs >> newOptionsJson;
        newOptions = newOptionsJson;
    }
    catch (...) {
        // Display an error message if loading options fails
        std::cerr << "Couldn't load options from file. " << std::endl;
        ifs.close();
        return;
    }
    ifs.close();

    // Apply the new options and resize the window accordingly
    g_options = Json::wrap<Options>(newOptions);
    IO::ResizeWindow(g_options.windowWidth, g_options.windowHeight);

    // Display a message indicating successful options loading
    size_t len = wcslen(path.c_str()) + 1;
    char* charPath = new char[len];
    wcstombs(charPath, path.c_str(), len);
    std::cout << "Loaded options from: " << charPath << std::endl;
    delete[] charPath;
}

int main() {
    IO::OpenWindow(g_options.windowWidth, g_options.windowHeight);

    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        // Handle events
        IO::HandleEvents();
        g_options.windowWidth = IO::GetWindowWidth();
        g_options.windowHeight = IO::GetWindowHeight();
        vec2 normalizedMousePos = IO::NormalizePixel(IO::GetMousePos().x, IO::GetMousePos().y);

        // Draw GUI
        ShowPropertiesWindow();

        // Reset camera when space is pressed
        const Uint8* state = SDL_GetKeyboardState(nullptr);
        if (state[SDL_SCANCODE_SPACE]) {
            g_options.SetProperty({ 0, 0 });
            g_options.camera.position = { 0, 0 };
            g_options.camera.zoom = 0.2;
        }

        // Begin dragging if left mouse button is clicked
        static vec2 z0Start;
        static bool dragDisabled = false;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            dragDisabled = g_propertiesWindowHovered;
            z0Start = g_options.GetProperty();
            dragStart = normalizedMousePos;
        }

        // Update fractal property during mouse drag (if dragging is allowed)
        if (IO::IsButtonDown(SDL_BUTTON_LEFT) && !dragDisabled) {
            g_options.SetProperty(z0Start + 0.25L * (normalizedMousePos - dragStart) / g_options.camera.zoom);
        }

        // Handle zooming based on mouse wheel movement
        static Float zoom = 1.L;
        static Float zoomDP = 1.02;
        static Float zoomDN = 1.035;
        if (IO::GetMouseWheel() > 0)
            zoom = zoomDP;
        else if (IO::GetMouseWheel() < 0)
            zoom = 1.L / zoomDN;

        // Update properties considering zoom level and smooth zoom decrease
        if (abs(zoom - 1.L) > 0.0001f)
            g_options.camera.position = g_options.camera.position - 0.5L * normalizedMousePos / g_options.camera.zoom + 0.5L * normalizedMousePos / (g_options.camera.zoom * zoom);
        g_options.camera.zoom *= zoom;
        zoom = 1.L + (zoom - 1.L) * 0.975L;

        // Invoke CUDA function for Mandelbrot set computation
        mandelbrotCuda(g_options, (int)((float)g_options.baseIterations * powl(g_options.camera.zoom, 1.0L / g_options.iterationIncreaseFallOff)));

        IO::Render();

        // Handle dropped files
        if (IO::FileDropped()) {
            HandleDroppedFile(IO::GetDroppedFilePath());
        }
    }

    IO::Quit();

    return 0;
}


