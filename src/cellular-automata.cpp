#include "automata.h"

#include "utils/ExternalResource.h"
#include "IO.h"

#include <SDL2/SDL.h>
#undef main

auto& g_options = external_resource<"options.json", Json::wrap<Options>>::value;

bool g_doShowPropertiesWindow = true;
bool g_propertiesWindowHovered = false;

ConwayGrid g_grid{ 1000, 1000 };

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

    ImGui::SeparatorText("State Transition Tick Delay:");
    ImGui::DragInt("##stateTransitionTickDelay", &g_options.stateTransitionTickDelay, 1, 0, 100);

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
        vec2 normalizedMousePos = IO::NormalizePixel((int)IO::GetMousePos().x, (int)IO::GetMousePos().y);
        vec2 mouseWorldPos = normalizedMousePos / 2 / g_options.camera.zoom - g_options.camera.position;

        // Draw GUI
        ShowPropertiesWindow();

        // Reset camera when space is pressed
        const Uint8* state = SDL_GetKeyboardState(nullptr);
        if (state[SDL_SCANCODE_SPACE]) {
            g_options.camera.position = { 0, 0 };
            g_options.camera.zoom = 1;
        }

        // Begin dragging if left mouse button is clicked
        static vec2 dragStart;
        static bool dragDisabled = false;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            dragDisabled = g_propertiesWindowHovered;
            dragStart = mouseWorldPos;
        }

        // Update fractal property during mouse drag (if dragging is allowed)
        if (IO::IsButtonDown(SDL_BUTTON_LEFT) && !dragDisabled) {
            g_options.camera.position = (mouseWorldPos - dragStart);
        }

        if (IO::IsButtonDown(SDL_BUTTON_RIGHT) && !g_propertiesWindowHovered) {
            g_grid.set(floor(mouseWorldPos.x), floor(mouseWorldPos.y), 1);
        }

        // Handle zooming based on mouse wheel movement
        static Float zoom = 1.L;
        static Float zoomDP = 1.055;
        static Float zoomDN = 1.055;
        if (IO::GetMouseWheel() > 0)
            zoom = zoomDP;
        else if (IO::GetMouseWheel() < 0)
            zoom = 1.L / zoomDN;

        // Update properties considering zoom level and smooth zoom decrease
        if (abs(zoom - 1.L) > 0.0001f)
            g_options.camera.position = g_options.camera.position - 0.5L * normalizedMousePos / g_options.camera.zoom + 0.5L * normalizedMousePos / (g_options.camera.zoom * zoom);
        g_options.camera.zoom *= zoom;
        zoom = 1.L + (zoom - 1.L) * 0.75L;

        // Invoke CUDA function for Mandelbrot set computation
        static long c_tickCount = 0;
        conwayCuda(g_options, &g_grid, g_options.stateTransitionTickDelay == 0 ? false : ((c_tickCount++) % g_options.stateTransitionTickDelay == 0));

        IO::Render();

        // Handle dropped files
        if (IO::FileDropped()) {
            HandleDroppedFile(IO::GetDroppedFilePath());
        }
    }

    IO::Quit();

    return 0;
}
