#include "utils/IO.h"
#include "utils.h"
#include "device_helpers.h"
#include "kernels.h"
#include "grid.h"
#include "cell.h"

auto& g_options = external_resource<"options.json", Json::wrap<Options>>::value;

bool g_doShowPropertiesWindow = true;
bool g_propertiesWindowHovered = false;

long long g_updateDuration;
long long g_renderDuration;

/**
 * Shows the properties window, allowing users to modify Mandelbrot set options.
 * Updates global variables based on user interactions.
 */
void ShowPropertiesWindow() {
    if (!g_doShowPropertiesWindow) return;

    // Initialize window state and set its size
    bool isWindowOpen = true;
    bool c_initedSize = false;
    ImGui::Begin("Properties", &isWindowOpen, (c_initedSize) ? ImGuiWindowFlags_NoResize : 0);
    ImGui::SetWindowSize({ 0 , 0 });
    c_initedSize = true;

    // Check if the properties window is hovered
    g_propertiesWindowHovered = ImGui::IsWindowHovered();

    ImGui::SeparatorText("Simulation Speed:");
    ImGui::Text("Enabled: "); ImGui::SameLine(); ImGui::Checkbox("##simulationEnableCheckBox", &g_options.simulationEnabled);
    ImGui::Text("Simulation step delay: "); ImGui::DragInt("##stateTransitionTickDelay", &g_options.stateTransitionTickDelay, sqrtf(g_options.stateTransitionTickDelay) / 3.f, 1, 250, "%d tick(s)");
    static int c_tickDelayBeforeDisable = 1;
    static bool c_disabled = false;
    if (!g_options.simulationEnabled) {
        if (!c_disabled)
            c_tickDelayBeforeDisable = g_options.stateTransitionTickDelay;
        c_disabled = true;
        g_options.stateTransitionTickDelay = 0;
    } 
    else if (c_disabled) {
        c_disabled = false;
        g_options.stateTransitionTickDelay = c_tickDelayBeforeDisable ? c_tickDelayBeforeDisable : 1;
    }

    // Display brush options
    ImGui::SeparatorText("Brush:");
    ImGui::Text("Size: "); ImGui::SameLine(); ImGui::DragInt("##brushSize", &g_options.brushSize, 1, 1, 64);

    // Display render and update durations
    ImGui::SeparatorText("Performance:");
    ImGui::Text("Update: "); ImGui::SameLine(); ImGui::Text("%f", (float)g_updateDuration / 1000.f); ImGui::SameLine(); ImGui::Text("ms");
    ImGui::Text("Render: "); ImGui::SameLine(); ImGui::Text("%f", (float)g_renderDuration / 1000.f); ImGui::SameLine(); ImGui::Text("ms");

    ImGui::End();
}

void UpdateCamera(vec2 mouseWorldPos, const Uint8* state, vec2 normalizedMousePos) {
    // Reset camera when space is pressed
    if (state[SDL_SCANCODE_SPACE]) {
        g_options.camera.position = { 0, 0 };
        g_options.camera.zoom = 1;
    }

    // Begin dragging if left mouse button is clicked
    static vec2 c_dragStart;
    static bool dragDisabled = false;
    if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
        dragDisabled = g_propertiesWindowHovered;
        c_dragStart = mouseWorldPos;
    }

    // Move camera while dragging
    if (IO::IsButtonDown(SDL_BUTTON_LEFT) && !dragDisabled) {
        g_options.camera.position = g_options.camera.position + (mouseWorldPos - c_dragStart);
        //c_dragStart = mouseWorldPos;
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

int g_simulationStepCount = 0;
void UpdateGrid(Grid<Cell>& grid) {
    bool evenTick = (g_simulationStepCount++) % 2;
    for (int _x = rand() % 3; _x < grid.Width() * 3; _x += 3)
    for (int _y = grid.Height() - 1; _y >= 0; --_y)
    //for (int _y = 0; _y < grid.Height() * 3; _y += 3)
    {
        int x = _x % grid.Width();
        //int y = _y % grid.Height();
        int y = _y;

        Cell cell = grid.Get(x, y);

        if (cell.type == 0 || cell.updatedOnEven == evenTick) {
            cell.updatedOnEven = evenTick;
            grid.Set(x, y, cell);
            continue;
        }
        cell.updatedOnEven = evenTick;

        Cell destination;
        if (cell.type == 1) {
            destination = grid.Get(x, y + 1, { evenTick, 1 });
            if (destination.type == 0 || destination.type == 2) {
                grid.Set(x, y + 1, { evenTick, cell.type });
                destination.updatedOnEven = evenTick;
                grid.Set(x, y, destination);
                continue;
            }

            char displacement = evenTick * 2 - 1;
            destination = grid.Get(x + displacement, y + 1, { evenTick, 1 });
            if (destination.type == 0 || destination.type == 2) {
                grid.Set(x + displacement, y + 1, { evenTick, cell.type });
                destination.updatedOnEven = evenTick;
                grid.Set(x, y, destination);
                continue;
            }

            destination = grid.Get(x - displacement, y + 1, { evenTick, 1 });
            if (destination.type == 0 || destination.type == 2) {
                grid.Set(x - displacement, y + 1, { evenTick, cell.type });
                destination.updatedOnEven = evenTick;
                grid.Set(x, y, destination);
                continue;
            }
        }
        else if (cell.type == 2) {
            if (grid.Get(x, y + 1, { evenTick, 1 }).type == 0) {
                grid.Set(x, y + 1, { evenTick, cell.type });
                grid.Set(x, y, { evenTick, 0 });
                continue;
            }

            //char displacement = evenTick * 2 - 1;
            char displacement = cell.waterDirection * 2 - 1;
            bool displacement2 = evenTick;
            if (grid.Get(x + displacement, y + displacement2, { evenTick, 1 }).type == 0) {
                grid.Set(x + displacement, y + displacement2, { evenTick, cell.type, cell.waterDirection });
                grid.Set(x, y, { evenTick, 0 });
                continue;
            }

            //if (grid.Get(x - displacement, y + displacement2, { evenTick, 1 }).type == 0) {
            //    grid.Set(x - displacement, y + displacement2, { evenTick, cell.type, cell.waterDirection });
            //    grid.Set(x, y, { evenTick, 0 });
            //    continue;
            //}

            if (grid.Get(x + displacement, y + !displacement2, { evenTick, 1 }).type == 0) {
                grid.Set(x + displacement, y + !displacement2, { evenTick, cell.type, cell.waterDirection });
                grid.Set(x, y, { evenTick, 0 });
                continue;
            }

            //if (grid.Get(x - displacement, y + !displacement2, { evenTick, 1 }).type == 0) {
            //    grid.Set(x - displacement, y + !displacement2, { evenTick, cell.type, cell.waterDirection });
            //    grid.Set(x, y, { evenTick, 0 });
            //    continue;
            //}

            grid.Set(x, y, { evenTick, cell.type, !cell.waterDirection });
        }
    }
}

int main() {
    std::cout << sizeof(Cell) << std::endl;

    IO::OpenWindow(g_options.windowWidth, g_options.windowHeight);

    Grid<Cell> grid(1024, 1024);
    grid.Load("grid.bin");

    GlobalBuffer<IO::RGB> outputBuffer;
    outputBuffer.Init(IO::GetWindowWidth() * IO::GetWindowHeight(), IO::GetOutputBuffer());

    auto rendererKernel = Kernel(drawGrid, &outputBuffer);

    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        // Handle events
        IO::HandleEvents();
        g_options.windowWidth = IO::GetWindowWidth();
        g_options.windowHeight = IO::GetWindowHeight();
        vec2 normalizedMousePos = IO::NormalizePixel((int)IO::GetMousePos().x, (int)IO::GetMousePos().y);
        vec2 mouseWorldPos = normalizedMousePos / 2 / g_options.camera.zoom - g_options.camera.position;

        // Resize buffer when window resized
        if (IO::Resized())
            outputBuffer.Init(IO::GetWindowWidth() * IO::GetWindowHeight(), IO::GetOutputBuffer(), false);

        // Draw GUI
        ShowPropertiesWindow();

        const Uint8* state = SDL_GetKeyboardState(nullptr);
        UpdateCamera(mouseWorldPos, state, normalizedMousePos);

        static short unsigned int c_valueToSet = 1;
        static short unsigned int c_valueBeforeClearStart = 1;
        if (IO::IsKeyDown(SDL_SCANCODE_1))
            c_valueToSet = 1;
        if (IO::IsKeyDown(SDL_SCANCODE_2))
            c_valueToSet = 2;

        int mouseWorldPosFlooredX = (int)floor(mouseWorldPos.x);
        int mouseWorldPosFlooredY = (int)floor(mouseWorldPos.y);
        if (IO::MouseClicked(SDL_BUTTON_RIGHT) && grid.Get(mouseWorldPosFlooredX, mouseWorldPosFlooredY).type != 0) {
            std::cout << grid.Get(mouseWorldPosFlooredX, mouseWorldPosFlooredY).type << std::endl;
            c_valueBeforeClearStart = c_valueToSet;
            c_valueToSet = 0;
        }
        if (IO::MouseReleased(SDL_BUTTON_RIGHT) && c_valueToSet == 0)
            c_valueToSet = c_valueBeforeClearStart;

        int& brushSize = g_options.brushSize;
        if (IO::IsButtonDown(SDL_BUTTON_RIGHT) && !g_propertiesWindowHovered) {
            int x = mouseWorldPosFlooredX - brushSize / 2;
            int y = mouseWorldPosFlooredY - brushSize / 2;
            for (int i = 0; i < brushSize * brushSize; ++i) {
                int dX = i % brushSize;
                int dY = i / brushSize;

                float distance = length(vec2(x + dX, y + dY) - vec2(x + brushSize / 2, y + brushSize / 2));
                if (distance < brushSize / 2)
                    grid.Set(x + dX, y + dY, { false, c_valueToSet });
            }
        }

        // Invoke CUDA function
        static long c_tickCount = 0;
        if (g_options.stateTransitionTickDelay != 0 && (IO::GetTicks()) % g_options.stateTransitionTickDelay == 0) {
            auto updateStart = std::chrono::high_resolution_clock::now();

            UpdateGrid(grid);
            grid.UpdateState();

            auto updateEnd = std::chrono::high_resolution_clock::now();
            g_updateDuration = std::chrono::duration_cast<std::chrono::microseconds>(updateEnd - updateStart).count();
        }
        else
            grid.UpdateState();

        // Render and draw to screen
        auto renderStart = std::chrono::high_resolution_clock::now();

        rendererKernel.Execute(grid, g_options);
        rendererKernel.SyncAllFromDevice(true);

        IO::Render();

        auto renderEnd = std::chrono::high_resolution_clock::now();
        g_renderDuration = std::chrono::duration_cast<std::chrono::microseconds>(renderEnd - renderStart).count();


        // Handle dropped files
        if (IO::FileDropped()) {
            HandleDroppedFile(IO::GetDroppedFilePath());
        }
    }

    grid.Save("grid.bin");

    IO::Quit();

    return 0;
}
