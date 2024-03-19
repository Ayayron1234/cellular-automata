#include "utils/IO.h"
#include "utils.h"
#include "simulation.h"
#include "device.h"

auto& g_options = external_resource<"data/options.json", Json::wrap<Options>>::value;

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

    // Show chunk borders
    ImGui::Text("Chunk borders:"); ImGui::SameLine(); ImGui::Checkbox("##showChunkBorders", &g_options.showChunkBorders);

    // Display brush options
    ImGui::SeparatorText("Brush:");
    ImGui::Text("Size: "); ImGui::SameLine(); ImGui::DragInt("##brushSize", &g_options.brushSize, 1, 1, 64);

    // Display render and update durations
    ImGui::SeparatorText("Performance:");
    ImGui::Text("Update: "); ImGui::SameLine(); ImGui::Text("%f", (float)g_updateDuration / 1000.f); ImGui::SameLine(); ImGui::Text("ms");
    ImGui::Text("Render: "); ImGui::SameLine(); ImGui::Text("%f", (float)g_renderDuration / 1000.f); ImGui::SameLine(); ImGui::Text("ms");

    // Display mouse position
    std::stringstream mousePosSStream;
    auto mousePos = g_options.camera.screenToWorld(IO::GetWindowWidth(), IO::GetWindowHeight(), IO::GetMousePos());
    mousePosSStream << "Mouse position: {" << mousePos.x << "," << mousePos.y << "}";
    ImGui::Text(mousePosSStream.str().c_str());

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
void UpdateChunk(Options options, Chunk& chunk) {
    using namespace SimValues;

    int x0 = chunk.getFirstCellCoord().x, y0 = chunk.getFirstCellCoord().y;

    bool evenTick = (g_simulationStepCount) % 2;
    char step = rand() % 2 == 0 ? 3 : 5;
    for (int _x = rand() % step; _x < CHUNK_SIZE * step; _x += step)
    for (int _y = 0; _y < CHUNK_SIZE; ++_y)
    //for (int _y = CHUNK_SIZE - 1; _y >= 0; --_y)
    {
        int x = x0 + (_x % CHUNK_SIZE);
        int y = y0 + _y;

        Cell cell = chunk.getCell(x, y);

        if (cell.type == Cell::Type::AIR || cell.updatedOnEvenTick == evenTick) {
            //cell.updatedOnEvenTick = evenTick;
            //chunk.setCell(x, y, cell);
            continue;
        }

        if (cell.type == Cell::Type::SAND) {
            CellCoord destination(x, y + 1);

            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }

            char displacement = evenTick * 2 - 1;
            destination.x += displacement;
            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }

            destination.x -= 2 * displacement;
            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }
        } 

        if (cell.type == Cell::Type::WATER) {
            CellCoord destination(x, y + 1);

            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }

            char displacement = evenTick * 2 - 1;
            destination.x += displacement;
            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }

            destination.x -= 2 * displacement;
            if (chunk.getCell(destination).isLighter(cell)) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }

            destination.y = y;
            int directionVec = (int)cell.water.direction * 2 - 1;
            destination.x = x + directionVec;

            for (destination.x = x + directionVec; chunk.getCell(destination).isLighter(cell) && abs(destination.x - x) < 7; destination.x += directionVec) { }
            destination.x -= directionVec;

            if (destination.x != x) {
                chunk.swapCells({ x, y }, destination);
                chunk.markCellAsUpdated(destination, evenTick);
                continue;
            }
            
            cell.updatedOnEvenTick = evenTick;
            cell.water.direction += 1;
            chunk.setCell({ x, y }, cell);
        }
    }
}

int main() {
    std::string worldPath = "data/worlds/test";
#ifdef _DEBUG
    worldPath = "data/worlds/dbgTest";
#endif
    if (worldPath.empty()) {
        std::cout << "World name: ";
        std::stringstream worldPathStream;
        worldPathStream << "data/worlds/";
        std::string worldName;
        std::cin >> worldName;
        worldPathStream << worldName;
        worldPath = worldPathStream.str();
    }

    IO::OpenWindow(g_options.windowWidth, g_options.windowHeight);

    World world;
    world.load(worldPath);

    auto rendererKernel = Kernel(drawWorld, new GlobalBuffer<IO::RGB>(IO::GetOutputBuffer(), IO::GetWindowWidth() * IO::GetWindowHeight()));

    Bitmap colorPaletteBitmap("data/color_palett.bmp");
    ColorPalette colorPalette(colorPaletteBitmap);

    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        // Handle events
        IO::HandleEvents();
        g_options.windowWidth = IO::GetWindowWidth();
        g_options.windowHeight = IO::GetWindowHeight();
        vec2 normalizedMousePos = IO::NormalizePixel((int)IO::GetMousePos().x, (int)IO::GetMousePos().y);
        vec2 mouseWorldPos = g_options.camera.screenToWorld(IO::GetWindowWidth(), IO::GetWindowHeight(), IO::GetMousePos());
        //vec2 mouseWorldPos = normalizedMousePos / 2 / g_options.camera.zoom - g_options.camera.position;

        const Uint8* state = SDL_GetKeyboardState(nullptr);
        UpdateCamera(mouseWorldPos, state, normalizedMousePos);

        static Cell::Type c_valueToSet = Cell::Type::SAND;
        static Cell::Type c_valueBeforeClearStart = Cell::Type::SAND;
        if (IO::IsKeyDown(SDL_SCANCODE_1))
            c_valueToSet = Cell::Type::SAND;
        if (IO::IsKeyDown(SDL_SCANCODE_2))
            c_valueToSet = Cell::Type::WATER;
        if (IO::IsKeyDown(SDL_SCANCODE_3))
            c_valueToSet = Cell::Type::BEDROCK;

        int mouseWorldPosFlooredX = (int)floor(mouseWorldPos.x);
        int mouseWorldPosFlooredY = (int)floor(mouseWorldPos.y);
        if (IO::MouseClicked(SDL_BUTTON_RIGHT) && world.getCell(mouseWorldPosFlooredX, mouseWorldPosFlooredY).type != Cell::Type::AIR) {
            std::cout << (int)world.getCell(mouseWorldPosFlooredX, mouseWorldPosFlooredY).type << std::endl;
            c_valueBeforeClearStart = c_valueToSet;
            c_valueToSet = Cell::Type::AIR;
        }
        if (IO::MouseReleased(SDL_BUTTON_RIGHT) && c_valueToSet == Cell::Type::AIR)
            c_valueToSet = c_valueBeforeClearStart;

        int& brushSize = g_options.brushSize;
        if (IO::IsButtonDown(SDL_BUTTON_RIGHT) && !g_propertiesWindowHovered) {
            int x = mouseWorldPosFlooredX - brushSize / 2;
            int y = mouseWorldPosFlooredY - brushSize / 2;
            for (int i = 0; i < brushSize * brushSize; ++i) {
                int dX = i % brushSize;
                int dY = i / brushSize;

                float distance = length(vec2(x + dX, y + dY) - vec2(x + brushSize / 2, y + brushSize / 2));
                if (distance < brushSize / 2) {
                    Cell newCell = Cell::create(Cell::Type(c_valueToSet));
                    world.setCell(x + dX, y + dY, newCell);
                }
            }
        }

        if (IO::IsKeyDown(SDL_SCANCODE_F11)) {
            IO::ToggleFullscreen();
        }

        bool shouldDraw = false;
        float framesPerSec = 75.f;
        {
            static auto prevDrawTime = std::chrono::high_resolution_clock::now();
            auto now = std::chrono::high_resolution_clock::now();

            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - prevDrawTime).count();
            if (elapsedTime > 1000 / framesPerSec) {
                shouldDraw = true;
                prevDrawTime = now;
            }
        }

        // Invoke CUDA function
        static long c_tickCount = 0;
        if (g_options.stateTransitionTickDelay != 0 && (IO::GetTicks()) % g_options.stateTransitionTickDelay == 0) {
            auto updateStart = std::chrono::high_resolution_clock::now();

            if (shouldDraw)
                world.updateAndDraw(g_options, UpdateChunk);
            else
                world.update(g_options, UpdateChunk);
            ++g_simulationStepCount;

            auto updateEnd = std::chrono::high_resolution_clock::now();
            g_updateDuration = std::chrono::duration_cast<std::chrono::microseconds>(updateEnd - updateStart).count();

        }
        else if (shouldDraw)
            world.draw(g_options);

        if (shouldDraw) {
            // Draw GUI
            ShowPropertiesWindow();

            // Render and draw to screen
            auto renderStart = std::chrono::high_resolution_clock::now();

            if (IO::Resized())
                rendererKernel.data().changeBuffer(IO::GetOutputBuffer(), IO::GetWindowWidth() * IO::GetWindowHeight());

            rendererKernel.execute(world, g_options, colorPalette);

            IO::Render();

            auto renderEnd = std::chrono::high_resolution_clock::now();
            g_renderDuration = std::chrono::duration_cast<std::chrono::microseconds>(renderEnd - renderStart).count();
        }

        // Handle dropped files
        if (IO::FileDropped()) {
            HandleDroppedFile(IO::GetDroppedFilePath());
        }
    }

    world.save(worldPath);

    IO::Quit();

    return 0;
}
