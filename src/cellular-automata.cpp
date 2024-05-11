#include "utils/IO.h"
#include "utils.h"
#include "simulation.h"
#include "application.h"
#include "device.h"
#include <array>
#include <string>

/*
TODO: 
    BUG FIXES: 
        - multithreading: sometimes freezes in release mode
        - multithreading: always freezes in debug mode with more than one chunks
        - multithreading: nonAirCount is not always correct
        - fix camera jump when resizing window

    OPTIMIZATION: 
        - track whether chunks need to be updated
        - track whether chunk data changed (if not don't reupload to device)
        - dirty rectangle

    SIMULATION:
        - more cell types
        - cell velocity
        - textured cell insertion

    OTHER:
        - cleanup this file

*/

/*
2024.05.06 (8h): glsl cell shade error fix, Eraser, SolidBrush, RandomizerBrush
2024.05.07 (8h): Texture, TexturedBrush, imgui brush widget, Cell::update(), bitmap hot reload (for color palette and textures), isFreefalling
2024.05.08 (8h): textured brush imgui widget, PerformanceMonitor, GUI, main file cleanup
*/

void UpdateChunk(Options options, Chunk& chunk) {
    using namespace SimValues;

    int x0 = chunk.getFirstCellCoord().x, y0 = chunk.getFirstCellCoord().y;

    bool evenTick = chunk.evenTick();
    char step = rand() % 2 == 0 ? 3 : 5;
    for (int _x = rand() % step; _x < CHUNK_SIZE * step; _x += step)
    //for (int _y = 0; _y < CHUNK_SIZE; ++_y)
    for (int _y = CHUNK_SIZE - 1; _y >= 0; --_y)
    {
        int x = x0 + (_x % CHUNK_SIZE);
        int y = y0 + _y;

        Cell& cell = chunk.cell({ x, y });

        Cell::update(cell, chunk, x, y);
    }
}

int main() {
    // Init GUI
    GUI gui;
    gui.getWorldPath();
    gui.loadOptions("data/options.json");
    gui.createWindow();

    // Load world
    World world;
    LOG_TASK("World Load", {
        world.load(gui.getWorldPath());
        world.setPalette(ColorPalette::loadFromFile("data/color_palett.bmp"));
    });

    // Main loop
    while (!SDL_QuitRequested()) {
        BEGIN_TASK("Other");

        // Handle user input
        gui.handleInputs(world);

        // Decide whether to update world
        bool shouldUpdate = true;
        if (!gui.options().simulationEnabled)
            shouldUpdate = false;
        if (PERF_MONITOR.duration("Update") / 1000.f < gui.options().updateWaitTimeMs)
            shouldUpdate = false;

        // Update world
        if (shouldUpdate) TASK("Update",
            world.update(gui.options(), UpdateChunk); );

        // Render and draw to screen
        TASK("Render", {
            world.draw(gui.options());
            gui.render();
        });

        END_TASK("Other", { "Render", "Update" });
    }

    LOG_TASK("Save world",
        world.save(gui.getWorldPath()); );

    return 0;
}
