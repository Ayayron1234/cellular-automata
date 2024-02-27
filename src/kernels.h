#pragma once
#include "grid.h"
#include "utils.h"
#include "IO.h"

void gameOfLife(Grid<int> grid);

void fallingSand(Grid<int> grid, bool pipes = false);

void drawGrid(GlobalBuffer<IO::RGB> pixelBuffer, Grid<int> grid, Options options);
