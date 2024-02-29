#pragma once
#include "grid.h"
#include "utils.h"
#include "utils/IO.h"
#include "cell.h"

void drawGrid(GlobalBuffer<IO::RGB> pixelBuffer, Grid<Cell> grid, Options options);
