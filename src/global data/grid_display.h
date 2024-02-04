#pragma once
#include "grid.h"
#include "../options.h"

class GridDisplay {
public:
	static void DrawGrid(const Grid<int>& grid, Options options);

};
