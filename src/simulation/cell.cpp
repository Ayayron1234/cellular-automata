#include "cell.h"
#include "chunk.h"

void Cell::update(Cell cell, Chunk& chunk, int x, int y) {
	// Return if cell was allready updated in this tick
	if (chunk.isCellUpdated(cell))
		return;

	switch (cell.type)
	{
	case Cell::Type::AIR:
		break;
	case Cell::Type::WATER:
		break;
	case Cell::Type::BEDROCK:
		break;
	case Cell::Type::STONE:
		break;
	case Cell::Type::DIRT:
		movePowder(cell, chunk, x, y);
		break;
	case Cell::Type::SAND:
		movePowder(cell, chunk, x, y);
		break;
	default:
		break;
	}
}

void Cell::movePowder(Cell cell, Chunk& chunk, int x, int y) {
	CellCoord destination(x, y - 1);

	if (chunk.getCell(destination).isLighter(cell)) {
		chunk.swapCells({ x, y }, destination);
		chunk.markCellAsUpdated(destination);
		return;
	}

	char displacement = chunk.evenTick() * 2 - 1;
	destination.x += displacement;
	if (chunk.getCell(destination).isLighter(cell)) {
		chunk.swapCells({ x, y }, destination);
		chunk.markCellAsUpdated(destination);
		return;
	}

	destination.x -= 2 * displacement;
	if (chunk.getCell(destination).isLighter(cell)) {
		chunk.swapCells({ x, y }, destination);
		chunk.markCellAsUpdated(destination);
		return;
	}
}

void Cell::moveLiquid(Cell cell, Chunk& chunk, int x, int y) {

}
