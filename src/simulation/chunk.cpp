#include "chunk.h"
#include "world.h"
#include "../utils/options.h"

inline Cell Chunk::getCell(coord_t x, coord_t y) const {
	if (isCoordInside(x, y)) {
		if (empty())
			return Cell();
		return m_cells[coordToIndex(x, y)];
	}
	return m_world->getCell(x, y);
}

void Chunk::setCell(coord_t x, coord_t y, Cell cell) {
	if (isCoordInside(x, y)) {
		// Break uniformity if needed
		if (empty())
			allocate();

		// Access the specified Cell position within the Chunk
		Cell& cellPos = m_cells[coordToIndex(x, y)];

		// Update cell and cell type counts
		m_nonAirCount += cellPos.type == Cell::Type::AIR;
		cellPos = cell;
		m_nonAirCount -= cellPos.type == Cell::Type::AIR;
	}
	else 
		// Delegate the task to the World if the coordinates are outside the Chunk
		m_world->setCell(x, y, cell);
}

void Chunk::swapCells(CellCoord a, CellCoord b) {
	if (!isCoordInside(a)) { m_world->swapCells(a, b); return; }
	if (!isCoordInside(b)) { m_world->swapCells(a, b); return; }

	Cell aCell = m_cells[coordToIndex(a)];
	m_cells[coordToIndex(a)] = m_cells[coordToIndex(b)];
	m_cells[coordToIndex(b)] = aCell;
}

void Chunk::process(Options options, SimulationUpdateFunction updateFunction, bool doDraw) {
	if (doDraw) {
		// Check whether the chunk will be drawn to the screen 
		if (m_world->isChunkDrawn(m_coord)) {
			commitToDevice();
			m_world->addChunkToDrawnChunks(this);
		}
		else
			m_deviceBuffer.freeDevice();
	}

	if (updateFunction != nullptr) {
		// Advance chunk simulation state
		updateFunction(options, *this);

		// Update lear chunk if it is empty
		if (m_nonAirCount == 0)
			clear();
	}
}
