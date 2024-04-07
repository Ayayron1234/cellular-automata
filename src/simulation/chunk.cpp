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
		// Allocate chunk if empty
		if (empty())
			allocate();

		// Tell the chunk to update
		requestUpdate();

		// Access the specified Cell position within the Chunk
		Cell& cellPos = m_cells[coordToIndex(x, y)];

		// Update cell and cell type counts
		m_airCountMutex.lock();
		m_nonAirCount += cellPos.type == Cell::Type::AIR;
		cellPos = cell;
		m_nonAirCount -= cellPos.type == Cell::Type::AIR;
		m_airCountMutex.unlock();

		// Update neighbouring chunk if the cell is near chunk edge
		updateChunksNearCell({ x, y });
	}
	else 
		// Delegate the task to the World if the coordinates are outside the Chunk
		m_world->setCell(x, y, cell);
}

void Chunk::swapCells(CellCoord a, CellCoord b) {
	// If atleast one of the coords is outside, delegate the task to world.
	if (!isCoordInside(a)) { m_world->swapCells(a, b); return; }
	if (!isCoordInside(b)) { m_world->swapCells(a, b); return; }

	// Tell the chunk to update
	requestUpdate();
	
	// Swap cells
	Cell aCell = m_cells[coordToIndex(a)];
	m_cells[coordToIndex(a)] = m_cells[coordToIndex(b)];
	m_cells[coordToIndex(b)] = aCell;

	// Update neighbouring chunk if the cell is near chunk edge
	updateChunksNearCell(a);
	updateChunksNearCell(b);
}

void Chunk::process(Options options, SimulationUpdateFunction updateFunction, bool doDraw) {
	if (updateFunction != nullptr)
		checkForUpdates();

	if (doDraw) {
		// Check whether the chunk will be drawn to the screen 
		if (m_world->isChunkDrawn(m_coord)) {
			commitToDevice();
			m_world->drawChunk(this);
		}
		else
			m_deviceBuffer.freeDevice();
	}

	if (updated() && updateFunction != nullptr) {
		// Advance chunk simulation state
		updateFunction(options, *this);

		// Update lear chunk if it is empty
		if (m_nonAirCount == 0)
			clear();
	}
}

void Chunk::requestUpdate() {
	m_shouldUpdate = true;
	m_updatedOnEvenTick = m_world->isEvenTick();
}

void Chunk::updateChunksNearCell(CellCoord coord) {
	if (abs(coord.x) % CHUNK_SIZE == 0) m_world->requestChunkUpdate({ m_coord.x - 1, m_coord.y });
	if (abs(coord.y) % CHUNK_SIZE == 0) m_world->requestChunkUpdate({ m_coord.x, m_coord.y - 1 });
	if (abs(coord.x) % CHUNK_SIZE == CHUNK_SIZE - 1) m_world->requestChunkUpdate({ m_coord.x + 1, m_coord.y });
	if (abs(coord.y) % CHUNK_SIZE == CHUNK_SIZE - 1) m_world->requestChunkUpdate({ m_coord.x, m_coord.y + 1 });
}

void Chunk::checkForUpdates() {
	m_updated = m_shouldUpdate;
	if (m_world->isEvenTick() != m_updatedOnEvenTick)
		m_shouldUpdate = false;
}
