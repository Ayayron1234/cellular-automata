#include "chunk.h"
#include "world.h"
#include "chunk_view.h"
#include "../utils/options.h"

#ifdef _DEBUG 
#define DBG_BREAK_PROCESS if (m_dbgBreakOnProcess) { __debugbreak(); m_dbgBreakOnProcess = false; }
#else
#define DBG_BREAK_PROCESS
#endif // _DEBUG

inline Cell Chunk::getCell(const CellCoord& coord) const {
	if (isCoordInside(coord)) {
		if (empty())
			return Cell();
		return m_cells[coordToIndex(coord)];
	}
	return m_world->getCell(coord);
}

void Chunk::setCell(const CellCoord& coord, Cell cell) {
	if (isCoordInside(coord)) {
		// Access the specified Cell position within the Chunk
		Cell& cellPos = m_cells[coordToIndex(coord)];

		// Update cell and cell type counts
		Cell::Type prevType = cellPos.type;
		Cell::Type currentType = cell.type;
		cellPos = cell;
		m_state->cellSet(prevType, currentType);

		// Update neighbouring chunk if the cell is near chunk edge
		updateChunksNearCell(coord);
	}
	else 
		// Delegate the task to the World if the coordinates are outside the Chunk
		m_world->setCell(coord, cell);
}

Chunk* Chunk::deserialise(std::ifstream& is, World* world) {
	// Read chunk size
	decltype(m_size) size;
	is.read((char*)&size, sizeof(m_size));
	if (size != CHUNK_SIZE)
		throw;

	// Read coordinates
	ChunkCoord coord;
	is.read((char*)&coord.x, sizeof(coord.x));
	is.read((char*)&coord.y, sizeof(coord.y));

	// Create and allocate chunk
	Chunk* chunk = new Chunk(world, coord);

	// Read nonAir cell count
	chunk->m_state->deserialise(is);

	// Read cells
	is.read((char*)chunk->m_cells, CHUNK_SIZE * CHUNK_SIZE * sizeof(Cell));

	// Upate view position
	chunk->m_view->updatePosition();

	return chunk;
}

void Chunk::swapCells(const CellCoord& a, const CellCoord& b) {
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

void Chunk::process(Options options, SimulationUpdateFunction updateFunction) {
	DBG_BREAK_PROCESS

	if (m_state->shouldUpdate()) {
		// Advance chunk simulation state
		updateFunction(options, *this);
	}

	m_state->nextState();
}

void Chunk::draw(Options options) {
	m_view->draw(options);
	m_state->drawn();
}

bool Chunk::isCellUpdated(Cell cell) const {
	return m_world->isCellUpdated(cell);
}

bool Chunk::evenTick() const {
	return m_world->evenTick();
}

void Chunk::updateChunksNearCell(const CellCoord& coord) {
	CellCoord firstCell = getFirstCellCoord();

	if ((coord.x - firstCell.x) == 0)				m_world->requestChunkUpdate({ m_coord.x - 1, m_coord.y });
	if ((coord.y - firstCell.y) == 0)				m_world->requestChunkUpdate({ m_coord.x, m_coord.y - 1 });
	if ((coord.x - firstCell.x) == CHUNK_SIZE - 1)  m_world->requestChunkUpdate({ m_coord.x + 1, m_coord.y });
	if ((coord.y - firstCell.y) == CHUNK_SIZE - 1)  m_world->requestChunkUpdate({ m_coord.x, m_coord.y + 1 });
}

Chunk::Chunk(World* world, ChunkCoord coord) {
	m_world = world;
	m_coord = coord;
	m_size	= CHUNK_SIZE;

	m_cells = new Cell[CHUNK_SIZE * CHUNK_SIZE];
	m_view	= new ChunkView(this);
	m_state = new ChunkState();
}
