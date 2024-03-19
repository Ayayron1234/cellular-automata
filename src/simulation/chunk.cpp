#include "chunk.h"
#include "world.h"
#include "../utils/options.h"

Chunk& Chunk::operator=(const Json& json) {
	m_coord = json["coord"];
	m_isVaried = !json["uniform"];

	if (isUniform()) {
		m_uniformCell = Cell(json["uniformCell"]);
	}
	else {
		m_cells = new Cell[CHUNK_SIZE * CHUNK_SIZE];

		const std::string& path = m_world->m_saveDirectory;
		if (path.empty())
			return *this;

		std::string fileName = Chunk::chunkFileName(m_coord);
		std::ifstream ifs(path + fileName, std::ios::binary);
		if (!ifs.is_open()) {
			std::cerr << "[World]: Couldn't open " << path << fileName << " for deserialisation. " << std::endl;
			return *this;
		}

		deserialise(ifs);
	}

	return *this;
}

inline Cell Chunk::getCell(coord_t x, coord_t y) const {
	if (isCoordInside(x, y)) {
		if (isUniform())
			return m_uniformCell;
		return m_cells[coordToIndex(x, y)];
	}
	return m_world->getCell(x, y);
}

void Chunk::setCell(coord_t x, coord_t y, Cell cell) {
	if (isCoordInside(x, y)) {
		// Break uniformity if needed
		if (isUniform())
			makeVaried();

		// Access the specified Cell position within the Chunk
		Cell& cellPos = m_cells[coordToIndex(x, y)];

		// Update cell and cell type counts
		--m_cellTypeCounts[(unsigned int)cellPos.type];
		cellPos = cell;
		++m_cellTypeCounts[(unsigned int)cellPos.type];
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
		// Update chunk uniformity information
		int uniformType = getUniformType();
		if (uniformType != -1)
			makeUniform(Cell::Type(uniformType));

		// Advance chunk simulation state
			updateFunction(options, *this);
	}
}
