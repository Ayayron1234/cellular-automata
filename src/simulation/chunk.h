#pragma once
#include "common.h"
#include "color_palette.h"
#include <mutex>
#include "chunk_state.h"
#include "../device.h"

class Options;

class Chunk {
public:
	/**
	 * Retrieves the Cell at the specified CellCoord within the Chunk.
	 * Delegates the task to the overloaded getCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @return Cell - The Cell object at the specified coordinates.
	 */
	Cell getCell(const CellCoord& coord) const;

	/**
	 * Sets the properties of the Cell at the specified CellCoord within the Chunk.
	 * Delegates the task to the overloaded setCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(const CellCoord& coord, Cell cell);

	void markCellAsUpdated(const CellCoord& coord) {
		if (!isCoordInside(coord))
			return;

		m_cells[coordToIndex(coord)].updatedOnEvenTick = evenTick();
	}

	/**
	 * Checks whether the specified CellCoord is within the boundaries of the current Chunk.
	 *
	 * @param coord - The CellCoord to be checked.
	 * @return bool - Returns true if the coordinates of the CellCoord are inside the Chunk, false otherwise.
	 */
	bool isCoordInside(const CellCoord& coord) const {
		return 0 <= (coord.x - m_coord.x * (coord_t)CHUNK_SIZE) && (coord.x - m_coord.x * (coord_t)CHUNK_SIZE) < CHUNK_SIZE
			&& 0 <= (coord.y - m_coord.y * (coord_t)CHUNK_SIZE) && (coord.y - m_coord.y * (coord_t)CHUNK_SIZE) < CHUNK_SIZE;
	}

	/**
	 * Serializes the non-uniform Chunk data and writes it to the provided output file stream.
	 * Skips serialization if the Chunk is uniform.
	 *
	 * @param os - The output file stream to write the serialized data.
	 */
	void serialise(std::ofstream& os) const {
		// Write chunk size
		os.write((char*)&m_size, sizeof(m_size));

		// Write coordinates
		os.write((char*)&m_coord.x, sizeof(m_coord.x));
		os.write((char*)&m_coord.y, sizeof(m_coord.y));

		// Write nonAir cell count
		m_state->serialise(os);
		//os.write((char*)&m_nonAirCount, sizeof(m_nonAirCount));

		// Write cells
		os.write((char*)m_cells, CHUNK_SIZE * CHUNK_SIZE * sizeof(Cell));
	}

	/**
	 * Deserializes non-uniform Chunk data from the provided input file stream.
	 * Skips deserialization if the Chunk is uniform.
	 *
	 * @param is - The input file stream containing the serialized data.
	 */
	static Chunk* deserialise(std::ifstream& is, World* world);

	/**
	 * Swaps the properties of the Cells at the specified CellCoords within the Chunk.
	 * If either of the specified coordinates is outside the Chunk, the task is delegated to the World.
	 *
	 * @param a - The first CellCoord specifying the coordinates of the first Cell to be swapped.
	 * @param b - The second CellCoord specifying the coordinates of the second Cell to be swapped.
	 */
	void swapCells(const CellCoord& a, const CellCoord& b);

	/**
	 * Processes the Chunk, checking its visibility, updating uniformity, and advancing simulation state if specified.
	 *
	 * @param options - The rendering and simulation options.
	 * @param updateFunction - The function to update the simulation state, if provided.
	 */
	void process(Options options, SimulationUpdateFunction updateFunction);

	void draw(Options options);

	/**
	 * Retrieves the ChunkCoord of the Chunk.
	 *
	 * @return ChunkCoord - The coordinates of the Chunk.
	 */
	ChunkCoord getCoord() const {
		return m_coord;
	}

	/**
	 * Retrieves the CellCoord of the first cell within the Chunk.
	 *
	 * @return CellCoord - The coordinates of the first cell in the Chunk.
	 */
	CellCoord getFirstCellCoord() const {
		return CellCoord(m_coord.x * CHUNK_SIZE, m_coord.y * CHUNK_SIZE);
	}

	bool isCellUpdated(Cell cell) const;

	bool evenTick() const;

#ifdef _DEBUG
	void _dbgInsertBreakOnProcess() {
		m_dbgBreakOnProcess = true;
	}
#endif

	bool empty() const {
		return m_state->empty();
	}

	bool updatedSinceDraw() const {
		return m_state->updatedSinceDraw();
	}

	void requestUpdate() {
		m_state->requestUpdate();
	}

	Chunk& operator=(const Chunk&) = delete;
	Chunk(const Chunk&) = delete;

	~Chunk() {
		if (m_cells)	delete[] m_cells;	m_cells = nullptr;
		if (m_view)		delete m_view;		m_view = nullptr;
		if (m_state)	delete m_state;		m_state = nullptr;
	}

private:
	World*		m_world;			// Pointer to the World containing the Chunk
	ChunkCoord	m_coord;			// Coordinates of the Chunk within the World
	size_t		m_size;				// Size of the Chunk

	Cell*		m_cells = nullptr;  // Array storing individual Cell properties

	ChunkState* m_state = nullptr;
	ChunkView*	m_view = nullptr;

#ifdef _DEBUG
	bool		m_dbgBreakOnProcess = false;
#endif

	size_t coordToIndex(const CellCoord& coord) const {
		return (coord.y - m_coord.y * CHUNK_SIZE) * CHUNK_SIZE + (coord.x - m_coord.x * CHUNK_SIZE);
	}

	void updateChunksNearCell(const CellCoord& coord);

	Chunk(World* world, ChunkCoord coord);

	static std::string chunkFileName(const ChunkCoord& coord) {
		std::stringstream sstream;
		sstream << "/chunk" << coord.x << ";" << coord.y << ".bin";
		return sstream.str();
	}

	friend class World;
	friend class ChunkView;
	friend Json& Json::operator=<Chunk>(Chunk chunk);
};
