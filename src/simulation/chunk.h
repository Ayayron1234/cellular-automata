#pragma once
#include "common.h"
#include "color_palette.h"
#include <mutex>
#include "../device.h"

class Options;

class ChunkView;

class Chunk {
public:
	/**
	 * Retrieves the Cell at the specified coordinates within the Chunk.
	 * Returns the uniform cell if the Chunk is uniform; otherwise, retrieves the individual cell.
	 * If the coordinates are outside the Chunk, delegates the task to the World.
	 *
	 * @param x - The x-coordinate of the Cell.
	 * @param y - The y-coordinate of the Cell.
	 * @return Cell - The Cell object at the specified coordinates.
	 */
	__host__ __device__
	Cell getCell(coord_t x, coord_t y) const;

	/**
	 * Retrieves the Cell at the specified CellCoord within the Chunk.
	 * Delegates the task to the overloaded getCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @return Cell - The Cell object at the specified coordinates.
	 */
	__host__ __device__
	Cell getCell(CellCoord coord) const {
		return getCell(coord.x, coord.y);
	}

	/**
	 * Sets the properties of the Cell at the specified coordinates within the Chunk.
	 * If the Chunk is uniform and the specified coordinates are inside the Chunk,
	 * the uniformity is broken, and a new dynamic array is allocated for individual cell storage.
	 * Updates the cell type counts accordingly.
	 *
	 * @param x - The x-coordinate of the Cell.
	 * @param y - The y-coordinate of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(coord_t x, coord_t y, Cell cell);

	/**
	 * Sets the properties of the Cell at the specified CellCoord within the Chunk.
	 * Delegates the task to the overloaded setCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(CellCoord coord, Cell cell) {
		setCell(coord.x, coord.y, cell);
	}

	void markCellAsUpdated(CellCoord coord, bool updatedOnEvenTick) {
		if (isCoordInside(coord))
			m_cells[coordToIndex(coord)].updatedOnEvenTick = updatedOnEvenTick;
	}

	/**
	 * Checks whether the specified coordinates (x, y) are within the boundaries of the current Chunk.
	 *
	 * @param x - The x-coordinate to be checked.
	 * @param y - The y-coordinate to be checked.
	 * @return bool - Returns true if the coordinates are inside the Chunk, false otherwise.
	 */
	__host__ __device__
	bool isCoordInside(coord_t x, coord_t y) const {
		return 0 <= (x - m_coord.x * (coord_t)CHUNK_SIZE) && (x - m_coord.x * (coord_t)CHUNK_SIZE) < CHUNK_SIZE
			&& 0 <= (y - m_coord.y * (coord_t)CHUNK_SIZE) && (y - m_coord.y * (coord_t)CHUNK_SIZE) < CHUNK_SIZE;
	}

	/**
	 * Checks whether the specified CellCoord is within the boundaries of the current Chunk.
	 *
	 * @param coord - The CellCoord to be checked.
	 * @return bool - Returns true if the coordinates of the CellCoord are inside the Chunk, false otherwise.
	 */
	__host__ __device__
	bool isCoordInside(CellCoord coord) const {
		return isCoordInside(coord.x, coord.y);
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
		os.write((char*)&m_nonAirCount, sizeof(m_nonAirCount));

		// Write cells
		os.write((char*)m_cells, CHUNK_SIZE * CHUNK_SIZE * sizeof(Cell));
	}

	/**
	 * Deserializes non-uniform Chunk data from the provided input file stream.
	 * Skips deserialization if the Chunk is uniform.
	 *
	 * @param is - The input file stream containing the serialized data.
	 */
	void deserialise(std::ifstream& is) {
		// Read chunk size
		decltype(m_size) size;
		is.read((char*)&size, sizeof(m_size));
		if (size != m_size)
			return;

		// Read coordinates
		is.read((char*)&m_coord.x, sizeof(m_coord.x));
		is.read((char*)&m_coord.y, sizeof(m_coord.y));

		// Read nonAir cell count
		is.read((char*)&m_nonAirCount, sizeof(m_nonAirCount));

		// Read cells
		is.read((char*)m_cells, CHUNK_SIZE * CHUNK_SIZE * sizeof(Cell));
	}

	/**
	 * Swaps the properties of the Cells at the specified CellCoords within the Chunk.
	 * If either of the specified coordinates is outside the Chunk, the task is delegated to the World.
	 *
	 * @param a - The first CellCoord specifying the coordinates of the first Cell to be swapped.
	 * @param b - The second CellCoord specifying the coordinates of the second Cell to be swapped.
	 */
	void swapCells(CellCoord a, CellCoord b);

	/**
	 * Processes the Chunk, checking its visibility, updating uniformity, and advancing simulation state if specified.
	 *
	 * @param options - The rendering and simulation options.
	 * @param updateFunction - The function to update the simulation state, if provided.
	 */
	void process(Options options, SimulationUpdateFunction updateFunction);

	void draw(Options options);

	void render(Options options, ColorPalette* palette);

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

	__host__ __device__
	bool empty() const {
		return m_cells == nullptr;
	}

	__host__ __device__
	bool updated() const {
		return m_updated;
	}

	bool updatedSinceDraw() const {
		// TODO: implement this
		return m_updated;
	}

	void requestUpdate();

	__host__ __device__
	Chunk(const Chunk& chunk) 
		: m_world(chunk.m_world)
		, m_coord(chunk.m_coord)
		, m_size(chunk.m_size)
		, m_nonAirCount(chunk.m_nonAirCount)
		, m_cells(chunk.m_cells)
		, m_shouldUpdate(chunk.m_shouldUpdate)
		, m_updated(chunk.m_updated)
	{ }

	Chunk& operator=(const Chunk& chunk) {
		m_world = chunk.m_world;
		m_coord = chunk.m_coord;
		m_size = chunk.m_size;
		m_nonAirCount = chunk.m_nonAirCount;
		m_cells = chunk.m_cells;
		m_shouldUpdate = chunk.m_shouldUpdate;
		m_updated = chunk.m_updated;
		return *this;
	}

private:
	World*		m_world;	// Pointer to the World containing the Chunk
	ChunkCoord	m_coord;	// Coordinates of the Chunk within the World
	size_t		m_size;     // Size of the Chunk

	std::mutex m_airCountMutex;
	volatile unsigned m_nonAirCount = 0;	// Number of cells in chunk that are not air

	Cell* m_cells = nullptr;    // Dynamic array storing individual Cell properties

	bool m_shouldUpdate = true;
	bool m_updated = true;
	bool m_updatedOnEvenTick;

	ChunkView* m_view = nullptr;

	__host__ __device__
	size_t coordToIndex(coord_t x, coord_t y) const {
		return (y - m_coord.y * CHUNK_SIZE) * CHUNK_SIZE + (x - m_coord.x * CHUNK_SIZE);
	}

	__host__ __device__
	size_t coordToIndex(CellCoord coord) const {
		return coordToIndex(coord.x, coord.y);
	}

	void updateChunksNearCell(CellCoord coord);

	void checkForUpdates();

	void allocate();

	void clear() {
		delete[] m_cells;
		m_cells = nullptr;
	}

	Chunk() 
		: m_world(nullptr)
		, m_size(CHUNK_SIZE)
	{ }

	Chunk(World* world)
		: m_world(world)
		, m_size(CHUNK_SIZE)
	{
		allocate();
	}

	Chunk(World* world, ChunkCoord coord)
		: m_world(world)
		, m_coord(coord)
		, m_size(CHUNK_SIZE)
	{
		allocate();
	}

	static std::string chunkFileName(ChunkCoord coord) {
		std::stringstream sstream;
		sstream << "/chunk" << coord.x << ";" << coord.y << ".bin";
		return sstream.str();
	}

	friend class World;
	friend Json& Json::operator=<Chunk>(Chunk chunk);
};
