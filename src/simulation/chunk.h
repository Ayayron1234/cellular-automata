#pragma once
#include "common.h"
#include "../device.h"

class Options;

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
	Cell getCell(coord_t x, coord_t y) const
#ifndef __CUDA_ARCH__
		;
#else
	{
		if (isCoordInside(x, y)) {
			if (isUniform())
				return m_uniformCell;
			return m_deviceBuffer[coordToIndex(x, y)];
		}
		return Cell();
	}
#endif

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
	 * Checks if the Chunk is uniform, meaning all cells in the Chunk have the same properties.
	 *
	 * @return bool - Returns true if the Chunk is uniform, false otherwise.
	 */
	__host__ __device__
	bool isUniform() const {
		return !m_isVaried;
	}

	/**
	 * Serializes the non-uniform Chunk data and writes it to the provided output file stream.
	 * Skips serialization if the Chunk is uniform.
	 *
	 * @param os - The output file stream to write the serialized data.
	 */
	void serialise(std::ofstream& os) const {
		if (isUniform())
			return;

		// Write chunk size
		os.write((char*)&m_size, sizeof(m_size));

		// Write coordinates
		os.write((char*)&m_coord.x, sizeof(m_coord.x));
		os.write((char*)&m_coord.y, sizeof(m_coord.y));

		// Write cell type counts
		os.write((char*)m_cellTypeCounts, Cell::MaxTypeCount() * sizeof(m_cellTypeCounts[0]));

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
		if (isUniform())
			return;

		// Read chunk size
		is.read((char*)&m_size, sizeof(m_size));

		// Read coordinates
		is.read((char*)&m_coord.x, sizeof(m_coord.x));
		is.read((char*)&m_coord.y, sizeof(m_coord.y));

		// Read cell type counts
		is.read((char*)m_cellTypeCounts, Cell::MaxTypeCount() * sizeof(m_cellTypeCounts[0]));

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
	void process(Options options, SimulationUpdateFunction updateFunction, bool doDraw);

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

	Chunk& operator=(const Json& json);

	__host__ __device__
	Chunk(const Chunk&) = default;

private:
	World* m_world;				// Pointer to the World containing the Chunk
	ChunkCoord m_coord;			// Coordinates of the Chunk within the World
	size_t m_size;              // Size of the Chunk

	bool m_isVaried = false;    // Flag indicating if the Chunk is uniform
	Cell m_uniformCell;         // Uniform Cell properties if the Chunk is uniform
	Cell* m_cells = nullptr;    // Dynamic array storing individual Cell properties

	unsigned int m_cellTypeCounts[Cell::MaxTypeCount()] = { 0 };  // Array counting the occurrences of each Cell type

	DeviceBuffer<Cell> m_deviceBuffer;

	__host__ __device__
	size_t coordToIndex(coord_t x, coord_t y) const {
		return (y - m_coord.y * CHUNK_SIZE) * CHUNK_SIZE + (x - m_coord.x * CHUNK_SIZE);
	}

	__host__ __device__
	size_t coordToIndex(CellCoord coord) const {
		return coordToIndex(coord.x, coord.y);
	}

	__host__ __device__
	DeviceBuffer<Cell> getDeviceBuffer() const {
		return m_deviceBuffer;
	}

	void makeUniform(Cell::Type type) {
		m_isVaried = false;
		m_deviceBuffer.freeDevice();
		m_uniformCell = Cell::create(type);
	}

	void makeVaried() {
		m_isVaried = true;
		m_cells = new Cell[CHUNK_SIZE * CHUNK_SIZE];
		for (int i = 0; i < CHUNK_SIZE * CHUNK_SIZE; ++i)
			m_cells[i] = Cell::create(m_uniformCell.type);
	}

	void commitToDevice() {
		if (isUniform()) {
			m_deviceBuffer.freeDevice();
			return;
		}

		if (!m_deviceBuffer.isAllocated())
			m_deviceBuffer.alloc(CHUNK_SIZE * CHUNK_SIZE);

		m_deviceBuffer.uploadAsync(m_cells, CHUNK_SIZE * CHUNK_SIZE);
	}

	// Returns -1 if varied
	int getUniformType() const {
		for (int i = 0; i < Cell::MaxTypeCount(); ++i)
			if (m_cellTypeCounts[i] == CHUNK_SIZE * CHUNK_SIZE)
				return i;
		return -1;
	}

	Chunk() 
		: m_isVaried(false)
		, m_size(CHUNK_SIZE)
		, m_uniformCell(SimValues::Air())
	{ }

	Chunk(World* world, ChunkCoord coord, bool isUniform)
		: m_world(world)
		, m_coord(coord)
		, m_size(CHUNK_SIZE)
		, m_isVaried(!isUniform)
	{
		if (!isUniform)
			m_cells = new Cell[CHUNK_SIZE * CHUNK_SIZE];
		m_cellTypeCounts[(unsigned int)Cell::Type::AIR] = CHUNK_SIZE * CHUNK_SIZE;
	}

	static std::string chunkFileName(ChunkCoord coord) {
		std::stringstream sstream;
		sstream << "/chunk" << coord.x << ";" << coord.y << ".bin";
		return sstream.str();
	}

	friend class World;
	friend Json& Json::operator=<Chunk>(Chunk chunk);
};
