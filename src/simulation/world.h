#pragma once
#include "common.h"
#include "chunk.h"
#include "chunk_worker.h"
#include <unordered_map>

class World {
public:
	/**
	 * Retrieves the Cell at the specified coordinates within the World.
	 * Delegates the task to the corresponding Chunk if it exists; otherwise, returns a default Cell.
	 *
	 * @param x - The x-coordinate of the Cell.
	 * @param y - The y-coordinate of the Cell.
	 * @return Cell - The Cell object at the specified coordinates.
	*/
	__host__ __device__
	Cell getCell(coord_t x, coord_t y) const {
		ChunkCoord chunkCoord = chunkCoordOf(x, y);
		
		// Check if the corresponding Chunk exists
		if (!hasChunk(chunkCoord))
			return SimValues::Air();

		// Delegate the task to the corresponding Chunk
#ifndef __CUDA_ARCH__
		return m_chunks.at(chunkCoord)->getCell(x, y);
#else
		ChunkCoord coord;
		coord.x = chunkCoord.x - m_minDrawnChunk.x;
		coord.y = chunkCoord.y - m_minDrawnChunk.y;
		return m_deviceChunks[coord.y * (m_maxDrawnChunk.x - m_minDrawnChunk.x + 1) + coord.x].getCell(x, y);
#endif
	}

	/**
	 * Retrieves the Cell at the specified CellCoord within the World.
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
	 * Sets the properties of the Cell at the specified coordinates within the World.
	 * Delegates the task to the corresponding Chunk if it exists; otherwise, creates a new Chunk and sets the Cell.
	 *
	 * @param x - The x-coordinate of the Cell.
	 * @param y - The y-coordinate of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(coord_t x, coord_t y, Cell cell) {
		ChunkCoord chunkCoord = chunkCoordOf(x, y);
		if (!hasChunk(chunkCoord)) {
			createChunk(chunkCoord)->setCell(x, y, cell);
			return;
		}

		// Delegate the task to the corresponding Chunk
		m_chunks.at(chunkCoord)->setCell(x, y, cell);
	}

	/**
	 * Sets the properties of the Cell at the specified CellCoord within the World.
	 * Delegates the task to the overloaded setCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(CellCoord coord, Cell cell) {
		setCell(coord.x, coord.y, cell);
	}

	/**
	 * Retrieves the Chunk at the specified ChunkCoord within the World.
	 * Creates a new Chunk if it does not exist; otherwise, returns the existing Chunk.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return Chunk* - A pointer to the retrieved or newly created Chunk.
	 */
	__host__ __device__
	Chunk* getChunk(ChunkCoord coord) {
		if (!hasChunk(coord))
			#ifndef __CUDA_ARCH__
			return createChunk(coord);
			#else
			return nullptr;
			#endif

		#ifndef __CUDA_ARCH__
		return m_chunks.at(coord);
		#else
		coord.x -= m_minDrawnChunk.x;
		coord.y -= m_minDrawnChunk.y;
		return &m_deviceChunks[coord.y * (m_maxDrawnChunk.x - m_minDrawnChunk.x + 1) + coord.x];
		#endif
	}

	/**
	 * Retrieves the const pointer to the Chunk at the specified ChunkCoord within the World.
	 * Creates a new Chunk if it does not exist; otherwise, returns the existing Chunk.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return const Chunk* - A const pointer to the retrieved or newly created Chunk.
	 */
	__host__ __device__
	const Chunk* getChunk(ChunkCoord coord) const {
		return getChunk(coord);
	}

	/**
	 * Retrieves the Chunk at the specified ChunkCoord within the World if it is populated.
	 * Returns a pointer to the Chunk if it exists; otherwise, returns nullptr.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return Chunk* - A pointer to the retrieved Chunk if populated; otherwise, nullptr.
	 */
	Chunk* getChunkIfPopulated(ChunkCoord coord) {
		if (!hasChunk(coord))
			return nullptr;

		return m_chunks.at(coord);
	}

	/**
	 * Retrieves the const pointer to the Chunk at the specified ChunkCoord within the World if it is populated.
	 * Returns a const pointer to the Chunk if it exists; otherwise, returns nullptr.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return const Chunk* - A const pointer to the retrieved Chunk if populated; otherwise, nullptr.
	 */
	const Chunk* getChunkIfPopulated(ChunkCoord coord) const {
		if (!hasChunk(coord))
			return nullptr;

		return m_chunks.at(coord);
	}

	/**
	 * Swaps the properties of the Cells at the specified CellCoords within the World.
	 * If both CellCoords are within the same Chunk, delegates the task to that Chunk.
	 * Otherwise, swaps Cells between the corresponding Chunks.
	 *
	 * @param a - The first CellCoord specifying the coordinates of the first Cell to be swapped.
	 * @param b - The second CellCoord specifying the coordinates of the second Cell to be swapped.
	 */
	void swapCells(CellCoord a, CellCoord b) {
		ChunkCoord aChunkCoord = chunkCoordOf(a);
		ChunkCoord bChunkCoord = chunkCoordOf(b);
		if (aChunkCoord == bChunkCoord) {
			Chunk* chunk = getChunkIfPopulated(aChunkCoord);
			if (chunk != nullptr)
				chunk->swapCells(a, b);
		}
		else {
			Chunk* aChunk = getChunk(aChunkCoord);
			Chunk* bChunk = getChunk(bChunkCoord);

			Cell aCell = aChunk->getCell(a);
			aChunk->setCell(a, bChunk->getCell(b));
			bChunk->setCell(b, aCell);
		}
	}

	/**
	 * Checks if the Chunk at the specified ChunkCoord within the World is empty.
	 * Returns true if the Chunk does not exist; otherwise, returns false.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return bool - Returns true if the Chunk is empty (does not exist), false otherwise.
	 */
	__host__ __device__
	bool isChunkEmpty(ChunkCoord coord) const {
		return !hasChunk(coord);
	}

	/**
	 * Calculates and returns the ChunkCoord based on the specified coordinates.
	 *
	 * @param x - The x-coordinate used to calculate the ChunkCoord.
	 * @param y - The y-coordinate used to calculate the ChunkCoord.
	 * @return ChunkCoord - The calculated ChunkCoord.
	 */
	__host__ __device__
	static ChunkCoord chunkCoordOf(coord_t x, coord_t y) {
		return ChunkCoord{ (coord_t)floor(x / (double)CHUNK_SIZE), (coord_t)floor(y / (double)CHUNK_SIZE) };
	}

	/**
	 * Calculates and returns the ChunkCoord based on the specified CellCoord.
	 *
	 * @param coord - The CellCoord used to calculate the ChunkCoord.
	 * @return ChunkCoord - The calculated ChunkCoord.
	 */
	__host__ __device__
	static ChunkCoord chunkCoordOf(CellCoord coord) {
		return chunkCoordOf(coord.x, coord.y);
	}

	/**
	 * Saves the current state of the World to a JSON file at the specified path.
	 *
	 * @param path - The path where the World state should be saved.
	 */
	void save(const std::string& path) const {
		namespace fs = std::filesystem;

		// Create save path
		fs::create_directories(path);

		// Save world data
		std::ofstream worldOfs(path + "/world.json");
		worldOfs << Json(*this);

		// Save chunks
		std::string chunksPath = path + "/chunks/";
		fs::remove_all(chunksPath);
		fs::create_directories(chunksPath);
		for (auto& entry : m_chunks) {
			ChunkCoord coord = entry.first;
			Chunk& chunk = *entry.second;

			if (chunk.empty())
				continue;

			// Create chunk path
			std::stringstream chunkFileName;
			chunkFileName << "(" << coord.x << ";" << coord.y << ").bin";
			std::string chunkPath = chunksPath + chunkFileName.str();

			// Serialise chunk
			std::ofstream ofs(chunkPath, std::ios::binary);
			chunk.serialise(ofs);
		}
	}

	/**
	 * Loads the World state from a JSON file located in the specified path.
	 *
	 * @param path - The path where the World state should be loaded from.
	 */
	void load(const std::string& path) {
		namespace fs = std::filesystem;

		// Load world json
		std::ifstream worldFile(path + "/world.json");
		if (!worldFile.is_open())
			return;
		Json worldJson;
		worldFile >> worldJson;

		// Check world data
		if ((unsigned)worldJson["chunkSize"] != CHUNK_SIZE) return;
		if ((unsigned)worldJson["cellSize"] != sizeof(Cell)) return;

		// Load chunks
		std::string chunksPath = path + "/chunks";
		for (auto& chunkFilePath : fs::recursive_directory_iterator(chunksPath)) {
			if (chunkFilePath.is_directory())
				continue;

			// Open file
			std::ifstream chunkFile(chunkFilePath.path(), std::ios::binary);
			if (!chunkFile.is_open())
				continue;

			Chunk* chunk = new Chunk(this);
			chunk->deserialise(chunkFile);
			m_chunks.insert({ chunk->m_coord, chunk });
		}
	}

	/**
	 * Checks if the specified ChunkCoord is within the visible region determined by the given rendering options.
	 *
	 * @param options - The rendering options specifying camera and window parameters.
	 * @param coord - The ChunkCoord to be checked for visibility.
	 * @return bool - Returns true if the ChunkCoord is within the visible region; otherwise, returns false.
	 */
	__host__ __device__
	bool isChunkDrawn(ChunkCoord coord) const {
		return m_minDrawnChunk.x <= coord.x && coord.x <= m_maxDrawnChunk.x
			&& m_minDrawnChunk.y <= coord.y && coord.y <= m_maxDrawnChunk.y;
	}

	void update(Options options, SimulationUpdateFunction updateFunction) {
		// Update chunks without drawing them
		updateImpl(options, updateFunction, false);
	}

	void draw(Options options) {
		bool drawnChunksChanged = prepareDrawnChunksBuffer(options, false);

		// If the screen moved draw chunks without updating them
		if (drawnChunksChanged)
			updateImpl(options, nullptr, true);

		uploadDrawnChunksToDevice(drawnChunksChanged);
	}

	void updateAndDraw(Options options, SimulationUpdateFunction updateFunction) {
		bool drawnChunksChanged = prepareDrawnChunksBuffer(options, true);

		// Update and draw chunks
		updateImpl(options, updateFunction, true);

		uploadDrawnChunksToDevice(drawnChunksChanged);
	}

	bool isEvenTick() const {
		return m_evenTick;
	}

	void requestChunkUpdate(ChunkCoord coord) {
		if (hasChunk(coord))
			m_chunks.at(coord)->requestUpdate();
	}

	World() = default;

	World(const World& world)
		: m_chunks()
		, m_minDrawnChunk(world.m_minDrawnChunk)
		, m_maxDrawnChunk(world.m_maxDrawnChunk)
		, m_deviceChunks(world.m_deviceChunks)
	{ }

private:
	std::unordered_map<ChunkCoord, Chunk*, ChunkCoord::Hasher> m_chunks{};

	ChunkCoord			m_minDrawnChunk;
	ChunkCoord			m_maxDrawnChunk;
	DeviceBuffer<Chunk> m_deviceChunks;
	Chunk*				m_chunksToCommitToDevice = nullptr;

	bool m_evenTick;

#ifndef __CUDA_ARCH__
	bool hasChunk(ChunkCoord coord) const {
		return m_chunks.find(coord) != m_chunks.end();
	}
#else
	__device__
	bool hasChunk(ChunkCoord coord) const {
		return isChunkDrawn(coord);
	}
#endif

	void updateImpl(Options options, SimulationUpdateFunction updateFunction, bool doDraw) {
		if (updateFunction != nullptr)
			m_evenTick = !m_evenTick;

		ChunkWorkerPool::instance().setParams(options, updateFunction, doDraw);
		for (auto& chunk : m_chunks)
			//chunk.second->process(options, updateFunction, doDraw);
			ChunkWorkerPool::instance().processChunk(chunk.second);

		ChunkWorkerPool::instance().awaitAll();
	}

	void addChunkToDrawnChunks(Chunk* chunk) {
		ChunkCoord coord;
		coord.x = chunk->m_coord.x - m_minDrawnChunk.x;
		coord.y = chunk->m_coord.y - m_minDrawnChunk.y;

		m_chunksToCommitToDevice[coord.y * (m_maxDrawnChunk.x - m_minDrawnChunk.x + 1) + coord.x] = *chunk;
	}

	// returns true when reallocated the drawnChunksBuffer
	bool prepareDrawnChunksBuffer(Options options, bool clearChunks) {
		bool shouldReallocDrawnChunksBuffer = changeMinAndMaxDrawnChunks(options);

		if (shouldReallocDrawnChunksBuffer)
			reallocDrawnChunksBuffer();
		else if (clearChunks)
			resetDrawnChunksBuffer();

		return shouldReallocDrawnChunksBuffer;
	}

	void uploadDrawnChunksToDevice(bool drawnChunksChanged) {
		// Copy drawn chunks buffer to the device
		if (!m_deviceChunks.isAllocated() || drawnChunksChanged)
			m_deviceChunks.alloc(drawnChunkCount());
		m_deviceChunks.upload(m_chunksToCommitToDevice, drawnChunkCount());
	}

	void reallocDrawnChunksBuffer() {
		if (m_chunksToCommitToDevice != nullptr)
			delete[] m_chunksToCommitToDevice;

		m_chunksToCommitToDevice = new Chunk[drawnChunkCount() * sizeof(Chunk)];
		//m_chunksToCommitToDevice = (Chunk*)calloc(drawnChunkCount(), sizeof(Chunk));
	}

	// Memsets the buffer to 0
	void resetDrawnChunksBuffer() {
		//for (int i = 0; i < drawnChunkCount(); ++i)
		//	m_chunksToCommitToDevice[i] = Chunk();
		memset(m_chunksToCommitToDevice, 0, drawnChunkCount() * sizeof(Chunk));
	}

	// Returns true when value changed
	bool changeMinAndMaxDrawnChunks(Options options) {
		vec2 minCameraPos, maxCameraPos;
		options.camera.getMinWorldPos(options.windowWidth, options.windowHeight, minCameraPos.x, minCameraPos.y);
		options.camera.getMaxWorldPos(options.windowWidth, options.windowHeight, maxCameraPos.x, maxCameraPos.y);

		ChunkCoord prevMinDrawnChunk = m_minDrawnChunk, prevMaxDrawnChunk = m_maxDrawnChunk;
		m_minDrawnChunk = chunkCoordOf(minCameraPos.x, minCameraPos.y);
		m_maxDrawnChunk = chunkCoordOf(maxCameraPos.x, maxCameraPos.y);
		return (prevMinDrawnChunk != m_minDrawnChunk) || (prevMaxDrawnChunk != m_maxDrawnChunk);
	}

	size_t drawnChunkCount() const {
		return (m_maxDrawnChunk.x - m_minDrawnChunk.x + 1) * (m_maxDrawnChunk.y - m_minDrawnChunk.y + 1);
	}

	Chunk* createChunk(ChunkCoord coord) {
		Chunk* chunk = new Chunk(this, coord);
		m_chunks.insert({ coord, chunk });
		return chunk;
	}

	friend void Chunk::process(Options options, SimulationUpdateFunction updateFunction, bool doDraw);
};

template <>
inline Json& Json::operator=(World world) {
	*this = Json::CreateEmptyObject<World>();

	(*this)["chunkSize"] = CHUNK_SIZE;
	(*this)["cellSize"] = sizeof(Cell);

	return *this;
}
