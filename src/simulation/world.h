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
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @return Cell - The Cell object at the specified coordinates.
	 */
	Cell getCell(const CellCoord& coord) const {
		ChunkCoord chunkCoord = coord.getChunkCoord();

		// Check if the corresponding Chunk exists
		if (!hasChunk(chunkCoord))
			return SimValues::Air();

		// Delegate the task to the corresponding Chunk
		return m_chunks.at(chunkCoord)->getCell(coord);
	}

	/**
	 * Sets the properties of the Cell at the specified CellCoord within the World.
	 * Delegates the task to the overloaded setCell method with individual x and y coordinates.
	 *
	 * @param coord - The CellCoord specifying the coordinates of the Cell.
	 * @param cell - The Cell object containing the new properties.
	 */
	void setCell(const CellCoord& coord, Cell cell) {
		ChunkCoord chunkCoord = coord.getChunkCoord();
		if (!hasChunk(chunkCoord)) {
			createChunk(chunkCoord)->setCell(coord, cell);
			return;
		}

		// Delegate the task to the corresponding Chunk
		m_chunks.at(chunkCoord)->setCell(coord, cell);
	}

	/**
	 * Retrieves the Chunk at the specified ChunkCoord within the World.
	 * Creates a new Chunk if it does not exist; otherwise, returns the existing Chunk.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return Chunk* - A pointer to the retrieved or newly created Chunk.
	 */
	Chunk* getChunk(const ChunkCoord& coord) {
		if (!hasChunk(coord))
			return createChunk(coord);
		
		return m_chunks.at(coord);
	}

	/**
	 * Retrieves the const pointer to the Chunk at the specified ChunkCoord within the World.
	 * Creates a new Chunk if it does not exist; otherwise, returns the existing Chunk.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return const Chunk* - A const pointer to the retrieved or newly created Chunk.
	 */
	const Chunk* getChunk(const ChunkCoord& coord) const {
		return getChunk(coord);
	}

	/**
	 * Retrieves the Chunk at the specified ChunkCoord within the World if it is populated.
	 * Returns a pointer to the Chunk if it exists; otherwise, returns nullptr.
	 *
	 * @param coord - The ChunkCoord specifying the coordinates of the Chunk.
	 * @return Chunk* - A pointer to the retrieved Chunk if populated; otherwise, nullptr.
	 */
	Chunk* getChunkIfPopulated(const ChunkCoord& coord) {
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
	const Chunk* getChunkIfPopulated(const ChunkCoord& coord) const {
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
	void swapCells(const CellCoord& a, const CellCoord& b) {
		ChunkCoord aChunkCoord = a.getChunkCoord();
		ChunkCoord bChunkCoord = b.getChunkCoord();
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
	bool isChunkEmpty(const ChunkCoord& coord) const {
		return !hasChunk(coord);
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
		std::cout << "Loading world from " << path << "." << std::endl;

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

			// Deserialize chunk
			Chunk* chunk = Chunk::deserialise(chunkFile, this);
			m_chunks.insert({ chunk->m_coord, chunk });
		}

		std::cout << "Loaded " << m_chunks.size() << " chunks. " << std::endl;
	}

	/**
	 * Checks if the specified ChunkCoord is within the visible region determined by the given rendering options.
	 *
	 * @param options - The rendering options specifying camera and window parameters.
	 * @param coord - The ChunkCoord to be checked for visibility.
	 * @return bool - Returns true if the ChunkCoord is within the visible region; otherwise, returns false.
	 */
	bool isChunkDrawn(const ChunkCoord& coord) const;

	void setPalette(ColorPalette* palette) {
		m_palette = palette;
	}

	void draw(const Options& options);

	void update(const Options& options, SimulationUpdateFunction updateFunction) {
		m_evenTick = !m_evenTick;

		//for (auto& chunk : m_chunks) {
		//	chunk.second->process(options, updateFunction);
		//}

		static auto updateTask = new ChunkWorkerTask(&Chunk::process);
		updateTask->setArgs(options, updateFunction);
		m_workers->execute(updateTask);

		eraseEmptyChunks();
	}

	void requestChunkUpdate(ChunkCoord coord) {
		if (hasChunk(coord))
			m_chunks.at(coord)->requestUpdate();
	}

	World();

	// TODO: Serialize without copy constructor
	World(const World&) = default;

	World(World&&) = delete;
	World& operator=(const World&) = delete;

	bool isCellUpdated(Cell cell) const {
		return cell.updatedOnEvenTick == m_evenTick;
	}

	bool evenTick() const {
		return m_evenTick;
	}

private:
	using ChunkMap = std::unordered_map<ChunkCoord, Chunk*, ChunkCoord::Hasher>;
	using WorkerPool = std::shared_ptr<ChunkWorkerPool>;

	ChunkMap		m_chunks{};
	ColorPalette*	m_palette = nullptr;

	bool			m_evenTick = false;
	WorldView*		m_view;
	WorkerPool		m_workers;
	
	bool hasChunk(ChunkCoord coord) const {
		return m_chunks.find(coord) != m_chunks.end();
	}

	void eraseEmptyChunks() {
		for (auto it = m_chunks.begin(); it != m_chunks.end();) {
			if (it->second->empty()) {
				delete it->second;
				it = m_chunks.erase(it);
			}
			else
				++it;
		}
	}

	Chunk* createChunk(ChunkCoord coord) {
		// TODO: this is temp
		static std::mutex c_mut;
		std::lock_guard<std::mutex> lock(c_mut);

		Chunk* chunk = new Chunk(this, coord);
		m_chunks.insert({ coord, chunk });
		return chunk;
	}
};

template <>
inline Json& Json::operator=(World world) {
	*this = Json::CreateEmptyObject<World>();

	(*this)["chunkSize"] = CHUNK_SIZE;
	(*this)["cellSize"] = sizeof(Cell);

	return *this;
}
