#pragma once
#include "../utils.h"
#include "cell.h"

#define CHUNK_SIZE 0x80
//inline constexpr size_t g_chunkSize = 0x40;

using coord_t = int;

struct ChunkCoord;

struct CellCoord {
	coord_t x;
	coord_t y;

	ChunkCoord getChunkCoord() const;

	__host__ __device__
	CellCoord(coord_t _x = 0, coord_t _y = 0) : x(_x), y(_y) { }
	__host__ __device__
	CellCoord(vec2 v) : x(v.x), y(v.y) { }
	__host__ __device__
	CellCoord(const CellCoord&) = default;
	CellCoord(const ChunkCoord&) = delete;

	operator vec2() { return vec2(x, y); }
};

inline std::ostream& operator<<(std::ostream& os, const CellCoord& coord) {
	return os << "{" << coord.x << "," << coord.y << "}";
}

struct ChunkCoord {
	coord_t x;
	coord_t y;

	__host__ __device__
	ChunkCoord(coord_t _x = 0, coord_t _y = 0) : x(_x), y(_y) { }
	ChunkCoord(const ChunkCoord&) = default;
	ChunkCoord(const CellCoord&) = delete;

	operator vec2() { return vec2(x, y); }

	struct Hasher {
		int operator()(ChunkCoord coord) const {
			return coord.x + coord.y;
		}
	};

	bool operator==(const ChunkCoord& coord) const {
		return x == coord.x && y == coord.y;
	}
}; JSON_C(ChunkCoord, JSON_M(x), JSON_M(y))

inline std::ostream& operator<<(std::ostream& os, const ChunkCoord& coord) {
	return os << "{" << coord.x << "," << coord.y << "}";
}

inline ChunkCoord CellCoord::getChunkCoord() const {
	return ChunkCoord{ (coord_t)floor(x / (double)CHUNK_SIZE), (coord_t)floor(y / (double)CHUNK_SIZE) };
}

class World;
class WorldView;

class Chunk;
class ChunkView;

using SimulationUpdateFunction = void(*)(Options options, Chunk& chunk);
