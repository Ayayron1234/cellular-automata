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

	__host__ __device__
	CellCoord(coord_t _x = 0, coord_t _y = 0) : x(_x), y(_y) { }
	__host__ __device__
	CellCoord(const CellCoord&) = default;
	CellCoord(const ChunkCoord&) = delete;
};

struct ChunkCoord {
	coord_t x;
	coord_t y;

	__host__ __device__
	ChunkCoord(coord_t _x = 0, coord_t _y = 0) : x(_x), y(_y) { }
	__host__ __device__
	ChunkCoord(const ChunkCoord&) = default;
	ChunkCoord(const CellCoord&) = delete;

	struct Hasher {
		__host__ __device__
		int operator()(ChunkCoord coord) const {
			return coord.x + coord.y;
		}
	};

	__host__ __device__
	bool operator==(const ChunkCoord& coord) const {
		return x == coord.x && y == coord.y;
	}
}; JSON_C(ChunkCoord, JSON_M(x), JSON_M(y))

class World;

class Chunk;

using SimulationUpdateFunction = void(*)(Options options, Chunk& chunk);
