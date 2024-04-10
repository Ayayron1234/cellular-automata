#pragma once
#include "chunk.h"
#include "color_palette.h"
#include "gl/GL.h"

class ChunkView {
public:
	void alloc() {
		m_texture = SDL_CreateTexture(IO::GetRenderer(), SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, CHUNK_SIZE, CHUNK_SIZE);
		SDL_LockTexture(m_texture, NULL, (void**)&m_buffer, &m_pitch);
	}

	void destroy() {
		if (!isAllocated())
			return;

		SDL_DestroyTexture(m_texture);
		m_buffer = nullptr;
		m_texture = nullptr;
	}

	void draw(Chunk* chunk, Options options) {	// from main thread
		if (!isAllocated())
			alloc();

		// Send data to video memory
		SDL_UnlockTexture(m_texture);

		updateDstRect(chunk, options);
		SDL_RenderCopyF(IO::GetRenderer(), m_texture, NULL, &m_dstRect);

		//SDL_LockTexture(m_texture, NULL, (void**)&m_buffer, &m_pitch);
	}

	void render(Chunk* chunk, Options options, ColorPalette* palette) {		// multi threaded
		if (!isAllocated())
			return;

		// TODO: updateSinceDraw is not implemented
		if (chunk->updatedSinceDraw())
			renderCells(chunk, options, palette);
	}

	bool isAllocated() const {
		return m_buffer != nullptr;
	}

	~ChunkView() {
		destroy();
	}

private:
	SDL_Texture*	m_texture = nullptr;
	IO::RGB*		m_buffer = nullptr;
	int				m_pitch = 0;

	SDL_FRect		m_dstRect{};

	void updateDstRect(Chunk* chunk, Options options) {
		m_dstRect.w = CHUNK_SIZE * options.camera.zoom;
		m_dstRect.h = CHUNK_SIZE * options.camera.zoom;

		//vec2 pos = options.camera.worldToScreen(options.windowWidth, options.windowHeight, (vec2)chunk->getCoord() * CHUNK_SIZE);
		vec2 pos(0,0);
		m_dstRect.x = pos.x;
		m_dstRect.y = pos.y;
	}

	void renderCells(Chunk* chunk, Options options, ColorPalette* palette) {
		for (int i = 0; i < CHUNK_SIZE * CHUNK_SIZE; ++i) {
			CellCoord coord(i % CHUNK_SIZE, i / CHUNK_SIZE);

			Cell cell = chunk->getCell(coord);

			IO::RGB color;
			color = palette->getColor(cell);

			if (options.showChunkBorders) {
				// TODO: implement

				//ChunkCoord chunkCoord = world.chunkCoordOf(coord);
				//Chunk* chunk = world.getChunk(chunkCoord);
				//if (!chunk->empty()) {
				//	bool isChunkEdge = world.chunkCoordOf(cellCoordFromPixel(options, col + 1, row)).x != chunkCoord.x
				//		|| world.chunkCoordOf(cellCoordFromPixel(options, col, row + 1)).y != chunkCoord.y
				//		|| world.chunkCoordOf(cellCoordFromPixel(options, col - 1, row + 1)).x != chunkCoord.x
				//		|| world.chunkCoordOf(cellCoordFromPixel(options, col, row - 1)).y != chunkCoord.y;

				//	if (isChunkEdge)
				//		color = (chunk->updated()) ? IO::RGB::red() : IO::RGB(50, 50, 50);
				//}
			}

			m_buffer[i] = color;
		}
	}
};
