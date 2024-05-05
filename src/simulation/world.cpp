#include "world.h"
#include "world_view.h"
#include "../graphics_headers.h"

bool World::isChunkDrawn(const ChunkCoord& coord) const {
	return m_view->isChunkDrawn(coord);
}

World::World()
{
	m_workers = std::make_shared<ChunkWorkerPool>(m_chunks);
	m_view = new WorldView();
}

void World::draw(const Options& options) {
	m_view->draw(*m_palette, options);

	for (auto& chunk : m_chunks) 
		chunk.second->draw(options);

	// After OpenGL function calls
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		// Handle the error
		std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
	}
}
