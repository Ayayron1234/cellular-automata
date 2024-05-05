#include "chunk_view.h"
#include "world.h"

bool ChunkView::isDrawn() const {
    return m_chunk->m_world->isChunkDrawn(m_chunk->getCoord());
}
