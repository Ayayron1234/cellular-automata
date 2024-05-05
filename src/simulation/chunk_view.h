#pragma once
#include "chunk.h"
#include "color_palette.h"
#include "../graphics_headers.h"
#include "world_view.h"
#include <thread>
#include <mutex>

class ChunkView {
public:
	void draw(Options options) {
		// If chunk is outside of screen don't render it
		if (!isDrawn()) {
		//	destroy();
			return;
		}

		// Use cell shader
		WorldView::useShader(WorldView::ShaderType::CellTexture);

		// Create texture and vbo if not created yet
		createTexture();

		// Upload chunk data to gpu and create draw call
		uploadAndBind();
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		// Draw chunk border
		if (options.showChunkBorders)
			drawBorder();

		// Check for errors
		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
		}
	}

	void updatePosition() {
		CellCoord chunkCellCoord = m_chunk->getFirstCellCoord();

		// Calculate vertex positions
		vec2 a = (vec2)chunkCellCoord + vec2(0.f, 0.f),
			b = (vec2)chunkCellCoord + vec2(CHUNK_SIZE, 0.f),
			c = (vec2)chunkCellCoord + vec2(CHUNK_SIZE, CHUNK_SIZE),
			d = (vec2)chunkCellCoord + vec2(0.f, CHUNK_SIZE);

		// Set vertex positions and UVs
		// Vertex position							// UV
		m_vertices[0] = a.x; m_vertices[1] = a.y; m_vertices[2] = 0.f; m_vertices[3] = 0.f;
		m_vertices[4] = b.x; m_vertices[5] = b.y; m_vertices[6] = 1.f; m_vertices[7] = 0.f;
		m_vertices[8] = c.x; m_vertices[9] = c.y; m_vertices[10] = 1.f; m_vertices[11] = 1.f;
		m_vertices[12] = d.x; m_vertices[13] = d.y; m_vertices[14] = 0.f; m_vertices[15] = 1.f;

		// Upload quad vertices
		glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

		// Position
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// UV
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	ChunkView(Chunk* chunk)
		: m_chunk(chunk)
	{
		if (s_vbo == 0) {
			glGenBuffers(1, &s_vbo);
		}

		updatePosition();
	}

	~ChunkView() {
		destroyTexture();
	}

private:
	Chunk* m_chunk;

	inline static GLuint	s_vbo = 0;

	float					m_vertices[16]{};
	GLuint					m_textureUnit = 1;
	GLuint					m_texture = 0;

	bool isDrawn() const;

	void createTexture() {
		if (m_texture != 0)
			return;

		// Generate opengl texture
		glGenTextures(1, &m_texture);
		glActiveTexture(GL_TEXTURE0 + m_textureUnit);
		glBindTexture(GL_TEXTURE_2D, m_texture);

		// Set texture parameters (optional, depending on your requirements)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void destroyTexture() {
		if (m_texture == 0)
			return;

		// Delete texture
		glDeleteTextures(1, &m_texture);

		m_texture = 0;
	}

	void drawBorder() const {
		IO::RGB color = IO::RGB::white();
		//color.a = .8f;

		if (m_chunk->updatedSinceDraw())
			color = IO::RGB::red();

		float m_data[24] = { 
			m_vertices[0],  m_vertices[1],	color.r, color.g, color.b, color.a,
			m_vertices[4],  m_vertices[5],	color.r, color.g, color.b, color.a,
			m_vertices[8],  m_vertices[9],	color.r, color.g, color.b, color.a,
			m_vertices[12], m_vertices[13], color.r, color.g, color.b, color.a
		};

		// Upload quad vertices
		glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

		// Position
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// UV
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBufferData(GL_ARRAY_BUFFER, sizeof(m_data), m_data, GL_STATIC_DRAW);

		WorldView::useShader(WorldView::ShaderType::Color);
		glDrawArrays(GL_LINE_LOOP, 0, 4);
		WorldView::useShader(WorldView::ShaderType::CellTexture);
	}

	void uploadAndBind() const {
		// Bind texture
		glActiveTexture(GL_TEXTURE0 + m_textureUnit);
		glBindTexture(GL_TEXTURE_2D, m_texture);

		// Upload cell data as texture
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CHUNK_SIZE, CHUNK_SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)m_chunk->m_cells);

		// Set uniform
		GLuint uniformLocation = glGetUniformLocation(WorldView::getShaderID(WorldView::ShaderType::CellTexture), "cells");
		glUniform1i(uniformLocation, m_textureUnit);

		// Upload quad vertices
		glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

		// Position
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// UV
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(m_vertices), m_vertices);
	}
};
