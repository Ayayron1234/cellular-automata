#pragma once
#include "chunk.h"
#include "world.h"
#include "color_palette.h"
#include "../graphics_headers.h"
//#include "chunk_view.h"
#include "../utils/shader.h"
#include <fstream>

class WorldView {
public:
	enum class ShaderType { CellTexture, Color };

	void draw(const ColorPalette& palette, const Options& options) {
		cullChunks(options);

		genVaoAndBind();
		s_colorShader.use();
		updateCamera(options);
		s_chunkShader.use();
		updateCamera(options);

		// Load shaders if not loaded yet
		loadShaders();
		
		// create palette if not created yet or changed between draws
		createPalette(palette);
	}

	bool isChunkDrawn(const ChunkCoord& coord) const {
		return m_minDrawnChunk.x <= coord.x && coord.x <= m_maxDrawnChunk.x
			&& m_minDrawnChunk.y <= coord.y && coord.y <= m_maxDrawnChunk.y;
	}

	static GLuint getShaderID(ShaderType shader) {
		if (shader == ShaderType::CellTexture)
			return s_chunkShader.getID();
		if (shader == ShaderType::Color)
			return s_colorShader.getID();
	}

	static void useShader(ShaderType shader) {
		if (shader == ShaderType::CellTexture)
			s_chunkShader.use();
		else if (shader == ShaderType::Color)
			s_colorShader.use();
	}

private:
	ChunkCoord m_minDrawnChunk;
	ChunkCoord m_maxDrawnChunk;
	
	void*	m_paletteBufferPtr;	// just for detecting change

	GLuint	m_vao = 0;
	GLuint	m_paletteTexture;
	bool	m_paletteTextureGenerated = false;

	inline static Shader s_chunkShader{};
	inline static Shader s_colorShader{};

	void createPalette(const ColorPalette& palette) {
		if (m_paletteBufferPtr == palette.getBuffer())
			return;

		std::cout << "creating color palette" << std::endl;
	
		// Generate texture
		if (!m_paletteTextureGenerated) {
			glGenTextures(1, &m_paletteTexture);
			m_paletteTextureGenerated = true;
		}

		// Bind texture
		GLuint textureUnit = 0;
		glActiveTexture(GL_TEXTURE0 + textureUnit);
		glBindTexture(GL_TEXTURE_1D, m_paletteTexture);

		// Upload palette as texture
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, palette.getSize().x * palette.getSize().y, 0, GL_RGBA, GL_UNSIGNED_BYTE, palette.getBuffer());

		// Set min and max filters
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// Set uniform
		GLuint uniformLocation = glGetUniformLocation(s_chunkShader.getID(), "colorPalette");
		glUniform1i(uniformLocation, textureUnit);

		m_paletteBufferPtr = (void*)palette.getBuffer();

		// Check for errors
		checkForErrors();
	}

	void updateCamera(const Options& options) {
		GLuint positionLocation = glGetUniformLocation(s_chunkShader.getID(), "camera.position");
		glUniform2f(positionLocation, (float)options.camera.position.x, (float)options.camera.position.y);

		GLuint zoomLocation = glGetUniformLocation(s_chunkShader.getID(), "camera.zoom");
		glUniform1f(zoomLocation, options.camera.zoom);

		GLuint aspectRatioLocation = glGetUniformLocation(s_chunkShader.getID(), "camera.aspectRatio");
		glUniform1f(aspectRatioLocation, (float)options.windowWidth / (float)options.windowHeight);
	}

	static void checkForErrors() {
		// Check for errors
		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
		}
	}

	void genVaoAndBind() {
		if (m_vao == 0) {
			std::cout << "creating world draw data" << std::endl;
			glGenVertexArrays(1, &m_vao);
		}
		
		glBindVertexArray(m_vao);
	}

	void cullChunks(Options options) {
		vec2 minCameraPos, maxCameraPos;
		options.camera.getMinWorldPos(options.windowWidth, options.windowHeight, minCameraPos.x, minCameraPos.y);
		options.camera.getMaxWorldPos(options.windowWidth, options.windowHeight, maxCameraPos.x, maxCameraPos.y);

		m_minDrawnChunk = CellCoord(minCameraPos).getChunkCoord();
		m_maxDrawnChunk = CellCoord(maxCameraPos).getChunkCoord();
	}

	static void loadShaders() {
		if (s_chunkShader.getID() == 0) {
			s_chunkShader.setSourceFiles("data/shaders/chunk_vs.glsl", "data/shaders/chunk_fs.glsl");
			s_chunkShader.create();
			checkForErrors();
		}
		else {
			s_chunkShader.recompileIfSourceChanged();
		}

		if (s_colorShader.getID() == 0) {
			s_colorShader.setSourceFiles("data/shaders/color_vs.glsl", "data/shaders/color_fs.glsl");
			s_colorShader.create();
			checkForErrors();
		}
		else {
			s_colorShader.recompileIfSourceChanged();
		}
	}
};
