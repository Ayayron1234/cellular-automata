#pragma once
#include "../graphics_headers.h"
#include <memory>

class TextureID {
public:
	TextureID() {
		glGenTextures(1, &m_id);
	}

	~TextureID() {
		glDeleteTextures(1, &m_id);
	}

	operator GLuint() const {
		return m_id;
	}

private:
	GLuint m_id;

};

class Texture {
public:
	Texture()
		: m_id(std::make_shared<TextureID>())
	{ }

	static Texture loadFromFile(const char* path, int minFilter = GL_LINEAR, int magFilter = GL_LINEAR);

	GLuint id() const {
		return *m_id;
	}

	unsigned width() const {
		return m_width;
	}

	unsigned height() const {
		return m_height;
	}

	bool empty() const {
		return m_empty;
	}

private:
	bool						m_empty = true;

	std::shared_ptr<TextureID>	m_id;
	unsigned					m_width = 0;
	unsigned					m_height = 0;

};
