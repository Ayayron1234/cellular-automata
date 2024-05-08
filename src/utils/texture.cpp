#include "texture.h"
#include "bitmap.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture Texture::loadFromFile(const char* path, int minFilter, int magFilter)
{
    Texture texture;

    // Load from file
    Bitmap bitmap(path);
    if (!bitmap.loaded())
        return texture;

    std::shared_ptr<IO::RGB[]> buffer = bitmap.getRGBABuffer();
    texture.m_width = bitmap.width();
    texture.m_height = bitmap.height();

    // Create a OpenGL texture identifier
    glBindTexture(GL_TEXTURE_2D, texture.id());

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bitmap.width(), bitmap.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)buffer.get());

    texture.m_empty = false;
    return texture;
}
