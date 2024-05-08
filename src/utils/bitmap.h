#pragma once
#include <iostream>
#include <fstream>
#include <intrin.h>
#include <memory>
#include "IO.h"

class Bitmap {
private:
    constexpr static char signature[2] = { 'B', 'M' };

    struct BitmapFileHeader {
        short   signature;
        short   _fileSize[2];
        short : 16; short : 16;             // reserved
        short   _offsetToPixelArray[2];

        int fileSize() const {
            return *(unsigned int*)&_fileSize;
        }

        int offsetToPixelArray() const {
            return *(unsigned int*)&_offsetToPixelArray;
        }
    };

    struct DIBHeader {
        unsigned int dibHeaderSize;
        unsigned int imageWidth;
        unsigned int imageHeight;
        unsigned short : 16;
        unsigned short bitsPerPixel;
        unsigned int : 32;
        unsigned int imageSize;
        unsigned int xPixelsPerMeter;
        unsigned int yPixelsPerMeter;
        unsigned int : 32;
        unsigned int : 32;
        unsigned int redChannelBitmask;
        unsigned int greenChannelBitmask;
        unsigned int blueChannelBitmask;
        unsigned int alphaChannelBitmask;
    };

public:
	Bitmap(const char* path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open())
            return;

        readHeader(ifs);

        m_pixelArray = std::make_shared<unsigned char[]>(m_dibHeader.imageSize);
        ifs.seekg(ifs.beg);
        ifs.seekg(m_fileHeader.offsetToPixelArray());
        ifs.read((char*)m_pixelArray.get(), m_dibHeader.imageSize);

        setChanelOffsets();

        m_loaded = true;
	}

    Bitmap(const Bitmap&) = default;

    unsigned int getOriginal(unsigned int x, unsigned int y) const {
        if (x >= m_dibHeader.imageWidth || y >= m_dibHeader.imageHeight)
            return 0;

        unsigned int pixelSize = m_dibHeader.bitsPerPixel / 8;
        unsigned int padding = (4 - ((m_dibHeader.bitsPerPixel / 8 * m_dibHeader.imageWidth) % 4)) % 4;
        unsigned int rowSize = m_dibHeader.imageWidth * pixelSize + padding;

        unsigned int pixel;
        unsigned int position = (m_dibHeader.imageHeight - y - 1) * rowSize + x * pixelSize;

        memcpy(&pixel, &m_pixelArray.get()[position], pixelSize);
        return pixel;
    }

    unsigned int getRGBA(unsigned int x, unsigned int y) const {
        unsigned int original = getOriginal(x, y);
        unsigned int rgba = 0;
        ((unsigned char*)&rgba)[0] = ((unsigned char*)&original)[m_rChanelOffset];
        ((unsigned char*)&rgba)[1] = ((unsigned char*)&original)[m_gChanelOffset];
        ((unsigned char*)&rgba)[2] = ((unsigned char*)&original)[m_bChanelOffset];
        ((unsigned char*)&rgba)[3] = ((unsigned char*)&original)[m_aChanelOffset];
        return rgba;
    }

    std::shared_ptr<IO::RGB[]> getRGBABuffer() const {
        std::shared_ptr<IO::RGB[]> buffer = std::make_shared<IO::RGB[]>(m_dibHeader.imageWidth * m_dibHeader.imageHeight);

        for (int x = 0; x < m_dibHeader.imageWidth; ++x)
            for (int y = 0; y < m_dibHeader.imageHeight; ++y) {
                IO::RGB color;
                *(unsigned*)&color = getRGBA(x, y);
                buffer[y * m_dibHeader.imageWidth + x] = color;
            }

        return buffer;
    }

    unsigned int getARGB(unsigned int x, unsigned int y) const {
        unsigned int original = getOriginal(x, y);
        unsigned int argb = 0;
        ((unsigned char*)&argb)[0] = ((unsigned char*)&original)[m_aChanelOffset];
        ((unsigned char*)&argb)[1] = ((unsigned char*)&original)[m_rChanelOffset];
        ((unsigned char*)&argb)[2] = ((unsigned char*)&original)[m_gChanelOffset];
        ((unsigned char*)&argb)[3] = ((unsigned char*)&original)[m_bChanelOffset];
        return argb;
    }

    __host__ __device__
    unsigned width() const {
        return m_dibHeader.imageWidth;
    }

    __host__ __device__
    unsigned height() const {
        return m_dibHeader.imageHeight;
    }

    bool loaded() const {
        return m_loaded;
    }

private:

    std::shared_ptr<unsigned char[]> m_pixelArray;

    bool                m_loaded = false;
    BitmapFileHeader    m_fileHeader;
    DIBHeader           m_dibHeader;

    int m_rChanelOffset = 0;
    int m_gChanelOffset = 1;
    int m_bChanelOffset = 2;
    int m_aChanelOffset = 3;

    void readHeader(std::ifstream& ifs) {
        ifs.read((char*)&m_fileHeader, sizeof(BitmapFileHeader));
        ifs.read((char*)&m_dibHeader, sizeof(DIBHeader));
    }

    void setChanelOffsets() {
        unsigned int mask = m_dibHeader.redChannelBitmask;
        unsigned int offset = 0;
        while (mask != 0) { offset += 1; mask = mask >> 8; }
        m_rChanelOffset = offset - 1;

        mask = m_dibHeader.greenChannelBitmask;
        offset = 0;
        while (mask != 0) { offset += 1; mask = mask >> 8; }
        m_gChanelOffset = offset - 1;

        mask = m_dibHeader.blueChannelBitmask;
        offset = 0;
        while (mask != 0) { offset += 1; mask = mask >> 8; }
        m_bChanelOffset = offset - 1;

        mask = m_dibHeader.alphaChannelBitmask;
        offset = 0;
        while (mask != 0) { offset += 1; mask = mask >> 8; }
        m_aChanelOffset = offset - 1;
    }

    friend std::ostream& operator<<(std::ostream& os, Bitmap::BitmapFileHeader header);
};

inline std::ostream& operator<<(std::ostream& os, Bitmap::BitmapFileHeader header) {
    os << "Bitmap Header: ";
    os << "\n\tsignature:" << ((char*)&header.signature)[0] << ((char*)&header.signature)[1];
    os << "\n\tfile size:" << header.fileSize();
    os << "\n\toffset to pixel array:" << header.offsetToPixelArray();
    return os;
}
