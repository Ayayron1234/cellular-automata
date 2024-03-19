#pragma once
#include "../utils.h"
#include "../device.h"
#include "cell.h"

class ColorPalette {
public:
	ColorPalette(const Bitmap& image) {
		m_width = image.width();
		m_height = image.height();
		uploadToBuffer(image);
	}

	void update(const Bitmap& image) {
		m_width = image.width();
		m_height = image.height();
		uploadToBuffer(image);
	}

	__device__ __host__
	IO::RGB getColor(Cell cell) const {
#ifndef __CUDA_ARCH__
		return m_hostBuffer[(unsigned)cell.type * m_width + cell.shade];
#else
		return m_deviceBuffer[(unsigned)cell.type * m_width + cell.shade];
#endif
	}

private:
	unsigned m_width = 0;
	unsigned m_height = 0;
	IO::RGB* m_hostBuffer = nullptr;
	DeviceBuffer<IO::RGB> m_deviceBuffer;

	void uploadToBuffer(const Bitmap& image) {
		if (m_hostBuffer != nullptr)
			delete[] m_hostBuffer;

		m_hostBuffer = new IO::RGB[m_width * m_height];
		for (int y = 0; y < m_height; ++y)
			for (int x = 0; x < m_width; ++x) {
				IO::RGB pixel;
				*(unsigned*)&pixel = image.getRGBA(x, y);
				m_hostBuffer[y * m_width + x] = pixel;
			}

		m_deviceBuffer.upload(m_hostBuffer, m_width * m_height);
		delete[] m_hostBuffer;
	}

};
