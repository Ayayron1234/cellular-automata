#pragma once
#include "../utils.h"
#include "../device.h"
#include "cell.h"

class ColorPalette {
public:
	ColorPalette(const Bitmap& image) {
		m_width = image.width();
		m_height = image.height();
		init(image);
	}

	void update(const Bitmap& image) {
		m_width = image.width();
		m_height = image.height();
		init(image);
	}

	IO::RGB getColor(Cell cell) const {
		return m_buffer[(unsigned)cell.type * m_width + cell.shade];
	}

	IO::RGB* getBuffer() {
		return m_buffer;
	}

	const IO::RGB* getBuffer() const {
		return m_buffer;
	}

	vec2 getSize() const {
		return { (Float)m_width, (Float)m_height };
	}

private:
	unsigned m_width = 0;
	unsigned m_height = 0;
	IO::RGB* m_buffer = nullptr;
	DeviceBuffer<IO::RGB> m_deviceBuffer;

	void init(const Bitmap& image) {
		if (m_buffer != nullptr)
			delete[] m_buffer;

		m_buffer = new IO::RGB[m_width * m_height];
		for (int y = 0; y < m_height; ++y)
			for (int x = 0; x < m_width; ++x) {
				IO::RGB pixel;
				*(unsigned*)&pixel = image.getRGBA(x, y);
				m_buffer[y * m_width + x] = pixel;
			}
	}

};
