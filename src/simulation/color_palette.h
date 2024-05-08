#pragma once
#include "../utils.h"
#include "../device.h"
#include "cell.h"

class ColorPalette {
public:
	ColorPalette(const BitmapFileResource& file)
		: m_width(file.value().width())
		, m_height(file.value().height())
		, m_file(file)
	{
		init();
	}

	//void update(const Bitmap& image) {
	//	m_width = image.width();
	//	m_height = image.height();
	//	init(image);
	//}

	IO::RGB getColor(Cell cell) {
		if (m_file.updated()) {
			m_file.load();
			init();
		}

		return m_buffer[(unsigned)cell.type * m_width + cell.shade];
	}

	IO::RGB* getBuffer() {
		if (m_file.updated()) {
			m_file.load();
			init();
		}

		return m_buffer.get();
	}

	//const IO::RGB* getBuffer() const {
	//	return m_buffer.get();
	//}

	vec2 getSize() const {
		return { (Float)m_width, (Float)m_height };
	}

	static ColorPalette loadFromFile(const std::string& path) {
		return ColorPalette(BitmapFileResource("data/color_palett.bmp"));
	}

private:
	unsigned m_width = 0;
	unsigned m_height = 0;

	BitmapFileResource			m_file;
	std::shared_ptr<IO::RGB[]>	m_buffer;

	//IO::RGB* m_buffer = nullptr;

	void init() {
		std::shared_ptr<IO::RGB[]> newBuffer = std::make_shared<IO::RGB[]>(m_width * m_height);
		m_buffer.swap(newBuffer);

		//m_buffer = new IO::RGB[m_width * m_height];
		for (int y = 0; y < m_height; ++y)
			for (int x = 0; x < m_width; ++x) {
				IO::RGB pixel;
				*(unsigned*)&pixel = m_file.value().getRGBA(x, y);
				m_buffer[y * m_width + x] = pixel;
			}
	}

};
