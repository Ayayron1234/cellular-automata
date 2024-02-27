#pragma once
#include <array>

// Chunk size is 64x64
template <typename T>
class Chunk {
public:
	constexpr inline static int width = 64;
	constexpr inline static int height = 64;

	Chunk() 
		: m_data(new T[width * height])
	{ }

	const T& Read(int x, int y) const {
		return m_data[coordToIdx(x, y)];
	}

	void Write(int x, int y, T data) {
		m_data[coordToIdx(x, y)] = data;
	}

private:
	std::array<T, width * height> m_data;

	int coordToIdx(int x, int y) {
		return (y % height) * width + (x % width);
	}

};
