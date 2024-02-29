#pragma once
#include <memory>

template <typename T>
class ReadWriteBuffer {
public:
	void Init(size_t count, T* buffer = nullptr) {
		Destroy();

		m_bufferSize = count * sizeof(T);

		if (buffer == nullptr) {
			buffer = new T[count];
			memset(buffer, 0, Size());
		}

		m_readBuffer = buffer;
		m_writeBuffer = new T[count];
		memcpy(m_writeBuffer, m_readBuffer, Size());
	}

	const T& Read(int index) const {
		return m_readBuffer[index];
	}
	
	void Write(int index, const T& data) {
		m_writeBuffer[index] = data;
	}

	void Sync() {
		memcpy(m_readBuffer, m_writeBuffer, Size());
	}

	T* GetWriteBuffer() {
		return m_writeBuffer;
	}

	size_t Size() {
		return m_bufferSize;
	}

	void Destroy() {
		if (m_readBuffer != nullptr)
			delete[] m_readBuffer;
		if (m_writeBuffer != nullptr)
			delete[] m_writeBuffer;
	}

private:
	size_t m_bufferSize;	// Size of one of the buffers in bytes
	T* m_readBuffer;
	T* m_writeBuffer;

};
