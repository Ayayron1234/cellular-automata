#pragma once
#include "device_buffer.h"

template <typename T>
class GlobalBuffer {
public:
	__host__
	void changeBuffer(T* hostBuffer, size_t count) {
		m_hostBuffer = hostBuffer;
		alloc(count);
	}

	__host__
	void alloc(size_t count) {
		m_deviceBuffer.alloc(count);
	}
	
	__host__
	void freeDevice() {
		m_deviceBuffer.freeDevice();
	}

	__host__
	void upload(size_t count) {
		m_deviceBuffer.upload(m_hostBuffer, count);
	}

	__host__
	void uploadAsync(size_t count) {
		m_deviceBuffer.uploadAsync(m_hostBuffer, count);
	}

	__host__
	void download() {
		m_deviceBuffer.download(m_hostBuffer);
	}

	__host__
	bool downloadAsync() {
		m_deviceBuffer.downloadAsync(m_hostBuffer);
	}

#ifndef __CUDA_ARCH__
	__host__
	T& operator[](size_t index) {
		return m_hostBuffer[index];
	}
#else
	__device__
	T& operator[](size_t index) {
		return m_deviceBuffer[index];
	}
#endif

#ifndef __CUDA_ARCH__
	__host__
	const T& operator[](size_t index) const {
		return m_hostBuffer[index];
	}
#else
	__device__
	const T& operator[](size_t index) const {
		return m_deviceBuffer[index];
	}
#endif

	__host__ __device__
	size_t count() const {
		return m_deviceBuffer.count();
	}

	__host__ __device__
	bool isAllocated() const {
		return m_deviceBuffer.isAllocated();
	}

	GlobalBuffer(T* hostBuffer, size_t count)
		: m_deviceBuffer()
		, m_hostBuffer(hostBuffer)
	{ 
		m_deviceBuffer.alloc(count);
	}

private:
	DeviceBuffer<T> m_deviceBuffer;
	T* m_hostBuffer;

	__host__
	void reallocIfCountChanged(size_t count) {
		if (count != m_deviceBuffer.count()) {
			m_deviceBuffer.alloc(count);
			std::cout << "allocating:" << count << std::endl;
		}
	}
};
