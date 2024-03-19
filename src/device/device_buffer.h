#pragma once
#include "error_handler.h"

template <typename T>
class DeviceBuffer {
private:
	using error = ErrorHandler<DeviceBuffer>;

public:
	__host__
	void alloc(size_t count) {
		freeDevice();

		cudaError_t cudaStatus = cudaSetDevice(0);
		error::check("cudaSetDevice", cudaStatus);

		m_count = count;

		// Allocate the buffer on the gpu
		cudaStatus = cudaMalloc((void**)&m_deviceBuffer, sizeof(T) * this->count());
		error::check("cudaMalloc", cudaStatus);
	}

	__host__
	void freeDevice() {
		if (m_deviceBuffer != nullptr)
			cudaFree(m_deviceBuffer);

		m_deviceBuffer = nullptr;
	}

	__host__
	void upload(T* src, size_t count) {
		if (m_count != count)
			alloc(count);

		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(m_deviceBuffer, src, sizeof(T) * this->count(), cudaMemcpyHostToDevice);
		error::check("cudaMemcpy", cudaStatus);
	}

	__host__
	void uploadAsync(T* buffer, size_t count) {
		if (m_count != count)
			alloc(count);

		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpyAsync(m_deviceBuffer, buffer, sizeof(T) * this->count(), cudaMemcpyHostToDevice);
		error::check("cudaMemcpy", cudaStatus);
	}

	__host__ 
	void download(T* dst) {
		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(dst, m_deviceBuffer, sizeof(T) * count(), cudaMemcpyDeviceToHost);
		error::check("cudaMemcpy", cudaStatus);
	}

	__host__
	void downloadAsync(T* dst) {
		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpyAsync(dst, m_deviceBuffer, sizeof(T) * count(), cudaMemcpyDeviceToHost);
		error::check("cudaMemcpy", cudaStatus);
	}

	__device__
	T& operator[](size_t index) {
		return m_deviceBuffer[index];
	}

	__device__
	const T& operator[](size_t index) const {
		return m_deviceBuffer[index];
	}

	__host__ __device__
	size_t count() const {
		return m_count;
	}

	__host__ __device__
	bool isAllocated() const {
		return m_deviceBuffer != nullptr;
	}

protected:
	size_t m_count		= 0ull;
	T* m_deviceBuffer	= nullptr;

};
