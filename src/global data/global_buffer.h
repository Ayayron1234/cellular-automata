#pragma once
#include <utility>
#include <iostream>
#include <curand_kernel.h>

#include "global_object.h"

template <typename T>
class GlobalBuffer : public GlobalObject {
public:
	void SetUp(size_t count, T* buffer = nullptr) {
		if (m_deviceBuffer != nullptr)
			CleanupDevice();
		if (m_hostBuffer != nullptr)
			delete[] m_hostBuffer;

		if (buffer == nullptr)
			buffer = new T[count];

		m_hostBuffer = buffer;
		m_bufferSize = sizeof(T) * count;

		cudaError_t cudaStatus = cudaSetDevice(0);
		checkError("cudaSetDevice", cudaStatus);

		// Allocate the buffer on the gpu
		cudaStatus = cudaMalloc((void**)&m_deviceBuffer, Size());
		checkError("cudaMalloc", cudaStatus);

		// Copy data from the host buffer to the device buffer
		cudaStatus = cudaMemcpy(m_deviceBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	void CleanupDevice() {
		if (m_deviceBuffer != nullptr)
			cudaFree(m_deviceBuffer);

		m_deviceBuffer = nullptr;
	}

	size_t Size() const {
		return m_bufferSize;
	}

private:
	size_t m_bufferSize;
	T* m_hostBuffer;
	T* m_deviceBuffer;

	void checkErrorAndCleanup(const char* function, cudaError_t errorCode,
		bool shouldExit = true, const std::source_location location = std::source_location::current()) {

		if (errorCode == cudaSuccess)
			return;

		CleanupDevice();
		checkError(function, errorCode, shouldExit, location);
	}

};

// Double buffered object for working with data on the cpu and gpu simultaneously. 
// Contains utility functions for syncing and accessing data. 
// In order for the syncing to work the T type's operator=(const T&) function has to be implemented correctly. 
template <typename T>
class GlobalReadWriteBuffer : public GlobalObject {
public:
	void SetUp(size_t count, T* buffer = nullptr) {
		// Free allocated memory
		if (m_deviceReadBuffer != nullptr && m_deviceWriteBuffer != nullptr)
			CleanupDevice();
		if (m_hostBuffer != nullptr)
			delete[] m_hostBuffer;

		// Allocate host buffer if not provided
		if (buffer == nullptr)
			buffer = new T[count];

		m_hostBuffer = buffer;
		m_bufferSize = sizeof(T) * count;

		cudaError_t cudaStatus = cudaSetDevice(0);
		checkError("cudaSetDevice", cudaStatus);

		// Allocate the read buffer on the gpu
		cudaStatus = cudaMalloc((void**)&m_deviceReadBuffer, Size());
		checkError("cudaMalloc", cudaStatus);

		// Allocate the write buffer on the gpu
		cudaStatus = cudaMalloc((void**)&m_deviceWriteBuffer, Size());
		checkError("cudaMalloc", cudaStatus);

		// Copy data from the host buffer to the device read buffer
		cudaStatus = cudaMemcpy(m_deviceReadBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Sync the data on the cpu's object from the gpu's
	void SyncFromDevice() {
		cudaError_t cudaStatus = cudaMemcpy(m_hostBuffer, m_deviceWriteBuffer, Size(), cudaMemcpyDeviceToHost);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Sync the data on the gpu's object from the cpu's
	void SyncFromHost() {
		cudaError_t cudaStatus = cudaMemcpy(m_deviceReadBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Swap the read and write buffers on the device
	void SwapDeviceBuffers() {
		T* swapTmp = m_deviceReadBuffer;
		m_deviceReadBuffer = m_deviceWriteBuffer;
		m_deviceWriteBuffer = m_deviceReadBuffer;
	}

	// When called from the cpu returns the cpu's buffer and when called from the gpu returns the gpu's
	__host__ __device__
	const T& Read(int index) const {
		#ifndef __CUDA_ARCH__
		return m_hostBuffer[index];
		#else
		return m_deviceReadBuffer[index];
		#endif
	}

	// When called from the cpu updates the cpu's buffer and when called from the gpu updates the gpu's
	__host__ __device__
	void Write(int index, const T& data) {
		#ifndef __CUDA_ARCH__
		m_hostBuffer[index] = data;
		#else
		m_deviceWriteBuffer[index] = data;
		#endif
	}

	// Wait for the gpu to finish operations
	void AwaitDevice() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		checkError("cudaDeviceSynchronize", cudaStatus);
	}

	// Frees memory allocated on the gpu
	void CleanupDevice() {
		if (m_deviceReadBuffer != nullptr)
			cudaFree(m_deviceReadBuffer);
		
		if (m_deviceWriteBuffer != nullptr)
			cudaFree(m_deviceWriteBuffer);
		
		m_deviceReadBuffer = nullptr;
		m_deviceWriteBuffer = nullptr;
	}

	size_t Size() const {
		return m_bufferSize;
	}

private:
	size_t m_bufferSize;
	T* m_hostBuffer;
	T* m_deviceReadBuffer;
	T* m_deviceWriteBuffer;

	void checkErrorAndCleanup(const char* function, cudaError_t errorCode,
		bool shouldExit = true, const std::source_location location = std::source_location::current()) {

		if (errorCode == cudaSuccess)
			return;

		CleanupDevice();
		checkError(function, errorCode, shouldExit, location);
	}

};
