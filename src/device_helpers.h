#pragma once
#include <iostream>
#include <source_location>
#include <curand_kernel.h>
#include <utility>
#include <stack>
#include <stdlib.h>

class GlobalObject {
protected:
	static void checkError(const char* function, cudaError_t errorCode,
		bool shouldExit = true, const std::source_location location = std::source_location::current())
	{
		if (errorCode == cudaSuccess)
			return;

		std::cerr << "[GlobalObject] " << function << " returned error code " << errorCode << ":\n\t" << cudaGetErrorString(errorCode) << std::endl;
		std::clog << "\tFile: "
			<< location.file_name() << '('
			<< location.line() << ':'
			<< location.column() << ") `"
			<< location.function_name() << std::endl;

		if (shouldExit)
			exit(1);
	}

};

template <typename T>
concept GlobalDataHandler_C = requires(T t) {
	{ t.SyncFromDevice(true) };
	{ t.SyncFromHost(true) };
};

template <typename T>
class GlobalBuffer : public GlobalObject {
public:
	void Init(size_t count, T* buffer = nullptr, bool doFreeHostBuffer = true) {
		if (m_deviceBuffer != nullptr)
			CleanupDevice();
		if (doFreeHostBuffer && m_hostBuffer != nullptr)
			delete[] m_hostBuffer;

		if (buffer == nullptr) {
			buffer = new T[count];
			memset(buffer, 0, count * sizeof(T));
		}

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

	// Sync the data on the cpu's object from the gpu's
	void SyncFromDevice(bool async = false) {
		cudaError_t cudaStatus;
		if (async)
			cudaStatus = cudaMemcpyAsync(m_hostBuffer, m_deviceBuffer, Size(), cudaMemcpyDeviceToHost);
		else
			cudaStatus = cudaMemcpy(m_hostBuffer, m_deviceBuffer, Size(), cudaMemcpyDeviceToHost);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Sync the data on the gpu's object from the cpu's
	void SyncFromHost(bool async = false) {
		cudaError_t cudaStatus;
		if (async)
			cudaStatus = cudaMemcpyAsync(m_deviceBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		else
			cudaStatus = cudaMemcpy(m_deviceBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Wait for the gpu to finish operations
	void AwaitDevice() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		checkError("cudaDeviceSynchronize", cudaStatus);
	}

	__host__ __device__
	const T& Read(int index) const {
		#ifndef __CUDA_ARCH__
		return m_hostBuffer[index];
		#else
		return m_deviceBuffer[index];
		#endif
	}

	__host__ __device__
	void Write(int index, const T& data) {
		#ifndef __CUDA_ARCH__
		m_hostBuffer[index] = data;
		#else
		m_deviceBuffer[index] = data;
		#endif
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
	T* m_hostBuffer = nullptr;
	T* m_deviceBuffer = nullptr;

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
	void Init(size_t count, T* buffer = nullptr) {
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
	void SyncFromDevice(bool async = false) {
		cudaError_t cudaStatus;
		if (async)
			cudaStatus = cudaMemcpyAsync(m_hostBuffer, m_deviceWriteBuffer, Size(), cudaMemcpyDeviceToHost);
		else
			cudaStatus = cudaMemcpy(m_hostBuffer, m_deviceWriteBuffer, Size(), cudaMemcpyDeviceToHost);
		checkErrorAndCleanup("cudaMemcpy", cudaStatus);
	}

	// Sync the data on the gpu's object from the cpu's
	void SyncFromHost(bool async = false) {
		cudaError_t cudaStatus;
		if (async)
			cudaStatus = cudaMemcpyAsync(m_deviceReadBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
		else
			cudaStatus = cudaMemcpy(m_deviceReadBuffer, m_hostBuffer, Size(), cudaMemcpyHostToDevice);
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

class IKernel : public GlobalObject {
public:
	static void AwaitDevice() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		checkError("cudaDeviceSynchronize", cudaStatus);
	}

	static void SyncAllFromDevice(bool async = false) {
		AwaitDevice();

		while (!s_kernelsToSyncFromDevice.empty()) {
			s_kernelsToSyncFromDevice.top()->syncFromDevice();
			s_kernelsToSyncFromDevice.pop();
		}

		if (!async)
			AwaitDevice();
	}

protected:
	inline static std::stack<IKernel*> s_kernelsToSyncFromDevice{};

	virtual void syncFromDevice() = 0;
};

template <GlobalDataHandler_C T, class... Args>
class Kernel : public IKernel {
public:
	Kernel(void(*kernelFncPtr)(T, Args...), T* data)
		: m_function(kernelFncPtr)
		, m_data(data)
	{ }

	void Execute(Args ...args) {
		m_function(*m_data, args...);

		cudaError_t cudaStatus = cudaGetLastError();
		checkError("cudaGetLastError", cudaStatus);

		s_kernelsToSyncFromDevice.push(this);
	}

	void ExecuteAndSync(Args ...args) {
		m_function(*m_data, args...);

		cudaError_t cudaStatus = cudaGetLastError();
		checkError("cudaGetLastError", cudaStatus);

		AwaitDevice();

		syncFromDevice();
	}


private:
	void(*m_function)(T, Args...);
	T* m_data;

	using SyncFunction = void();

	virtual void syncFromDevice() override {
		m_data->SyncFromDevice(true);
	}

};
