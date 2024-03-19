#pragma once
#include <curand_kernel.h>
#include <stack>
#include "error_handler.h"

template <typename T>
concept GlobalData_C = requires(T t) {
	{ t.upload((size_t)0) };
	{ t.uploadAsync((size_t)0) };
	{ t.download() };
	{ t.downloadAsync() };
	{ t.alloc((size_t)0) };
};

template <GlobalData_C T, class... Args>
class Kernel {
private:
	using error = ErrorHandler<Kernel>;

public:
	Kernel(void(*kernelFncPtr)(T, Args...), T* data)
		: m_function(kernelFncPtr)
		, m_data(data)
	{ }

	T& data() {
		return *m_data;
	}

	void execute(Args ...args) {
		m_function(*m_data, args...);

		cudaError_t cudaStatus = cudaGetLastError();
		error::check("cudaGetLastError", cudaStatus);

		awaitDevice();
		m_data->download();
	}

private:
	void(*m_function)(T, Args...);
	T* m_data;

	using SyncFunction = void();

	static void awaitDevice() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		error::check("cudaDeviceSynchronize", cudaStatus);
	}
};
