#pragma once
#include "global_object.h"

template <class T, class... Args>
class Kernel : public GlobalObject {
public:
	Kernel(void(*kernelFncPtr)(T, Args...), T* data)
		: m_function(kernelFncPtr)
		, m_data(data)
	{ }

	void Execute(Args ...args) {
		m_function(*m_data, args...);

		cudaError_t cudaStatus = cudaGetLastError();
		checkError("cudaGetLastError", cudaStatus);

		cudaStatus = cudaDeviceSynchronize();
		checkError("cudaDeviceSynchronize", cudaStatus);

	}

	void AwaitDevice() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		checkError("cudaDeviceSynchronize", cudaStatus);
	}

private:
	void(*m_function)(T, Args...);
	T* m_data;

};
