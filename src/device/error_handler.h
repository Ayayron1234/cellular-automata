#pragma once
#include <iostream>
#include <source_location>
#include <curand_kernel.h>

template <typename T>
class ErrorHandler {
public:
	static void check(const char* cudaFunctionName, cudaError_t errorCode, const std::source_location location = std::source_location::current()) {
		if (errorCode == cudaSuccess)
			return;

		std::cerr << "[" << typeid(T).name() << "] " << cudaFunctionName << " returned error code " << errorCode 
			<< ":\n\t" << cudaGetErrorString(errorCode) << std::endl;
		std::cerr << "\tFile: "
			<< location.file_name() << '('
			<< location.line() << ':'
			<< location.column() << ") `"
			<< location.function_name() << std::endl;

		exit(1);
	}
};
