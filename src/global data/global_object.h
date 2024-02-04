#pragma once
#include <iostream>
#include <source_location>
#include <curand_kernel.h>

class GlobalObject {
protected:
	void checkError(const char* function, cudaError_t errorCode,
		bool shouldExit = true, const std::source_location location = std::source_location::current())
	{
		if (errorCode == cudaSuccess)
			return;

		std::cerr << "[GlobalReadWriteBuffer] " << function << " returned error code " << errorCode << ":\n\t" << cudaGetErrorString(errorCode) << std::endl;
		std::clog << "\tFile: "
			<< location.file_name() << '('
			<< location.line() << ':'
			<< location.column() << ") `"
			<< location.function_name() << std::endl;

		if (shouldExit)
			exit(1);
	}

};
