#pragma once
#include <iostream>

//std::cout << "Assertion faled at " << __FILE__ << "(" << __LINE__ << "):\n   ASSERT(" << #_e << ")\n" << std::endl;
#define ASSERT_DEVICE(_e)\
	if (!(_e)) {\
		__debugbreak();\
	}

template <typename T, T _OuterValue>
class Grid {
public:
    using value_t = T;
    constexpr static T outerValue = _OuterValue;

    Grid(int width, int height)
        : m_width(width)
        , m_height(height)
        , m_data(new T[width * height])
    {
        std::cout << "Initialized grid. " << std::endl;
    }

    __device__ __host__ int width() const {
        return m_width;
    }

    __device__ __host__ int height() const {
        return m_height;
    }

    __device__ __host__ T& operator[](int idx) {
        ASSERT_DEVICE((idx < m_width * m_height && idx > 0));
        return m_data[idx];
    }

    __device__ __host__ const T& operator[](int idx) const {
        if (idx >= m_width * m_height || idx < 0)
            return _OuterValue;
        return m_data[idx];
    }


    void set(int x, int y, value_t value) {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return;
        m_data[y * width() + x] = value;
    }

    __host__ const T& at(int x, int y) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return _OuterValue;
        return m_data[y * m_width + x];
    }

    __device__ const T& at_device(int x, int y) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return _OuterValue;
        return m_cudaBuffer[y * m_width + x];
    }

    __device__ __host__ T* getBuffer() const {
        return m_cudaBuffer;
    }

    cudaError_t setCudaBuffer(T* buffer) {
        m_cudaBuffer = buffer;

        cudaError_t cudaStatus = cudaMemcpy(m_cudaBuffer, m_data, width() * height() * sizeof(value_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }
        return cudaStatus;
    }

    cudaError_t loadCudaBuffer() {
        cudaError_t cudaStatus = cudaMemcpy(m_data, m_cudaBuffer, width() * height() * sizeof(value_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }
        return cudaStatus;
    }

private:
    int m_width;
    int m_height;
    T* m_data;
    mutable T* m_cudaBuffer = nullptr;

};

using ConwayGrid = Grid<int, 0>;