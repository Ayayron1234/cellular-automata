#pragma once
#include <iostream>

//std::cout << "Assertion faled at " << __FILE__ << "(" << __LINE__ << "):\n   ASSERT(" << #_e << ")\n" << std::endl;
//#define ASSERT_DEVICE(_e)\
//	if (!(_e)) {\
//		__debugbreak();\
//	}

template <typename T, T _OuterValue>
class Grid {
public:
    using cell_t = T;
    constexpr static T outerValue = _OuterValue;

    Grid(int width, int height)
        : m_width(width)
        , m_height(height)
        , m_data(new T[width * height])
    {
        std::cout << "Initialized grid. " << std::endl;
    }

    __device__ __host__ 
    int width() const {
        return m_width;
    }

    __device__ __host__ 
    int height() const {
        return m_height;
    }

    //T& operator[](int idx) {
    //    ASSERT_DEVICE((idx < m_width * m_height && idx > 0));
    //    return m_data[idx];
    //}
    //const T& operator[](int idx) const {
    //    if (idx >= m_width * m_height || idx < 0)
    //        return _OuterValue;
    //    return m_data[idx];
    //}

    __host__ __device__ 
    void set(int x, int y, cell_t value) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return;

        #ifndef  __CUDA_ARCH__
            m_data[y * width() + x] = value;
        #else
            m_cudaOutputBuffer[y * width() + x] = value;
        #endif
    }

    __host__ __device__
    const T get(int x, int y) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return _OuterValue;

        #ifndef __CUDA_ARCH__
            return m_data[y * m_width + x];
        #else
            return m_cudaInputBuffer[y * m_width + x];
        #endif
    }

    //__device__ const T& at_device(int x, int y) const {
    //    if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
    //        return _OuterValue;
    //    return m_cudaInputBuffer[y * m_width + x];
    //}
    //__device__ __host__ T* getInputBuffer() const {
    //    return m_cudaInputBuffer;
    //}
    //__device__ __host__ T* getOutputBuffer() const {
    //    return m_cudaOutputBuffer;
    //}

    cudaError_t setCudaBuffer() {
        T* buffer;

        cudaError_t cudaStatus = cudaMalloc((void**)&m_cudaInputBuffer, width() * height() * sizeof(cell_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&m_cudaOutputBuffer, width() * height() * sizeof(cell_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(m_cudaInputBuffer, m_data, width() * height() * sizeof(cell_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        // TODO: this should not be called. Implement in another way. 
        cudaStatus = cudaMemcpy(m_cudaOutputBuffer, m_data, width() * height() * sizeof(cell_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        return cudaStatus;
    }

    cudaError_t loadCudaBuffer() {
        cudaError_t cudaStatus = cudaMemcpy(m_data, m_cudaOutputBuffer, width() * height() * sizeof(cell_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }
        return cudaStatus;
    }

    void cleanupCudaBuffer() {
        cudaFree(m_cudaInputBuffer);
        cudaFree(m_cudaOutputBuffer);
    }

private:
    int m_width;
    int m_height;
    T* m_data;
    mutable T* m_cudaInputBuffer = nullptr;
    mutable T* m_cudaOutputBuffer = nullptr;

};

using ConwayGrid = Grid<int, 0>;