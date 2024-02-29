#pragma once
#include <iostream>
#include "utils/options.h"
#include "device_helpers.h"
#include "cellular-storage.h"

template <typename T>
class Grid {
public:
    using cell_t = T;

    Grid(int width, int height, cell_t* buffer = nullptr)
        : m_width(width)
        , m_height(height)
    {
        //m_hostBuffer.Init(width * height, buffer);
        //m_globalBuffer.Init(width * height, m_hostBuffer.GetWriteBuffer(), false);
        m_globalBuffer.Init(width * height, buffer);
    }

    __host__ __device__
    void Set(int x, int y, const cell_t& value) {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return;

        //#ifndef __CUDA_ARCH__
        //return m_hostBuffer.Write(y * Width() + x, value);
        //#else
        //return m_globalBuffer.Write(y * Width() + x, value);
        //#endif
        m_globalBuffer.Write(y * Width() + x, value);
    }

    __host__ __device__
    cell_t Get(int x, int y, const cell_t& outerValue = { 0 }) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return outerValue;

        //#ifndef __CUDA_ARCH__
        //return m_hostBuffer.Read(y * m_width + x);
        //#else
        //return m_globalBuffer.Read(y * m_width + x);
        //#endif
        return m_globalBuffer.Read(y * m_width + x);
    }

    void UpdateState() {
        //m_globalBuffer.SyncFromHost(false);
        //m_hostBuffer.Sync();
        //m_globalBuffer.AwaitDevice();
        m_globalBuffer.SyncFromHost(true);
    }

    __host__ __device__
    int Width() {
        return m_width;
    }

    __host__ __device__
    int Height() {
        return m_height;
    }

    void Save(const char* path) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            std::cout << "[Grid] Coundn't save to file: " << path << std::endl;
            return;
        }

        ofs.write((char*)&m_width, sizeof(m_width));
        ofs.write((char*)&m_height, sizeof(m_height));

        ofs.write((char*)&m_globalBuffer.Read(0), Width() * Height() * sizeof(T));
    }

    void Load(const char* path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            std::cout << "[Grid] Coundn't load from file: " << path << std::endl;
            return;
        }

        ifs.read((char*)&m_width, sizeof(m_width));
        ifs.read((char*)&m_height, sizeof(m_height));

        cell_t* buffer = new cell_t[Width() * Height()];
        ifs.read((char*)buffer, Width() * Height() * sizeof(cell_t));

        //m_hostBuffer.Init(Width() * Height(), buffer);
        //m_globalBuffer.Init(Width() * Height(), m_hostBuffer.GetWriteBuffer(), false);
        m_globalBuffer.Init(Width() * Height(), buffer);
    }

private:
    int m_width, m_height;
    //ReadWriteBuffer<cell_t> m_hostBuffer{};
    GlobalBuffer<cell_t> m_globalBuffer{};

};
