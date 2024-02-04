#pragma once
#include <iostream>
#include "global_buffer.h"
#include "../options.h"

template <typename T>
class Grid {
public:
    using cell_t = T;

    Grid(int width, int height, cell_t* buffer = nullptr)
        : m_width(width)
        , m_height(height)
    {
        m_buffer.SetUp(width * height, buffer);
    }

    __host__ __device__
    void Set(int x, int y, const cell_t& value) {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return;

        m_buffer.Write(y * Width() + x, value);
    }

    __host__ __device__
    cell_t Get(int x, int y, const cell_t& outerValue = { 0 }) const {
        if (!(0 <= x && x < m_width && 0 <= y && y < m_height))
            return outerValue;

        return m_buffer.Read(y * m_width + x);
    }

    void NextState(Options options) {
        m_buffer.SyncFromHost();

        advanceState(options);

        m_buffer.AwaitDevice();
        m_buffer.SyncFromDevice();
    }

    void Update() {
        m_buffer.SyncFromHost();
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

        ofs.write((char*)&m_buffer.Read(0), Width() * Height() * sizeof(T));
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

        m_buffer.SetUp(Width() * Height(), buffer);
    }

private:
    int m_width, m_height;
    GlobalReadWriteBuffer<cell_t> m_buffer{};
   
    void advanceState(Options options);

};
