#pragma once 
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include "chunk.h"
#include <functional>

template <typename... Args>
class ChunkWorkerPool {
public:
	ChunkWorkerPool() {
		unsigned nThreads = std::thread::hardware_concurrency();

		std::cout << "Initializing chunk worker pool with " << nThreads << " threads. " << std::endl;

		for (int i = 0; i < nThreads; ++i)
			m_threads.push_back(new std::thread(workerThreadFunction, this));
	}

	void awaitAll() const {
		using namespace std::chrono_literals;
		//std::this_thread::sleep_for(1ms);

		std::unique_lock<std::mutex> lock(m_chunksMutex);
		m_done.wait(lock, [this] { return m_chunks.empty() || m_shutdown; });
	}

	void processChunk(Chunk* chunk) {
		enqueueChunk(chunk);
	}

	void updateParams(Args... args) {
		m_args = std::make_tuple(args...);
	}

	~ChunkWorkerPool() {
		std::cout << "Joining threads. " << std::endl;

		// Request threads to stop
		shutdown();

		// Join and clear threads
		for (auto thread : m_threads) {
			thread->join();
			delete thread;
		}
	}

private:
	using cv = std::condition_variable;

	std::vector<std::thread*>	m_threads;
	std::atomic_bool			m_shutdown = false;

	std::queue<Chunk*>			m_chunks{};
	mutable std::mutex			m_chunksMutex;
	mutable cv					m_chunkCountCondition;
	mutable cv					m_done;

	std::tuple<Args...>			m_args;

	Chunk* dequeueChunk() {
		Chunk* res;

		// Wait until queue is not empty
		std::unique_lock<std::mutex> lock{ m_chunksMutex };
		m_chunkCountCondition.wait(lock, [this] { return !m_chunks.empty() || m_shutdown; });

		// Handle shutdown
		if (m_shutdown)
			return nullptr;

		// Pop front of queue
		res = m_chunks.front();
		m_chunks.pop();
		
		return res;
	}

	void enqueueChunk(Chunk* chunk) {
		std::unique_lock<std::mutex> lock{ m_chunksMutex };

		m_chunks.push(chunk);

		lock.unlock();
		m_chunkCountCondition.notify_one();
	}

	void shutdown() {
		m_shutdown = true;
		m_chunkCountCondition.notify_all();
	}

	static void workerThreadFunction(ChunkWorkerPool* pool) {
		while (!pool->m_shutdown) {
			Chunk* chunk = pool->dequeueChunk();
			
			// Handle shutdown
			if (chunk == nullptr)
				break;

			// Process chunk
			std::apply(&Chunk::process, std::tuple_cat(std::make_tuple(chunk), pool->m_args));
			
			// Chunk processed
			pool->m_done.notify_one();
		}
	}
};

template <typename... Args>
inline constexpr auto makeChunkWorkerPool(void(Chunk::*)(Args...)) {
	return std::make_shared<ChunkWorkerPool<Args...>>();
}

template <typename T>
struct ChunkWorkerPoolT;

template <typename... Args>
struct ChunkWorkerPoolT<void(Chunk::*)(Args...)> {
	using type = std::shared_ptr<ChunkWorkerPool<Args...>>;
};
