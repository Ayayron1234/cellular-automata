#pragma once 
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include "chunk.h"
#include <functional>

class ChunkTaskQueue {
public:
	void enqueue(Chunk* chunk) {
		std::lock_guard<std::mutex> queueLock{ m_lock };
		m_queue.push(chunk);
		m_cv.notify_one();
	}

	Chunk* dequeue() {
		std::unique_lock<std::mutex> queue_lock{ m_lock };
		while (m_queue.empty()) {
			m_cv.wait(queue_lock);
		}
		
		Chunk* res = m_queue.front();
		m_queue.pop();
		return res;
	}

private:
	std::queue<Chunk*> m_queue;
	std::condition_variable m_cv;
	std::mutex m_lock;
};

class ChunkWorkerPool {
public:
	ChunkWorkerPool() {
		unsigned nThreads = std::thread::hardware_concurrency();
		for (int i = 0; i < nThreads; ++i)
			m_threads.push_back(new std::thread(threadLoop, this));
	}
	
	void setParams(Options options, SimulationUpdateFunction updateFunction, bool doDraw) {
		m_paramOptions = options;
		m_paramUpdateFunction = updateFunction;
		m_paramDoDraw = doDraw;
	}

	void processChunk(Chunk* chunk) {
		m_allTasksDone = false;
		++m_taskCount;
		m_tasks.enqueue(chunk);
	}

	void awaitAll() const {
		m_allTasksDone.wait(false);
	}

	unsigned threadCount() const {
		return m_threads.size();
	}

	static ChunkWorkerPool& instance() {
		static ChunkWorkerPool c_instance;
		return c_instance;
	}

private:
	ChunkTaskQueue m_tasks;
	std::vector<std::thread*> m_threads;
	std::atomic_uint m_taskCount;
	std::atomic_bool m_allTasksDone = true;

	Options m_paramOptions;
	SimulationUpdateFunction m_paramUpdateFunction = nullptr;
	bool m_paramDoDraw = true;

	static void threadLoop(ChunkWorkerPool* pool) {
		while (true) {
			Chunk* chunk = pool->m_tasks.dequeue();

			chunk->process(pool->m_paramOptions, pool->m_paramUpdateFunction, pool->m_paramDoDraw);
			--pool->m_taskCount;

			pool->m_allTasksDone = pool->m_taskCount == 0;
			pool->m_allTasksDone.notify_one();
		}
	}
};
