#pragma once 
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include "chunk.h"
#include <functional>
//
//class IChunkWorkerTask {
//public:
//	// Call enclosed function with the provided arguments. 
//	virtual void call(Chunk* chunk) = 0;
//};
//
//// Basicly std::function with the exception that in this case
//// the chunk can be the argument and the arguments can be "captured". 
//template <typename... Args>
//class ChunkWorkerTask : public IChunkWorkerTask {
//public:
//	ChunkWorkerTask(void(Chunk::*fnc)(Args...))
//		: m_function(fnc)
//	{ }
//
//	// Set arguments with which the function can be called. 
//	ChunkWorkerTask* setArgs(Args... args) {
//		m_args = std::make_tuple(args...);
//		return this;
//	}
//
//	// Call enclosed function with the provided arguments. 
//	virtual void call(Chunk* chunk) override {
//		std::apply(m_function, std::tuple_cat(std::make_tuple(chunk), m_args));
//	}
//
//private:
//	std::tuple<Args...> m_args;
//	void(Chunk::*		m_function)(Args...);
//};
//
//class ChunkWorkerPool {
//public:
//	ChunkWorkerPool() {
//		unsigned nThreads = std::thread::hardware_concurrency();
//
//		std::cout << "Initializing chunk worker pool with " 
//			<< nThreads << " threads. " << std::endl;
//
//		// Initialize threads
//		for (int i = 0; i < nThreads; ++i)
//			m_threads.push_back(new std::thread(workerThreadFunction, this));
//	}
//
//	void addChunk(Chunk* chunk) {	// main thread only
//		m_chunks.push_back(chunk);
//	}
//
//	// Removes all chunks from collection
//	void clearChunks() {
//		m_chunks.clear();
//	}
//
//	// Execute task on all chunks with the provided arguments. Returns when all chunks are processed. 
//	void execute(IChunkWorkerTask* task) {
//		setTask(task);
//		begin();
//		waitAllDone();
//	}
//
//	~ChunkWorkerPool() {
//		std::cout << "Joining threads. " << std::endl;
//
//		// Request threads to stop
//		shutdown();
//
//		// Join and clear threads
//		for (auto thread : m_threads) {
//			thread->join();
//			delete thread;
//		}
//	}
//
//private:
//	using cv = std::condition_variable;
//
//	std::vector<std::thread*>	m_threads;
//	std::atomic_bool			m_shutdown = false;
//
//	std::vector<Chunk*>			m_chunks{};
//	using iterator = decltype(m_chunks)::iterator;
//
//	mutable iterator			m_iter;
//	mutable	std::mutex			m_iterMut;
//	mutable cv					m_iterBeginCV;
//
//	mutable size_t				m_doneCount = 0;
//	mutable std::mutex			m_doneCountMut;
//	mutable cv					m_doneAllCV;
//
//	IChunkWorkerTask*			m_task = nullptr;
//	mutable std::mutex			m_taskMut;
//	mutable cv					m_hasTaskCV;
//
//	Chunk* next() const {
//		// Wait untill iter is not at end
//		std::unique_lock lock(m_iterMut);
//		m_iterBeginCV.wait(lock, [this] { 
//			return (m_iter._Ptr != nullptr && m_iter != m_chunks.end()) || m_shutdown; 
//			});
//
//		// Handle shutdown
//		if (m_shutdown)
//			return nullptr;
//
//		// Get chunk and increase iter
//		Chunk* res = *m_iter;
//		++m_iter;
//
//		// Notify next worker
//		lock.unlock();
//		m_iterBeginCV.notify_one();
//
//		return res;
//	}
//
//	IChunkWorkerTask* getTask() const {
//		// Wait until task is provided
//		std::unique_lock lock(m_taskMut);
//		m_hasTaskCV.wait(lock, [this] { return m_task != nullptr; });
//
//		return m_task;
//	}
//
//	void setTask(IChunkWorkerTask* task) {
//		std::lock_guard lock(m_taskMut);
//		m_task = task;
//	}
//
//	void begin() {
//		std::lock_guard doneCountLock(m_doneCountMut);
//		std::lock_guard iterLock(m_iterMut);
//
//		m_doneCount = 0;
//		m_iter = m_chunks.begin();
//
//		// Notify first chunk
//		m_iterBeginCV.notify_one();
//	}
//
//	void notifyDone() const {
//		std::lock_guard lock(m_doneCountMut);
//		++m_doneCount;
//		m_doneAllCV.notify_one();
//	}
//
//	void waitAllDone() const {		// main thread only
//		std::unique_lock lock(m_doneCountMut);
//		m_doneAllCV.wait(lock, [this] { return m_doneCount == m_chunks.size(); });
//	}
//
//	void shutdown() {
//		m_shutdown = true;
//		m_iterBeginCV.notify_all();
//	}
//	
//	static void workerThreadFunction(ChunkWorkerPool* pool) {
//		while (!pool->m_shutdown) {
//			// Wait for unprocessed chunk
//			Chunk* chunk = pool->next();
//
//			// Handle shutdown
//			if (chunk == nullptr)
//				break;
//
//			// Wait for provided task
//			IChunkWorkerTask* task = pool->getTask();
//
//			task->call(chunk);
//			pool->notifyDone();
//		}
//	}
//};

class IChunkWorkerTask {
public:
	// Call enclosed function with the provided arguments. 
	virtual void call(Chunk* chunk) = 0;
};

// Basicly std::function with the exception that in this case
// the chunk can be the argument and the arguments can be "captured". 
template <typename... Args>
class ChunkWorkerTask : public IChunkWorkerTask {
public:
	ChunkWorkerTask(void(Chunk::* fnc)(Args...))
		: m_function(fnc)
	{ }

	// Set arguments with which the function can be called. 
	ChunkWorkerTask* setArgs(Args... args) {
		m_args = std::make_tuple(args...);
		return this;
	}

	// Call enclosed function with the provided arguments. 
	virtual void call(Chunk* chunk) override {
		std::apply(m_function, std::tuple_cat(std::make_tuple(chunk), m_args));
	}

private:
	std::tuple<Args...> m_args;
	void(Chunk::* m_function)(Args...);
};

template <typename T>
concept Iterable_C = requires(T t) {
	{ t.begin() };
	{ t.end() };
	{ t.size() };
};

class IChunkCollection {
public:
	virtual Chunk* next() = 0;
	virtual void begin() = 0;
	virtual size_t size() const = 0;

	void shutdown() {
		m_shutdown = true;
		m_iterBeginCV.notify_all();
	}

protected:
	using cv = std::condition_variable;

	bool				m_shutdown = false;

	mutable std::mutex	m_mut;
	mutable cv			m_iterBeginCV;

};

template <Iterable_C T>
class ChunkCollection : public IChunkCollection {
public:
	ChunkCollection(T& collection)
		: m_collection(collection)
		, m_iter(collection.begin())
	{ }

	virtual Chunk* next() override {
		// Wait untill iter is not at end
		std::unique_lock lock(m_mut);
		m_iterBeginCV.wait(lock, [this] {
			return (m_iter._Ptr != nullptr && m_iter != m_collection.end()) || m_shutdown;
			});

		// Handle shutdown
		if (m_shutdown)
			return nullptr;

		// Get chunk and increase iter
		Chunk* res = (*m_iter).second;
		++m_iter;

		// Notify next worker
		lock.unlock();
		m_iterBeginCV.notify_one();

		return res;
	}

	virtual void begin() override {
		std::lock_guard iterLock(m_mut);

		m_iter = m_collection.begin();

		// Notify first thread
		m_iterBeginCV.notify_one();
	}

	virtual size_t size() const override {
		return m_collection.size();
	}

private:
	T& m_collection;
	T::iterator	m_iter;

};

class ChunkWorkerPool {
public:
	template <typename T>
	ChunkWorkerPool(T& chunks) {
		unsigned nThreads = std::thread::hardware_concurrency();

		std::cout << "Initializing chunk worker pool with "
			<< nThreads << " threads. " << std::endl;

		m_chunks = new ChunkCollection(chunks);

		// Initialize threads
		for (int i = 0; i < nThreads; ++i)
			m_threads.push_back(new std::thread(workerThreadFunction, this));
	}

	// Execute task on all chunks with the provided arguments. Returns when all chunks are processed. 
	void execute(IChunkWorkerTask* task) {
		setTask(task);
		begin();
		waitAllDone();
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

	IChunkCollection* m_chunks;

	mutable size_t				m_doneCount = 0;
	mutable std::mutex			m_doneCountMut;
	mutable cv					m_doneAllCV;

	IChunkWorkerTask* m_task = nullptr;
	mutable std::mutex			m_taskMut;
	mutable cv					m_hasTaskCV;

	IChunkWorkerTask* getTask() const {
		// Wait until task is provided
		std::unique_lock lock(m_taskMut);
		m_hasTaskCV.wait(lock, [this] { return m_task != nullptr || m_shutdown; });

		// Handle shutdown
		if (m_shutdown)
			return nullptr;

		return m_task;
	}

	void setTask(IChunkWorkerTask* task) {
		std::lock_guard lock(m_taskMut);
		m_task = task;
	}

	void begin() {
		std::lock_guard doneCountLock(m_doneCountMut);
		m_doneCount = 0;

		m_chunks->begin();
	}

	void notifyDone() const {
		std::lock_guard lock(m_doneCountMut);
		++m_doneCount;
		m_doneAllCV.notify_one();
	}

	void waitAllDone() const {		// main thread only
		std::unique_lock lock(m_doneCountMut);
		m_doneAllCV.wait(lock, [this] { return m_doneCount == m_chunks->size(); });
	}

	void shutdown() {
		m_shutdown = true;
		m_chunks->shutdown();
	}

	static void workerThreadFunction(ChunkWorkerPool* pool) {
		while (!pool->m_shutdown) {
			// Wait for unprocessed chunk
			Chunk* chunk = pool->m_chunks->next();

			// Wait for provided task
			IChunkWorkerTask* task = pool->getTask();

			// Handle shutdown
			if (chunk == nullptr || task == nullptr)
				break;

			task->call(chunk);
			pool->notifyDone();
		}
	}
};
