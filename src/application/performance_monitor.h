#pragma once
#include <string>
#include <unordered_map>
#include <chrono>

#define PERF_MONITOR PerformanceMonitor::instance()

#define BEGIN_TASK(name) PERF_MONITOR.beginTask(name)
#define END_TASK(name, ...) PERF_MONITOR.endTask(name, __VA_ARGS__)
#define TASK(name, body) do { BEGIN_TASK(name); body END_TASK(name); } while (false)
#define LOG_TASK(name, body) do { BEGIN_TASK(name); body END_TASK(name); PERF_MONITOR.log(name); PERF_MONITOR.forget(name); } while (false)

class PerformanceMonitor {
	using time_point = std::chrono::steady_clock::time_point;

	struct Task;

	struct SubTask {
		Task*		task = nullptr;
		SubTask*	next = nullptr;
	};

	struct Task {
		time_point	startTime;
		time_point	endTime;
		long long	duration;
		SubTask*	subTasks;
	};

public:
	void beginTask(const std::string& name) {
		auto taskIt = m_tasks.find(name);
		Task& task = (taskIt == m_tasks.end())
			? m_tasks.insert({ name, Task{} }).first->second
			: taskIt->second;

		task.startTime = now();
	}

	void endTask(const std::string& name, std::vector<std::string> subTasks = std::vector<std::string>{}) {
		auto taskIt = m_tasks.find(name);
		Task& task = (taskIt == m_tasks.end())
			? m_tasks.insert({ name, Task{} }).first->second
			: taskIt->second;

		long long subTasksSumDuration = 0;
		for (auto& name : subTasks)
			if (auto taskIt = m_tasks.find(name); taskIt != m_tasks.end())
				subTasksSumDuration += taskIt->second.duration;

		task.endTime = now();
		task.duration = durationInMicroSeconds(task.startTime, task.endTime) - subTasksSumDuration;
	}

	long long duration(const std::string& name) const {
		auto taskIt = m_tasks.find(name);
		if (taskIt == m_tasks.end())
			return 0;

		return taskIt->second.duration;
	}

	void log(const std::string& name) const {
		std::cout << "Task: \"" << name << "\" completed in " << duration(name) / 1000.f << "ms. " << std::endl;
	}

	void forget(const std::string& name) {
		m_tasks.erase(name);
	}

	void showImGuiWidget() const {
		for (auto& [name, task] : m_tasks) {
			ImGui::Text(name.c_str()); ImGui::SameLine();
			ImGui::Text("%f", task.duration / 1000.f);
			ImGui::SameLine(); ImGui::Text("ms");
		}
	}

	static PerformanceMonitor& instance() {
		static PerformanceMonitor c_instance{};
		return c_instance;
	}

private:
	std::unordered_map<std::string, Task> m_tasks{};

	time_point now() const {
		return std::chrono::high_resolution_clock::now();
	}

	long long durationInMicroSeconds(time_point start, time_point end) const {
		return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}

};
