#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {

public:
	int numThreads = 0;

	ThreadPool(size_t numThreads) 
		: stop(false), activeTasks(0) 
	{
		this->numThreads = int(numThreads);

		for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
			workers.emplace_back([this, threadIndex] {
				while (true) {
					std::function<void(int)> task;
					{
						std::unique_lock<std::mutex> lock(this->queueMutex);
						this->condition.wait(lock, [this] {
							return this->stop || !this->tasks.empty();
						});

						if (this->stop && this->tasks.empty()) return;

						task = std::move(this->tasks.front());
						this->tasks.pop();
					}
					
					// Execute task and decrement the counter
					task(threadIndex);
					
					// Atomically decrement and notify the wait() function
					if (--activeTasks == 0) {
						std::vector<std::function<void()>> callbacks;
						{
							std::unique_lock<std::mutex> lock(waitMutex);
							callbacks.swap(emptyCallbacks);
							waitCondition.notify_all();
						}
						// Invoke outside the lock so callbacks may enqueue() or onEmpty()
						for (auto& callback : callbacks) {
							callback();
						}
					}
				}
			});
		}
	}

	void enqueue(std::function<void(int)> task) {
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			tasks.push(std::move(task));
			activeTasks++; // Increment active task count
		}
		condition.notify_one();
	}
	
	// Invoke callback once all tasks are finished. If no tasks are pending,
	// the callback is invoked immediately; otherwise it fires once when the
	// active task count drops to zero.
	void onEmpty(std::function<void()> callback) {
		{
			std::unique_lock<std::mutex> lock(waitMutex);
			if (activeTasks != 0) {
				emptyCallbacks.push_back(std::move(callback));
				return;
			}
		}

		callback();
	}

	// New: Block until all tasks are finished
	void wait() {
		std::unique_lock<std::mutex> lock(waitMutex);
		waitCondition.wait(lock, [this] {
			return activeTasks == 0;
		});
	}

	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread &worker : workers) {
			worker.join();
		}
	}

private:
	std::vector<std::thread> workers;
	std::queue<std::function<void(int)>> tasks;
	
	std::mutex queueMutex;
	std::condition_variable condition;
	
	// For the wait() functionality; waitMutex also guards emptyCallbacks
	std::mutex waitMutex;
	std::condition_variable waitCondition;
	std::vector<std::function<void()>> emptyCallbacks;
	std::atomic<size_t> activeTasks;
	
	bool stop;
};