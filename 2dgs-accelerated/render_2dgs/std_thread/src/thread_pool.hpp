#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace render_2dgs {

class ThreadPool {
public:
  using Job = std::function<void()>;

  ThreadPool(std::uint32_t number_of_threads, std::uint32_t queue_capacity);

  ~ThreadPool();

  void EnqueueJob(Job job);

  ThreadPool(const ThreadPool &) = delete;
  auto operator=(const ThreadPool &) -> ThreadPool & = delete;
  ThreadPool(ThreadPool &&) = delete;
  auto operator=(ThreadPool &&) -> ThreadPool & = delete;

private:
  void LoopForWorker();

  struct JobQueue {
    std::uint32_t capacity{0}; ///< Maximum queue capacity
    std::vector<Job> buffer;   ///< Circular buffer storage
    std::uint32_t head{0};     ///< Index for dequeue
    std::uint32_t tail{0};     ///< Index for enqueue
    std::uint32_t count{0};    ///< Current number of jobs in queue
    std::atomic<std::uint32_t> jobs_in_progress{0}; ///< Count of executing jobs
    std::mutex mutex;                               ///< Protects queue state
    std::condition_variable enqueue_condition; ///< Signals when space available
    std::condition_variable dequeue_condition; ///< Signals when jobs available
  };

  JobQueue job_queue_;                   ///< Job queue
  std::atomic<bool> stop_signal_{false}; ///< Flag to signal worker shutdown
  std::vector<std::thread> workers_;     ///< Worker threads
};

} // namespace render_2dgs
