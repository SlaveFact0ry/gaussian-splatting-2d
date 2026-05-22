#include "thread_pool.hpp"

#include <algorithm>
#include <utility>

namespace render_2dgs {

ThreadPool::ThreadPool(std::uint32_t number_of_threads,
                       std::uint32_t queue_capacity) {
  if (number_of_threads == 0) number_of_threads = 1;
  if (queue_capacity == 0) queue_capacity = 1;

  job_queue_.capacity = queue_capacity;
  job_queue_.buffer.resize(queue_capacity);

  workers_.reserve(number_of_threads);
  for (std::uint32_t i = 0; i < number_of_threads; ++i) {
    workers_.emplace_back([this] { LoopForWorker(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(job_queue_.mutex);
    stop_signal_.store(true, std::memory_order_release);
  }
  job_queue_.dequeue_condition.notify_all();
  job_queue_.enqueue_condition.notify_all();

  for (auto &w : workers_) {
    if (w.joinable()) w.join();
  }
}

void ThreadPool::EnqueueJob(Job job) {
  {
    std::unique_lock<std::mutex> lock(job_queue_.mutex);
    job_queue_.enqueue_condition.wait(lock, [this] {
      return job_queue_.count < job_queue_.capacity ||
             stop_signal_.load(std::memory_order_acquire);
    });
    if (stop_signal_.load(std::memory_order_acquire)) return;

    job_queue_.buffer[job_queue_.tail] = std::move(job);
    job_queue_.tail = (job_queue_.tail + 1) % job_queue_.capacity;
    ++job_queue_.count;
  }
  job_queue_.dequeue_condition.notify_one();
}

void ThreadPool::WaitAll() {
  std::unique_lock<std::mutex> lock(job_queue_.mutex);
  job_queue_.enqueue_condition.wait(lock, [this] {
    return job_queue_.count == 0 &&
           job_queue_.jobs_in_progress.load(std::memory_order_acquire) == 0;
  });
}

void ThreadPool::ParallelFor(
    std::uint32_t total,
    std::function<void(std::uint32_t, std::uint32_t, int)> fn) {
  if (total == 0) return;

  const std::uint32_t T = NumThreads();
  const std::uint32_t chunk = (total + T - 1) / T;

  for (std::uint32_t t = 0; t < T; ++t) {
    std::uint32_t start = t * chunk;
    if (start >= total) break;
    std::uint32_t end = std::min(start + chunk, total);
    int tid = static_cast<int>(t);
    EnqueueJob([fn, start, end, tid] { fn(start, end, tid); });
  }
  WaitAll();
}

void ThreadPool::LoopForWorker() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(job_queue_.mutex);
      job_queue_.dequeue_condition.wait(lock, [this] {
        return job_queue_.count > 0 ||
               stop_signal_.load(std::memory_order_acquire);
      });

      if (job_queue_.count == 0 &&
          stop_signal_.load(std::memory_order_acquire)) {
        return;
      }

      job = std::move(job_queue_.buffer[job_queue_.head]);
      job_queue_.head = (job_queue_.head + 1) % job_queue_.capacity;
      --job_queue_.count;
      job_queue_.jobs_in_progress.fetch_add(1, std::memory_order_acq_rel);
    }
    job_queue_.enqueue_condition.notify_one();

    job();

    job_queue_.jobs_in_progress.fetch_sub(1, std::memory_order_acq_rel);
    job_queue_.enqueue_condition.notify_all();
  }
}

} // namespace render_2dgs
