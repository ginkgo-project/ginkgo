// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_REUSED_THREAD_HPP_
#define GKO_PUBLIC_REUSED_THREAD_HPP_

#include <atomic>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace gko {


class reused_thread {
public:
    reused_thread()
        : stop_(false), worker_([&] {
              while (true) {
                  std::function<void()> task;
                  {
                      // get job in the queue
                      std::unique_lock<std::mutex> lock(this->queue_mutex_);
                      this->condition_.wait(lock, [this] {
                          return this->stop_ || !this->tasks_.empty();
                      });
                      if (this->stop_ && this->tasks_.empty()) {
                          return;
                      }
                      task = std::move(this->tasks_.front());
                      this->tasks_.pop();
                  }
                  task();
              }
          })
    {}

    ~reused_thread()
    {
        this->stop_ = true;
        this->condition_.notify_all();
        this->worker_.join();
    }


    template <typename Func, typename... Args>
    auto add_task(Func&& f, Args&&... args)
        -> std::future<typename std::result_of<Func(Args...)>::type>
    {
        using result_type = typename std::result_of<Func(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
        auto future_result = task->get_future();
        {
            // submit job
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            if (this->stop_) {
                throw std::runtime_error("submit job but the thread is closed");
            }
            this->tasks_.emplace([task]() { (*task)(); });
        }
        // notify one thread there is a job
        this->condition_.notify_one();
        return future_result;
    }

private:
    std::atomic<bool> stop_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    // ensure worker is the last one be constructed
    std::thread worker_;
};


}  // namespace gko

#endif  // GKO_PUBLIC_REUSED_THREAD_HPP_
