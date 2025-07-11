// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/reused_thread.hpp>


namespace gko {

reused_thread::reused_thread()
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


reused_thread::~reused_thread()
{
    this->stop_ = true;
    this->condition_.notify_all();
    this->worker_.join();
}


}  // namespace gko
