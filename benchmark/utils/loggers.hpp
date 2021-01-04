/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_BENCHMARK_UTILS_LOGGERS_HPP_
#define GKO_BENCHMARK_UTILS_LOGGERS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <mutex>
#include <regex>
#include <unordered_map>


#include "benchmark/utils/general.hpp"


// A logger that accumulates the time of all operations
struct OperationLogger : gko::log::Logger {
    void on_allocation_started(const gko::Executor *exec,
                               const gko::size_type &) const override
    {
        this->start_operation(exec, "allocate");
    }

    void on_allocation_completed(const gko::Executor *exec,
                                 const gko::size_type &,
                                 const gko::uintptr &) const override
    {
        this->end_operation(exec, "allocate");
    }

    void on_free_started(const gko::Executor *exec,
                         const gko::uintptr &) const override
    {
        this->start_operation(exec, "free");
    }

    void on_free_completed(const gko::Executor *exec,
                           const gko::uintptr &) const override
    {
        this->end_operation(exec, "free");
    }

    void on_copy_started(const gko::Executor *from, const gko::Executor *to,
                         const gko::uintptr &, const gko::uintptr &,
                         const gko::size_type &) const override
    {
        from->synchronize();
        this->start_operation(to, "copy");
    }

    void on_copy_completed(const gko::Executor *from, const gko::Executor *to,
                           const gko::uintptr &, const gko::uintptr &,
                           const gko::size_type &) const override
    {
        from->synchronize();
        this->end_operation(to, "copy");
    }

    void on_operation_launched(const gko::Executor *exec,
                               const gko::Operation *op) const override
    {
        this->start_operation(exec, op->get_name());
    }

    void on_operation_completed(const gko::Executor *exec,
                                const gko::Operation *op) const override
    {
        this->end_operation(exec, op->get_name());
    }

    void write_data(rapidjson::Value &object,
                    rapidjson::MemoryPoolAllocator<> &alloc,
                    gko::uint32 repetitions)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto &entry : total) {
            add_or_set_member(
                object, entry.first.c_str(),
                std::chrono::duration<double>(entry.second).count() /
                    repetitions,
                alloc);
        }
    }

    OperationLogger(std::shared_ptr<const gko::Executor> exec, bool nested_name)
        : gko::log::Logger(exec), use_nested_name{nested_name}
    {}

private:
    void start_operation(const gko::Executor *exec,
                         const std::string &name) const
    {
        exec->synchronize();
        const std::lock_guard<std::mutex> lock(mutex);
        auto nested_name = nested.empty() || !use_nested_name
                               ? name
                               : nested.back().first + "::" + name;
        nested.emplace_back(nested_name, std::chrono::steady_clock::duration{});
        start[nested_name] = std::chrono::steady_clock::now();
    }

    void end_operation(const gko::Executor *exec, const std::string &name) const
    {
        exec->synchronize();
        const std::lock_guard<std::mutex> lock(mutex);
        // if operations are properly nested, nested_name now ends with name
        auto nested_name = nested.back().first;
        const auto end = std::chrono::steady_clock::now();
        const auto diff = end - start[nested_name];
        // make sure timings for nested operations are not counted twice
        total[nested_name] += diff - nested.back().second;
        nested.pop_back();
        if (!nested.empty()) {
            nested.back().second += diff;
        }
    }

    bool use_nested_name;
    mutable std::mutex mutex;
    mutable std::map<std::string, std::chrono::steady_clock::time_point> start;
    mutable std::map<std::string, std::chrono::steady_clock::duration> total;
    // the position i of this vector holds the total time spend on child
    // operations on nesting level i
    mutable std::vector<
        std::pair<std::string, std::chrono::steady_clock::duration>>
        nested;
};


struct StorageLogger : gko::log::Logger {
    void on_allocation_completed(const gko::Executor *,
                                 const gko::size_type &num_bytes,
                                 const gko::uintptr &location) const override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        storage[location] = num_bytes;
    }

    void on_free_completed(const gko::Executor *,
                           const gko::uintptr &location) const override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        storage[location] = 0;
    }

    void write_data(rapidjson::Value &output,
                    rapidjson::MemoryPoolAllocator<> &allocator)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        gko::size_type total{};
        for (const auto &e : storage) {
            total += e.second;
        }
        add_or_set_member(output, "storage", total, allocator);
    }

    StorageLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec)
    {}

private:
    mutable std::mutex mutex;
    mutable std::unordered_map<gko::uintptr, gko::size_type> storage;
};


// Logs true and recurrent residuals of the solver
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *, const gko::size_type &,
                               const gko::LinOp *residual,
                               const gko::LinOp *solution,
                               const gko::LinOp *residual_norm) const override
    {
        timestamps.PushBack(std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count(),
                            alloc);
        if (residual_norm) {
            rec_res_norms.PushBack(
                get_norm(gko::as<vec<ValueType>>(residual_norm)), alloc);
        } else {
            rec_res_norms.PushBack(
                compute_norm2(gko::as<vec<ValueType>>(residual)), alloc);
        }
        if (solution) {
            true_res_norms.PushBack(
                compute_residual_norm(matrix, b,
                                      gko::as<vec<ValueType>>(solution)),
                alloc);
        } else {
            true_res_norms.PushBack(-1.0, alloc);
        }
    }

    ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                   const gko::LinOp *matrix, const vec<ValueType> *b,
                   rapidjson::Value &rec_res_norms,
                   rapidjson::Value &true_res_norms,
                   rapidjson::Value &timestamps,
                   rapidjson::MemoryPoolAllocator<> &alloc)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b},
          start{std::chrono::steady_clock::now()},
          rec_res_norms{rec_res_norms},
          true_res_norms{true_res_norms},
          timestamps{timestamps},
          alloc{alloc}
    {}

private:
    const gko::LinOp *matrix;
    const vec<ValueType> *b;
    std::chrono::steady_clock::time_point start;
    rapidjson::Value &rec_res_norms;
    rapidjson::Value &true_res_norms;
    rapidjson::Value &timestamps;
    rapidjson::MemoryPoolAllocator<> &alloc;
};


// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &num_iterations,
                               const gko::LinOp *, const gko::LinOp *,
                               const gko::LinOp *) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask)
    {}

    void write_data(rapidjson::Value &output,
                    rapidjson::MemoryPoolAllocator<> &allocator)
    {
        add_or_set_member(output, "iterations", this->num_iters, allocator);
    }

private:
    mutable gko::size_type num_iters{0};
};


#endif  // GKO_BENCHMARK_UTILS_LOGGERS_HPP_
