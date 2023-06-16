/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <cmath>
#include <mutex>
#include <regex>
#include <unordered_map>


#include "benchmark/utils/general.hpp"
#include "core/distributed/helpers.hpp"


struct JsonSummaryWriter : gko::log::ProfilerHook::SummaryWriter,
                           gko::log::ProfilerHook::NestedSummaryWriter {
    JsonSummaryWriter(rapidjson::Value& object,
                      rapidjson::MemoryPoolAllocator<>& alloc,
                      gko::uint32 repetitions)
        : object{&object}, alloc{&alloc}, repetitions{repetitions}
    {}

    void write(
        const std::vector<gko::log::ProfilerHook::summary_entry>& entries,
        std::chrono::nanoseconds overhead) override
    {
        for (const auto& entry : entries) {
            if (entry.name != "total") {
                add_or_set_member(*object, entry.name.c_str(),
                                  entry.exclusive.count() * 1e-9 / repetitions,
                                  *alloc);
            }
        }
        add_or_set_member(*object, "overhead",
                          overhead.count() * 1e-9 / repetitions, *alloc);
    }

    void write_nested(const gko::log::ProfilerHook::nested_summary_entry& root,
                      std::chrono::nanoseconds overhead) override
    {
        auto visit =
            [this](auto visit,
                   const gko::log::ProfilerHook::nested_summary_entry& node,
                   std::string prefix) -> void {
            auto exclusive = node.elapsed;
            auto new_prefix = prefix + node.name + "::";
            for (const auto& child : node.children) {
                visit(visit, child, new_prefix);
                exclusive -= child.elapsed;
            }
            add_or_set_member(*object, (prefix + node.name).c_str(),
                              exclusive.count() * 1e-9 / repetitions, *alloc);
        };
        // we don't need to annotate the total
        for (const auto& child : root.children) {
            visit(visit, child, "");
        }
        add_or_set_member(*object, "overhead",
                          overhead.count() * 1e-9 / repetitions, *alloc);
    }

    rapidjson::Value* object;
    rapidjson::MemoryPoolAllocator<>* alloc;
    gko::uint32 repetitions;
};


inline std::shared_ptr<gko::log::ProfilerHook> create_operations_logger(
    bool gpu_timer, bool nested, std::shared_ptr<gko::Executor> exec,
    rapidjson::Value& object, rapidjson::MemoryPoolAllocator<>& alloc,
    gko::uint32 repetitions)
{
    std::shared_ptr<gko::Timer> timer;
    if (gpu_timer) {
        timer = gko::Timer::create_for_executor(exec);
    } else {
        timer = std::make_unique<gko::CpuTimer>();
    }
    if (nested) {
        return gko::log::ProfilerHook::create_nested_summary(
            timer,
            std::make_unique<JsonSummaryWriter>(object, alloc, repetitions));
    } else {
        return gko::log::ProfilerHook::create_summary(
            timer,
            std::make_unique<JsonSummaryWriter>(object, alloc, repetitions));
    }
}


struct StorageLogger : gko::log::Logger {
    void on_allocation_completed(const gko::Executor*,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr& location) const override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        storage[location] = num_bytes;
    }

    void on_free_completed(const gko::Executor*,
                           const gko::uintptr& location) const override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        storage[location] = 0;
    }

    void write_data(rapidjson::Value& output,
                    rapidjson::MemoryPoolAllocator<>& allocator)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        gko::size_type total{};
        for (const auto& e : storage) {
            total += e.second;
        }
        add_or_set_member(output, "storage", total, allocator);
    }

#if GINKGO_BUILD_MPI
    void write_data(gko::experimental::mpi::communicator comm,
                    rapidjson::Value& output,
                    rapidjson::MemoryPoolAllocator<>& allocator)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        gko::size_type total{};
        for (const auto& e : storage) {
            total += e.second;
        }
        comm.reduce(gko::ReferenceExecutor::create(),
                    comm.rank() == 0
                        ? static_cast<gko::size_type*>(MPI_IN_PLACE)
                        : &total,
                    &total, 1, MPI_SUM, 0);
        add_or_set_member(output, "storage", total, allocator);
    }
#endif

private:
    mutable std::mutex mutex;
    mutable std::unordered_map<gko::uintptr, gko::size_type> storage;
};


// Logs true and recurrent residuals of the solver
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    using rc_vtype = gko::remove_complex<ValueType>;

    void on_iteration_complete(const gko::LinOp*,
                               const gko::LinOp* right_hand_side,
                               const gko::LinOp* solution,
                               const gko::size_type&,
                               const gko::LinOp* residual,
                               const gko::LinOp* residual_norm,
                               const gko::LinOp* implicit_sq_residual_norm,
                               const gko::array<gko::stopping_status>* status,
                               bool all_stopped) const override
    {
        timestamps.PushBack(std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count(),
                            alloc);
        if (residual_norm) {
            rec_res_norms.PushBack(
                get_norm(gko::as<vec<rc_vtype>>(residual_norm)), alloc);
        } else {
            gko::detail::vector_dispatch<rc_vtype>(
                residual, [&](const auto v_residual) {
                    rec_res_norms.PushBack(compute_norm2(v_residual), alloc);
                });
        }
        if (solution) {
            gko::detail::vector_dispatch<
                rc_vtype>(solution, [&](auto v_solution) {
                using concrete_type =
                    std::remove_pointer_t<std::decay_t<decltype(v_solution)>>;
                true_res_norms.PushBack(
                    compute_residual_norm(matrix, gko::as<concrete_type>(b),
                                          v_solution),
                    alloc);
            });
        } else {
            true_res_norms.PushBack(-1.0, alloc);
        }
        if (implicit_sq_residual_norm) {
            implicit_res_norms.PushBack(
                std::sqrt(get_norm(
                    gko::as<vec<rc_vtype>>(implicit_sq_residual_norm))),
                alloc);
            has_implicit_res_norm = true;
        } else {
            implicit_res_norms.PushBack(-1.0, alloc);
        }
    }

    ResidualLogger(gko::ptr_param<const gko::LinOp> matrix,
                   gko::ptr_param<const gko::LinOp> b,
                   rapidjson::Value& rec_res_norms,
                   rapidjson::Value& true_res_norms,
                   rapidjson::Value& implicit_res_norms,
                   rapidjson::Value& timestamps,
                   rapidjson::MemoryPoolAllocator<>& alloc)
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask),
          matrix{matrix.get()},
          b{b.get()},
          start{std::chrono::steady_clock::now()},
          rec_res_norms{rec_res_norms},
          true_res_norms{true_res_norms},
          has_implicit_res_norm{},
          implicit_res_norms{implicit_res_norms},
          timestamps{timestamps},
          alloc{alloc}
    {}

    bool has_implicit_res_norms() const { return has_implicit_res_norm; }

private:
    const gko::LinOp* matrix;
    const gko::LinOp* b;
    std::chrono::steady_clock::time_point start;
    rapidjson::Value& rec_res_norms;
    rapidjson::Value& true_res_norms;
    mutable bool has_implicit_res_norm;
    rapidjson::Value& implicit_res_norms;
    rapidjson::Value& timestamps;
    rapidjson::MemoryPoolAllocator<>& alloc;
};


// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp*, const gko::LinOp*,
                               const gko::LinOp*,
                               const gko::size_type& num_iterations,
                               const gko::LinOp*, const gko::LinOp*,
                               const gko::LinOp*,
                               const gko::array<gko::stopping_status>*,
                               bool) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger()
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}

    void write_data(rapidjson::Value& output,
                    rapidjson::MemoryPoolAllocator<>& allocator)
    {
        add_or_set_member(output, "iterations", this->num_iters, allocator);
    }

private:
    mutable gko::size_type num_iters{0};
};


#endif  // GKO_BENCHMARK_UTILS_LOGGERS_HPP_
