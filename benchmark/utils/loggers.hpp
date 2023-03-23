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
        gko::int64 overhead_ns) override
    {
        for (const auto& entry : entries) {
            if (entry.name != "total") {
                add_or_set_member(*object, entry.name.c_str(),
                                  entry.exclusive_ns * 1e-9 / repetitions,
                                  *alloc);
            }
        }
        add_or_set_member(*object, "overhead", overhead_ns * 1e-9 / repetitions,
                          *alloc);
    }

    void write_nested(const gko::log::ProfilerHook::nested_summary_entry& root,
                      gko::int64 overhead_ns) override
    {
        auto visit =
            [this](auto visit,
                   const gko::log::ProfilerHook::nested_summary_entry& node,
                   std::string prefix) -> void {
            auto exclusive_ns = node.elapsed_ns;
            auto new_prefix = prefix + node.name + "::";
            for (const auto& child : node.children) {
                visit(visit, child, new_prefix);
                exclusive_ns -= child.elapsed_ns;
            }
            add_or_set_member(*object, (prefix + node.name).c_str(),
                              exclusive_ns * 1e-9 / repetitions, *alloc);
        };
        // we don't need to annotate the total
        for (const auto& child : root.children) {
            visit(visit, child, "");
        }
        add_or_set_member(*object, "overhead", overhead_ns * 1e-9 / repetitions,
                          *alloc);
    }

    rapidjson::Value* object;
    rapidjson::MemoryPoolAllocator<>* alloc;
    gko::uint32 repetitions;
};


inline std::shared_ptr<gko::log::ProfilerHook> create_operations_logger(
    bool nested, rapidjson::Value& object,
    rapidjson::MemoryPoolAllocator<>& alloc, gko::uint32 repetitions)
{
    if (nested) {
        return gko::log::ProfilerHook::create_nested_summary(
            std::make_unique<JsonSummaryWriter>(object, alloc, repetitions));
    } else {
        return gko::log::ProfilerHook::create_summary(
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

    // TODO2.0: Remove when deprecating simple overload
    void on_iteration_complete(const gko::LinOp* solver,
                               const gko::size_type& it,
                               const gko::LinOp* residual,
                               const gko::LinOp* solution,
                               const gko::LinOp* residual_norm) const override
    {
        on_iteration_complete(solver, it, residual, solution, residual_norm,
                              nullptr);
    }

    void on_iteration_complete(
        const gko::LinOp*, const gko::size_type&, const gko::LinOp* residual,
        const gko::LinOp* solution, const gko::LinOp* residual_norm,
        const gko::LinOp* implicit_sq_residual_norm) const override
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
    void on_iteration_complete(const gko::LinOp*,
                               const gko::size_type& num_iterations,
                               const gko::LinOp*, const gko::LinOp*,
                               const gko::LinOp*) const override
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


struct StorageTracker : public gko::log::ProfilerHook {
    std::string format_bytes(gko::size_type bytes) const
    {
        std::map<int, std::string> suffix{
            {0, "  B"}, {1, " KB"}, {1, " MB"}, {3, " GB"}, {4, " TB"}};
        auto log1000 = [](auto num) { return std::log(num) / std::log(1'000); };
        auto magnitude = static_cast<int>(log1000(bytes));
        auto it = std::lower_bound(
            suffix.begin(), suffix.end(), magnitude,
            [](const auto& a, const auto& b) { return a.first < b; });
        return std::to_string(int(bytes / std::pow(1000, it->first))) +
               it->second;
    }

    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type& num_bytes) const override
    {
        gko::log::ProfilerHook::on_allocation_started(exec, num_bytes);
        const std::lock_guard<std::mutex> lock(mutex);
        if (num_bytes > 1'000'000'000) {
            std::cerr << "  Allocating " << format_bytes(num_bytes)
                      << std::endl;
            es->print();
        }
    }

    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr& location) const override
    {
        gko::log::ProfilerHook::on_allocation_completed(exec, num_bytes,
                                                        location);
        const std::lock_guard<std::mutex> lock(mutex);
        storage[location] = num_bytes;
        total_storage += num_bytes;
        if (num_bytes > 1'000'000'000) {
            std::cerr << "  Allocation done at " << location << std::endl;
            std::cerr << "Total bytes: " << format_bytes(total_storage)
                      << std::endl;
            if (total_storage > max_storage) {
                max_storage = total_storage;
                max_es = *es;
            }
        }
    }

    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr& location) const override
    {
        gko::log::ProfilerHook::on_free_completed(exec, location);
        const std::lock_guard<std::mutex> lock(mutex);
        auto num_bytes = storage.at(location);
        total_storage -= num_bytes;
        storage[location] = 0;
        if (num_bytes > 1'000'000'000) {
            std::cerr << "  Freed " << format_bytes(num_bytes) << " at "
                      << location << std::endl;
            std::cerr << "Total bytes: " << format_bytes(total_storage)
                      << std::endl;
            es->print();
        }
    }

    struct event_stack {
        std::deque<std::string> stack;

        void push(const char* name) { stack.emplace_back(name); }

        void pop(const char* name) { stack.pop_back(); }

        void print()
        {
            std::cerr << "  Current event stack:" << std::endl;
            std::string prefix{"  "};
            for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
                std::cerr << prefix << *it << std::endl;
                if (prefix.size() / 2 < 6) {
                    prefix.append("  ");
                }
            }
        }
    };

    static std::unique_ptr<StorageTracker> create()
    {
        auto es = std::make_shared<event_stack>();
        return std::unique_ptr<StorageTracker>(new StorageTracker(
            es,
            [es](const char* name, gko::log::profile_event_category) {
                es->push(name);
            },
            [es](const char* name, gko::log::profile_event_category) {
                es->pop(name);
            }));
    }

    ~StorageTracker()
    {
        std::cerr << "Maximal storage: " << format_bytes(max_storage)
                  << std::endl;
        max_es.print();
    }

private:
    template <typename F1, typename F2>
    explicit StorageTracker(std::shared_ptr<event_stack> es, F1&& begin_fn,
                            F2&& end_fn)
        : gko::log::ProfilerHook(std::forward<F1>(begin_fn),
                                 std::forward<F2>(end_fn)),
          es(es)
    {}

    mutable std::shared_ptr<event_stack> es;
    mutable std::mutex mutex;
    mutable std::unordered_map<gko::uintptr, gko::size_type> storage;
    mutable gko::size_type total_storage;
    mutable gko::size_type max_storage = 0;
    mutable event_stack max_es;
};


#endif  // GKO_BENCHMARK_UTILS_LOGGERS_HPP_
