// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    JsonSummaryWriter(json& object, gko::uint32 repetitions)
        : object{&object}, repetitions{repetitions}
    {}

    void write(
        const std::vector<gko::log::ProfilerHook::summary_entry>& entries,
        std::chrono::nanoseconds overhead) override
    {
        for (const auto& entry : entries) {
            if (entry.name != "total") {
                (*object)[entry.name] =
                    entry.exclusive.count() * 1e-9 / repetitions;
            }
        }
        (*object)["overhead"] = overhead.count() * 1e-9 / repetitions;
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
            (*object)[prefix + node.name] =
                exclusive.count() * 1e-9 / repetitions;
        };
        // we don't need to annotate the total
        for (const auto& child : root.children) {
            visit(visit, child, "");
        }
        (*object)["overhead"] = overhead.count() * 1e-9 / repetitions;
    }

    json* object;
    gko::uint32 repetitions;
};


inline std::shared_ptr<gko::log::ProfilerHook> create_operations_logger(
    bool gpu_timer, bool nested, std::shared_ptr<gko::Executor> exec,
    json& object, gko::uint32 repetitions)
{
    std::shared_ptr<gko::Timer> timer;
    if (gpu_timer) {
        timer = gko::Timer::create_for_executor(exec);
    } else {
        timer = std::make_unique<gko::CpuTimer>();
    }
    if (nested) {
        return gko::log::ProfilerHook::create_nested_summary(
            timer, std::make_unique<JsonSummaryWriter>(object, repetitions));
    } else {
        return gko::log::ProfilerHook::create_summary(
            timer, std::make_unique<JsonSummaryWriter>(object, repetitions));
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

    void write_data(json& output)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        gko::size_type total{};
        for (const auto& e : storage) {
            total += e.second;
        }
        output["storage"] = total;
    }

#if GINKGO_BUILD_MPI
    void write_data(gko::experimental::mpi::communicator comm, json& output)
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
        output["storage"] = total;
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
        timestamps->push_back(std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
        if (residual_norm) {
            rec_res_norms->push_back(
                get_norm(gko::as<vec<rc_vtype>>(residual_norm)));
        } else {
            gko::detail::vector_dispatch<rc_vtype>(
                residual, [&](const auto v_residual) {
                    rec_res_norms->push_back(compute_norm2(v_residual));
                });
        }
        if (solution) {
            gko::detail::vector_dispatch<
                rc_vtype>(solution, [&](auto v_solution) {
                using concrete_type =
                    std::remove_pointer_t<std::decay_t<decltype(v_solution)>>;
                true_res_norms->push_back(compute_residual_norm(
                    matrix, gko::as<concrete_type>(b), v_solution));
            });
        } else {
            true_res_norms->push_back(-1.0);
        }
        if (implicit_sq_residual_norm) {
            implicit_res_norms->push_back(std::sqrt(
                get_norm(gko::as<vec<rc_vtype>>(implicit_sq_residual_norm))));
            has_implicit_res_norm = true;
        } else {
            implicit_res_norms->push_back(-1.0);
        }
    }

    ResidualLogger(gko::ptr_param<const gko::LinOp> matrix,
                   gko::ptr_param<const gko::LinOp> b, json& rec_res_norms,
                   json& true_res_norms, json& implicit_res_norms,
                   json& timestamps)
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask),
          matrix{matrix.get()},
          b{b.get()},
          start{std::chrono::steady_clock::now()},
          rec_res_norms{&rec_res_norms},
          true_res_norms{&true_res_norms},
          has_implicit_res_norm{},
          implicit_res_norms{&implicit_res_norms},
          timestamps{&timestamps}
    {}

    bool has_implicit_res_norms() const { return has_implicit_res_norm; }

private:
    const gko::LinOp* matrix;
    const gko::LinOp* b;
    std::chrono::steady_clock::time_point start;
    json* rec_res_norms;
    json* true_res_norms;
    mutable bool has_implicit_res_norm;
    json* implicit_res_norms;
    json* timestamps;
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

    void write_data(json& output) { output["iterations"] = this->num_iters; }

private:
    mutable gko::size_type num_iters{0};
};


#endif  // GKO_BENCHMARK_UTILS_LOGGERS_HPP_
