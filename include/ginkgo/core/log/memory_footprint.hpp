// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_MEMORY_FOOTPRINT_HPP_
#define GKO_PUBLIC_CORE_LOG_MEMORY_FOOTPRINT_HPP_


#include <iostream>
#include <unordered_map>


#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/log/stack_capture.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace log {


class MemoryFootprint : public StackCapture {
public:
    struct data {
        size_type max_total = 0;
        size_type largest = 0;
        std::vector<size_type> timeline = {};
        std::vector<std::string> max_total_stack = {};
        std::vector<std::string> largest_stack = {};
    };

    void on_allocation_completed(const Executor* exec,
                                 const size_type& num_bytes,
                                 const uintptr& location) const override;

    void on_free_completed(const Executor* exec,
                           const uintptr& location) const override;

    data get_data() const;

private:
    struct intermediate_data {
        size_type max_total = 0;
        size_type largest = 0;
        std::vector<size_type> timeline = {};
        std::vector<int64> max_total_stack = {};
        std::vector<int64> largest_stack = {};

        data create_data(const MemoryFootprint* mt) const;
    };

    mutable intermediate_data im_data_ = {};
    mutable size_type current_total_ = 0;
    mutable std::unordered_map<std::uintptr_t, size_type> allocated_bytes = {};
    size_type threshold_ = sizeof(default_precision) * 5;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_MEMORY_FOOTPRINT_HPP_
