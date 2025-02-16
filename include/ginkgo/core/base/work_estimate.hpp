// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_WORK_ESTIMATE_HPP_
#define GKO_PUBLIC_CORE_BASE_WORK_ESTIMATE_HPP_


#include <variant>

#include <ginkgo/core/base/types.hpp>


namespace gko {


/** Work estimate for a kernel that is likely compute-bound. */
struct compute_bound_work_estimate {
    size_type flops;

    friend compute_bound_work_estimate operator+(compute_bound_work_estimate a,
                                                 compute_bound_work_estimate b);

    compute_bound_work_estimate& operator+=(compute_bound_work_estimate other);
};


/** Work estimate for a kernel that is likely memory-bound. */
struct memory_bound_work_estimate {
    size_type bytes_read;
    size_type bytes_written;

    friend memory_bound_work_estimate operator+(memory_bound_work_estimate a,
                                                memory_bound_work_estimate b);

    memory_bound_work_estimate& operator+=(memory_bound_work_estimate other);
};


/** Work estimate based on a custom operation count. */
struct custom_work_estimate {
    std::string operation_count_name;
    size_type operations;

    friend custom_work_estimate operator+(custom_work_estimate a,
                                          custom_work_estimate b);

    custom_work_estimate& operator+=(custom_work_estimate other);
};


using kernel_work_estimate =
    std::variant<compute_bound_work_estimate, memory_bound_work_estimate,
                 custom_work_estimate>;


kernel_work_estimate operator+(kernel_work_estimate a, kernel_work_estimate b);


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_WORK_ESTIMATE_HPP_
