// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/work_estimate.hpp>


namespace gko {


compute_bound_work_estimate operator+(compute_bound_work_estimate a,
                                      compute_bound_work_estimate b)
{
    return {a.flops + b.flops};
}


compute_bound_work_estimate& compute_bound_work_estimate::operator+=(
    compute_bound_work_estimate other)
{
    *this = *this + other;
    return *this;
}


memory_bound_work_estimate operator+(memory_bound_work_estimate a,
                                     memory_bound_work_estimate b)
{
    return {a.bytes_read + b.bytes_read, a.bytes_written + b.bytes_written};
}


memory_bound_work_estimate& memory_bound_work_estimate::operator+=(
    memory_bound_work_estimate other)
{
    *this = *this + other;
    return *this;
}


custom_work_estimate operator+(custom_work_estimate a, custom_work_estimate b)
{
    GKO_ASSERT(a.operation_count_name == b.operation_count_name);
    return {a.operation_count_name, a.operations + b.operations};
}


custom_work_estimate& custom_work_estimate::operator+=(
    custom_work_estimate other)
{
    *this = *this + other;
    return *this;
}


kernel_work_estimate operator+(kernel_work_estimate a, kernel_work_estimate b)
{
    // this fails with std::bad_variant_access if the two estimates are of
    // different types
    return std::visit(
        [b](auto a) -> kernel_work_estimate {
            return a + std::get<decltype(a)>(b);
        },
        a);
}


}  // namespace gko
