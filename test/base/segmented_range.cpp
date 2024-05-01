// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/base/segmented_range.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class SegmentedRange : public CommonTestFixture {
public:
    SegmentedRange()
        : ptrs{exec, {0, 0, 1, 3, 4, 9}},
          values{exec, {1, 2, 3, 4, 5, 6, 7, 8, 9}},
          output{exec, 2 * values.get_size()}
    {}

    gko::array<int> ptrs;
    gko::array<int> values;
    gko::array<int> output;
};


// nvcc doesn't like device lambdas declared in complex classes, move it out
void run_segmented_range(std::shared_ptr<gko::EXEC_TYPE> exec,
                         const gko::array<int>& ptrs,
                         const gko::array<int>& values, gko::array<int>& output)
{
    gko::kernels::EXEC_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto ptrs, auto values, auto output, auto size) {
            gko::segmented_range<int> range{ptrs, size};
            for (auto [row, segment] : range) {
                for (auto nz : segment) {
                    output[nz] = row;
                }
            }
            auto num_values = ptrs[size];
            gko::segmented_value_range<int, const int*> vrange{ptrs, values,
                                                               size};
            for (auto [row, segment] : vrange.enumerated()) {
                for (auto [nz, value] : segment) {
                    output[nz + num_values] = row * 10 + value;
                }
            }
        },
        1, ptrs, values, output, static_cast<int>(ptrs.get_size() - 1));
}


TEST_F(SegmentedRange, KernelRunsSegmentedRange)
{
    gko::array<int> expected{
        ref, {1, 2, 2, 3, 4, 4, 4, 4, 4, 11, 22, 23, 34, 45, 46, 47, 48, 49}};

    run_segmented_range(exec, ptrs, values, output);

    GKO_ASSERT_ARRAY_EQ(output, expected);
}
