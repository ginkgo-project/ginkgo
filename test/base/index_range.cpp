// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/index_range.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class IndexRange : public CommonTestFixture {
public:
    IndexRange() : result_array{exec, 16}, expected_array{ref, 16} {}

    gko::array<int> result_array;
    gko::array<int> expected_array;
};


// nvcc doesn't like device lambdas declared in complex classes, move it out
void run_range_for(std::shared_ptr<gko::EXEC_TYPE> exec,
                   gko::array<int>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto result, auto size) {
            for (auto i : gko::irange<int>{size}) {
                result[i] = i * i;
            }
        },
        1, result_array, static_cast<int>(result_array.get_size()));
}


TEST_F(IndexRange, KernelRunsRangeFor)
{
    auto size = static_cast<int>(result_array.get_size());
    for (int i = 0; i < size; i++) {
        expected_array.get_data()[i] = i * i;
    }

    run_range_for(exec, result_array);

    GKO_ASSERT_ARRAY_EQ(result_array, expected_array);
}
