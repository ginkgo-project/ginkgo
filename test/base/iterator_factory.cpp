// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include "core/base/iterator_factory.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class IteratorFactory : public CommonTestFixture {
public:
    IteratorFactory()
        : key_array{exec, {6, 2, 3, 8, 1, 0, 2}},
          value_array{exec, {9, 5, 7, 2, 4, 7, 2}},
          expected_key_array{ref, {7, 1, 2, 2, 3, 6, 8}},
          expected_value_array{ref, {7, 4, 2, 5, 7, 9, 2}}
    {}

    gko::array<int> key_array;
    gko::array<int> value_array;
    gko::array<int> expected_key_array;
    gko::array<int> expected_value_array;
};


// nvcc doesn't like device lambdas declared in complex classes, move it out
void run_zip_iterator(std::shared_ptr<gko::EXEC_TYPE> exec,
                      gko::array<int>& key_array, gko::array<int>& value_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto keys, auto values, auto size) {
            auto begin = gko::detail::make_zip_iterator(keys, values);
            auto end = begin + size;
            using std::swap;
            for (auto it = begin; it != end; ++it) {
                auto min_it = it;
                for (auto it2 = it; it2 != end; ++it2) {
                    if (*it2 < *min_it) {
                        min_it = it2;
                    }
                }
                swap(*it, *min_it);
            }
            // check structured bindings
            auto [key, value] = *begin;
            static_assert(std::is_same<std::remove_reference_t<decltype(key)>,
                                       int>::value,
                          "incorrect type");
            gko::get<0>(*begin) = value;
        },
        1, key_array, value_array, static_cast<int>(key_array.get_size()));
}


TEST_F(IteratorFactory, KernelRunsZipIterator)
{
    run_zip_iterator(exec, key_array, value_array);

    GKO_ASSERT_ARRAY_EQ(key_array, expected_key_array);
    GKO_ASSERT_ARRAY_EQ(value_array, expected_value_array);
}
