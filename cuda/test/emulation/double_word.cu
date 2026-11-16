// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/double_word.hpp"

#include <gtest/gtest.h>

#include "cuda/test/emulation/float_float_helper.cuh"


namespace {


using namespace float_float_helper;


// the arithmetic of two words
struct arithmetic_operations {
    static constexpr int num_results = 2 * 13;

    template <typename MW>
    __host__ __device__ static void apply(const MW& x, const MW& y, float a,
                                          double, double* results)
    {
        store(x + y, results);
        store(x - y, results + 2);
        store(x + a, results + 4);
        store(x - a, results + 6);
        store(a - x, results + 8);
        store(x * a, results + 10);
        store(x / a, results + 12);
        store(normalize(x), results + 14);
        store(sqrt(abs(x)), results + 16);
        // NaN for negative values
        store(sqrt(x), results + 18);
        if constexpr (has_product<MW>) {
            store(x * y, results + 20);
            store(x / y, results + 22);
            store(a / x, results + 24);
        }
    }
};


template <typename MW>
class DoubleWord : public fixture<MW> {};

TYPED_TEST_SUITE(DoubleWord, types, TypenameNameGenerator);


TYPED_TEST(DoubleWord, DeviceResultsAreBitIdenticalToHost)
{
    this->template check_device_against_host<arithmetic_operations>();
}


}  // namespace
