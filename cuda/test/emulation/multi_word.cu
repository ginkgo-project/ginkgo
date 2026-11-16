// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/multi_word.hpp"

#include <gtest/gtest.h>

#include "core/emulation/double_word.hpp"
#include "cuda/test/emulation/float_float_helper.cuh"


namespace {


using namespace float_float_helper;


// the operations of the class itself: conversions, negation, abs, comparison
// and evaluation
struct class_operations {
    static constexpr int num_results = 9;

    template <typename MW>
    __host__ __device__ static void apply(const MW& x, const MW& y, float,
                                          double d, double* results)
    {
        store(MW{d}, results);
        store(-x, results + 2);
        store(abs(x), results + 4);
        results[6] = (x == y) | (x != y) << 1 | (x < y) << 2 | (x <= y) << 3 |
                     (x > y) << 4 | (x >= y) << 5;
        results[7] = mw::eval<double>(x);
        results[8] = static_cast<double>(x);
    }
};


template <typename MW>
class MultiWord : public fixture<MW> {};

TYPED_TEST_SUITE(MultiWord, types, TypenameNameGenerator);


TYPED_TEST(MultiWord, DeviceResultsAreBitIdenticalToHost)
{
    this->template check_device_against_host<class_operations>();
}


}  // namespace
