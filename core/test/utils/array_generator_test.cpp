// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/array_generator.hpp"


#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"

namespace {


template <typename T>
class ArrayGenerator : public ::testing::Test {
protected:
    using value_type = T;

    ArrayGenerator() : exec(gko::ReferenceExecutor::create())
    {
        array = gko::test::generate_random_array<T>(
            500, std::normal_distribution<gko::remove_complex<T>>(20.0, 5.0),
            std::default_random_engine(42), exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::array<T> array;

    template <typename InputIterator, typename ValueType, typename Closure>
    ValueType get_nth_moment(int n, ValueType c, InputIterator sample_start,
                             InputIterator sample_end, Closure closure_op)
    {
        using std::pow;
        ValueType res = 0;
        ValueType num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(closure_op(tmp) - c, n);
            num_elems += 1;
        }
        return res / num_elems;
    }

    template <typename ValueType, typename InputIterator, typename Closure>
    void check_average_and_deviation(
        InputIterator sample_start, InputIterator sample_end,
        gko::remove_complex<ValueType> average_ans,
        gko::remove_complex<ValueType> deviation_ans, Closure closure_op)
    {
        auto average =
            this->get_nth_moment(1, gko::zero<gko::remove_complex<ValueType>>(),
                                 sample_start, sample_end, closure_op);
        auto deviation = sqrt(this->get_nth_moment(2, average, sample_start,
                                                   sample_end, closure_op));

        // check that average & deviation is within 10% of the required amount
        ASSERT_NEAR(average, average_ans, average_ans * 0.1);
        ASSERT_NEAR(deviation, deviation_ans, deviation_ans * 0.1);
    }
};

TYPED_TEST_SUITE(ArrayGenerator, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ArrayGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(this->array.get_size(), 500);
}


TYPED_TEST(ArrayGenerator, OutputHasCorrectAverageAndDeviation)
{
    using std::sqrt;
    using T = typename TestFixture::value_type;

    // check the real part
    this->template check_average_and_deviation<T>(
        this->array.get_const_data(),
        this->array.get_const_data() + this->array.get_size(), 20.0, 5.0,
        [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            this->array.get_const_data(),
            this->array.get_const_data() + this->array.get_size(), 20.0, 5.0,
            [](T& val) { return gko::imag(val); });
    }
}


}  // namespace
