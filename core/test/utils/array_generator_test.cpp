/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
            500, std::normal_distribution<>(20.0, 5.0),
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
    ASSERT_EQ(this->array.get_num_elems(), 500);
}


TYPED_TEST(ArrayGenerator, OutputHasCorrectAverageAndDeviation)
{
    using std::sqrt;
    using T = typename TestFixture::value_type;

    // check the real part
    this->template check_average_and_deviation<T>(
        this->array.get_const_data(),
        this->array.get_const_data() + this->array.get_num_elems(), 20.0, 5.0,
        [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            this->array.get_const_data(),
            this->array.get_const_data() + this->array.get_num_elems(), 20.0,
            5.0, [](T& val) { return gko::imag(val); });
    }
}


}  // namespace
