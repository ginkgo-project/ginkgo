/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/components/absolute_array.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class AbsoluteArray : public ::testing::Test {
protected:
    using value_type = T;
    using abs_type = gko::remove_complex<T>;
    AbsoluteArray()
        : ref(gko::ReferenceExecutor::create()),
          total_size(6344),
          inplace_expected(ref, total_size),
          outplace_expected(ref, total_size),
          vals(ref, total_size)
    {
        std::fill_n(inplace_expected.get_data(), total_size, T(6453));
        std::fill_n(vals.get_data(), total_size, T(-6453));
        std::fill_n(outplace_expected.get_data(), total_size, abs_type(6453));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::size_type total_size;
    gko::Array<value_type> inplace_expected;
    gko::Array<abs_type> outplace_expected;
    gko::Array<value_type> vals;
};

TYPED_TEST_SUITE(AbsoluteArray, gko::test::ValueTypes);


TYPED_TEST(AbsoluteArray, InplaceEqualsExpected)
{
    using T = typename TestFixture::value_type;

    gko::kernels::reference::components::inplace_absolute_array(
        this->ref, this->vals.get_data(), this->total_size);

    GKO_ASSERT_ARRAY_EQ(this->vals, this->inplace_expected);
}


TYPED_TEST(AbsoluteArray, OutplaceEqualsExpected)
{
    using T = typename TestFixture::value_type;
    using AbsT = typename TestFixture::abs_type;
    gko::Array<AbsT> abs_vals(this->ref, this->total_size);

    gko::kernels::reference::components::outplace_absolute_array(
        this->ref, this->vals.get_const_data(), this->total_size,
        abs_vals.get_data());

    GKO_ASSERT_ARRAY_EQ(abs_vals, this->outplace_expected);
}


}  // namespace
