/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/base/array.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes, TypenameNameGenerator);


TYPED_TEST(Array, CanBeFilledWithValue)
{
    this->x.fill(TypeParam{42});

    ASSERT_EQ(this->x.get_num_elems(), 2);
    ASSERT_EQ(this->x.get_data()[0], TypeParam{42});
    ASSERT_EQ(this->x.get_data()[1], TypeParam{42});
    ASSERT_EQ(this->x.get_const_data()[0], TypeParam{42});
    ASSERT_EQ(this->x.get_const_data()[1], TypeParam{42});
}


TYPED_TEST(Array, CanBeReduced)
{
    auto out = gko::array<TypeParam>(this->exec, I<TypeParam>{1});

    gko::reduce_add(this->x, out);

    ASSERT_EQ(out.get_data()[0], TypeParam{8});
}


TYPED_TEST(Array, CanBeReduced2)
{
    auto out = gko::reduce_add(this->x, TypeParam{2});

    ASSERT_EQ(out, TypeParam{9});
}


}  // namespace
