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

#include <ginkgo/core/base/array.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "cuda/test/utils.hpp"


template <typename T>
class Array : public CudaTestFixture {
protected:
    Array() : x(ref, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::array<T>& a)
    {
        ASSERT_EQ(a.get_num_elems(), 2);
        EXPECT_EQ(a.get_data()[0], T{5});
        EXPECT_EQ(a.get_data()[1], T{2});
        EXPECT_EQ(a.get_const_data()[0], T{5});
        EXPECT_EQ(a.get_const_data()[1], T{2});
    }

    gko::array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes, TypenameNameGenerator);


TYPED_TEST(Array, CanCreateTemporaryCloneOnDifferentExecutor)
{
    auto tmp_clone = make_temporary_clone(this->exec, &this->x);

    ASSERT_NE(tmp_clone.get(), &this->x);
    tmp_clone->set_executor(this->ref);
    this->assert_equal_to_original_x(*tmp_clone.get());
}


TYPED_TEST(Array, CanCopyBackTemporaryCloneOnDifferentExecutor)
{
    {
        auto tmp_clone = make_temporary_clone(this->exec, &this->x);
        // change x, so it no longer matches the original x
        // the copy-back will overwrite it again with the correct value
        this->x.get_data()[0] = 0;
    }

    this->assert_equal_to_original_x(this->x);
}


TYPED_TEST(Array, CanBeReduced)
{
    using T = TypeParam;
    auto arr = gko::array<TypeParam>(this->exec, I<T>{4, 6});
    auto out = gko::array<TypeParam>(this->exec, I<T>{2});

    gko::reduce_add(arr, out);

    out.set_executor(this->exec->get_master());
    ASSERT_EQ(out.get_data()[0], T{12});
}


TYPED_TEST(Array, CanBeReduced2)
{
    using T = TypeParam;
    auto arr = gko::array<TypeParam>(this->exec, I<T>{4, 6});

    auto out = gko::reduce_add(arr, T{3});

    ASSERT_EQ(out, T{13});
}
