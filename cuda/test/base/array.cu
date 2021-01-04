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

#include <ginkgo/core/base/array.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


template <typename T>
class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::Array<T> &a)
    {
        ASSERT_EQ(a.get_num_elems(), 2);
        EXPECT_EQ(a.get_data()[0], T{5});
        EXPECT_EQ(a.get_data()[1], T{2});
        EXPECT_EQ(a.get_const_data()[0], T{5});
        EXPECT_EQ(a.get_const_data()[1], T{2});
    }

    std::shared_ptr<gko::Executor> exec;
    gko::Array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes);


TYPED_TEST(Array, CanCreateTemporaryCloneOnDifferentExecutor)
{
    auto cuda = gko::CudaExecutor::create(0, this->exec);

    auto tmp_clone = make_temporary_clone(cuda, &this->x);

    ASSERT_NE(tmp_clone.get(), &this->x);
    tmp_clone->set_executor(this->exec);
    this->assert_equal_to_original_x(*tmp_clone.get());
}


TYPED_TEST(Array, CanCopyBackTemporaryCloneOnDifferentExecutor)
{
    auto cuda = gko::CudaExecutor::create(0, this->exec);

    {
        auto tmp_clone = make_temporary_clone(cuda, &this->x);
        // change x, so it no longer matches the original x
        // the copy-back will overwrite it again with the correct value
        this->x.get_data()[0] = 0;
    }

    this->assert_equal_to_original_x(this->x);
}
