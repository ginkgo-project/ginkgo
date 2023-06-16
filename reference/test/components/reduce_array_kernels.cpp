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

#include "core/components/reduce_array_kernels.hpp"


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class ReduceArray : public ::testing::Test {
protected:
    using value_type = T;
    ReduceArray()
        : ref{gko::ReferenceExecutor::create()},
          out{ref, I<T>{3}},
          vals{ref, I<T>{1, 4, 6}}
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::array<value_type> out;
    gko::array<value_type> vals;
};

TYPED_TEST_SUITE(ReduceArray, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(ReduceArray, KernelWorks)
{
    using T = typename TestFixture::value_type;

    gko::kernels::reference::components::reduce_add_array(this->ref, this->vals,
                                                          this->out);

    ASSERT_EQ(this->out.get_data()[0], T{14});
}


TYPED_TEST(ReduceArray, CoreWorks)
{
    using T = typename TestFixture::value_type;

    gko::reduce_add(this->vals, this->out);

    ASSERT_EQ(this->out.get_data()[0], T{14});
}


TYPED_TEST(ReduceArray, CoreWorks2)
{
    using T = typename TestFixture::value_type;

    auto result = gko::reduce_add(this->vals, T{1});

    ASSERT_EQ(result, T{12});
}


}  // namespace
