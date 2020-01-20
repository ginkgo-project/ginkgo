/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/matrix/coo_builder.hpp"


#include <memory>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>


namespace {


template <typename ValueIndexType>
class CooBuilder : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Coo<value_type, index_type>;

    CooBuilder()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};


TYPED_TEST_CASE(CooBuilder, gko::test::ValueIndexTypes);


TYPED_TEST(CooBuilder, ReturnsCorrectArrays)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix::CooBuilder<value_type, index_type> builder{this->mtx.get()};

    auto builder_row_idxs = builder.get_row_idx_array().get_data();
    auto builder_col_idxs = builder.get_col_idx_array().get_data();
    auto builder_values = builder.get_value_array().get_data();
    auto ref_row_idxs = this->mtx->get_row_idxs();
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();

    ASSERT_EQ(builder_row_idxs, ref_row_idxs);
    ASSERT_EQ(builder_col_idxs, ref_col_idxs);
    ASSERT_EQ(builder_values, ref_values);
}


}  // namespace
